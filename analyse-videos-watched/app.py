import streamlit as st
import os
import tempfile
import gc
import base64
import time
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from youtube_api import authenticate_youtube, get_liked_videos, get_watched_videos
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables from .env file
load_dotenv()

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool

# Create transcripts directory if it doesn't exist
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# Initialize session state variables for Streamlit
if "all_transcripts" not in st.session_state:
    st.session_state.all_transcripts = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "response" not in st.session_state:
    st.session_state.response = None

if "crew" not in st.session_state:
    st.session_state.crew = None

@st.cache_resource
def load_llm():
    """Initialize and cache the LLM instance for better performance."""
    llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    )
    return llm

def save_transcript(video_id, transcript_text):
    """
    Save a video transcript to a file and return the file path.
    Each transcript is saved with the video ID as the filename.
    """
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    return transcript_path

def get_video_transcript(video_id):
    """
    Fetch transcript for a video using youtube_transcript_api.
    Returns None silently if transcript cannot be retrieved.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception:
        return None

def validate_transcripts():
    """
    Validate that transcripts exist and are readable.
    Returns True if validation passes, raises appropriate exceptions if not.
    """
    if not TRANSCRIPTS_DIR.exists():
        raise FileNotFoundError("Transcripts directory not found")
    
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    if not transcript_files:
        raise FileNotFoundError("No transcript files found in directory")
    
    # Validate each transcript file is readable
    for file in transcript_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                f.read()
        except Exception as e:
            raise IOError(f"Error reading transcript file {file}: {str(e)}")
    
    return True

def create_agents_and_tasks(file_paths):
    """
    Creates a Crew of agents for analyzing YouTube video transcripts.
    Includes enhanced transcript handling and detailed task descriptions.
    Returns a properly configured Crew instance with correct verbosity settings.
    """
    # Load agent and task configurations
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Create an enhanced tool for reading transcript files
    transcript_reader = FileReadTool(
        name="TranscriptReader",
        description="""
        Read and parse YouTube video transcripts from files.
        Each file contains the full transcript of a single video.
        Use this tool to access the content of transcript files in the transcripts directory.
        Files are named as [video_id].txt
        """,
        directory=str(TRANSCRIPTS_DIR)
    )

    # Create the analysis agent with improved configuration
    analysis_agent = Agent(
        role="YouTube Video Analyst",
        goal=f"Analyze YouTube video transcripts located in {file_paths} to identify trends, patterns, and insights",
        backstory="""
        You are an expert in analyzing video content and extracting meaningful insights.
        Your task is to thoroughly analyze the transcripts stored in the transcripts directory.
        Use the TranscriptReader tool to access each transcript file listed in metadata.json.
        """,
        verbose=True,  # Using boolean instead of integer
        tools=[transcript_reader],
        llm=load_llm(),
        allow_delegation=False
    )

    # Create the response synthesizer agent
    response_synthesizer_agent = Agent(
        role="Response Synthesizer",
        goal="Create a comprehensive summary report based on the video analysis",
        backstory="""
        You are skilled at synthesizing complex information into clear and concise summaries.
        Your role is to take the analysis of multiple video transcripts and create a cohesive report
        that highlights key themes and patterns across all videos.
        """,
        verbose=True, 
        tools=[transcript_reader],
        llm=load_llm(),
        allow_delegation=False
    )

    # Create detailed tasks with explicit instructions
    analysis_task = Task(
        description=f"""
        Analyze the transcripts of the provided YouTube videos located in {file_paths}.
        Steps:
        1. Use the TranscriptReader tool to read each transcript file
        2. For each transcript, identify:
           - Main topics and themes
           - Key points and arguments
           - Speaking style and presentation patterns
        3. Look for patterns and commonalities across all videos
        4. Document your findings in detail
        """,
        expected_output="""
        A detailed analysis document containing:
        - Summary of key themes across all videos
        - Pattern analysis of content and presentation
        - Notable insights and observations
        - Supporting evidence from specific transcripts
        """,
        agent=analysis_agent
    )

    response_task = Task(
        description="""
        Create a comprehensive summary report based on the analysis of all video transcripts in {file_paths}.
        The report should:
        1. Synthesize the key findings from the analysis
        2. Highlight the most significant patterns and trends
        3. Present insights in a clear, organized manner
        4. Include specific examples to support conclusions
        """,
        expected_output="""
        A well-structured report containing:
        - Executive summary of findings
        - Major themes and patterns identified
        - Supporting evidence and examples
        - Conclusions and insights
        """,
        agent=response_synthesizer_agent
    )

    # Create and return the crew with proper verbose setting
    return Crew(
        agents=[analysis_agent, response_synthesizer_agent],
        tasks=[analysis_task, response_task],
        process=Process.sequential,
        verbose=True 
    )

def start_analysis():
    """
    Main function to handle the complete video analysis process.
    Includes improved error handling and transcript validation.
    """
    try:
        # Step 1: Authenticate with YouTube
        with st.spinner('Authenticating with YouTube...'):
            youtube = authenticate_youtube()

        # Step 2: Fetch and process videos
        with st.spinner('Fetching videos...'):
            status_container = st.empty()
            status_container.info("Fetching videos from your YouTube account...")

            # Clear previous transcripts
            st.session_state.all_transcripts = {}
            for file in TRANSCRIPTS_DIR.glob("*.txt"):
                file.unlink()

            # Get videos based on selected source
            if st.session_state.video_source == "watched":
                status_container.warning("Watch history is not available. Switching to liked videos...")
                videos = get_liked_videos(youtube, max_results=50)
            elif st.session_state.video_source == "liked":
                videos = get_liked_videos(youtube, max_results=50)
            else:
                status_container.error("Invalid video source selected.")
                return

            if not videos:
                status_container.error("No videos found.")
                return

            # Step 3: Process videos and get transcripts
            status_container.info(f"Found {len(videos)} videos. Fetching transcripts...")
            progress_bar = st.progress(0)
            successful_transcripts = 0
            
            for idx, video in enumerate(videos):
                try:
                    video_id = video['id']
                    transcript = get_video_transcript(video_id)
                    if transcript:
                        transcript_path = save_transcript(video_id, transcript)
                        st.session_state.all_transcripts[video_id] = str(transcript_path)
                        successful_transcripts += 1
                    progress_bar.progress((idx + 1) / len(videos))
                except Exception:
                    continue

            if successful_transcripts == 0:
                status_container.error("No transcripts could be retrieved. Please try different videos.")
                return

            status_container.success(f"Successfully processed {successful_transcripts} out of {len(videos)} videos.")

            # Step 4: Create metadata for agents
            metadata = {
                "transcript_count": successful_transcripts,
                "transcript_files": list(st.session_state.all_transcripts.values())
            }
            with open(TRANSCRIPTS_DIR / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Step 5: Validate transcripts before analysis
            validate_transcripts()

            # Step 6: Initialize crew and start analysis
            file_paths = ", ".join(st.session_state.all_transcripts.values())
            if file_paths:
                status_container = st.empty()
                with st.spinner('The agent is analyzing the videos... This may take a moment.'):
                    # Create crew with file_paths
                    st.session_state.crew = create_agents_and_tasks(file_paths)
                    crew_output = st.session_state.crew.kickoff(inputs={"file_paths": file_paths})
                    # Convert CrewOutput to string
                    st.session_state.response = str(crew_output)

    except FileNotFoundError as e:
        st.error(f"Transcript access error: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        raise e

# UI Setup
st.markdown("# YouTube Trend Analysis powered by CrewAI", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("YouTube Account")
    
    # Video source selection
    st.subheader("Video Source")
    video_source = st.radio(
        "Choose video source",
        ["watched", "liked"],
        key="video_source"
    )

    st.divider()
    
    # Date range selection
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
        st.session_state.start_date = start_date.strftime("%Y-%m-%d")
    with col2:
        end_date = st.date_input("End Date")
        st.session_state.end_date = end_date.strftime("%Y-%m-%d")

    st.divider()
    # Analysis trigger button
    st.button("Start Analysis 🚀", type="primary", on_click=start_analysis)

# Main content area for displaying results
if st.session_state.response:
    with st.spinner('Generating content...'):
        try:
            st.markdown("### Generated Analysis")
            st.markdown(st.session_state.response)
            
            # Download button for analysis
            st.download_button(
                label="Download Analysis",
                data=st.session_state.response,
                file_name="youtube_trend_analysis.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"Error displaying analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with CrewAI, Bright Data, and Streamlit")