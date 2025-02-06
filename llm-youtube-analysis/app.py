
import streamlit as st
import os
import tempfile
import gc
import base64
import time
import yaml

from tqdm import tqdm
from scraper import *
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool

docs_tool = FileReadTool()

# Create transcripts directory if it doesn't exist
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

bright_data_api_key = "23b78375c0dcb16ea17bd05bafe20f74e8bc966640008f86e3eeedf3455aac97"

@st.cache_resource
def load_llm():

    llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    )
    return llm

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks():
    """Creates a Crew for analysis of the channel scrapped output"""

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    analysis_agent = Agent(
        role=config["agents"][0]["role"],
        goal=config["agents"][0]["goal"],
        backstory=config["agents"][0]["backstory"],
        verbose=True,
        tools=[docs_tool],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role=config["agents"][1]["role"],
        goal=config["agents"][1]["goal"],
        backstory=config["agents"][1]["backstory"],
        verbose=True,
        llm=load_llm()
    )

    analysis_task = Task(
        description=config["tasks"][0]["description"],
        expected_output=config["tasks"][0]["expected_output"],
        agent=analysis_agent
    )

    response_task = Task(
        description=config["tasks"][1]["description"],
        expected_output=config["tasks"][1]["expected_output"],
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[analysis_agent, response_synthesizer_agent],
        tasks=[analysis_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ===========================
#   Streamlit Setup
# ===========================

st.markdown(
    "# YouTube Trend Analysis powered by CrewAI", 
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "response" not in st.session_state:
    st.session_state.response = None

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def start_analysis():
    with st.spinner('Scraping videos... This may take a moment.'):
        status_container = st.empty()
        status_container.info("Extracting videos from the channels...")
        
        # Filter out empty channel URLs
        valid_channels = [channel for channel in st.session_state.youtube_channels if channel.strip()]
        
        # Debug: Print channel information
        print(f"Valid Channels: {valid_channels}")
        print(f"Start Date: {st.session_state.start_date}")
        print(f"End Date: {st.session_state.end_date}")
        
        # Check if there are any valid channels
        if not valid_channels:
            status_container.error("Please enter at least one valid YouTube channel URL.")
            return
        
        channel_snapshot_id = trigger_scraping_channels(
            bright_data_api_key, 
            valid_channels, 
            10, 
            st.session_state.start_date, 
            st.session_state.end_date, 
            "Latest", 
            ""
        )
        
        # Debug: Print the entire returned object
        print("Channel Snapshot ID Response:")
        print(channel_snapshot_id)
        
        # Add more robust error handling
        if not channel_snapshot_id:
            status_container.error("Failed to trigger scraping. Check your Bright Data configuration.")
            return
        
        # Check if the response contains an error
        if isinstance(channel_snapshot_id, dict) and 'error' in channel_snapshot_id:
            status_container.error(f"Scraping error: {channel_snapshot_id.get('error', 'Unknown error')}")
            return

        # Extract snapshot ID safely
        if isinstance(channel_snapshot_id, dict) and 'snapshot_id' in channel_snapshot_id:
            snapshot_id = channel_snapshot_id['snapshot_id']
        else:
            status_container.error("Failed to extract snapshot ID. Unexpected API response structure.")
            return

        # Continue with progress tracking using the extracted snapshot ID
        status = get_progress(bright_data_api_key, snapshot_id)

        while status['status'] != "ready":
            status_container.info(f"Current status: {status['status']}")
            time.sleep(10)
            status = get_progress(bright_data_api_key, snapshot_id)

            if status['status'] == "failed":
                status_container.error(f"Scraping failed: {status}")
                return
        
        if status['status'] == "ready":
            status_container.success("Scraping completed successfully!")
            
            channel_scrapped_output = get_output(bright_data_api_key, snapshot_id, format="json")

            # Filter valid videos that have a URL
            valid_videos = []
            if channel_scrapped_output and isinstance(channel_scrapped_output, list):
                for page in channel_scrapped_output:
                    if isinstance(page, list):
                        for video in page:
                            if isinstance(video, dict) and 'url' in video:
                                valid_videos.append(video)

            st.markdown("## YouTube Videos Extracted")
            
            if valid_videos:
                # Create a container for the carousel
                carousel_container = st.container()
                videos_per_row = 3
                num_videos = len(valid_videos)

                with carousel_container:
                    num_rows = (num_videos + videos_per_row - 1) // videos_per_row
                    
                    for row in range(num_rows):
                        cols = st.columns(videos_per_row)
                        
                        for col_idx in range(videos_per_row):
                            video_idx = row * videos_per_row + col_idx
                            
                            if video_idx < num_videos:
                                with cols[col_idx]:
                                    video = valid_videos[video_idx]
                                    st.video(video['url'])
                                    if 'title' in video:
                                        st.caption(video['title'][:50] + "...")
            else:
                st.warning("No valid videos found in the scraped data")

            # NEW CODE TO ADD
            status_container.info("Processing transcripts...")
            st.session_state.all_files = []

            # Use the filtered valid_videos list instead of original output
            for video in tqdm(valid_videos):
                try:
                    # Verify required fields exist
                    if 'shortcode' not in video or 'formatted_transcript' not in video:
                        continue
                        
                    youtube_video_id = video['shortcode']
                    file_path = os.path.join(TRANSCRIPTS_DIR, f"{youtube_video_id}.txt")
                    
                    # Write transcript to file
                    with open(file_path, "w") as f:
                        for segment in video['formatted_transcript']:
                            text = segment.get('text', '')
                            start = segment.get('start_time', 0)
                            end = segment.get('end_time', 0)
                            f.write(f"({start:.2f}-{end:.2f}): {text}\n")
                        
                    st.session_state.all_files.append(file_path)
                    
                except Exception as e:
                    print(f"Error processing video {video.get('url', 'unknown')}: {str(e)}")
                    continue

            if not st.session_state.all_files:
                status_container.error("No valid transcripts found")
                return

            status_container.success(f"Processed {len(st.session_state.all_files)} transcripts")

        else:
            status_container.error(f"Scraping failed with status: {status}")

    if status['status'] == "ready":

        status_container = st.empty()
        with st.spinner('The agent is analyzing the videos... This may take a moment.'):
            # create crew
            st.session_state.crew = create_agents_and_tasks()
            st.session_state.response = st.session_state.crew.kickoff(inputs={"file_paths": ", ".join(st.session_state.all_files)})       


# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("YouTube Channels")
    
    # Initialize the channels list in session state if it doesn't exist
    if "youtube_channels" not in st.session_state:
        st.session_state.youtube_channels = [""]  # Start with one empty field
    
    # Function to add new channel field
    def add_channel_field():
        st.session_state.youtube_channels.append("")
    
    # Create input fields for each channel
    for i, channel in enumerate(st.session_state.youtube_channels):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.session_state.youtube_channels[i] = st.text_input(
                "Channel URL",
                value=channel,
                key=f"channel_{i}",
                label_visibility="collapsed"
            )
        # Show remove button for all except the first field
        with col2:
            if i > 0:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.youtube_channels.pop(i)
                    st.rerun()
    
    # Add channel button
    st.button("Add Channel ➕", on_click=add_channel_field)
    
    st.divider()
    
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
        st.session_state.start_date = start_date
        # store date as string
        st.session_state.start_date = start_date.strftime("%Y-%m-%d")
    with col2:
        end_date = st.date_input("End Date")
        st.session_state.end_date = end_date
        st.session_state.end_date = end_date.strftime("%Y-%m-%d")

    st.divider()
    st.button("Start Analysis 🚀", type="primary", on_click=start_analysis)
    # st.button("Clear Chat", on_click=reset_chat)

# ===========================
#   Main Chat Interface
# ===========================

# Main content area
if st.session_state.response:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = st.session_state.response
            st.markdown("### Generated Analysis")
            st.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"youtube_trend_analysis.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with CrewAI, Bright Data and Streamlit")
