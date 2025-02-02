import streamlit as st
import os
import tempfile
import gc
import base64
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ConfigDict
import xml.etree.ElementTree as ET
import requests

from crewai import Agent, Crew, Process, Task, LLM
from custom_tool import FireCrawlWebSearchTool, ArxivSearchTool

@st.cache_resource
def load_llm():
    llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    )
    return llm

#  Define Agent Tasks and Goals
def create_agents_and_tasks():
    """Creates a Crew with arXiv and web search tools."""
    web_search_tool = FireCrawlWebSearchTool()
    arxiv_tool = ArxivSearchTool()

    retriever_agent = Agent(
        role="Academic Research Retrieval Specialist",
        goal=(
            "Analyze the user query '{query}' to identify core technical concepts and domain. "
            "Retrieve papers that explicitly mention the query terms in titles/abstracts. "
            "Prioritize papers from the last 6 months with clear empirical validation. "
            "Verify paper relevance to '{query}' before including results."
        ),
        backstory=(
            "You're a dedicated research assistant with deep knowledge in data science and machine learning "
            "and academic literature. You specialize in finding cutting-edge research "
            "on topics like '{query}' in data science and machine learning."
        ),
        verbose=True,
        tools=[arxiv_tool, web_search_tool],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role="Research Synthesis Expert",
        goal=(
            "Analyze papers for direct relevance to '{query}'. "
            "Exclude papers that only tangentially mention the topic. "
            "Highlight papers with experimental results and clear methodology. "
            "Verify URLs point to actual PDFs of the mentioned papers."
        ),
        backstory=(
            "You're a research scientist with expertise in distilling complex academic "
            "findings into actionable insights related to topics like '{query}'. "
            "Your summaries help practitioners stay updated with the latest advancements."
        ),
        verbose=True,
        llm=load_llm()
    )

    retrieval_task = Task(
        description=(
            "Search for recent research papers on '{query}' from arXiv "
            "and supplementary web sources. Focus on papers published in the last 6 months."
        ),
        expected_output=(
            "A collection of relevant paper metadata for '{query}' including titles, authors, "
            "abstracts, publication dates, and arXiv URLs."
        ),
        agent=retriever_agent,
        context=[], 
        inputs={"query": "The user's research query"}
    )

    response_task = Task(
        description=(
            "Synthesize the research findings about '{query}' into a comprehensive summary. "
            "Include key technical details and practical implications."
        ),
        expected_output=(
            "A well-structured summary about '{query}' with:\n"
            "1. Key papers and their contributions\n"
            "2. Common methodologies and innovations\n"
            "3. Potential applications and future directions\n"
            "4. Links to the full papers"
        ),
        agent=response_synthesizer_agent,
        context=[retrieval_task],
        inputs={"query": "The user's research query"}
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew


if "messages" not in st.session_state:
    st.session_state.messages = []

if "crew" not in st.session_state:
    st.session_state.crew = None

def reset_chat():
    st.session_state.messages = []
    gc.collect()

# UI Interface
st.markdown("""
    # Research Assistant Data Science & ML
    *Search and analyse research papers in any AI/ML and Data Science topic.*  
    *Powered by arXiv & CrewAI.*
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about recent research in data science and machine learning...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Searching arXiv and analyzing papers..."):
            inputs = {"query": prompt}
            result = st.session_state.crew.kickoff(inputs=inputs).raw
        
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line + ('\n' if i < len(lines)-1 else '')
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.1)
        
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": result})
