import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List

import json
import requests
from langchain.tools import tool

MODEL_NAME = "ollama/llama3.2"  # The name of the LLM model to be used for inference, in this case, Ollama's LLaMA 3.2.
BASE_URL = "http://127.0.0.1:11434"  # The base URL for the LLM server, running locally on port 11434.
PROVIDER = "ollama"  # Specifies the LLM provider; here, it indicates Ollama as the provider.
SERPER_API_KEY = "44bcc750e7c15b2ca21386fb93fd3acfbbd81111"  # API key for the Serper API, used to perform internet searches.

class SearchTools:
    @tool("Search internet")
    def search_internet(query: str) -> List[Dict[str, str]]:
        """Search the internet for a given topic and return relevant results."""
        print(f"[DEBUG] Received query: {query}")  # Debug log to inspect input
        return SearchTools.search(query)

    @staticmethod
    def search(query: str, n_results=5) -> List[Dict[str, str]]:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': SERPER_API_KEY,  
            'content-type': 'application/json',
        }
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            results = response.json().get('organic', [])
            formatted_results = []
            for result in results[:n_results]:
                formatted_results.append({
                    "title": result.get("title", "No Title Available"),
                    "link": result.get("link", "No Link Available"),
                    "snippet": result.get("snippet", "No Snippet Available")
                })
            return formatted_results
        except requests.exceptions.RequestException as e:
            return [{"error": f"Error during search: {e}"}]

# Define LLM
llm = LLM(model=MODEL_NAME, base_url=BASE_URL, provider=PROVIDER)

# Define the agents with roles and goals
def create_planner(from_location, destination):
    return Agent(
        role='Travel Research Specialist',
        goal=f'Plan a vacation from {from_location} to {destination} for {num_people} people.',
        backstory=f"""You are an expert travel planner specializing in budget-friendly vacations.
        Your role is to find the best options for travel, accommodation, and activities from {from_location} to {destination}.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        tools=[SearchTools.search_internet],
        llm=llm,
    )

def create_writer(from_location, destination):
    return Agent(
        role='Vacation Itinerary Creator',
        goal=f"Create a detailed vacation itinerary from {from_location} to {destination}.",
        backstory=f"""You are a travel content creator who crafts engaging and informative itineraries.
        Your job is to use the data provided by the Travel Planner to create a clear and structured itinerary for a trip from {from_location} to {destination}.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        max_iter=5,
        tools=[SearchTools.search_internet],
    )

# Define tasks dynamically
def create_planning_task(planner, from_location, destination):
    return Task(
        description=f"""Find the best vacation options from {from_location} to {destination} for {num_people} people.

        STEPS TO COMPLETE THE TASK:
        1. Search for affordable flights from {from_location} to {destination} and provide 3 options with links and costs.
        2. Find 4 budget-friendly accommodations in {destination} for {duration} days, with links and total costs.
        3. Identify affordable activities and excursions in {destination} for {num_people} people, with estimated costs.
        4. Calculate total costs (flights, accommodation, and activities).""",
        expected_output=f"""A detailed breakdown of costs for a {destination} vacation for {num_people} people from {from_location}, including:
        - Links to 3 flight options with costs for return tickets from {from_location} to {destination}.
        - Links to 4 accommodation options with {duration}-day total costs in {destination}.
        - Activities with individual and total costs.
        - Final cost breakdown.""",
        agent=planner,
        max_iter=5
    )

def create_itinerary_task(writer, from_location, destination, duration):
    return Task(
        description=f"""Create a detailed {duration}-day vacation itinerary from {from_location} to {destination}.

        Itinerary Requirements:
        1. Provide a day-by-day breakdown of activities.
        2. Include details on accommodations, transport, meals, and excursions with costs and links.
        3. Ensure the itinerary is engaging and provides a comprehensive travel experience from {from_location} to {destination}.""",
        expected_output=f"""A complete {duration}-day vacation itinerary from {from_location} to {destination}, including:
        - Day-by-day schedule
        - Activities and excursions
        - Accommodation and meal details with costs
        - Transport details between {from_location} and {destination} with costs.""",
        agent=writer,
        max_iter=5
    )

# Streamlit App
st.title("AI-Powered Travel Planner")
st.write("Plan your vacation with AI agents!")

# Inputs
from_location = st.text_input("Departure Location", "Johannesburg, South Africa")
destination = st.text_input("Destination", "Mozambique")
num_people = st.number_input("Number of Travelers", min_value=1, max_value=10, value=2)
duration = st.slider("Travel Duration (days)", min_value=3, max_value=14, value=7)

# Button to Trigger Agents
if st.button("Generate Travel Plan"):
    # Create dynamic agents and tasks
    planner = create_planner(from_location, destination)
    writer = create_writer(from_location, destination)
    
    planning_task = create_planning_task(planner, from_location, destination)
    itinerary_task = create_itinerary_task(writer, from_location, destination, duration)
    
    # Execute the Crew
    with st.spinner("Generating your travel plan..."):
        crew = Crew(
            agents=[planner, writer],
            tasks=[planning_task, itinerary_task],
            verbose=False,
            process='sequential',
        )
        result = crew.kickoff()

    # Log the result structure for debugging
    st.write("Result:")
    st.json(result)

    # Safely access the task results
    try:
        planning_result = result.get("tasks", {}).get("planning_task", {}).get("output", "No result found for Planning Task")
        itinerary_result = result.get("tasks", {}).get("itinerary_task", {}).get("output", "No result found for Itinerary Task")
    except AttributeError as e:
        st.write(f"Error accessing task results: {e}")
        planning_result = "Error occurred while fetching planning result"
        itinerary_result = "Error occurred while fetching itinerary result"


    # Display Results in Streamlit
    st.header("Travel Plan Breakdown")
    
    # Format the planning result as a readable text
    if isinstance(planning_result, str):
        st.write(planning_result)
    elif isinstance(planning_result, dict):
        # Convert dictionary to formatted text if needed
        formatted_plan = f"""
**Travel Plan from {from_location} to {destination}**

**Flights:**
{chr(10).join([f"{i+1}. " + line for i, line in enumerate(planning_result.get('flights', ['No flight information available']))])}

**Accommodations:**
{chr(10).join([f"{i+1}. " + line for i, line in enumerate(planning_result.get('accommodations', ['No accommodation information available']))])}

**Activities:**
{chr(10).join([f"{i+1}. " + line for i, line in enumerate(planning_result.get('activities', ['No activity information available']))])}

**Total Budget Breakdown:**
{planning_result.get('budget_summary', 'No budget summary available')}
        """
        st.markdown(formatted_plan)
    else:
        st.write("Unexpected result format")

    # Format the itinerary result as a readable text
    st.header("Travel Itinerary")
    
    if isinstance(itinerary_result, str):
        st.write(itinerary_result)
    elif isinstance(itinerary_result, dict):
        # Convert dictionary to formatted text if needed
        formatted_itinerary = f"""
**Detailed Itinerary for {duration} Days in {destination}**

{chr(10).join([f"**Day {i+1}:** {day_details}" for i, day_details in enumerate(itinerary_result.get('daily_schedule', ['No daily schedule available']))])}

**Additional Details:**
- Accommodations: {itinerary_result.get('accommodation', 'No accommodation details')}
- Meals: {itinerary_result.get('meals', 'No meal information')}
- Transport: {itinerary_result.get('transport', 'No transport details')}
        """
        st.markdown(formatted_itinerary)
    else:
        st.write("Unexpected itinerary format")