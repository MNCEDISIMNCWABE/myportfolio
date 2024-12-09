import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from typing import List, Dict
import json
import requests
from dotenv import load_dotenv

import os
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List

import json
import requests
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

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
            'X-API-KEY': "44bcc750e7c15b2ca21386fb93fd3acfbbd81111",  
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
llm = LLM(model="ollama/llama3.2", base_url="http://127.0.0.1:11434", provider="ollama")

# Define the agents with roles and goals
planner = Agent(
    role='Travel Planner',
    goal='Plan a vacation to Mozambique for 2 people within a budget of R14,000.',
    backstory="""You are an expert travel planner specializing in budget-friendly vacations.
    Your role is to find the best options for travel, accommodation, and activities while staying within budget.""",
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    tools=[SearchTools.search_internet],
    llm=llm,
)

writer = Agent(
    role='Vacation Itinerary Creator',
    goal="Create a detailed vacation itinerary for a trip to Mozambique.",
    backstory="""You are a travel content creator who crafts engaging and informative itineraries.
    Your job is to use the data provided by the Travel Planner to create a clear and structured itinerary.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    max_iter=5,
    tools=[SearchTools.search_internet],
)

# Define tasks
planning_task = Task(
    description="""Find the best vacation options for Mozambique for 2 people within a budget of R14,000.

    STEPS TO COMPLETE THE TASK:
    1. Search for affordable flights from South Africa to Mozambique and provide 3 options with links and costs.
    2. Find 4 budget-friendly accommodations in Mozambique for one week, with links and total costs.
    3. Identify affordable activities and excursions in Mozambique for two people, with estimated costs.
    4. Calculate total costs (flights, accommodation, and activities) to ensure they fit within the R14,000 budget.""",
    expected_output="""A detailed breakdown of costs for a Mozambique vacation for 2 people, including:
    - Links to 3 flight options with costs for return tickets.
    - Links to 4 accommodation options with 7-day total costs.
    - Activities with individual and total costs.
    - Final cost breakdown ensuring it fits within the R14,000 budget.""",
    agent=planner,
    max_iter=5,
)

itinerary_task = Task(
    description="""Create a detailed one-week vacation itinerary for Mozambique.

    Itinerary Requirements:
    1. Provide a day-by-day breakdown of activities.
    2. Include details on accommodations, transport, meals, and excursions.
    3. Ensure the itinerary is engaging and fits within the provided budget.""",
    expected_output="""A complete one-week vacation itinerary for Mozambique, including:
    - Day-by-day schedule
    - Activities and excursions
    - Accommodation and meal details.""",
    agent=writer,
)

# Streamlit App
st.title("AI-Powered Travel Planner")
st.write("Plan your dream vacation with AI agents!")

# Inputs
destination = st.text_input("Destination", "Mozambique")
budget = st.number_input("Budget (in R)", min_value=5000, max_value=50000, value=14000)
num_people = st.number_input("Number of Travelers", min_value=1, max_value=10, value=2)
duration = st.slider("Travel Duration (days)", min_value=3, max_value=14, value=7)

# Button to Trigger Agents
if st.button("Generate Travel Plan"):
    # Update tasks with inputs
    planning_task.description = f"""
    Plan a vacation to {destination} for {num_people} people within a budget of R{budget}.
    Include flights, accommodations, activities, and ensure all costs fit within the budget."""
    itinerary_task.description = f"""
    Create a detailed {duration}-day travel itinerary for {destination}.
    Include activities, transport, meals, and accommodation."""
    
    # Execute the Crew
    with st.spinner("Generating your travel plan..."):
        crew = Crew(
            agents=[planner, writer],
            tasks=[planning_task, itinerary_task],
            verbose=False,
            process='sequential',
        )
        result = crew.kickoff()

    # Safely access the result and handle possible errors
    try:
        # Inspect and debug the result structure
        st.write("Debug: Raw result object:")
        st.json(result.__dict__)  # This displays the raw structure of the result object

        # Access task results based on the structure
        planning_result = getattr(result, "planning_task_output", "No result found for Planning Task")
        itinerary_result = getattr(result, "itinerary_task_output", "No result found for Itinerary Task")
    except AttributeError as e:
        planning_result = f"Error accessing planning task result: {e}"
        itinerary_result = f"Error accessing itinerary task result: {e}"

    # Display Results in Streamlit
    st.header("Travel Plan Breakdown")
    if isinstance(planning_result, dict):
        st.json(planning_result)
    else:
        st.write(planning_result)

    st.header("Travel Itinerary")
    if isinstance(itinerary_result, dict):
        st.json(itinerary_result)
    else:
        st.write(itinerary_result)
