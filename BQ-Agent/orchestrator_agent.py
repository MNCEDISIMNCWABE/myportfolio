from typing import Optional, Union, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import re
import os

# Import agents
from bq_agent import bigquery_agent
from spanner_agent import spanner_agent

# Initialize Vertex AI LLM for the orchestrator
orchestrator_llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0,
    max_output_tokens=2048,
    project="hallowed-span-459710-s1",
    location="us-central1"
)


# Define the routing prompt
ROUTING_PROMPT = """You are an intelligent query router that determines which database system should handle a user's question.

Available Systems:
1. BigQuery - Handles user engagement analytics and time-series data
   - Tables: user-engagement, dim_date, fact_user_metrics
   - Example questions: 
     * "Show me monthly active users"
     * "What's the average engagement time by country?"
     * "Compare Q1 vs Q2 retention rates"

2. Spanner - Handles transactional and operational data
   - Tables: users, countries
   - Example questions:
     * "Show me users with high transaction counts"
     * "List countries by average transaction value"
     * "Find users who haven't transacted in 30 days"

User Question: {question}

Analyze the question and determine which system is most appropriate. Respond ONLY with either "bigquery" or "spanner".
"""

def route_question(question: str) -> str:
    """Determine which agent should handle the question"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTING_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | orchestrator_llm | StrOutputParser()
    destination = chain.invoke({"question": question})
    
    # Clean and validate the response
    destination = destination.strip().lower()
    if destination not in ["bigquery", "spanner"]:
        # Default to BigQuery if routing fails
        print(f"Warning: Invalid routing destination '{destination}', defaulting to BigQuery")
        return "bigquery"
    return destination

def moosa_orchestrator(question: str, max_attempts: int = 3) -> Union[pd.DataFrame, str]:
    """Main orchestrator function that routes questions to appropriate agents"""
    print(f"\n{'='*50}\nProcessing: {question}\n{'='*50}")
    
    #  Route the question
    destination = route_question(question)
    print(f"Routing to: {destination}")
    
    # Execute with the appropriate agent
    if destination == "bigquery":
        result = bigquery_agent(question, max_attempts)
    else:
        result = spanner_agent(question, max_attempts)
    
    # Return results
    print("\nFinal Result:")
    if isinstance(result, pd.DataFrame):
        print(result.head())
    else:
        print(result)
    
    return result

# Test cases
test_questions = [
    "Give me a list of top 5 users by engagement time", # expected to route to BQ agent
    "Show me the top 5 countries by transaction count" # expected to route to Spanner agent
]
for question in test_questions:
    moosa_orchestrator(question)