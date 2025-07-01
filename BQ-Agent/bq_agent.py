from langchain_google_community import BigQueryLoader
from google.cloud import bigquery
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import Graph, END
from typing import TypedDict, Annotated, Union, Optional
import pandas as pd
import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/mncedisimncwabe/Downloads/Everything else/hallowed-span-459710-s1-c41d79c9b56b.json"

# Initialize BigQuery client
client = bigquery.Client()

def get_all_schemas(dataset_id):
    """Fetch schemas for all tables in the dataset"""
    tables = client.list_tables(dataset_id)
    schemas = {}
    
    for table in tables:
        try:
            table_ref = client.get_table(table)
            schemas[table.table_id] = [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "description": field.description or "No description"
                }
                for field in table_ref.schema
            ]
        except Exception as e:
            print(f"Error fetching schema for {table.table_id}: {str(e)}")
    
    return schemas

# Get all schemas
schemas = get_all_schemas("test_clustering")

# Update these
BQ_PROJECT = "hallowed-span-459710-s1" 
BQ_DATASET = "test_clustering"   
LOCATION =  "us-central1"  
TARGET_TABLES = {                      
    "user-engagement": "User engagement data",
    "dim_date": "Date dimension table",
    "fact_user_metrics": "Aggregated user metrics"
}

# Initialize Vertex AI LLM
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0, # Controls the randomness/creativity of the AI's output. With higher values, it might sometimes add unnecessary clauses.
    max_output_tokens=2048, # Set the maximum length of the generated response in tokens (≈ words/word parts). 1,500-2,000 words
    project=BQ_PROJECT,
    location=LOCATION
)

def format_schema_for_prompt(schema_data):
    """Format schema data for human-readable prompt"""
    formatted = []
    for table_name, columns in schema_data.items():
        formatted.append(f"Table {table_name}:")
        for col in columns:
            formatted.append(f"  - {col['name']} ({col['type']}): {col['description']}")
    return "\n".join(formatted)

def get_table_reference(table_name: str) -> str:
    """Generate properly formatted BigQuery table reference"""
    return f"`{BQ_PROJECT}.{BQ_DATASET}.{table_name}`"

# agent instructions prompt 
system_prompt = f"""You are an advanced BigQuery SQL expert with data modeling intuition. Key capabilities:

1. Schema Reasoning:
- Automatically detect date fields that should join to dim_date (e.g., first_seen_date → dim_date.date)
- Recognize common patterns (user_id for joins, *_date for time dimensions)
- Identify fact vs dimension tables based on structure

2. Intelligent Defaults:
- For time-based questions, default to appropriate date granularity (month/quarter/year)
- For user metrics, consider both raw (user-engagement) and aggregated (fact_user_metrics) sources
- When counting distinct values, automatically add LIMIT based on expected cardinality

3. Self-Correction:
- If initial query returns unexpected zeros/null values:
  1. Check date formatting
  2. Verify join conditions
  3. Consider alternative source tables

4. Analytical Best Practices:
- Prefer COUNT(DISTINCT) over COUNT() for user metrics
- Use appropriate date functions (EXTRACT, DATE_TRUNC)
- Apply CASE WHEN for conditional logic

Available tables:
{format_schema_for_prompt(schemas)}

Examples of Intelligent Behavior:
Q: "How many unique months per user?"
A: SELECT 
     u.user_id,
     COUNT(DISTINCT FORMAT_DATE('%Y-%m', d.date)) AS unique_months
   FROM {get_table_reference('user-engagement')} u
   JOIN {get_table_reference('dim_date')} d 
     ON u.first_seen_date = d.date
   GROUP BY u.user_id

Q: "Find users active in Q2 but not Q3"
A: WITH q2_users AS (
     SELECT DISTINCT user_id 
     FROM {get_table_reference('user-engagement')} u
     JOIN {get_table_reference('dim_date')} d 
       ON u.first_seen_date = d.date
     WHERE d.quarter = 'Q2'
   )
   SELECT q2.user_id
   FROM q2_users q2
   WHERE NOT EXISTS (
     SELECT 1 
     FROM {get_table_reference('user-engagement')} u2
     JOIN {get_table_reference('dim_date')} d2 
       ON u2.first_seen_date = d2.date
     WHERE d2.quarter = 'Q3'
     AND u2.user_id = q2.user_id
   )
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Define the agent state
class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    query_result: Union[pd.DataFrame, str, None]
    validation_errors: list[str]
    attempts: int
    needs_correction: bool
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    needs_modification: Optional[bool]

# Helper function to clean SQL query
def clean_sql_query(query: str) -> str:
    """Clean and normalize SQL query by removing backticks, etc."""
    if query is None:
        return ""
    if query.startswith("```") and query.endswith("```"):
        query = query.strip("`")
        if query.lower().startswith("sql"):
            query = query[3:].strip()
    return query.strip()

# Add new helper function for human interaction
def get_human_approval(state: AgentState) -> AgentState:
    """Get human approval for generated SQL with ability to provide feedback"""
    print("\n" + "="*50)
    print("HUMAN APPROVAL REQUIRED")
    print("Generated SQL Query:")
    print(state["sql_query"])
    
    question = state.get("question", "Unknown question")
    print("\nQuestion being answered:", question)
    
    # Allow for human nuanced feedback
    print("\nYou can type:")
    print("- 'Y' or 'Yes' to approve")
    print("- 'N' or 'No' to reject")
    print("- 'Yes but...' followed by your feedback/suggestions to modify the query")
    
    feedback = input("Your response: ").strip()
    
    # Process feedback
    feedback_lower = feedback.lower()
    if feedback_lower.startswith('y') or feedback_lower.startswith('yes'):
        is_approved = True
        # Check if there's feedback beyond just approval
        if 'but' in feedback_lower:
            # Extract the modification suggestion
            suggestion = feedback[feedback.lower().find('but') + 3:].strip()
            return {
                **state,
                "human_approved": True,
                "human_feedback": suggestion,
                "needs_modification": True
            }
        else:
            return {**state, "human_approved": True, "needs_modification": False}
    else:
        # Rejection with potential feedback
        suggestion = feedback[2:].strip() if len(feedback) > 2 else ""
        return {
            **state,
            "human_approved": False,
            "human_feedback": suggestion if suggestion else "Query rejected",
            "needs_modification": False  # Will go to correction process
        }

# Initialize the graph
workflow = Graph()

# Define nodes
def generate_sql(state: AgentState) -> AgentState:
    # LLM generates SQL query
    print(f"\nGenerating SQL for: {state['question']}")
    chain = (
        {"question": lambda x: x["question"]}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    sql_query = chain.invoke(state)
    print(f"Generated SQL (raw): {sql_query}")
    return {
        **state,  # Preserve all existing state fields, including question
        "sql_query": sql_query, 
        "attempts": state.get("attempts", 0) + 1,
        "query_result": None, 
        "validation_errors": [] 
    }

def validate_sql(state: AgentState) -> AgentState:
    query = clean_sql_query(state["sql_query"])
    errors = []
    
    table_ref_pattern = re.compile(
        r"(`" + re.escape(BQ_PROJECT) + r"\." + re.escape(BQ_DATASET) + r"\.[a-zA-Z0-9_-]+`|`[a-zA-Z0-9_-]+`)", 
        re.IGNORECASE
    )
    
    # Validation checks
    if not query.lower().startswith(("select", "with")):
        errors.append("Must start with SELECT/WITH")
    
    if not table_ref_pattern.search(query):
        errors.append(f"Missing valid table reference (expected format: `{BQ_PROJECT}.{BQ_DATASET}.table_name` or `table_name`)")
    
    if "join" in query.lower() and not re.search(r"\bjoin\b(.|\n)+?\bon\b", query, re.IGNORECASE):
        errors.append("JOIN missing ON clause")
    
    return {
        **state,  
        "validation_errors": errors, 
        "sql_query": state["sql_query"],
        "query_result": None  
    }

def execute_query(state: AgentState) -> AgentState:
    if state["validation_errors"]:
        print(f"Validation errors: {state['validation_errors']}")
        return {
            **state,  
            "query_result": f"Validation errors: {', '.join(state['validation_errors'])}",
            "sql_query": state["sql_query"],
            "needs_correction": True
        }
    
    query = clean_sql_query(state["sql_query"])
    print(f"Executing query: {query}")
    try:
        query_job = client.query(query)
        result = query_job.result().to_dataframe()
        return {
            **state, 
            "query_result": result,
            "sql_query": state["sql_query"],
            "needs_correction": False
        }
    except Exception as e:
        return {
            **state,  
            "query_result": f"Execution Error: {str(e)}",
            "sql_query": state["sql_query"],
            "needs_correction": True
        }

def analyze_results(state: AgentState) -> AgentState:
    result_update = {
        **state, 
        "sql_query": state["sql_query"],
        "query_result": state["query_result"]
    }
    
    if isinstance(state["query_result"], str):
        print(f"Problem detected: {state['query_result']}")
        result_update["needs_correction"] = True
        return result_update
    
    if isinstance(state["query_result"], pd.DataFrame):
        if state["query_result"].empty:
            result_update["needs_correction"] = True
            result_update["query_result"] = "Query returned empty results"
        elif (state["query_result"].iloc[:, 1:] == 0).all().all():
            result_update["needs_correction"] = True
            result_update["query_result"] = "Query returned all zeros"
        else:
            result_update["needs_correction"] = False
    
    return result_update

def correct_query(state: AgentState) -> AgentState:
    """Enhanced correction with human feedback context"""
    error_context = state.get("query_result", "Unknown error")
    
    # Special case for human disapproval
    if state.get("human_approved") is False:
        error_context = "Human reviewer rejected the generated SQL query"
    
    original_query = state.get("sql_query", "No query generated yet")
    print(f"\nAttempting to correct query. Error: {error_context}")
    
    question = state.get("question", "Unknown question") 
    
    correction_prompt = f"""Correct this SQL query based on the feedback:
    
    Error Context: {error_context}
    Original Query: {original_query}
    
    User Question: {question}
    
    Provide ONLY the corrected SQL query:"""
    
    corrected = llm.invoke(correction_prompt)
    return {
        **state,  
        "sql_query": corrected.content.strip(),
        "attempts": state.get("attempts", 0),
        "query_result": None,
        "validation_errors": [],
        "human_approved": None  # Reset approval state
    }

# Add a new node for handling human modification suggestions
def modify_query_based_on_feedback(state: AgentState) -> AgentState:
    """Modify query based on human feedback"""
    original_query = state["sql_query"]
    feedback = state.get("human_feedback", "")
    
    print(f"\nModifying query based on feedback: {feedback}")
    
    # Construct prompt for the LLM to modify the query
    modification_prompt = f"""
    I have a SQL query that needs to be modified based on human feedback.
    
    Original SQL query:
    {original_query}
    
    Human feedback: {feedback}
    
    Please modify the query according to this feedback. Return ONLY the modified SQL query.
    """
    
    # Use the LLM to modify the query
    modified_query = llm.invoke(modification_prompt)
    
    return {
        **state,
        "sql_query": modified_query.content.strip(),
        "needs_modification": False,  # Reset flag after modification
        "human_approved": True  # Consider it approved after modification
    }

# Add nodes to workflow
workflow.add_node("generate", generate_sql)
workflow.add_node("validate", validate_sql)
workflow.add_node("execute", execute_query)
workflow.add_node("analyze", analyze_results)
workflow.add_node("correct", correct_query)
workflow.add_node("human_approval", get_human_approval)
workflow.add_node("modify", modify_query_based_on_feedback)

# Set up workflow edges
workflow.set_entry_point("generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", "human_approval") 

# Add conditional edges
workflow.add_conditional_edges(
    "human_approval",
    lambda x: "modify" if x.get("human_approved") and x.get("needs_modification", False) else 
              "execute" if x.get("human_approved") else "correct",
    {"modify": "modify", "execute": "execute", "correct": "correct"}
)

workflow.add_edge("modify", "execute")
workflow.add_edge("execute", "analyze")

workflow.add_conditional_edges(
    "analyze",
    lambda x: "correct" if x.get("needs_correction", False) and x.get("attempts", 0) < 3 else END,
    {"correct": "correct", END: END}
)

workflow.add_edge("correct", "validate")

# Compile the graph
app = workflow.compile()

def bigquery_agent(question: str, max_attempts: int = 3) -> Union[pd.DataFrame, str]:
    state = {
        "question": question,
        "sql_query": None,
        "query_result": None,
        "validation_errors": [],
        "attempts": 0,
        "needs_correction": False,
        "human_approved": None,
        "human_feedback": None,
        "needs_modification": False
    }
    
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        result_state = app.invoke(state, {"recursion_limit": 50})
        
        # Ensure result_state is not None
        if result_state is None:
            print("Warning: Workflow returned None state. Using previous state.")
            break
        else:
            state = result_state  
        
        # Handle early exit if human rejects final attempt
        if state.get("human_approved") is False and attempt == max_attempts - 1:
            return "Query rejected by human reviewer"
        
        if not state.get("needs_correction", False):
            result = state.get("query_result")
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return f"Final result: {result}"
    
    return state.get("query_result", "Max attempts reached without success")

# # Test questions
questions = [
    "Give me a list of top 5 users by engagement time"
]

for question in questions:
    print(f"\n{'='*50}\nProcessing: {question}\n{'='*50}")
    result = bigquery_agent(question)
    
    print("\nFinal Result:")
    if isinstance(result, pd.DataFrame):
        print(result.head())
    else:
        print(result)