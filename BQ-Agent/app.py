from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
import uuid
from datetime import datetime
import pandas as pd
import re
from langgraph.graph import Graph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
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
import re
import warnings

# Initialize FastAPI
app = FastAPI(title="BigQuery Agent API")

# --- BigQuery Agent Core Logic ---
class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    query_result: Union[pd.DataFrame, str, None]
    validation_errors: list[str]
    attempts: int
    needs_correction: bool

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/mncedisimncwabe/Downloads/hallowed-span-459710-s1-c41d79c9b56b.json"

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

schemas = get_all_schemas("test_clustering")

# Initialize Vertex AI LLM
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0, # Controls the randomness/creativity of the AI's output. With higher values, it might sometimes add unnecessary clauses.
    max_output_tokens=2048, # Set the maximum length of the generated response in tokens (≈ words/word parts). 1,500-2,000 words
    project="hallowed-span-459710-s1",
    location="us-central1"
)

def format_schema_for_prompt(schema_data):
    """Format schema data for human-readable prompt"""
    formatted = []
    for table_name, columns in schema_data.items():
        formatted.append(f"Table {table_name}:")
        for col in columns:
            formatted.append(f"  - {col['name']} ({col['type']}): {col['description']}")
    return "\n".join(formatted)

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
   FROM `hallowed-span-459710-s1.test_clustering.user-engagement` u
   JOIN `hallowed-span-459710-s1.test_clustering.dim_date` d 
     ON u.first_seen_date = d.date
   GROUP BY u.user_id

Q: "Find users active in Q2 but not Q3"
A: WITH q2_users AS (
     SELECT DISTINCT user_id 
     FROM `hallowed-span-459710-s1.test_clustering.user-engagement` u
     JOIN `hallowed-span-459710-s1.test_clustering.dim_date` d 
       ON u.first_seen_date = d.date
     WHERE d.quarter = 'Q2'
   )
   SELECT q2.user_id
   FROM q2_users q2
   WHERE NOT EXISTS (
     SELECT 1 
     FROM `hallowed-span-459710-s1.test_clustering.user-engagement` u2
     JOIN `hallowed-span-459710-s1.test_clustering.dim_date` d2 
       ON u2.first_seen_date = d2.date
     WHERE d2.quarter = 'Q3'
     AND u2.user_id = q2.user_id
   )
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting"""
    if query is None: return ""
    if query.startswith("```") and query.endswith("```"):
        query = query.strip("`")
        if query.lower().startswith("sql"):
            query = query[3:].strip()
    return query.strip()

# Initialize the graph
workflow = Graph()

# Define nodes
def generate_sql(state: AgentState):
    print(f"\nGenerating SQL for: {state['question']}")
    chain = prompt_template | llm | StrOutputParser()
    sql_query = chain.invoke({"question": state["question"]})
    print(f"Generated SQL (raw): {sql_query}")
    return {
        "sql_query": sql_query, 
        "attempts": state.get("attempts", 0) + 1,
        "query_result": None,
        "validation_errors": []
    }

def validate_sql(state: AgentState):
    query = clean_sql_query(state["sql_query"])
    errors = []
    if not query.lower().startswith(("select", "with")):
        errors.append("Must start with SELECT/WITH")
    if "join" in query.lower() and not re.search(r"\bjoin\b(.|\n)+?\bon\b", query, re.IGNORECASE):
        errors.append("JOIN missing ON clause")
    return {
        "validation_errors": errors, 
        "sql_query": state["sql_query"],
        "query_result": None  
    }

def execute_query(state: AgentState):
    if state["validation_errors"]:
        print(f"Validation errors: {state['validation_errors']}")
        return {
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
            "query_result": result,
            "sql_query": state["sql_query"],
            "needs_correction": False
        }
    except Exception as e:
        return {
            "query_result": f"Execution Error: {str(e)}",
            "sql_query": state["sql_query"],
            "needs_correction": True
        }

def analyze_results(state: AgentState):
    result_update = {
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

def correct_query(state: AgentState):
    # Uses LLM + past error to improve SQL
    error_context = state.get("query_result", "Unknown error")
    original_query = state.get("sql_query", "No query generated yet")
    print(f"\nAttempting to correct query. Error: {error_context}")
    
    correction_prompt = f"""Correct this SQL query based on the error:
    
    Error: {error_context}
    Original Query: {original_query}
    
    Question: {state['question']}
    
    Please provide the corrected SQL query only:"""
    
    corrected = llm.invoke(correction_prompt)
    return {
        "sql_query": corrected.content.strip(),
        "attempts": state.get("attempts", 0),
        "query_result": None,
        "validation_errors": []
    }

# Build workflow
# Add nodes to the graph
workflow.add_node("generate", generate_sql)
workflow.add_node("validate", validate_sql)
workflow.add_node("execute", execute_query)
workflow.add_node("analyze", analyze_results)
workflow.add_node("correct", correct_query)

# Set up edges
workflow.set_entry_point("generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", "execute")
workflow.add_edge("execute", "analyze")

# Conditional edges
workflow.add_conditional_edges(
    "analyze",
    lambda x: "correct" if x["needs_correction"] and x.get("attempts", 0) < 3 else END,
    {"correct": "correct", "__end__": END}
)
workflow.add_edge("correct", "validate")

agent_app = workflow.compile() 

# --- API Endpoints ---
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    sql_query: Optional[str] = None
    timestamp: str

sessions = {}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        if session_id not in sessions:
            sessions[session_id] = {
                "question": request.question,
                "sql_query": None,
                "query_result": None,
                "validation_errors": [],
                "attempts": 0,
                "needs_correction": False,
                "status": "processing"
            }
        
        state = sessions[session_id]
        result = agent_app.invoke(state)
        
        response_data = {
            "status": "completed" if isinstance(result["query_result"], pd.DataFrame) else "error",
            "sql_query": clean_sql_query(result["sql_query"])
        }

        # Handle the result properly
        if isinstance(result["query_result"], pd.DataFrame):
            # Convert DataFrame to a dictionary format that matches your schema
            response_data["result"] = {
                "columns": list(result["query_result"].columns),
                "data": result["query_result"].to_dict(orient="records")
            }
        else:
            response_data["error"] = str(result["query_result"])
        
        sessions[session_id].update(response_data)
        
        return QueryResponse(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            **response_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}", response_model=QueryResponse)
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return QueryResponse(
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        status=sessions[session_id]["status"],
        result=sessions[session_id].get("result"),
        error=sessions[session_id].get("error"),
        sql_query=sessions[session_id].get("sql_query")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)