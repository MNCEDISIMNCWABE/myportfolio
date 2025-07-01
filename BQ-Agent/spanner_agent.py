from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, Optional, Union, List, Dict, Any
from langchain_core.output_parsers import StrOutputParser, StrOutputParser
import pandas as pd
import re
from google.cloud import spanner
from google.cloud.spanner_v1 import param_types
import json
import os

GCP_PROJECT = "hallowed-span-459710-s1"
SPANNER_INSTANCE = "lee-test"
SPANNER_DATABASE = "transactions-db"
LOCATION = "us-central1"  

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/mncedisimncwabe/Downloads/hallowed-span-459710-s1-c41d79c9b56b.json"

# Initialize Vertex AI LLM
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0,
    max_output_tokens=2048,
    project=GCP_PROJECT,
    location=LOCATION
)

# Initialize Spanner client
spanner_client = spanner.Client(project=GCP_PROJECT)
instance = spanner_client.instance(SPANNER_INSTANCE)
database = instance.database(SPANNER_DATABASE)

# Specify the tables to retrieve schema for
tables_of_interest = ["users", "countries"]

def format_schema_for_prompt(schema_data):
    """Format schema data for human-readable prompt"""
    formatted = []
    for table_name, columns in schema_data.items():
        formatted.append(f"Table {table_name}:")
        for col in columns:
            formatted.append(f"  - {col['name']} ({col['type']}): {col['description']}")
    return "\n".join(formatted)

def get_spanner_schema() -> Dict[str, List[Dict[str, str]]]:
    """Extract schema information from Spanner database for specific tables"""
    schemas = {}

    with database.snapshot(multi_use=True) as snapshot:
        # Get schema information for the specified tables
        for table in tables_of_interest:
            columns_result = snapshot.execute_sql(
                """
                SELECT
                    column_name,
                    spanner_type,
                    IFNULL(is_nullable, '') as nullable
                FROM information_schema.columns
                WHERE table_name = @table_name
                ORDER BY ordinal_position
                """,
                params={"table_name": table},
                param_types={"table_name": param_types.STRING}
            )

            columns = []
            for row in columns_result:
                col_name, col_type, nullable = row
                description = f"{col_name} field ({col_type})" + (" - nullable" if nullable == "YES" else "")
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "description": description
                })

            schemas[table] = columns

    return schemas

# Get schema information
schemas = get_spanner_schema()

# Create system prompt
system_prompt = f"""You are an advanced Google Cloud Spanner SQL expert with data modeling intuition. Key capabilities:

1. Schema Understanding:
- You understand the structure of all tables in the Spanner database
- You can identify primary keys, foreign keys and common join patterns
- You recognize common field types and their appropriate usage in queries

2. Intelligent Defaults:
- For time-based questions, you'll use appropriate date/timestamp functions
- You'll automatically limit results to reasonable sizes unless specified otherwise
- You'll include appropriate WHERE clauses to filter out irrelevant data

3. SQL Generation Best Practices:
- Use Spanner SQL syntax (similar to standard SQL with some Google Cloud extensions)
- Avoid anti-patterns that would cause performance issues
- Use appropriate JOINs and avoid unnecessary CROSS JOINs
- Include comments to explain complex logic

4. Self-Correction:
- If initial query has issues, you'll analyze error messages and correct the problems
- You'll verify column names and types match the schema
- You'll ensure joins use compatible data types

Available tables:
{format_schema_for_prompt(schemas)}

Examples of SQL queries for Spanner:

1. Basic SELECT:
```sql
SELECT user_id, date 
FROM users
WHERE country = 'USA'
LIMIT 10;
```

2. JOINS
```sql
SELECT o.country_id, o.user_id
FROM countries c
JOIN users u ON c.country_name = u.country
WHERE u.transaction_count > 2
LIMIT 100;
```

3. Aggregation:
```sql
SELECT 
  user_id,
  SUM(transaction_count) as total_transactions,
FROM users
GROUP BY user_id
ORDER BY total_transactions DESC
LIMIT 20;
```

Remember to follow these Spanner-specific best practices:
- Use parameterized queries when possible (though you'll provide the raw SQL here)
- Avoid using subqueries when JOINS would be more efficient
- Include appropriate LIMIT clauses on large tables
- Consider using ARRAY_AGG() for generating arrays from grouped results
"""

# Define prompt template
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
    """Clean and normalize SQL query by removing code fences, etc."""
    if query is None:
        return ""
    if query.startswith("```") and query.endswith("```"):
        query = query.strip("`")
        if query.lower().startswith("sql"):
            query = query[3:].strip()
    return query.strip()

# Human approval function
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

# Function to generate SQL from question
def generate_sql(state: AgentState) -> AgentState:
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
        **state, 
        "sql_query": sql_query, 
        "attempts": state.get("attempts", 0) + 1,
        "query_result": None, 
        "validation_errors": [] 
    }

# Validate the SQL query
def validate_sql(state: AgentState) -> AgentState:
    query = clean_sql_query(state["sql_query"])
    errors = []
    
    # Basic validation checks
    if not query.lower().startswith(("select", "with")):
        errors.append("Must start with SELECT/WITH")
    
    referenced_tables = set(re.findall(r'\bFROM\s+([a-zA-Z0-9_]+)', query, re.IGNORECASE))
    referenced_tables.update(re.findall(r'\bJOIN\s+([a-zA-Z0-9_]+)', query, re.IGNORECASE))
    
    for table in referenced_tables:
        if table not in schemas:
            errors.append(f"Referenced table '{table}' not found in schema")
    
    return {
        **state,  
        "validation_errors": errors, 
        "sql_query": state["sql_query"],
        "query_result": None  
    }

# Execute the query against Spanner
def execute_query(state: AgentState) -> AgentState:
    if state["validation_errors"]:
        print(f"Validation errors: {state['validation_errors']}")
        return {
            **state,  
            "query_result": f"Validation errors: {', '.join(state['validation_errors'])}",
            "needs_correction": True
        }
    
    query = clean_sql_query(state["sql_query"])
    print(f"Executing query: {query}")
    
    try:
        with database.snapshot() as snapshot:
            result = snapshot.execute_sql(query)
            
            # Properly consume the result stream
            rows = list(result)
            
            # Get column names from the result metadata
            columns = [field.name for field in result.fields]
            
            if not rows:
                return {
                    **state,
                    "query_result": pd.DataFrame(columns=columns),
                    "needs_correction": False
                }
            
            df = pd.DataFrame(rows, columns=columns)
            
            return {
                **state, 
                "query_result": df,
                "needs_correction": False
            }
            
    except Exception as e:
        print(f"Full error details: {str(e)}")
        return {
            **state,  
            "query_result": f"Execution Error: {str(e)}",
            "needs_correction": True
        }


# Analyze query results
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
        elif (
            state["query_result"].shape[1] > 1 and 
            all(state["query_result"][col].nunique() <= 1 for col in state["query_result"].columns[1:])
        ):
            result_update["needs_correction"] = True
            result_update["query_result"] = "Query returned non-diverse results"
        else:
            result_update["needs_correction"] = False
    
    return result_update

# Correct query based on errors
def correct_query(state: AgentState) -> AgentState:
    """Enhanced correction with human feedback context"""
    error_context = state.get("query_result", "Unknown error")
    
    # Special case for human disapproval
    if state.get("human_approved") is False:
        error_context = "Human reviewer rejected the generated SQL query"
        if state.get("human_feedback"):
            error_context += f": {state.get('human_feedback')}"
    
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

# Modify query based on human feedback
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
        "needs_modification": False, 
        "human_approved": True 
    }

# Main function to use the Spanner agent
def spanner_agent(question: str, max_attempts: int = 3) -> Union[pd.DataFrame, str]:
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
    
    # Generate initial SQL
    state = generate_sql(state)
    
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        
        # Validate SQL first
        state = validate_sql(state)
        if state["validation_errors"]:
            print("Validation errors found, correcting...")
            state = correct_query(state)
            continue
            
        # Get human approval
        state = get_human_approval(state)
        if state.get("human_approved") is False:
            print("Human rejected query, correcting...")
            state = correct_query(state)
            continue
            
        if state.get("needs_modification", False):
            state = modify_query_based_on_feedback(state)
            continue
            
        # Execute only if approved
        if state.get("human_approved") is True:
            state = execute_query(state)
            
            # Check if execution was successful
            if isinstance(state["query_result"], pd.DataFrame):
                print("Query executed successfully")
                break
            else:
                print(f"Execution problem: {state['query_result']}")
                state = correct_query(state)
                continue
    
    # Final result processing
    result = state.get("query_result")
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Query returned empty results"
        return result
    elif result is None:
        return "Failed to generate valid query after maximum attempts"
    else:
        return str(result)


# Test
# questions = [
#         "Show me the top 5 countries by transaction count"
#     ]
    
# for question in questions:
#     print(f"\n{'='*50}\nProcessing: {question}\n{'='*50}")
#     result = spanner_agent(question)
    
#     print("\nFinal Result:")
#     if isinstance(result, pd.DataFrame):
#         print(result.head())
#     else:
#         print(result)