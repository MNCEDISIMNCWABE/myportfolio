from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import uuid
from datetime import datetime, timedelta
import pandas as pd
import importlib.util
import sys
import os
from pathlib import Path
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --------------------------
# Configuration
# --------------------------

GCP_PROJECT = "hallowed-span-459710-s1"
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

# --------------------------
# Protocol Definitions
# --------------------------

class ModelCapability(BaseModel):
    """Describes what an agent can do"""
    name: str
    description: str
    data_sources: List[str] = []  # What data this agent can access
    query_types: List[str] = []   # Types of queries it can handle
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any] = {}

class ModelContext(BaseModel):
    """Standardized context object for MCP"""
    session_id: str
    timestamp: datetime
    context_data: Dict[str, Any]
    source_agent: Optional[str] = None
    target_agents: Optional[List[str]] = None

class A2AMessage(BaseModel):
    """Standardized agent-to-agent message"""
    message_id: str
    sender: str
    recipients: List[str]
    content: Dict[str, Any]
    context: Optional[ModelContext] = None
    requires_response: bool = False
    expiration: Optional[datetime] = None

# --------------------------
# Wrappers
# --------------------------

class MCPWrapper:
    """Wraps existing agents with MCP capabilities"""

    def __init__(self, agent_instance, agent_name: str, agent_description: str,
                 data_sources: List[str] = None, query_types: List[str] = None):
        self.agent = agent_instance
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.data_sources = data_sources or []
        self.query_types = query_types or []
        self.capabilities = self._define_capabilities()
        self.mcp_server = self._create_mcp_server()
        self.fastapi_app = self._create_fastapi_app()

    def _define_capabilities(self) -> List[ModelCapability]:
        """Define what this agent can do with detailed information"""
        return [
            ModelCapability(
                name="sql_generation_and_execution",
                description=f"Generates and executes SQL queries for {', '.join(self.data_sources)} data",
                data_sources=self.data_sources,
                query_types=self.query_types,
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "max_attempts": {"type": "integer", "default": 3}
                    },
                    "required": ["question"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string"},
                        "query_result": {"type": ["object", "string"]},
                        "status": {"type": "string"}
                    }
                }
            )
        ]

    def _create_mcp_server(self):
        """Create an MCP server for this agent"""
        mcp = FastMCP(self.agent_name)

        @mcp.tool()
        def execute_query(question: str, max_attempts: int = 3) -> Dict[str, Any]:
            """Execute a query using the agent"""
            context = ModelContext(
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                context_data={
                    "question": question,
                    "max_attempts": max_attempts
                }
            )
            result_context = self.execute(context)
            return result_context.context_data

        return mcp

    def _create_fastapi_app(self):
        """Create a FastAPI app for this agent"""
        app = FastAPI()

        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/route_task")
        async def route_task_endpoint(request: Request):
            """Route a task to the best agent"""
            data = await request.json()
            task_description = data.get("task_description")

            # Use the registry to find the best agent for the task
            best_agent = registry.find_best_agent_for_task(task_description)
            if not best_agent:
                return {"error": "No suitable agent found"}

            # Route the task to the best agent
            result = registry.route_task(task_description)
            if result:
                return result.context_data
            else:
                return {"error": "Failed to execute task"}

        return app


    def execute(self, context: ModelContext) -> ModelContext:
        """Execute agent with MCP context"""
        question = context.context_data.get("question")
        max_attempts = context.context_data.get("max_attempts", 3)

        # Call the original agent
        try:
            if "spanner" in self.agent_name.lower():
                result = self.agent.spanner_agent(question, max_attempts)
            else:
                result = self.agent.bigquery_agent(question, max_attempts)

            # Convert result to dict if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                result_data = result.to_dict(orient='records')
            else:
                result_data = str(result)

            status = "success"
        except Exception as e:
            result_data = str(e)
            status = "error"

        # Package results into MCP format
        result_context = ModelContext(
            session_id=context.session_id,
            timestamp=datetime.now(),
            context_data={
                "question": question,
                "result": result_data,
                "status": status
            },
            source_agent=self.agent_name
        )

        return result_context

    def run_server(self):
        """Run the MCP server"""
        uvicorn.run(self.fastapi_app, host="0.0.0.0", port=8000)

class A2AWrapper:
    """Enables agent-to-agent communication"""

    def __init__(self, mcp_wrapper: MCPWrapper):
        self.mcp_agent = mcp_wrapper
        self.inbox: List[A2AMessage] = []

    def send_message(self, recipients: List[str], content: Dict[str, Any], requires_response: bool = False) -> A2AMessage:
        """Send a message to other agents"""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender=self.mcp_agent.agent_name,
            recipients=recipients,
            content=content,
            requires_response=requires_response,
            expiration=datetime.now() + timedelta(hours=1)
        )
        return message

    def receive_message(self, message: A2AMessage):
        """Receive a message from another agent"""
        self.inbox.append(message)

    def process_messages(self) -> Optional[A2AMessage]:
        """Process all received messages"""
        responses = []

        for message in self.inbox[:]:
            if message.expiration and message.expiration < datetime.now():
                self.inbox.remove(message)
                continue

            if message.requires_response:
                # Create context from message
                context = ModelContext(
                    session_id=message.message_id,
                    timestamp=datetime.now(),
                    context_data=message.content,
                    source_agent=message.sender,
                    target_agents=[self.mcp_agent.agent_name]
                )

                # Execute the agent
                response_context = self.mcp_agent.execute(context)

                # Send response
                response = A2AMessage(
                    message_id=str(uuid.uuid4()),
                    sender=self.mcp_agent.agent_name,
                    recipients=[message.sender],
                    content=response_context.context_data,
                    context=response_context
                )

                responses.append(response)
                self.inbox.remove(message)

        return responses[0] if responses else None

# --------------------------
# Registry with LLM Routing
# --------------------------

class AgentRegistry:
    """Central registry for agent discovery and management with LLM-based routing"""

    def __init__(self, llm: ChatVertexAI, heartbeat_timeout: int = 300):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.llm = llm
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at matching tasks to agent capabilities. Analyze the task description and available agents to determine the best match.

Available Agents:
{agents}

Instructions:
1. Match the task to agent capabilities, data sources, and query types
2. Consider keywords like "users", "engagement", "countries", "transactions"
3. Return ONLY the exact agent name that best matches
4. If multiple agents could work, pick the most specialized one
5. If no good match exists, return "NO_MATCH"

Examples:
- "top 5 users by engagement" → look for agents with user/engagement data
- "countries by transaction count" → look for agents with transaction/country data
- "sales analysis" → look for agents with sales data"""),
            ("human", "Task: {task}")
        ])

    def register_agent(self, mcp_wrapper: MCPWrapper, endpoint: Optional[str] = None):
        """Register an agent in the registry"""
        self.agents[mcp_wrapper.agent_name] = {
            "description": mcp_wrapper.agent_description,
            "capabilities": [cap.dict() for cap in mcp_wrapper.capabilities],
            "data_sources": mcp_wrapper.data_sources,
            "query_types": mcp_wrapper.query_types,
            "endpoint": endpoint,
            "last_heartbeat": datetime.now(),
            "wrapper": mcp_wrapper,
            "active": True
        }

    def update_heartbeat(self, agent_name: str):
        """Update the last active timestamp for an agent"""
        if agent_name in self.agents:
            self.agents[agent_name]["last_heartbeat"] = datetime.now()
            self.agents[agent_name]["active"] = True

    def check_agent_online(self, agent_name: str) -> bool:
        """Check if an agent is currently online"""
        agent = self.agents.get(agent_name)
        if not agent:
            return False
        return (datetime.now() - agent["last_heartbeat"]).total_seconds() < self.heartbeat_timeout

    def find_best_agent_for_task(self, task_description: str) -> Optional[str]:
        """
        Use LLM to find the best agent for a task
        Returns agent name or None if no suitable agent found
        """
        # Format agent information for the prompt with detailed capabilities
        agents_info = []
        for name, details in self.agents.items():
            if not self.check_agent_online(name):
                continue

            # Build detailed agent description
            agent_desc = []
            agent_desc.append(f"Agent: {name}")
            agent_desc.append(f"Description: {details['description']}")

            if details['data_sources']:
                agent_desc.append(f"Data Sources: {', '.join(details['data_sources'])}")
            if details['query_types']:
                agent_desc.append(f"Query Types: {', '.join(details['query_types'])}")

            capabilities = [cap['name'] for cap in details['capabilities']]
            agent_desc.append(f"Capabilities: {', '.join(capabilities)}")

            last_active_mins = (datetime.now() - details['last_heartbeat']).total_seconds() / 60
            agent_desc.append(f"Status: ONLINE (last active {last_active_mins:.1f} minutes ago)")

            agents_info.append('\n'.join(agent_desc))

        if not agents_info:
            print("DEBUG: No online agents found")
            return None

        agents_text = '\n\n'.join(agents_info)
        print(f"DEBUG: Sending to LLM:\nTask: {task_description}\nAgents:\n{agents_text}")

        # Get LLM routing
        try:
            chain = self.routing_prompt | self.llm | StrOutputParser()
            recommended_agent = chain.invoke({
                "agents": agents_text,
                "task": task_description
            }).strip()

            print(f"DEBUG: LLM recommended: '{recommended_agent}'")

            # Validate the routing
            if recommended_agent == "NO_MATCH" or recommended_agent not in self.agents:
                print(f"DEBUG: No valid match found. Available agents: {list(self.agents.keys())}")
                return None

            return recommended_agent

        except Exception as e:
            print(f"DEBUG: Error in LLM routing: {e}")
            return None

    def route_task(self, task_description: str) -> Optional[ModelContext]:
        """
        Automated task routing
        1. Finds best agent
        2. Creates execution context
        3. Returns result context
        """
        agent_name = self.find_best_agent_for_task(task_description)
        if not agent_name:
            return None

        # Create and execute context
        context = ModelContext(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context_data={
                "question": task_description,
                "max_attempts": 3
            },
            target_agents=[agent_name]
        )

        wrapper = self.agents[agent_name]["wrapper"]
        return wrapper.execute(context)

    def discover_agents(self) -> List[str]:
        """Discover all registered agents"""
        return list(self.agents.keys())

    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self.agents.get(agent_name)

    def get_agent_wrapper(self, agent_name: str) -> Optional[MCPWrapper]:
        """Get agent wrapper for direct execution"""
        agent_info = self.agents.get(agent_name)
        return agent_info["wrapper"] if agent_info else None

# --------------------------
# Test Cases
# --------------------------

def test_llm_routing():
    """Test LLM-based agent routing"""
    print("\n=== Testing LLM Routing ===")

    tasks = [
        "Give me a list of top 5 users by engagement time",
        "Show me the top 5 countries by transaction count"
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        best_agent = registry.find_best_agent_for_task(task)
        if best_agent:
            print(f"✅ Recommended agent: {best_agent}")

            # Execute the task
            print("Executing task...")
            result = registry.route_task(task)
            if result:
                print(f"Status: {result.context_data['status']}")
                if result.context_data['status'] == 'success':
                    result_str = str(result.context_data['result'])
                    print(f"Result preview: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")
                else:
                    print(f"Error: {result.context_data['result']}")
        else:
            print("❌ No suitable agent found")

def test_heartbeat_system():
    """Test agent heartbeat monitoring"""
    print("\n=== Testing Heartbeat System ===")

    # Check initial status
    print("\nAgent statuses:")
    for name in registry.agents:
        status = "ONLINE" if registry.check_agent_online(name) else "OFFLINE"
        last_seen = registry.agents[name]["last_heartbeat"].strftime('%H:%M:%S')
        print(f"- {name}: {status} (last seen {last_seen})")

    # Simulate agent going offline
    print("\nSimulating Spanner agent going offline...")
    offline_time = datetime.now() - timedelta(seconds=400)
    registry.agents["SpannerSQLAgent"]["last_heartbeat"] = offline_time

    # Verify routing skips offline agents
    task = "Get transaction counts by country"
    print(f"\nRouting task: {task}")
    best_agent = registry.find_best_agent_for_task(task)
    print(f"Selected agent: {best_agent}")

    # Reset agent status
    registry.update_heartbeat("SpannerSQLAgent")

def test_a2a_communication():
    """Test agent-to-agent communication"""
    print("\n=== Testing Agent-to-Agent Communication ===")

    # Create A2A wrappers
    spanner_a2a = A2AWrapper(spanner_mcp)
    bq_a2a = A2AWrapper(bq_mcp)

    # Test message sending
    print("\n1. Testing message sending between agents")

    # Spanner agent sends a message to BigQuery agent
    message = spanner_a2a.send_message(
        recipients=["BigQuerySQLAgent"],
        content={
            "request_type": "data_validation",
            "query": "Can you validate user engagement data for user_id '6c3dbd5cb2393a74d1b5d1fc3289f4b92deea4f92b9b2994399aabf172c500d5'?",
            "context": "Cross-referencing transaction data with engagement metrics"
        },
        requires_response=True
    )

    print(f"✅ Message sent from {message.sender} to {message.recipients}")
    print(f"   Message ID: {message.message_id}")
    print(f"   Content: {message.content['request_type']}")

    # BigQuery agent receives the message
    bq_a2a.receive_message(message)
    print(f"✅ Message received by BigQuerySQLAgent")
    print(f"   Inbox size: {len(bq_a2a.inbox)}")

    # Process messages
    print("\n2. Processing messages")
    response = bq_a2a.process_messages()

    if response:
        print(f"✅ Response generated by {response.sender}")
        print(f"   Response to: {response.recipients}")
        print(f"   Response content preview: {str(response.content)[:100]}...")
    else:
        print("❌ No response generated")

def test_advanced_a2a_scenarios():
    """Test advanced agent-to-agent scenarios"""
    print("\n=== Testing Advanced A2A Scenarios ===")

    # Create A2A wrappers
    spanner_a2a = A2AWrapper(spanner_mcp)
    bq_a2a = A2AWrapper(bq_mcp)

    print("\n1. Testing data correlation scenario")

    # Scenario: Find users with high engagement but low transaction counts
    correlation_message = spanner_a2a.send_message(
        recipients=["BigQuerySQLAgent"],
        content={
            "request_type": "data_correlation",
            "question": "Get top 10 users by engagement time",
            "purpose": "Will cross-reference with transaction data"
        },
        requires_response=True
    )

    print("✅ Correlation request sent to BigQuery agent")

    # BigQuery processes the request
    bq_a2a.receive_message(correlation_message)
    engagement_response = bq_a2a.process_messages()

    if engagement_response:
        print("✅ Engagement data retrieved")

        # Spanner agent uses this data to find patterns
        follow_up_context = ModelContext(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context_data={
                "question": "Show transaction counts for users with high engagement",
                "context_data": engagement_response.content
            }
        )

        spanner_result = spanner_mcp.execute(follow_up_context)
        print(f"✅ Cross-reference analysis completed")
        print(f"   Status: {spanner_result.context_data['status']}")

    print("\n2. Testing message expiration")

    # Create message with short expiration
    expired_message = A2AMessage(
        message_id=str(uuid.uuid4()),
        sender="TestAgent",
        recipients=["SpannerSQLAgent"],
        content={"test": "This message should expire"},
        expiration=datetime.now() - timedelta(seconds=1)
    )

    spanner_a2a.receive_message(expired_message)
    print(f"✅ Expired message added to inbox (size: {len(spanner_a2a.inbox)})")

    spanner_a2a.process_messages()
    print(f"✅ After processing: inbox size: {len(spanner_a2a.inbox)}")

    print("\n3. Testing multi-agent broadcast")

    # Create a broadcast message
    broadcast_message = bq_a2a.send_message(
        recipients=["SpannerSQLAgent"],
        content={
            "broadcast_type": "system_status",
            "message": "System maintenance scheduled",
            "timestamp": datetime.now().isoformat()
        },
        requires_response=False
    )

    print(f"✅ Broadcast message created")
    print(f"   Recipients: {broadcast_message.recipients}")
    print(f"   Requires response: {broadcast_message.requires_response}")

def test_registry_management():
    """Test registry management functions"""
    print("\n=== Testing Registry Management ===")

    print("\n1. Agent Discovery")
    agents = registry.discover_agents()
    print(f"✅ Discovered agents: {agents}")

    print("\n2. Agent Details")
    for agent_name in agents:
        agent_info = registry.get_agent(agent_name)
        if agent_info:
            print(f"\n{agent_name}:")
            print(f"   Description: {agent_info['description']}")
            print(f"   Data Sources: {agent_info['data_sources']}")
            print(f"   Query Types: {agent_info['query_types']}")
            print(f"   Online: {registry.check_agent_online(agent_name)}")

    print("\n3. Capability Matching")
    test_queries = [
        "Find users with most transactions",
        "Analyze engagement patterns over time"
    ]

    for query in test_queries:
        best_agent = registry.find_best_agent_for_task(query)
        print(f"   '{query}' → {best_agent or 'No match'}")

def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")

    print("\n1. Invalid SQL query handling")
    error_context = ModelContext(
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        context_data={
            "question": "SELECT * FROM nonexistent_table_xyz",
            "max_attempts": 1
        }
    )

    try:
        result = spanner_mcp.execute(error_context)
        print(f"✅ Error handled gracefully")
        print(f"   Status: {result.context_data['status']}")
        print(f"   Error message: {result.context_data['result'][:100]}...")
    except Exception as e:
        print(f"❌ Unhandled error: {e}")

    print("\n2. Offline agent handling")
    original_heartbeat = registry.agents["SpannerSQLAgent"]["last_heartbeat"]
    registry.agents["SpannerSQLAgent"]["last_heartbeat"] = datetime.now() - timedelta(seconds=400)

    task_result = registry.route_task("Get transaction data")
    if task_result is None:
        print("✅ Offline agent correctly excluded from routing")
    else:
        print("❌ Offline agent was still used")

    # Restore agent status
    registry.agents["SpannerSQLAgent"]["last_heartbeat"] = original_heartbeat

# --------------------------
# Main Execution
# --------------------------
def run_servers():
    """Run MCP servers for all registered agents"""
    print("Starting MCP servers...")
    for agent_name in registry.discover_agents():
        agent_wrapper = registry.get_agent_wrapper(agent_name)
        if agent_wrapper:
            print(f"Starting server for {agent_name}...")
            agent_wrapper.run_server()

if __name__ == "__main__":
    # Load existing agents
    def load_agent(agent_file: str):
        """Dynamically load an agent module"""
        module_name = Path(agent_file).stem
        spec = importlib.util.spec_from_file_location(module_name, agent_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    try:
        spanner_agent = load_agent("spanner_agent.py")
        bq_agent = load_agent("bq_agent.py")
    except Exception as e:
        print(f"Error loading agents: {e}")
        sys.exit(1)

    # Initialize wrappers with detailed metadata
    spanner_mcp = MCPWrapper(
        agent_instance=spanner_agent,
        agent_name="SpannerSQLAgent",
        agent_description="Generates and executes Google Cloud Spanner SQL queries for transactional data",
        data_sources=["transactions", "users", "countries", "payments", "orders"],
        query_types=["aggregation", "filtering", "grouping", "joins", "analytics"]
    )

    bq_mcp = MCPWrapper(
        agent_instance=bq_agent,
        agent_name="BigQuerySQLAgent",
        agent_description="Generates and executes BigQuery SQL queries for analytics and reporting",
        data_sources=["user_engagement", "web_analytics", "logs", "events", "metrics"],
        query_types=["analytics", "reporting", "time_series", "aggregation", "data_warehouse"]
    )

    # Initialize registry with LLM
    registry = AgentRegistry(llm=llm, heartbeat_timeout=300)
    registry.register_agent(spanner_mcp)
    registry.register_agent(bq_mcp)

    # Update heartbeats to ensure agents are online
    registry.update_heartbeat("SpannerSQLAgent")
    registry.update_heartbeat("BigQuerySQLAgent")

    # Run tests
    # test_llm_routing()
    # test_a2a_communication()
    # test_advanced_a2a_scenarios()
    # test_heartbeat_system()

    # Run MCP servers
    run_servers()