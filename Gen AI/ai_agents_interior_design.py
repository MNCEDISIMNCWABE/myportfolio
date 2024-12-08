import os
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
#from tools.search_tool import SearchTools
#from llm_response_logger.response_logger import ResponseLogger

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List

import json
import requests
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

class ResponseLogger(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
       None

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # we can print tokens as they are being streamed from LLM server
        None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(response.flatten())


class SearchTools:
    @tool("Search internet")
    def search_internet(query: str) -> str:
        """Search the internet for a given topic and return relevant results."""
        print(f"[DEBUG] Received query: {query}")  # Debug log to inspect input
        return SearchTools.search(query)

    @staticmethod
    def search(query: str, n_results=5) -> str:
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
            if not results:
                return "No search results found."

            formatted_results = []
            for result in results[:n_results]:
                formatted_results.append(
                    f"Title: {result['title']}\n"
                    f"Link: {result['link']}\n"
                    f"Snippet: {result.get('snippet', 'No snippet available.')}\n"
                    "\n-----------------"
                )

            return "\nSearch result:\n" + "\n".join(formatted_results)

        except requests.exceptions.RequestException as e:
            return f"Error during search: {e}"



# api_key = "44bcc750e7c15b2ca21386fb93fd3acfbbd81111"  # serper API for google searchs
# api_base = "http://127.0.0.1:1234/v1/" # local url from LLM studio
# model_name = "llama-3.2-3b-instruct" # the LLM model used to power the AI Agents - downloaded on LM Studio


# llm = ChatOpenAI(
#      model = model_name,
#      base_url = api_base,
#      api_key = api_key, # doesn't matter for LM Studio
#      temperature=0.01,
#      )

llm = LLM(model="ollama/llama3.2", base_url="http://127.0.0.1:11434", provider="ollama")

# Define the agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science related to Interior Design',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,  # Researcher cannot delegate tasks
    max_iter=5,
    tools=[SearchTools.search_internet],
    llm=llm,
)

writer = Agent(
    role='Tech Content Strategist in Interior Design sphere',
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for creating insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    max_iter=5,
    tools=[SearchTools.search_internet],
)

# Define tasks for each agent
research_task = Task(
    description="""Gather comprehensive and relevant information about the most interesting applications of AI in Interior Design in 2024.

    STEPS TO COMPLETE THE TASK:
    1. Use the internet search tool to gather insights about 4-5 cutting-edge AI applications in Interior Design.
    2. Structure the findings in a clear and concise format for further use.
    3. Focus on identifying how each application is transforming the design process.
    4. Ensure the information is accurate, up-to-date, and suitable for a tech-savvy audience.
    5. Stop searching once sufficient information has been gathered.

    Specific requirements:
    - Highlight at least 3-4 unique AI applications in Interior Design
    - Provide examples or case studies where possible
    - Avoid using overly complex technical jargon""",
    expected_output="A structured report summarizing 3-4 AI applications in Interior Design with key insights and examples",
    agent=researcher,
    max_iter=5,
)

writing_task = Task(
    description="""Develop a compelling blog post about AI innovations in Interior Design.

    Blog Post Requirements:
    1. Title: Catchy and descriptive of AI's role in interior design
    2. Introduction: Hook readers with the transformative potential of AI
    3. Body:
       - Discuss 3-4 specific AI technologies
       - Explain how each technology revolutionizes design processes
       - Use clear, engaging language, and consider including creative elements like anecdotes or metaphors
    4. Conclusion: Inspire readers about the future of AI-driven design, and optionally include a call-to-action for further exploration
    5. Tone: Conversational, exciting, and accessible to tech-enthusiast readers

    Additional Instructions:
    - If you lack sufficient information about the topic, delegate a research task to the Senior Research Analyst to gather the necessary insights before proceeding.""",
    expected_output="""A complete blog post of 200 to 300 words covering:
    - Innovative AI technologies in interior design
    - Practical applications and benefits
    - Engaging narrative structure
    - Forward-looking perspective""",
    agent=writer,
)

# Set up the crew for sequential task execution
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
    process='sequential',
)

# Run the crew task
result = crew.kickoff()

print("######################")
print(result)