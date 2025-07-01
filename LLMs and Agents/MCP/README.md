We’ll build a simple MCP weather server and connect it to a host, Claude for Desktop. We’ll start with a basic setup, and then progress to more complex use cases.

​
Many LLMs do not currently have the ability to fetch the forecast and severe weather alerts. Let’s use MCP to solve that!

We’ll build a server that exposes two tools: ``get-alerts`` and ``get-forecast``. Then we’ll connect the server to an MCP host (in this case, Claude for Desktop):

### Core MCP Concepts
MCP servers can provide three main types of capabilities:

- Resources: File-like data that can be read by clients (like API responses or file contents)
- Tools: Functions that can be called by the LLM
- Prompts: Pre-written templates that help users accomplish specific tasks