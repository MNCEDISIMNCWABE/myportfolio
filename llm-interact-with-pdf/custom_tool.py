
import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient
import requests

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # For small experiments
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        
        docs = [chunk.text for chunk in chunks]
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> list:
        """Search the document with a query string."""
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        return separator.join(docs)
    

class FireCrawlSearchInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query: str = Field(..., description="Search query for FireCrawl API")

class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Search the web using FireCrawl API for given query"
    args_schema: Type[BaseModel] = FireCrawlSearchInput
    model_config = ConfigDict(extra="allow")

    def __init__(self):
        super().__init__()
        self.api_key = "fc-8b879c61ecbd4bd38ee4a201cd9cc3c1"
        self.base_url = "https://api.firecrawl.dev/v0/search"

    # Rest of the implementation remains the same
    def _run(self, query: str) -> str:
        try:
            payload = {
                "query": query,
                "pageOptions": {
                    "fetchPageContent": True
                }
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json()
            
            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append(
                    f"Title: {result.get('title')}\n"
                    f"URL: {result.get('url')}\n"
                    f"Content: {result.get('pageContent')}\n"
                )
            
            return "\n---\n".join(formatted_results) if formatted_results else "No results found."
            
        except requests.RequestException as e:
            return f"Error performing web search: {str(e)}"

# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = "/Users/lindani/Downloads/Self_Evaluation_779313.pdff"
    
    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)
    
    # Test search
    result = searcher._run("What is the purpose of DSpy?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()
