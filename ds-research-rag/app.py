import os
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict

FIRE_CRAWL_API_KEY = "fc-8b879c61ecbd4bd38ee4a201cd9cc3c1"
LIMIT_RESEARCH_PAPERS = 10  # Limit to top 10 relevant papers

# FireCrawl web search as fallback
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
        self.api_key = FIRE_CRAWL_API_KEY
        self.base_url = "https://api.firecrawl.dev/v0/search"

    def _run(self, query: str) -> str:
        try:
            payload = {"query": query, "pageOptions": {"fetchPageContent": True}}
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
            formatted_results = [
                f"Title: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('pageContent')}"
                for res in results.get("results", [])
            ]
            return "\n---\n".join(formatted_results) or "No results found."
            
        except requests.RequestException as e:
            return f"Error performing web search: {str(e)}"

# ArxivSearchTool for precise category handling
class ArxivSearchInput(BaseModel):
    """Input schema for ArxivSearchTool."""
    query: str = Field(..., description="Search query for arXiv API")

class ArxivSearchTool(BaseTool):
    name: str = "ArxivSearchTool"
    description: str = "Search recent arXiv papers with precise category matching and relevance filtering"
    args_schema: Type[BaseModel] = ArxivSearchInput
    model_config = ConfigDict(extra="allow")

    def _detect_category(self, query: str) -> str:
        """Determine arXiv category using expanded technical vocabulary."""
        categories = {
            "multi-agent": [
                "multi-agent", "multiagent", "agent-based", "swarm intelligence",
                "distributed ai", "mas ", "multi-robot", "cooperative systems",
                "decentralized ai", "collective intelligence"
            ],
            "machine learning": [
                "machine learning", "deep learning", "neural network", "llm",
                "supervised learning", "unsupervised learning", "reinforcement learning",
                "transfer learning", "self-supervised learning"
            ],
            "nlp": [
                "nlp", "natural language", "transformer", "language model",
                "text mining", "named entity", "sentiment analysis", "text generation",
                "question answering", "summarization"
            ],
            "cv": [
                "computer vision", "cv", "image processing", "cnn", "object detection",
                "segmentation", "yolo", "resnet", "vision transformer", "image generation"
            ],
            "ai": [
                "artificial intelligence", "ai", "cognitive systems",
                "knowledge representation", "automated reasoning", "automated planning",
                "expert systems", "theorem proving"
            ],
            "robotics": [
                "robotics", "robot control", "motion planning", "slam",
                "manipulation", "autonomous systems", "robot learning"
            ],
            "stats": [
                "statistical", "bayesian", "regression", "time series",
                "hypothesis testing", "markov chain", "monte carlo", "causal inference"
            ]
        }
        
        query_lower = f" {query.lower()} "  
        for category, keywords in categories.items():
            if any(f" {keyword} " in query_lower for keyword in keywords):
                return category
        return "ai"  # Default to general AI

    def _run(self, query: str) -> str:
        try:
            category_mapping = {
                "multi-agent": "cs.MA",
                "machine learning": "cs.LG",
                "nlp": "cs.CL",
                "cv": "cs.CV",
                "ai": "cs.AI",
                "robotics": "cs.RO",
                "stats": "stat.ML"
            }
            
            detected_category = self._detect_category(query)
            arxiv_category = category_mapping.get(detected_category, "cs.AI")
            
            query_terms = [
                f'abs:"{term}"' if ' ' in term else f'ti:{term}'  # Search phrases in abstract, single terms in title
                for term in query.lower().split()
                if len(term) > 3  
            ]
            formatted_query = ' AND '.join(query_terms)
            
            recent_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            params = {
                "search_query": (
                    f"({formatted_query}) AND "
                    f"cat:{arxiv_category} AND "
                    f"submittedDate:[{recent_date} TO NOW]"
                ),
                "start": 0,
                "max_results": 15,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            response = requests.get("http://export.arxiv.org/api/query?", params=params)
            response.raise_for_status()
            return self._filter_results(response.text, query)

        except Exception as e:
            return f"arXiv API Error: {str(e)}"

    def _filter_results(self, xml_response: str, original_query: str) -> str:
        """Filter results based on query presence in title/abstract."""
        root = ET.fromstring(xml_response)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        query_terms = set(original_query.lower().split())
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                title = entry.find('atom:title', ns).text.strip().lower()
                abstract = entry.find('atom:summary', ns).text.strip().lower()
                
                # Calculate match score with weighting
                title_score = sum(3 for term in query_terms if term in title)
                abstract_score = sum(1 for term in query_terms if term in abstract)
                total_score = title_score + abstract_score
                
                if total_score >= len(query_terms) * 2:  # Require strong match
                    papers.append(self._format_paper(entry, ns))
                    
                if len(papers) >= LIMIT_RESEARCH_PAPERS:  # Limit to top 10 relevant papers
                    break
                    
            except AttributeError:
                continue 
        
        return "\n\n".join(papers)[:15000] if papers else "No relevant papers found."

    def _format_paper(self, entry, ns):
        """Format individual paper entry with validation."""
        try:
            paper_info = {
                'title': entry.find('atom:title', ns).text.strip(),
                'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
                'published': entry.find('atom:published', ns).text,
                'summary': entry.find('atom:summary', ns).text.strip(),
                'url': entry.find('atom:id', ns).text,
                'pdf': entry.find('atom:id', ns).text.replace('abs', 'pdf') + '.pdf'
            }
            
            # Validate URL format
            if not paper_info['url'].startswith('http://arxiv.org/abs/'):
                return ""
                
            return (
                f"Title: {paper_info['title']}\n"
                f"Authors: {', '.join(paper_info['authors'])}\n"
                f"Published: {paper_info['published']}\n"
                f"Abstract: {paper_info['summary'][:500]}...\n"
                f"PDF: {paper_info['pdf']}\n"
                f"URL: {paper_info['url']}\n"
                "────────────────────"
            )
        except:
            return ""  # Skip invalid entries

# Test the arXiv implementation
def test_arxiv_searcher():
    arxiv_tool = ArxivSearchTool()
    
    test_cases = [
        ("multi-agent AI systems", "Should return cs.MA papers on multi-agent systems"),
        ("transformer architectures in CV", "Should return cs.CV papers with transformers"),
        ("bayesian neural networks", "Should return stat.ML papers"),
        ("robot motion planning", "Should return cs.RO papers")
    ]
    
    for query, description in test_cases:
        print(f"\nTesting: {description}")
        result = arxiv_tool._run(query)
        print(f"Results:\n{result[:1000]}...")  

if __name__ == "__main__":
    test_arxiv_searcher()
