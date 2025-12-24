from typing import List, Optional, Dict, Any
from pydantic import BaseModel #allows us to validate data very easily


#These Schema classes are to be passed to the LLM so it knows what kind of structured data it should look for and retrieve.
class CompanyAnalysis(BaseModel):
    """A Structured output for LLM company analysis focused on developer tools"""
    pricing_model: str #Free, Freemium, Paid, Enterprise, Unknown
    is_open_source: Optional[bool] = None
    tech_stack: List[str] = str
    description: str = ""
    api_available: Optional[bool] = None
    language_support: List[str] = []
    integration_capabilities: List[str] = []

class CompanyInfo(BaseModel):
    name: str
    description: str
    website: str
    pricing_model: Optional[str] = None
    is_open_source: Optional[bool] = None
    tech_stack: List[str] = str
    api_available: Optional[bool] = None
    language_support: List[str] = []
    integration_capabilities: List[str] = []
    developer_experience_rating: Optional[str] = None

class ResearchState(BaseModel):
    query: str
    extracted_tools: List[str] = []
    companies: List[str] = []
    search_results : List[Dict[str, Any]] = []
    analysis: Optional[str] = None



