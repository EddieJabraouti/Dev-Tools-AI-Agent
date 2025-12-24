from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts
from urllib.parse import urlparse

class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.1)
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _is_valid_url(self, url:str)->bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and "." in parsed.netloc

    #initialize all agent nodes
    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze",  self._analyze_step)
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    #first step in graph
    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"Finding articles about: {state.query}")

        article_query = f"{state.query} tools comparison best alternatives" #finding the urls
        search_results = self.firecrawl.search_companies(article_query, num_results = 3)

        #scrape the markdown content from found pages
        all_content = ""
        for url, _ in search_results:
            scraped = self.firecrawl.scrape_company_pages(url)
            if scraped:
                all_content + scraped.markdown[:1500] + "\n\n"

        #create a message to be passed to the llm
        messages = [
            SystemMessage(content = self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content = self.prompts.tool_extraction_user(state.query, all_content))
        ]
        try: #parse all the tools from the articles and return + update state
            response = self.llm.invoke(messages)
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(e)
            return {"extracted_tools": []}

    #analyze company urls
    def _analyze_company_content(self, company_name:str, content:str) -> CompanyAnalysis:
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)

        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(company_name, content))
        ]

        try:
            analysis = structured_llm.invoke(messages)
            return analysis
        except Exception as e:
            print(e)
            return CompanyAnalysis(pricing_model="Unknown",
                                   is_open_source=None,
                                   tech_stack=[],
                                   description="Failed",
                                   api_available=None,
                                   language_support=[],
                                   integration_capabilities=[],)

        #research step

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        # Retrieve extracted tool names from state, if available
        extracted_tools = getattr(state, "extracted_tools", [])

        # Fallback to direct search if no tools were extracted
        if not extracted_tools:
            print("No extracted tools found, falling back to direct search")
            search_results = self.firecrawl.search_companies(state.query, num_results=4)

            tool_names = []
            for item in search_results:
                if isinstance(item, tuple):
                    _, metadata = item
                    if isinstance(metadata, dict):
                        title = metadata.get("title")
                        if title:
                            tool_names.append(title)

            tool_names = tool_names[:4]
        else:
            tool_names = extracted_tools[:4]

        print(f"Researching specific tools: {', '.join(tool_names)}")

        companies: list[CompanyInfo] = []

        # Perform research for each identified tool
        for tool_name in tool_names:
            tool_search_results = self.firecrawl.search_companies(
                f"{tool_name} official site",
                num_results=1
            )

            url = None
            markdown_preview = ""

            # Iterate over search results to extract a valid URL and metadata
            for item in tool_search_results:
                if not isinstance(item, tuple):
                    continue

                candidate_url, metadata = item

                if isinstance(candidate_url, str) and candidate_url.startswith("http"):
                    url = candidate_url
                    if isinstance(metadata, dict):
                        markdown_preview = metadata.get("markdown", "")
                    break

            # Skip processing if no valid URL was found
            if not url or not self._is_valid_url(url):
                continue

            # Initialize company information
            company = CompanyInfo(
                name=tool_name,
                description=markdown_preview,
                website=url,
                tech_stack=[]
            )

            # Scrape the official website for detailed content
            scraped = self.firecrawl.scrape_company_pages(
                url
            )
            if not scraped or not scraped.markdown:
                continue

            # Analyze scraped content using structured LLM output
            analysis = self._analyze_company_content(company.name, scraped.markdown)

            # Populate company fields from analysis results
            company.pricing_model = analysis.pricing_model
            company.is_open_source = analysis.is_open_source
            company.tech_stack = analysis.tech_stack
            company.description = analysis.description
            company.api_available = analysis.api_available
            company.language_support = analysis.language_support
            company.integration_capabilities = analysis.integration_capabilities

            companies.append(company)

        # Return updated state with researched companies
        return {"companies": companies}

    #analyze and given recommendations
    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating recommendations")

        company_data = ", ".join([
            company.json() for company in state.companies
        ])

        messages = [
            SystemMessage(content = self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data))
        ]

        response = self.llm.invoke(messages)
        return {"analysis": response.content}

    def run(self, query:str) ->ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.workflow.invoke(initial_state)
        return ResearchState(**final_state)




