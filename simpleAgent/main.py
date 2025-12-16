from click import command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model = ChatOpenAI( #initalize openAI model
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key = os.getenv("OPENAI_API_KEY")
)

server_params = StdioServerParameters( #establish background process with Stdio for communication with mcp tools
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"]
)

#connect to mcp client

async def main():
    async with stdio_client(server_params) as (read, write): #we can read the result of the tool & write to it y calling the tool
        async with ClientSession(read, write) as session: #create new session
            await session.initialize() #initalize session
            tools = await load_mcp_tools(session) #load tools
            agent = create_react_agent(model, tools) #create agent

            messages = [
                {
                    "role":"system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use appropriate tools to help the user"
                }
            ]

            print("Available Tools -", *[tool.name for tool in tools])
            print("- * 60")

            while True:
                user_input = input("\nYou: ")
                if user_input == "quit":
                    print("Goodbye")
                    break

                messages.append({"role":"user", "content":user_input[:175000]})

                try:
                    agent_response = await agent.ainvoke({"messages":messages})

                    ai_message = agent_response["messages"][-1].content
                    print("\nAgent: ", ai_message)
                except Exception as e:
                    print("error: ", e)

if __name__ == "__main__":
    asyncio.run(main())


