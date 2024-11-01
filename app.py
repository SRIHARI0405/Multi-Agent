import streamlit as st
import os
from dotenv import load_dotenv
from tools.custom_tools import TavilySearchTool, KaggleDatasetSearchTool, HuggingFaceSearchTool, GitHubSearchTool
from agents.research_agent import create_research_agent
from agents.usecase_agent import create_usecase_agent
from agents.resource_agent import create_resource_agent
from datetime import datetime

def initialize_tools():
    return [
        TavilySearchTool(),
        KaggleDatasetSearchTool(),
        HuggingFaceSearchTool(),
        GitHubSearchTool(),
    ]

def format_markdown_report(company, research_results, use_cases, resources):
    return f"""# AI Implementation Analysis for {company}

## 1. Industry and Company Research
{research_results}

## 2. AI/ML Use Cases
{use_cases}

## 3. Implementation Resources
{resources}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def main():
    load_dotenv()
    st.title("Multi-Agent AI Implementation Analysis")

    # Input fields
    company = st.text_input("Enter company name:")
    industry = st.text_input("Enter industry:")

    if st.button("Generate Analysis"):
        if not company or not industry:
            st.error("Please enter both company and industry.")
            return

        with st.spinner("Analyzing..."):
            tools = initialize_tools()

            # Create agents
            research_agent = create_research_agent(tools)
            usecase_agent = create_usecase_agent(tools)
            resource_agent = create_resource_agent(tools)

            # Execute research
            research_prompt = f"Research {company} in the {industry} industry. Focus on their business model, key offerings, and strategic focus areas."
            research_results = research_agent.invoke({"input": research_prompt})["output"]
            st.write("Research completed.")

            # Generate use cases
            usecase_prompt = f"Based on this research about {company}: {research_results}\nGenerate specific AI/ML use cases."
            use_cases = usecase_agent.invoke({"input": usecase_prompt})["output"]
            st.write("Use cases generated.")

            # Collect resources
            resource_prompt = f"Find implementation resources for these use cases: {use_cases}"
            resources = resource_agent.invoke({"input": resource_prompt})["output"]
            st.write("Resources collected.")

            # Generate final report
            report = format_markdown_report(company, research_results, use_cases, resources)
            st.markdown(report)

            # Save report
            with open(f"{company}_analysis_{datetime.now().strftime('%Y%m%d')}.md", "w") as f:
                f.write(report)
            st.success("Analysis complete! Report has been saved.")

if __name__ == "__main__":
    main()