from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

def create_usecase_agent(tools):
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Use Case Generation Agent.
        Based on the research provided, generate specific AI/ML use cases.
        Format each use case with:
        1. Title
        2. Business Problem
        3. Proposed Solution
        4. Expected Benefits
        5. Implementation Considerations"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor