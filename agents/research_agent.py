from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

def create_research_agent(tools):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Research Agent specialized in analyzing industries and companies.
        Your tasks are:
        1. Industry Analysis
        2. Company Analysis
        3. Key Offerings
        4. Strategic Focus Areas

        Provide a structured analysis with clear sections and bullet points."""),
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