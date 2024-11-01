
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

def create_resource_agent(tools):
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Resource Collection Agent.
        For each use case:
        1. Find relevant datasets
        2. Identify implementation resources
        3. Suggest GenAI solutions
        Provide direct links and clear implementation guidelines."""),
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