import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.tools import DuckDuckGoSearchResults

from qa_tools import load_documents, build_vectorstore

load_dotenv()

# Load documents and retriever
docs = load_documents()
retriever = build_vectorstore(docs)

# LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.4, "max_length": 512}
)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Tools
kb_tool = Tool(
    name="InternalDocs",
    func=lambda q: qa_chain.run(q),
    description="Answer internal HR and onboarding questions."
)

web_search = Tool(
    name="WebSearch",
    func=DuckDuckGoSearchResults().run,
    description="Search the web for up-to-date or external information."
)

# Initialize Agent
agent_executor = initialize_agent(
    tools=[kb_tool, web_search],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

def run_agent(query: str):
    return agent_executor.run(query)
