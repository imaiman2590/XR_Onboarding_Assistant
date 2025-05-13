import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def load_documents():
    faq_loader = TextLoader("knowledge/onboarding_faq.txt")
    faq_docs = faq_loader.load()

    notion_loader = NotionDBLoader(
        integration_token=os.getenv("NOTION_INTEGRATION_TOKEN"),
        database_id=os.getenv("NOTION_DATABASE_ID")
    )
    notion_docs = notion_loader.load()

    return faq_docs + notion_docs

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_type="similarity", k=3)
