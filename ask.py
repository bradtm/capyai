#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Validate arguments
if len(sys.argv) < 2:
    print(f"Syntax: {sys.argv[0]} [query...]")
    sys.exit(1)

query = " ".join(sys.argv[1:])

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# FAISS configuration (same as rag-faiss.py)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# Model configuration (same as rag-faiss.py)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY must be set in the environment.")
    sys.exit(1)

# Determine which vector store to use (same logic as rag-faiss.py)
use_pinecone = bool(PINECONE_API_KEY and PINECONE_INDEX)

if use_pinecone:
    print(f"*** Using Pinecone vector store: {PINECONE_INDEX} ***")
else:
    print(f"*** Using local FAISS vector store: {FAISS_INDEX_PATH} ***")
    # Check if FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index not found at '{FAISS_INDEX_PATH}'. Run rag-faiss.py first to create the index.")
        sys.exit(1)

print(f"Using OpenAI model: {OPENAI_MODEL}")
print(f"Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")

# Setup LangChain components
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
parser = StrOutputParser()
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# Setup LangChain components
template = """
You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
- You must ONLY use information from the context provided below
- If the context does not contain information to answer the question, you MUST respond with "I don't know"
- Do NOT use your general knowledge or training data
- Do NOT make assumptions or provide information not explicitly in the context

Context: {context}

Question: {question}

Answer based ONLY on the context above:"""


prompt = ChatPromptTemplate.from_template(template)

# Query vector store
try:
    if use_pinecone:
        print(f"*** Querying Pinecone index: {PINECONE_INDEX} ***")
        
        vectorstore = PineconeVectorStore.from_existing_index(
            PINECONE_INDEX, embeddings
        )
    else:
        print(f"*** Loading FAISS index from: {FAISS_INDEX_PATH} ***")
        
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        total_docs = vectorstore.index.ntotal
        print(f"*** Loaded FAISS index with {total_docs} documents ***")
    
    # Create chain with source tracking (same for both vector stores)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(
        answer=prompt | model | parser
    )
    
    print(f"*** Searching for: {query} ***")
    result = chain.invoke(query)
    print(f"\nAnswer: {result['answer']}")
    
    # Show source documents
    if result.get('context'):
        print(f"\nSources found: {len(result['context'])} documents")
        for i, doc in enumerate(result['context'], 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')
            source_file = doc.metadata.get('source_file', 'unknown')
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            print(f"  {i}. {doc_id} (from {source_file}): {preview}")
    
except Exception as e:
    print(f"Error querying vector store: {e}")
    sys.exit(1)
