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

print(f"*** Using OpenAI model: {OPENAI_MODEL}")
print(f"*** Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")

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

# Custom retriever function to get documents with scores
def get_documents_with_scores(vectorstore, query, k=4):
    """Get documents with similarity scores"""
    if use_pinecone:
        # For Pinecone, use similarity_search_with_score
        return vectorstore.similarity_search_with_score(query, k=k)
    else:
        # For FAISS, use similarity_search_with_score
        return vectorstore.similarity_search_with_score(query, k=k)

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
    
    print(f"*** Searching for: {query} ***")
    
    # Get documents with similarity scores
    docs_with_scores = get_documents_with_scores(vectorstore, query, k=4)
    
    # Extract just the documents for the chain
    docs = [doc for doc, score in docs_with_scores]
    
    # Create the context string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use the model directly since we already have the context
    formatted_prompt = prompt.format_messages(context=context, question=query)
    response = model.invoke(formatted_prompt)
    answer = parser.invoke(response)
    
    print(f"\nAnswer: {answer}")
    
    # Show source documents with similarity scores
    if docs_with_scores:
        print(f"\nReferences: {len(docs_with_scores)} documents")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')
            source_file = doc.metadata.get('source_file', 'unknown')
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            
            if use_pinecone:
                # Pinecone scores are typically between 0-1 (higher is better)
                print(f"  {i}. {doc_id} (from {source_file}) [similarity: {score:.4f}]: {preview}")
            else:
                # FAISS scores are distances (lower is better), convert to similarity
                similarity = 1 / (1 + score)  # Convert distance to similarity-like score
                print(f"  {i}. {doc_id} (from {source_file}) [similarity: {similarity:.4f}]: {preview}")
    
except Exception as e:
    print(f"Error querying vector store: {e}")
    sys.exit(1)
