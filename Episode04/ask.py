#!/usr/bin/env python3

import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Optional Chroma imports
try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Query RAG system with FAISS, Pinecone, or Chroma vector stores",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Query FAISS index (default)
  %(prog)s "What is artificial intelligence?"
  
  # Query Pinecone index
  %(prog)s --store pinecone "What is machine learning?"
  
  # Query Chroma index
  %(prog)s --store chroma --chroma-path ./my_chroma "Explain neural networks"
    """
)
parser.add_argument("query", nargs="+", help="Query text to search for")
parser.add_argument("--store", choices=["faiss", "pinecone", "chroma"], default="faiss",
                   help="Vector store to use (default: faiss)")
parser.add_argument("--faiss-path", help="FAISS index directory path (default: FAISS_INDEX_PATH env or 'faiss_index')")
parser.add_argument("--pinecone-key", help="Pinecone API key")
parser.add_argument("--pinecone-index", help="Pinecone index name")
parser.add_argument("--chroma-path", help="Chroma database path (default: CHROMA_PATH env or './chroma_db')")
parser.add_argument("--chroma-index", help="Chroma collection name (default: CHROMA_INDEX env or 'default_index')")
parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of similar documents to retrieve (default: 4)")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

args = parser.parse_args()
query = " ".join(args.query)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY must be set in the environment.")
    sys.exit(1)

# Configure vector store based on selection
if args.store == "faiss":
    FAISS_INDEX_PATH = args.faiss_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index not found at '{FAISS_INDEX_PATH}'. Run rag-improved.py first to create the index.")
        sys.exit(1)
    if args.verbose:
        print(f"*** Using local FAISS vector store: {FAISS_INDEX_PATH} ***")
elif args.store == "pinecone":
    PINECONE_API_KEY = args.pinecone_key or os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = args.pinecone_index or os.getenv("PINECONE_INDEX")
    if not PINECONE_API_KEY or not PINECONE_INDEX:
        print("Error: PINECONE_API_KEY and PINECONE_INDEX must be set for Pinecone store")
        sys.exit(1)
    if args.verbose:
        print(f"*** Using Pinecone vector store: {PINECONE_INDEX} ***")
elif args.store == "chroma":
    if not CHROMA_AVAILABLE:
        print("Error: Chroma dependencies not installed. Run: pip install langchain-chroma chromadb")
        sys.exit(1)
    CHROMA_PATH = args.chroma_path or os.getenv("CHROMA_PATH", "./chroma_db")
    CHROMA_INDEX = args.chroma_index or os.getenv("CHROMA_INDEX", "default_index")
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Chroma database not found at '{CHROMA_PATH}'. Run rag-improved.py first to create the database.")
        sys.exit(1)
    if args.verbose:
        print(f"*** Using Chroma vector store: {CHROMA_PATH}/{CHROMA_INDEX} ***")

if args.verbose:
    print(f"*** Using OpenAI model: {OPENAI_MODEL}")
    print(f"*** Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")

# Setup LangChain components
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
parser_output = StrOutputParser()
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
def get_documents_with_scores(vectorstore, query, store_type, k=4):
    """Get documents with similarity scores"""
    return vectorstore.similarity_search_with_score(query, k=k)

# Query vector store
try:
    if args.store == "faiss":
        if args.verbose:
            print(f"*** Loading FAISS index from: {FAISS_INDEX_PATH} ***")
        
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        if args.verbose:
            total_docs = vectorstore.index.ntotal
            print(f"*** Loaded FAISS index with {total_docs} documents ***")
    
    elif args.store == "pinecone":
        if args.verbose:
            print(f"*** Connecting to Pinecone index: {PINECONE_INDEX} ***")
        
        vectorstore = PineconeVectorStore.from_existing_index(
            PINECONE_INDEX, embeddings
        )
    
    elif args.store == "chroma":
        if args.verbose:
            print(f"*** Loading Chroma database from: {CHROMA_PATH} ***")
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        vectorstore = Chroma(
            client=client,
            collection_name=CHROMA_INDEX,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )
        
        if args.verbose:
            collection = client.get_collection(CHROMA_INDEX)
            total_docs = collection.count()
            print(f"*** Loaded Chroma collection with {total_docs} documents ***")
    
    if args.verbose:
        print(f"*** Searching for: {query} ***")
    
    # Get documents with similarity scores
    docs_with_scores = get_documents_with_scores(vectorstore, query, args.store, k=args.top_k)
    
    # Extract just the documents for the chain
    docs = [doc for doc, score in docs_with_scores]
    
    # Create the context string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use the model directly since we already have the context
    formatted_prompt = prompt.format_messages(context=context, question=query)
    response = model.invoke(formatted_prompt)
    answer = parser_output.invoke(response)
    
    print(f"\nAnswer: {answer}")
    
    # Show source documents with similarity scores
    if docs_with_scores:
        print(f"\nReferences: {len(docs_with_scores)} documents")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            
            if args.store == "pinecone":
                # Pinecone scores are typically between 0-1 (higher is better)
                print(f"  {i}. {doc_id} (from {source}) [similarity: {score:.4f}]: {preview}")
            elif args.store == "faiss":
                # FAISS scores are distances (lower is better), convert to similarity
                similarity = 1 / (1 + score)  # Convert distance to similarity-like score
                print(f"  {i}. {doc_id} (from {source}) [similarity: {similarity:.4f}]: {preview}")
            elif args.store == "chroma":
                # Chroma scores are distances (lower is better), convert to similarity
                similarity = 1 / (1 + score)  # Convert distance to similarity-like score
                print(f"  {i}. {doc_id} (from {source}) [similarity: {similarity:.4f}]: {preview}")
    
except Exception as e:
    print(f"Error querying vector store: {e}")
    sys.exit(1)