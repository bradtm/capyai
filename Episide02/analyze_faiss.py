#!/usr/bin/env python3

import os
import sys
from collections import Counter
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def analyze_faiss_index(faiss_path):
    """Analyze a FAISS index directory and provide comprehensive statistics"""
    
    # Check if directory exists
    if not os.path.exists(faiss_path):
        print(f"Error: FAISS directory '{faiss_path}' does not exist.")
        return
    
    if not os.path.isdir(faiss_path):
        print(f"Error: '{faiss_path}' is not a directory.")
        return

    files_in_dir = os.listdir(faiss_path)
    required_files = ['index.faiss', 'index.pkl']
    
    for file in required_files:
        if file not in files_in_dir:
            print(f"{file}: Missing")
    
    # List all files in directory
    print(f"\nAll files in directory:")
    for file in sorted(files_in_dir):
        file_path = os.path.join(faiss_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   • {file} ({size:,} bytes)")
    
    # Try to load the index
    try:
        print(f"\nLoading FAISS index...")
        
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        # print(f"   Using embedding model: {embedding_model}")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("Successfully loaded FAISS index")
        
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return


    total_docs = vectorstore.index.ntotal
    embedding_dim = vectorstore.index.d
    
    print(f"\nTotal documents: {total_docs:,}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Index type: {type(vectorstore.index).__name__}")
    
    if hasattr(vectorstore.index, 'metric_type'):
        print(f"Distance metric: {vectorstore.index.metric_type}")


    # Access the docstore to get metadata
    if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
        docstore_dict = vectorstore.docstore._dict
        
        if docstore_dict:
            # Analyze metadata
            source_files = Counter()
            
            print(f"Document store contains {len(docstore_dict)} documents")
            
            for doc in docstore_dict.values():
                if hasattr(doc, 'metadata'):
                    if 'source' in doc.metadata:
                        source_files[doc.metadata['source']] += 1
            
            # Source file statistics
            print(f"\nSOURCE FILES ({len(source_files)} unique files):")
            for file, count in source_files.most_common():
                print(f"   • {file}: {count} chunks")
            
            # Show first 3 entries
            print(f"\nFIRST 3 ENTRIES:")
            print("-" * len('FIRST 3 ENTRIES'))
            
            doc_items = list(docstore_dict.items())[:3]
            for i, (doc_key, doc) in enumerate(doc_items, 1):
                print(f"\nEntry {i}:")
                print(f"   Key: {doc_key}")
                
                if hasattr(doc, 'metadata'):
                    print(f"   Metadata:")
                    # Show all metadata fields, sorted for consistency
                    for key, value in sorted(doc.metadata.items()):
                        print(f"      {key}: {value}")
                else:
                    print(f"   Metadata: None")
                
                if hasattr(doc, 'page_content'):
                    content = doc.page_content.strip()
                    if len(content) > 100:
                        truncated_content = content[:100] + "..."
                    else:
                        truncated_content = content
                    # Replace newlines with spaces for cleaner display
                    truncated_content = truncated_content.replace('\n', ' ').replace('\r', ' ')
                    print(f"   Content: {truncated_content}")
                else:
                    print(f"   Content: None")
        else:
            print("️   Document store is empty or inaccessible")


def test_search(faiss_path, query="test search"):
    """Test search functionality on the FAISS index"""
    
    print(f"\nSEARCH TEST")
    print("-" * len('SEARCH TEST'))
    
    try:
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        results = vectorstore.similarity_search(query, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            if hasattr(doc, 'metadata'):
                print(f"   Metadata: {doc.metadata}")
            if hasattr(doc, 'page_content'):
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"   Content: {preview}")
        
        # Test with scores
        print(f"\n\nTesting search with similarity scores...")
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\nResult {i} (score: {score:.4f})")
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                print(f"   Source: {doc.metadata['source']}")
        
        return True
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def main():
    # Get FAISS index path from environment or command line
    default_faiss_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    
    if len(sys.argv) == 1:
        # No arguments - use environment default
        faiss_path = default_faiss_path
        search_query = None
        print(f"Using FAISS index path from environment: {faiss_path}")
    elif len(sys.argv) == 2:
        # Check if arg is a search query (not a path)
        arg = sys.argv[1]
        if os.path.exists(arg) or '/' in arg or '\\' in arg:
            # Looks like a path
            faiss_path = arg
            search_query = None
        else:
            # Assume it's a search query, use default path
            faiss_path = default_faiss_path
            search_query = arg
            print(f"Using FAISS index path from environment: {faiss_path}")
    elif len(sys.argv) == 3:
        # Traditional usage: path and query
        faiss_path = sys.argv[1]
        search_query = sys.argv[2]
    else:
        print(f"Usage: {sys.argv[0]} [faiss_index_path] [search_query]")
        print(f"Examples:")
        print(f"  {sys.argv[0]}                              # Use FAISS_INDEX_PATH from .env")
        print(f"  {sys.argv[0]} 'search query'               # Use FAISS_INDEX_PATH from .env with query")
        print(f"  {sys.argv[0]} ./faiss_index                # Use specific path")
        print(f"  {sys.argv[0]} ./faiss_index 'search query' # Use specific path with query")
        sys.exit(1)
    
    # Main analysis
    analyze_faiss_index(faiss_path)
    
    # Optional search test
    if search_query:
        test_search(faiss_path, search_query)
    else:
        # Ask if user wants to test search
        print(f"\n" + "=" * 80)
        print("TIP: Run with a search query to test search functionality:")
        print(f"   python {sys.argv[0]} {faiss_path} 'your search query'")

if __name__ == "__main__":
    main()
