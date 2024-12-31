#!/usr/bin/env python3

import os
import sys
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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
        load_dotenv()
        
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
            doc_ids = []
            
            print(f"Document store contains {len(docstore_dict)} documents")
            
            for doc in docstore_dict.values():
                if hasattr(doc, 'metadata'):
                    if 'source_file' in doc.metadata:
                        source_files[doc.metadata['source_file']] += 1
                    if 'doc_id' in doc.metadata:
                        doc_ids.append(doc.metadata['doc_id'])
            
            # Source file statistics
            print(f"\nSOURCE FILES ({len(source_files)} unique files):")
            for file, count in source_files.most_common():
                print(f"   • {file}: {count} chunks")
        else:
            print("️   Document store is empty or inaccessible")


    # 7. Health Check
    print(f"\nHEALTH CHECK")
    print("-" * len('HEALTH CHECK'))

    health_score = 0
    max_score = 4
    
    # Check 1: Required files exist
    if all(f in files_in_dir for f in required_files):
        print("✅ All required FAISS files present")
        health_score += 1
    else:
        print("❌ Missing required FAISS files")
    
    # Check 2: Index loads successfully
    if 'vectorstore' in locals():
        print("✅ Index loads without errors")
        health_score += 1
    else:
        print("❌ Index failed to load")
    
    # Check 3: Contains documents
    if total_docs > 0:
        print(f"✅ Index contains documents ({total_docs:,})")
        health_score += 1
    else:
        print("❌ Index is empty")

    # Check 4: Recent activity
    if required_files and all(f in files_in_dir for f in required_files):
        newest_file = max([os.path.join(faiss_path, f) for f in required_files], 
                         key=os.path.getmtime)
        days_old = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(newest_file))).days
        if days_old < 30:
            print(f"✅ Index recently updated ({days_old} days ago)")
            health_score += 1
        else:
            print(f"⚠️  Index last updated {days_old} days ago")
    
    print(f"\nOverall Health Score: {health_score}/{max_score}")
    
    if health_score >= 4:
        print("🟢 Index appears healthy and ready for use")
    elif health_score >= 2:
        print("🟡 Index has some issues but may still be usable")
    else:
        print("🔴 Index has significant issues and may need rebuilding")

def test_search(faiss_path, query="test search"):
    """Test search functionality on the FAISS index"""
    
    print(f"\nSEARCH TEST")
    print("-" * len('SEARCH TEST'))
    
    try:
        load_dotenv()
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
            if hasattr(doc, 'metadata') and 'source_file' in doc.metadata:
                print(f"   Source: {doc.metadata['source_file']}")
        
        return True
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <faiss_index_path> [search_query]")
        print(f"Example: {sys.argv[0]} ./faiss_index")
        print(f"Example: {sys.argv[0]} ./faiss_index 'machine learning'")
        sys.exit(1)
    
    faiss_path = sys.argv[1]
    search_query = sys.argv[2] if len(sys.argv) > 2 else None
    
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
