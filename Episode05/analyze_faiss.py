#!/usr/bin/env python3

import os
import sys
import json
import argparse
from collections import Counter
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def analyze_faiss_index(faiss_path, num_entries=3, truncate_content=True, keys_only=False):
    """Analyze a FAISS index directory and provide comprehensive statistics"""
    
    # Check if directory exists
    if not os.path.exists(faiss_path):
        print(f"Error: FAISS directory '{faiss_path}' does not exist.")
        sys.exit()
        return
    
    if not os.path.isdir(faiss_path):
        print(f"Error: '{faiss_path}' is not a directory.")
        return

    files_in_dir = os.listdir(faiss_path)
    required_files = ['index.faiss', 'index.pkl']
    
    for file in required_files:
        if file not in files_in_dir:
            print(f"{file}: Missing")
    
    # Try to load the index
    try:
        # print(f"\nLoading FAISS index...")
        
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        # print(f"   Using embedding model: {embedding_model}")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        # print("Successfully loaded FAISS index")
        
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return


    total_docs = vectorstore.index.ntotal
    embedding_dim = vectorstore.index.d
    
    print(f"Index: {os.path.basename(faiss_path)}")
    print(f"Document store contains {total_docs:,} chunks")
    
    # Load and display index metadata
    metadata_path = os.path.join(faiss_path, "index_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"\nINDEX METADATA:")
                for key, value in sorted(metadata.items()):
                    print(f"   {key}: {value}")
        except Exception as e:
            print(f"*** Warning: Could not load index metadata: {e} ***")
    
    if total_docs == 0:
        print("Index is empty")
        return

    # Access the docstore to get metadata
    if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
        docstore_dict = vectorstore.docstore._dict
        
        if docstore_dict:
            # Analyze metadata
            source_files = Counter()
            
            for doc in docstore_dict.values():
                if hasattr(doc, 'metadata'):
                    if 'source' in doc.metadata:
                        source_files[doc.metadata['source']] += 1
            
            # Source file statistics
            if source_files:
                print(f"\nSOURCE FILES ({len(source_files)} unique files):")
                for file, count in source_files.most_common():
                    print(f"   • {file}: {count} chunks")
            
            # Show sample entries
            if num_entries > 0:
                sample_size = min(num_entries, len(docstore_dict))
                
                if keys_only:
                    entries_text = f"FIRST {sample_size} KEYS" if sample_size != 1 else "FIRST KEY"
                    print(f"\n{entries_text}:")
                    print("-" * len(entries_text))
                    
                    doc_items = list(docstore_dict.items())[:sample_size]
                    for i, (doc_key, doc) in enumerate(doc_items, 1):
                        print(f"Key {i}: {doc_key}")
                else:
                    entries_text = f"FIRST {sample_size} ENTRIES" if sample_size != 1 else "FIRST ENTRY"
                    print(f"\n{entries_text}:")
                    print("-" * len(entries_text))
                    
                    doc_items = list(docstore_dict.items())[:sample_size]
                    for i, (doc_key, doc) in enumerate(doc_items, 1):
                        print(f"\nEntry {i}:")
                        print(f"   Key: {doc_key}")
                        
                        if hasattr(doc, 'metadata') and doc.metadata:
                            print(f"   Metadata:")
                            # Show all metadata fields, sorted for consistency
                            for key, value in sorted(doc.metadata.items()):
                                print(f"      {key}: {value}")
                        else:
                            print(f"   Metadata: None")
                        
                        if hasattr(doc, 'page_content'):
                            content = doc.page_content.strip()
                            if truncate_content and len(content) > 100:
                                display_content = content[:100] + "..."
                            else:
                                display_content = content
                            # Replace newlines with spaces for cleaner display (only if truncating)
                            if truncate_content:
                                display_content = display_content.replace('\n', ' ').replace('\r', ' ')
                            print(f"   Content: {display_content}")
                        else:
                            print(f"   Content: None")
        else:
            print("No documents found in index")
    else:
        print("Document store is empty or inaccessible")


def test_search(faiss_path, query="test search"):
    """Test search functionality on the FAISS index"""
    
    print(f"\nSEARCH TEST")
    print("-" * len('SEARCH TEST'))
    
    try:
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\nResult {i} (score: {score:.4f})")
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                print(f"   Source: {doc.metadata['source']}")
            # if hasattr(doc, 'metadata'):
            #     print(f"   Metadata: {doc.metadata}")
            if hasattr(doc, 'page_content'):
                # preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                preview = doc.page_content
                print(f"   Content: {preview}")
        
        return True
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def main():
    # Get FAISS index path from environment
    default_faiss_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    
    parser = argparse.ArgumentParser(
        description="Analyze FAISS index statistics and content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use FAISS_INDEX_PATH from .env
  %(prog)s --entries 5                       # Show first 5 entries instead of 3
  %(prog)s --no-truncate                     # Show full content without truncation
  %(prog)s --keys-only                       # Show only document keys
  %(prog)s --search "machine learning"       # Search with default path
  %(prog)s ./custom_index                    # Use specific path
  %(prog)s ./custom_index --entries 10 --search "AI"
  %(prog)s --keys-only --entries 10          # Show first 10 keys only
        """
    )
    
    parser.add_argument(
        'faiss_path', 
        nargs='?', 
        default=default_faiss_path,
        help=f'Path to FAISS index directory (default: {default_faiss_path} from FAISS_INDEX_PATH)'
    )
    
    parser.add_argument(
        '--entries', '-e',
        type=int,
        default=3,
        help='Number of sample entries to display (default: 3, use 0 to disable)'
    )
    
    parser.add_argument(
        '--no-truncate', '-n',
        action='store_true',
        help='Show full content without truncation (default: truncate at 100 characters)'
    )
    
    parser.add_argument(
        '--keys-only', '-k',
        action='store_true',
        help='Show only document keys without metadata or content'
    )
    
    parser.add_argument(
        '--search', '-s',
        metavar='QUERY',
        help='Search query to test after analysis'
    )
    
    args = parser.parse_args()
    
    # Show which path is being used if it's the default
    # if args.faiss_path == default_faiss_path:
    #    print(f"Using FAISS index path from environment: {args.faiss_path}")
    
    # Main analysis
    analyze_faiss_index(args.faiss_path, args.entries, not args.no_truncate, args.keys_only)
    
    # Optional search test
    if args.search:
        test_search(args.faiss_path, args.search)

if __name__ == "__main__":
    main()
