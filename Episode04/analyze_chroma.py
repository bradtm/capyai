#!/usr/bin/env python3

import os
import sys
import argparse
from collections import Counter
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

# Optional Chroma imports
try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Load environment variables
load_dotenv()

class OllamaEmbeddings:
    """Ollama embeddings using local models."""
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url.rstrip('/')

    def embed_documents(self, texts):
        import requests
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text):
        import requests
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

def get_collection_embedding_model(chroma_path, collection_name):
    """Get embedding model from collection metadata"""
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=collection_name)
        metadata = collection.metadata
        return metadata.get("embedding_model")
    except Exception:
        return None

def analyze_chroma_index(chroma_path, collection_name, num_entries=3, truncate_content=True, keys_only=False):
    """Analyze a Chroma index and provide comprehensive statistics"""
    
    if not CHROMA_AVAILABLE:
        print("Error: Chroma dependencies not installed. Run: pip install langchain-chroma chromadb")
        sys.exit(1)
    
    # Check if directory exists
    if not os.path.exists(chroma_path):
        print(f"Error: Chroma directory '{chroma_path}' does not exist.")
        sys.exit(1)
        return
    
    if not os.path.isdir(chroma_path):
        print(f"Error: '{chroma_path}' is not a directory.")
        return

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        
        # List available collections
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Error: Collection '{collection_name}' not found.")
            print(f"Available collections: {collection_names}")
            return
        
        collection = client.get_collection(name=collection_name)
        
    except Exception as e:
        print(f"Error connecting to Chroma database: {e}")
        return

    # Get collection info
    total_docs = collection.count()
    metadata = collection.metadata or {}
    
    print(f"Collection: {collection_name}")
    print(f"Document store contains {total_docs:,} chunks")
    
    if metadata:
        print(f"\nCOLLECTION METADATA:")
        for key, value in sorted(metadata.items()):
            print(f"   {key}: {value}")
    
    if total_docs == 0:
        print("Collection is empty")
        return

    # Get all documents with metadata
    try:
        results = collection.get(
            include=['metadatas', 'documents'],
            # limit=total_docs if total_docs < 10000 else 10000  # Limit to prevent memory issues
        )
        
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        
        if documents:
            # Analyze source files
            source_files = Counter()
            
            for metadata in metadatas:
                if metadata and 'source' in metadata:
                    source_files[metadata['source']] += 1
            
            # Source file statistics
            if source_files:
                print(f"\nSOURCE FILES ({len(source_files)} unique files):")
                for file, count in source_files.most_common():
                    print(f"   • {file}: {count} chunks")
            
            # Show sample entries
            if num_entries > 0:
                sample_size = min(num_entries, len(documents))
                
                if keys_only:
                    entries_text = f"FIRST {sample_size} KEYS" if sample_size != 1 else "FIRST KEY"
                    print(f"\n{entries_text}:")
                    print("-" * len(entries_text))
                    
                    for i in range(sample_size):
                        print(f"Key {i+1}: {ids[i]}")
                else:
                    entries_text = f"FIRST {sample_size} ENTRIES" if sample_size != 1 else "FIRST ENTRY"
                    print(f"\n{entries_text}:")
                    print("-" * len(entries_text))
                    
                    for i in range(sample_size):
                        print(f"\nEntry {i+1}:")
                        print(f"   Key: {ids[i]}")
                        
                        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                        if metadata:
                            print(f"   Metadata:")
                            # Show all metadata fields, sorted for consistency
                            for key, value in sorted(metadata.items()):
                                print(f"      {key}: {value}")
                        else:
                            print(f"   Metadata: None")
                        
                        document = documents[i] if i < len(documents) else ""
                        if document:
                            content = document.strip()
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
            print("No documents found in collection")
            
    except Exception as e:
        print(f"Error retrieving documents: {e}")

def test_search(chroma_path, collection_name, query="test search"):
    """Test search functionality on the Chroma index"""
    
    print(f"\nSEARCH TEST")
    print("-" * len('SEARCH TEST'))
    
    try:
        # Determine embedding model
        stored_embedding_model = get_collection_embedding_model(chroma_path, collection_name)
        embedding_model = stored_embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize embeddings based on model type
        if embedding_model in ["bge-m3", "nomic-embed-text"]:
            embeddings = OllamaEmbeddings(model=embedding_model)
        else:
            embeddings = OpenAIEmbeddings(model=embedding_model)
        
        client = chromadb.PersistentClient(path=chroma_path)
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_path
        )

        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            # Convert distance to similarity-like score
            similarity = 1 / (1 + score)
            print(f"\nResult {i} (similarity: {similarity:.4f})")
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                print(f"   Source: {doc.metadata['source']}")
            if hasattr(doc, 'page_content'):
                preview = doc.page_content
                print(f"   Content: {preview}")
        
        return True
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def main():
    # Get Chroma paths from environment
    default_chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    default_collection = os.getenv("CHROMA_INDEX", "default_index")
    
    parser = argparse.ArgumentParser(
        description="Analyze Chroma index statistics and content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use CHROMA_PATH and CHROMA_INDEX from .env
  %(prog)s --entries 5                       # Show first 5 entries instead of 3
  %(prog)s --no-truncate                     # Show full content without truncation
  %(prog)s --keys-only                       # Show only document keys
  %(prog)s --search "machine learning"       # Search with default path
  %(prog)s ./custom_chroma                   # Use specific path
  %(prog)s ./custom_chroma --collection my_docs --entries 10 --search "AI"
  %(prog)s --keys-only --entries 10          # Show first 10 keys only
        """
    )
    
    parser.add_argument(
        'chroma_path', 
        nargs='?', 
        default=default_chroma_path,
        help=f'Path to Chroma database directory (default: {default_chroma_path} from CHROMA_PATH)'
    )
    
    parser.add_argument(
        '--collection', '-c',
        default=default_collection,
        help=f'Chroma collection name (default: {default_collection} from CHROMA_INDEX)'
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
    
    # Main analysis
    analyze_chroma_index(args.chroma_path, args.collection, args.entries, not args.no_truncate, args.keys_only)
    
    # Optional search test
    if args.search:
        test_search(args.chroma_path, args.collection, args.search)

if __name__ == "__main__":
    main()
