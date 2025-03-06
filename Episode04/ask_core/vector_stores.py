"""Vector store implementations for the ask_core system."""

import os
from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore

# Optional imports
try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStoreManager:
    """Manager for different vector store implementations."""
    
    def __init__(self, store_type: str, embeddings: Embeddings, **kwargs):
        self.store_type = store_type
        self.embeddings = embeddings
        self.vectorstore = None
        
        if store_type == "faiss":
            self._setup_faiss(**kwargs)
        elif store_type == "pinecone":
            self._setup_pinecone(**kwargs)
        elif store_type == "chroma":
            self._setup_chroma(**kwargs)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def _setup_faiss(self, faiss_path: str = None, **kwargs):
        """Setup FAISS vector store."""
        faiss_path = faiss_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at '{faiss_path}'. Run rag.py first to create the index.")
        
        self.vectorstore = FAISS.load_local(
            faiss_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.store_path = faiss_path
    
    def _setup_pinecone(self, pinecone_key: str = None, pinecone_index: str = None, **kwargs):
        """Setup Pinecone vector store."""
        pinecone_key = pinecone_key or os.getenv("PINECONE_API_KEY")
        pinecone_index = pinecone_index or os.getenv("PINECONE_INDEX")
        
        if not pinecone_key or not pinecone_index:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set for Pinecone store")
        
        self.vectorstore = PineconeVectorStore.from_existing_index(
            pinecone_index, self.embeddings
        )
        self.store_path = pinecone_index
    
    def _setup_chroma(self, chroma_path: str = None, chroma_index: str = None, **kwargs):
        """Setup Chroma vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma dependencies not installed. Run: pip install langchain-chroma chromadb")
        
        chroma_path = chroma_path or os.getenv("CHROMA_PATH", "./chroma_db")
        chroma_index = chroma_index or os.getenv("CHROMA_INDEX", "default_index")
        
        if not os.path.exists(chroma_path):
            raise FileNotFoundError(f"Chroma database not found at '{chroma_path}'. Run rag.py first to create the database.")
        
        client = chromadb.PersistentClient(path=chroma_path)
        self.vectorstore = Chroma(
            client=client,
            collection_name=chroma_index,
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
        self.store_path = chroma_path
        self.chroma_client = client
        self.chroma_index = chroma_index
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Get documents with similarity scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_document_count(self) -> int:
        """Get total number of documents in the vector store."""
        if self.store_type == "faiss":
            return self.vectorstore.index.ntotal
        elif self.store_type == "chroma":
            collection = self.chroma_client.get_collection(self.chroma_index)
            return collection.count()
        else:  # Pinecone doesn't have a direct count method
            return -1  # Unknown count
    
    def get_chunks_by_source_and_range(self, source: str, start_idx: int, end_idx: int) -> List[Document]:
        """
        Retrieve chunks from a specific source file within a chunk index range.
        This is used for context expansion.
        """
        try:
            if self.store_type == "faiss":
                # For FAISS, we need to scan through all documents
                all_docs = []
                total_docs = self.vectorstore.index.ntotal
                
                for i in range(total_docs):
                    try:
                        doc_id = self.vectorstore.index_to_docstore_id[i]
                        doc = self.vectorstore.docstore.search(doc_id)
                        if (doc and 
                            doc.metadata.get('source') == source and
                            doc.metadata.get('chunk_index') is not None and
                            start_idx <= doc.metadata.get('chunk_index') <= end_idx):
                            all_docs.append(doc)
                    except (KeyError, IndexError):
                        continue
                        
                return sorted(all_docs, key=lambda d: d.metadata.get('chunk_index', 0))
                
            elif self.store_type == "chroma":
                # For Chroma, we can use metadata filtering
                collection = self.vectorstore._collection
                
                # Query with metadata filter
                results = collection.get(
                    where={
                        "$and": [
                            {"source": {"$eq": source}},
                            {"chunk_index": {"$gte": start_idx}},
                            {"chunk_index": {"$lte": end_idx}}
                        ]
                    },
                    include=['documents', 'metadatas']
                )
                
                docs = []
                for doc_text, metadata in zip(results['documents'], results['metadatas']):
                    docs.append(Document(page_content=doc_text, metadata=metadata))
                
                return sorted(docs, key=lambda d: d.metadata.get('chunk_index', 0))
                
            else:
                # Pinecone doesn't support efficient metadata range queries
                # Fall back to similarity search and filter
                return []
                
        except Exception as e:
            print(f"Warning: Could not retrieve chunks for range expansion: {e}")
            return []

    def get_store_info(self) -> dict:
        """Get information about the vector store."""
        info = {
            "store_type": self.store_type,
            "store_path": getattr(self, 'store_path', 'unknown'),
            "document_count": self.get_document_count()
        }
        
        if self.store_type == "chroma":
            info["collection_name"] = getattr(self, 'chroma_index', 'unknown')
        
        return info