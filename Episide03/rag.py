#!/usr/bin/env python3

"""
RAG Improved - Universal Document and Web Page Processor

Combines the best features of rag.py and rag-chroma.py:
- Supports FAISS, Pinecone, and Chroma vector stores
- Processes media files (audio, video, text, PDF) and web pages
- Configurable chunk size, overlap, and embedding models
- Smart web content extraction with garbage filtering
- Progress bars and cost estimation
- Environment variable configuration with CLI overrides
"""

import os

# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import json
import whisper
import requests
import re
import argparse
import hashlib
import datetime
from typing import List, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

# Vector store imports (with optional handling)
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Load environment variables
load_dotenv()


class EmbeddingModel:
    """Base class for embedding models with cost tracking."""
    name: str
    cost_per_1k_tokens: float
    max_tokens: int
    dimension: int

    @classmethod
    def estimate_cost(cls, token_count: int) -> float:
        return (token_count / 1000.0) * cls.cost_per_1k_tokens


class OpenAIAda002(EmbeddingModel):
    name = "text-embedding-ada-002"
    cost_per_1k_tokens = 0.0001
    max_tokens = 8191
    dimension = 1536


class OpenAI3Small(EmbeddingModel):
    name = "text-embedding-3-small"
    cost_per_1k_tokens = 0.00002
    max_tokens = 8191
    dimension = 1536


class OpenAI3Large(EmbeddingModel):
    name = "text-embedding-3-large"
    cost_per_1k_tokens = 0.00013
    max_tokens = 8191
    dimension = 3072


class OllamaModel(EmbeddingModel):
    name = "bge-m3"
    cost_per_1k_tokens = 0.0
    max_tokens = 8192
    dimension = 1024


class NomicEmbedText(EmbeddingModel):
    name = "nomic-embed-text"
    cost_per_1k_tokens = 0.0
    max_tokens = 8192
    dimension = 768


class OllamaEmbeddings(Embeddings):
    """Ollama embeddings using local models."""
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Ollama."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Ollama."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]


def load_processed_files(processed_files_path):
    """Load the list of already processed files from disk"""
    if os.path.exists(processed_files_path):
        try:
            with open(processed_files_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"*** Warning: Could not load processed files metadata ({e}). Treating all files as new. ***")
            return {}
    return {}


def save_processed_files(processed_files, processed_files_path):
    """Save the list of processed files to disk"""
    try:
        dirname = os.path.dirname(processed_files_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(processed_files_path, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"*** Warning: Could not save processed files metadata ({e}). ***")


def get_file_info(filepath):
    """Get file modification time and size for change detection"""
    stat = os.stat(filepath)
    return {
        'mtime': stat.st_mtime,
        'size': stat.st_size
    }


def file_has_changed(filepath, processed_files):
    """Check if a file has changed since last processing"""
    filename = os.path.basename(filepath)
    if filename not in processed_files:
        return True
    
    current_info = get_file_info(filepath)
    stored_info = processed_files[filename]
    
    return (current_info['mtime'] != stored_info['mtime'] or 
            current_info['size'] != stored_info['size'])


def extract_web_content(url):
    """Extract text content from a web page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"*** Fetching web page: {url} ***")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove unwanted elements more aggressively
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside', 
            'noscript', 'iframe', 'form', 'button', 'input',
            # Wikipedia-specific noise
            '.navbox', '.infobox', '.mbox', '.sidebar', '.toc', 
            '.navigation-not-searchable', '.printfooter', '.catlinks',
            '#toc', '#siteSub', '#contentSub', '#jump-to-nav', '#mw-navigation',
            '.reference', '.reflist', '.citation', '.hatnote', '.dablink',
            # Common site elements
            '.advertisement', '.ad', '.ads', '.social-share', '.share-buttons',
            '.comments', '.related-articles', '.author-bio', '.tags'
        ]
        
        # Remove by tag name (first 6 items)
        for tag_name in unwanted_tags[:6]:
            for element in soup(tag_name):
                element.decompose()
        
        # Remove by CSS selector (remaining items)
        for selector in unwanted_tags[6:]:
            for element in soup.select(selector):
                element.decompose()
        
        # Get page title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "Untitled"
        
        # Improved content selectors with site-specific handling
        content_selectors = [
            # Wikipedia specific
            '#mw-content-text .mw-parser-output',
            '#bodyContent #mw-content-text',
            # Generic article content
            'article', '[role="main"]', 'main',
            # Blog/CMS patterns
            '.post-content', '.entry-content', '.article-content',
            '.content', '#content', '.main-content',
            # News sites
            '.story-body', '.article-body', '.post-body'
        ]
        
        content_text = ""
        selected_selector = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_text = content_elem.get_text()
                selected_selector = selector
                break
        
        # If no main content found, try more aggressive extraction
        if not content_text:
            print("*** Warning: Using fallback content extraction ***")
            # Try to find the longest text block
            text_blocks = []
            for elem in soup.find_all(['p', 'div', 'article', 'section']):
                text = elem.get_text().strip()
                if len(text) > 100:  # Only consider substantial text blocks
                    text_blocks.append(text)
            
            if text_blocks:
                # Take the longest text blocks (likely main content)
                text_blocks.sort(key=len, reverse=True)
                content_text = '\n\n'.join(text_blocks[:5])  # Top 5 longest blocks
            else:
                body = soup.find('body')
                if body:
                    content_text = body.get_text()
                else:
                    content_text = soup.get_text()
        
        print(f"*** Used selector: {selected_selector or 'fallback'} ***")
        
        # Enhanced text cleaning
        lines = []
        for line in content_text.splitlines():
            line = line.strip()
            # Skip very short lines (likely navigation/UI elements)
            if len(line) < 3:
                continue
            # Skip lines that are mostly punctuation or numbers
            if len([c for c in line if c.isalpha()]) < len(line) * 0.5:
                continue
            # Skip common navigation text
            nav_keywords = ['edit', 'view', 'history', 'talk', 'skip to', 'jump to', 'menu', 'search']
            if any(keyword in line.lower() for keyword in nav_keywords) and len(line) < 50:
                continue
            lines.append(line)
        
        # Join lines and clean up excessive whitespace
        content_text = '\n'.join(lines)
        # Remove excessive blank lines
        content_text = re.sub(r'\n\s*\n\s*\n', '\n\n', content_text)
        content_text = content_text.strip()
        
        # Combine title and content
        full_text = f"{title_text}\n\n{content_text}"
        
        print(f"*** Extracted {len(full_text)} characters from web page ***")
        return full_text, title_text
        
    except requests.exceptions.RequestException as e:
        print(f"*** Error fetching URL {url}: {e} ***")
        return None, None
    except Exception as e:
        print(f"*** Error processing web page {url}: {e} ***")
        return None, None


def get_url_filename(url):
    """Generate a safe filename from URL"""
    parsed = urlparse(url)
    # Create filename from domain and path
    filename = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace(':', '_')
    # Remove special characters and limit length
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(filename) > 100:
        filename = filename[:100]
    if not filename or filename == '_':
        filename = "webpage"
    return filename


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False


class UniversalRAGProcessor:
    def __init__(self, args):
        self.args = args
        self.whisper_model = None
        self.embeddings = None
        self.vectorstore = None
        self.model_info = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        self.processed_files = {}
        self.processed_files_path = None
        self.new_files_processed = 0
        self.all_documents = []
        
        # Initialize model and embeddings
        self._setup_model()
        
    def _setup_model(self):
        """Setup embedding model and related components"""
        model_map = {
            "ada002": OpenAIAda002,
            "3-small": OpenAI3Small,
            "3-large": OpenAI3Large,
            "bge-m3": OllamaModel,
            "nomic-embed-text": NomicEmbedText
        }
        
        self.model_info = model_map[self.args.model]
        
        # Initialize embeddings based on model type
        if self.args.model in ["bge-m3", "nomic-embed-text"]:
            self.embeddings = OllamaEmbeddings(model=self.args.model)
        else:
            self.embeddings = OpenAIEmbeddings(model=self.model_info.name)
    
    def _setup_vector_store(self):
        """Setup the appropriate vector store"""
        if self.args.store == "faiss":
            self._setup_faiss()
        elif self.args.store == "pinecone":
            self._setup_pinecone()
        elif self.args.store == "chroma":
            self._setup_chroma()
    
    def _setup_faiss(self):
        """Setup FAISS vector store"""
        faiss_path = self.args.faiss_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
        self.processed_files_path = os.path.join(os.path.dirname(faiss_path), "processed_files.json")
        self.processed_files = load_processed_files(self.processed_files_path)
        
        if os.path.exists(faiss_path):
            print(f"*** Loading existing FAISS index from: {faiss_path} ***")
            self.vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"*** Loaded existing index with {self.vectorstore.index.ntotal} documents ***")
        else:
            print(f"*** No existing FAISS index found. Will create new index ***")
            self.vectorstore = None
        
        self.store_path = faiss_path
    
    def _setup_pinecone(self):
        """Setup Pinecone vector store"""
        pinecone_api_key = self.args.pinecone_key or os.getenv("PINECONE_API_KEY")
        pinecone_index = self.args.pinecone_index or os.getenv("PINECONE_INDEX")
        
        if not pinecone_api_key or not pinecone_index:
            raise ValueError("Pinecone API key and index name are required for Pinecone store")
        
        faiss_path = self.args.faiss_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
        self.processed_files_path = os.path.join(os.path.dirname(faiss_path), "processed_files.json")
        self.processed_files = load_processed_files(self.processed_files_path)
        
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index_name = pinecone_index
        print(f"*** Using Pinecone vector store: {pinecone_index} ***")
    
    def _setup_chroma(self):
        """Setup Chroma vector store"""
        chroma_path = self.args.chroma_path or os.getenv("CHROMA_PATH", "./chroma_db")
        index_name = self.args.chroma_index or os.getenv("CHROMA_INDEX", "default_index")
        
        os.makedirs(chroma_path, exist_ok=True)
        
        # ChromaDB setup
        client = chromadb.PersistentClient(path=chroma_path)
        
        collection_metadata = {
            "embedding_model": self.model_info.name,
            "embedding_dimension": self.model_info.dimension,
            "created_at": str(datetime.datetime.now(datetime.UTC)),
            "chunk_size": self.args.chunk_size,
            "chunk_overlap": self.args.chunk_overlap,
            "hnsw:space": "cosine"
        }
        
        try:
            collection = client.get_collection(name=index_name)
            print(f"*** Found existing Chroma collection: {index_name} ***")
        except:
            collection = client.create_collection(name=index_name, metadata=collection_metadata)
            print(f"*** Created new Chroma collection: {index_name} ***")
        
        self.vectorstore = Chroma(
            client=client,
            collection_name=index_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
        
        self.chroma_collection = collection
        self.store_path = chroma_path
    
    def process_input(self, input_path):
        """Process a single input (file, directory, or URL)"""
        if is_url(input_path):
            self._process_url(input_path)
        elif os.path.isfile(input_path):
            self._process_file(input_path)
        elif os.path.isdir(input_path):
            self._process_directory(input_path)
        else:
            print(f"*** Error: '{input_path}' is not a valid file, directory, or URL ***")
    
    def _process_url(self, url):
        """Process a single URL"""
        url_filename = get_url_filename(url)
        transcript_dir = self.args.transcript_dir or os.getenv("TRANSCRIPT_DIR", "./transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        
        transcript_filename = os.path.join(transcript_dir, url_filename + ".txt")
        
        print(f"\n*** Processing URL: {url} ***")
        
        # Check if we already have this URL cached and processed
        url_key = f"url_{url_filename}"
        if self.args.store != "chroma" and url_key in self.processed_files:
            print("*** Skipping URL (already processed and cached) ***")
            return
        
        # URL needs processing
        if os.path.exists(transcript_filename):
            print("*** Using existing transcript file ***") 
            loader = TextLoader(transcript_filename, encoding='utf-8')
        else:
            # Extract web content
            web_content, page_title = extract_web_content(url)
            if web_content:
                with open(transcript_filename, "w", encoding='utf-8') as file:
                    file.write(web_content)
                loader = TextLoader(transcript_filename, encoding='utf-8')
            else:
                print("*** Failed to extract content from URL ***")
                return
        
        if loader:
            raw_documents = loader.load()
            split_docs = self.text_splitter.split_documents(raw_documents)
            
            # Assign document IDs and fix metadata
            for idx, doc in enumerate(split_docs):
                # Store original source (transcript file path) before overwriting
                original_source = doc.metadata.get("source", transcript_filename)
                
                doc.metadata["doc_id"] = f"{url_filename}_{idx}"
                doc.metadata["source"] = url  # Original source (URL)
                doc.metadata["transcript_file"] = original_source  # Transcript file path
                doc.metadata["source_type"] = "url"
            
            print(f"*** Split web page into {len(split_docs)} documents ***")
            self.all_documents.extend(split_docs)
            
            # Mark URL as processed (for FAISS/Pinecone)
            if self.args.store != "chroma":
                self.processed_files[url_key] = {"url": url, "processed_time": time.time()}
            
            self.new_files_processed += 1
    
    def _process_file(self, filepath):
        """Process a single file"""
        filename = os.path.basename(filepath)
        
        if not filename.lower().endswith(('.mp3', '.wav', '.txt', '.pdf', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp')):
            print(f"*** Skipping unsupported file: {filename} ***")
            return
        
        # Check if file has changed or is new (skip for Chroma - it handles duplicates)
        if self.args.store != "chroma" and not file_has_changed(filepath, self.processed_files):
            print(f"*** Skipping {filename} (already processed and unchanged) ***")
            return
        
        base_name = os.path.splitext(filename)[0]
        transcript_dir = self.args.transcript_dir or os.getenv("TRANSCRIPT_DIR", "./transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        transcript_filename = os.path.join(transcript_dir, base_name + ".txt")
        
        print(f"\n*** Processing file: {filename} ***")
        print(f"*** Transcript file: {transcript_filename} ***")
        
        if os.path.exists(transcript_filename):
            print("*** Skipping extraction/transcription; text file already exists ***")
            loader = TextLoader(transcript_filename)
        elif filename.lower().endswith('.txt'):
            with open(filepath) as file:
                tokens = file.read()
            with open(transcript_filename, "w") as file:
                file.write(tokens)
            loader = TextLoader(transcript_filename)
        elif filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            tokens = "\n".join([doc.page_content for doc in loader.load()])
            with open(transcript_filename, "w") as file:
                file.write(tokens)
        else:
            # Handle audio and video files with Whisper transcription
            file_type = "audio" if filename.lower().endswith(('.mp3', '.wav')) else "video"
            print(f"*** Transcribing {file_type}: {filepath} ***")
            if self.whisper_model is None:
                whisper_model_name = self.args.whisper_model or os.getenv("WHISPER_MODEL", "base")
                print(f"*** Loading Whisper model: {whisper_model_name} ***")
                self.whisper_model = whisper.load_model(whisper_model_name)
            start = time.time()
            transcription = self.whisper_model.transcribe(filepath, fp16=False)
            end = time.time()
            tokens = transcription["text"].strip()
            print(f"*** Transcribed in {end - start:.2f} seconds; {len(tokens)} characters")
            with open(transcript_filename, "w") as file:
                file.write(tokens)
            loader = TextLoader(transcript_filename)
        
        raw_documents = loader.load()
        split_docs = self.text_splitter.split_documents(raw_documents)
        
        # Assign document IDs and fix metadata
        for idx, doc in enumerate(split_docs):
            # Store original source (transcript file path) before overwriting
            original_source = doc.metadata.get("source", transcript_filename)
            
            doc.metadata["doc_id"] = f"{base_name}_{idx}"
            doc.metadata["source"] = filename  # Original source (filename)
            doc.metadata["transcript_file"] = original_source  # Transcript file path
            doc.metadata["source_type"] = "file"
        
        print(f"*** Split transcript into {len(split_docs)} documents ***")
        self.all_documents.extend(split_docs)
        
        # Mark file as processed (for FAISS/Pinecone)
        if self.args.store != "chroma":
            self.processed_files[filename] = get_file_info(filepath)
        
        self.new_files_processed += 1
    
    def _process_directory(self, directory):
        """Process all supported files in a directory"""
        print(f"*** Processing directory: {directory} ***")
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                self._process_file(filepath)
    
    def save_to_vector_store(self):
        """Save processed documents to the vector store"""
        if not self.all_documents:
            print("*** No new documents to process ***")
            return
        
        print(f"\n*** Processing {self.new_files_processed} new/changed items with {len(self.all_documents)} document chunks ***")
        
        if self.args.store == "faiss":
            self._save_to_faiss()
        elif self.args.store == "pinecone":
            self._save_to_pinecone()
        elif self.args.store == "chroma":
            self._save_to_chroma()
    
    def _save_to_faiss(self):
        """Save documents to FAISS"""
        # Convert documents into (id, content, metadata) tuples
        docs_with_ids = [
            {
                "id": doc.metadata["doc_id"],
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in self.all_documents
        ]
        
        if self.vectorstore is None:
            # Create new FAISS index
            print(f"*** Creating new FAISS index ***")
            self.vectorstore = FAISS.from_texts(
                texts=[doc["page_content"] for doc in docs_with_ids],
                embedding=self.embeddings,
                metadatas=[doc["metadata"] for doc in docs_with_ids],
                ids=[doc["id"] for doc in docs_with_ids]
            )
        else:
            # Add new documents to existing index
            print(f"*** Adding new documents to existing FAISS index ***")
            print(f"*** Index had {self.vectorstore.index.ntotal} documents before merge ***")
            new_vectorstore = FAISS.from_texts(
                texts=[doc["page_content"] for doc in docs_with_ids],
                embedding=self.embeddings,
                metadatas=[doc["metadata"] for doc in docs_with_ids],
                ids=[doc["id"] for doc in docs_with_ids]
            )
            self.vectorstore.merge_from(new_vectorstore)
        
        # Save updated index to disk
        print(f"*** Saving FAISS index to: {self.store_path} ***")
        self.vectorstore.save_local(self.store_path)
        
        total_docs = self.vectorstore.index.ntotal
        print(f"*** FAISS index now contains {total_docs} total documents ***")
        
        # Save processed files metadata
        save_processed_files(self.processed_files, self.processed_files_path)
    
    def _save_to_pinecone(self):
        """Save documents to Pinecone"""
        try:
            index_list = self.pinecone_client.list_indexes().names()
            
            if self.pinecone_index_name not in index_list:
                # Create index and upload documents
                print(f"*** Creating Pinecone index: {self.pinecone_index_name} ***")
                self.pinecone_client.create_index(
                    name=self.pinecone_index_name,
                    dimension=self.model_info.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                
                # Wait for index to become available
                for _ in range(10):
                    time.sleep(1)
                    if self.pinecone_index_name in self.pinecone_client.list_indexes().names():
                        break
                else:
                    raise Exception("Index did not become available within 10 seconds.")
            
            # Upload documents to Pinecone
            print(f"*** Uploading documents to Pinecone index '{self.pinecone_index_name}' ***")
            
            docs_with_ids = [
                {
                    "id": doc.metadata["doc_id"],
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.all_documents
            ]
            
            vectorstore = PineconeVectorStore.from_texts(
                texts=[doc["page_content"] for doc in docs_with_ids],
                embedding=self.embeddings,
                metadatas=[doc["metadata"] for doc in docs_with_ids],
                ids=[doc["id"] for doc in docs_with_ids],
                index_name=self.pinecone_index_name
            )
            
            print(f"*** Uploaded {len(self.all_documents)} documents successfully ***")
            
            # Save processed files metadata
            save_processed_files(self.processed_files, self.processed_files_path)
            
        except Exception as e:
            print(f"Error with Pinecone operation: {e}")
            raise
    
    def _save_to_chroma(self):
        """Save documents to Chroma"""
        print(f"*** Adding documents to Chroma collection ***")
        
        added_count = 0
        with tqdm(total=len(self.all_documents), desc="Adding to Chroma") as pbar:
            for doc in self.all_documents:
                try:
                    # Generate deterministic ID based on content
                    chunk_id = hashlib.sha256(doc.page_content.encode()).hexdigest()
                    
                    # Check if chunk already exists
                    existing = self.chroma_collection.get(
                        ids=[chunk_id],
                        include=[],
                    )
                    
                    if existing['ids']:
                        pbar.update(1)
                        continue
                    
                    # Add metadata for Chroma
                    doc.metadata.update({
                        "embedding_model": self.model_info.name,
                        "embedding_dimension": self.model_info.dimension,
                        "processed_at": str(datetime.datetime.now(datetime.UTC))
                    })
                    
                    self.vectorstore.add_texts(
                        texts=[doc.page_content],
                        metadatas=[doc.metadata],
                        ids=[chunk_id]
                    )
                    added_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError adding document: {e}")
                    pbar.update(1)
                    continue
        
        print(f"*** Added {added_count} new documents to Chroma ***")


def main():
    parser = argparse.ArgumentParser(
        description="Universal RAG processor supporting FAISS, Pinecone, and Chroma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files with FAISS (default)
  %(prog)s file1.txt file2.pdf ./media_dir/
  
  # Process web page with Pinecone
  %(prog)s --store pinecone https://en.wikipedia.org/wiki/Artificial_intelligence
  
  # Use Chroma with custom chunk settings
  %(prog)s --store chroma --chunk-size 2000 --chunk-overlap 500 documents/
  
  # Use different embedding model
  %(prog)s --model 3-large --store faiss research_papers/
        """
    )
    
    # Input arguments
    parser.add_argument("inputs", nargs="+", help="Files, directories, or URLs to process")
    
    # Vector store selection
    parser.add_argument("--store", choices=["faiss", "pinecone", "chroma"], default="faiss",
                       help="Vector store to use (default: faiss)")
    
    # Model configuration
    parser.add_argument("--model", choices=["ada002", "3-small", "3-large", "bge-m3", "nomic-embed-text"],
                       default="3-small", help="Embedding model to use (default: 3-small)")
    parser.add_argument("--whisper-model", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model for audio/video transcription (default: base)")
    
    # Chunking configuration
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Size of text chunks in characters (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=400,
                       help="Overlap between adjacent chunks in characters (default: 400)")
    
    # Storage paths
    parser.add_argument("--faiss-path", help="FAISS index directory path (default: FAISS_INDEX_PATH env or 'faiss_index')")
    parser.add_argument("--pinecone-key", help="Pinecone API key")
    parser.add_argument("--pinecone-index", help="Pinecone index name")
    parser.add_argument("--chroma-path", help="Chroma database path (default: CHROMA_PATH env or './chroma_db')")
    parser.add_argument("--chroma-index", help="Chroma collection name (default: CHROMA_INDEX env or 'default_index')")
    parser.add_argument("--transcript-dir", help="Directory for cached transcripts (default: TRANSCRIPT_DIR env or './transcripts')")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Validate vector store availability
    if args.store == "faiss" and not FAISS_AVAILABLE:
        print("Error: FAISS dependencies not available. Install with: pip install faiss-cpu")
        sys.exit(1)
    elif args.store == "pinecone" and not PINECONE_AVAILABLE:
        print("Error: Pinecone dependencies not available. Install with: pip install langchain-pinecone pinecone")
        sys.exit(1)
    elif args.store == "chroma" and not CHROMA_AVAILABLE:
        print("Error: Chroma dependencies not available. Install with: pip install langchain-chroma chromadb")
        sys.exit(1)
    
    # Validate required environment variables
    if args.model in ["ada002", "3-small", "3-large"] and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable must be set for OpenAI models")
        sys.exit(1)
    
    if args.dry_run:
        print("\n=== DRY RUN - No changes will be made ===")
        print(f"Vector Store: {args.store}")
        print(f"Embedding Model: {args.model}")
        print(f"Chunk Size: {args.chunk_size}")
        print(f"Chunk Overlap: {args.chunk_overlap}")
        print(f"Inputs: {args.inputs}")
        print("=== End of dry run ===")
        return
    
    try:
        # Create processor
        processor = UniversalRAGProcessor(args)
        
        # Setup vector store
        processor._setup_vector_store()
        
        # Process all inputs
        with tqdm(total=len(args.inputs), desc="Processing inputs", unit="item") as pbar:
            for input_path in args.inputs:
                if args.verbose:
                    print(f"\nProcessing: {input_path}")
                processor.process_input(input_path)
                pbar.update(1)
        
        # Save to vector store
        processor.save_to_vector_store()
        
        print("*** Processing complete ***")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()