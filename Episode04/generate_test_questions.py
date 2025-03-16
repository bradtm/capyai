#!/usr/bin/env python3

"""
This script generates natural-sounding test questions from text chunks to evaluate
retrieval system quality. It supports multiple LLM providers

Dependencies:
  - langchain.text_splitter: For consistent chunking
  - requests: For API calls to Ollama
  - openai: For OpenAI API calls
  - transformers: For HuggingFace models
  - torch: For HuggingFace model inference
  - tqdm: For progress bars

Usage:
  # Using FAISS vector store with OpenAI (API key from environment variable)
  python text_question_generator.py --store faiss --provider openai --model gpt-4o-mini
  
  # Using Pinecone vector store
  python text_question_generator.py --store pinecone --provider openai --model gpt-4o-mini
  
  # Using Chroma vector store
  python text_question_generator.py --store chroma --chroma-path ./my_chroma --provider openai
  
  # Using Ollama with custom settings
  python text_question_generator.py --store faiss --provider ollama --model llama3.2 --questions-per-chunk 1
  
  # Using HuggingFace models
  python text_question_generator.py --store faiss --provider huggingface --model microsoft/DialoGPT-medium
  
  # Limit total questions and chunks
  python text_question_generator.py --store faiss --provider openai --total-questions 50
  
  # Limit questions per source file (randomly selects 5 chunks from each file)
  python text_question_generator.py --store faiss --provider openai --questions-per-file 5
  
  # Random document selection (select 100 random documents, 1 question each)
  python text_question_generator.py --store faiss --provider openai --random-docs 100
"""

import os
import sys
import json
import random
import argparse
import hashlib
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import the same chunking mechanism as in popchroma.py

# Vector store imports
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore

# Optional Chroma imports
try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try to load .env manually
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        try:
            # Start with minimal parameters that should work with all models
            completion_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Try the simplest call first (no temperature, no max tokens)
            # Many newer models don't support custom temperature or token limits
            try:
                response = self.client.chat.completions.create(**completion_params)
            except Exception:
                # Try adding max_completion_tokens for models that support it
                try:
                    completion_params["max_completion_tokens"] = max_tokens
                    response = self.client.chat.completions.create(**completion_params)
                except Exception:
                    # Try max_tokens instead for older models
                    completion_params.pop("max_completion_tokens", None)
                    completion_params["max_tokens"] = max_tokens
                    response = self.client.chat.completions.create(**completion_params)
            
            content = response.choices[0].message.content
            if content is None:
                return ""
            return content.strip()
        except Exception as e:
            print(f"Error generating text with OpenAI: {e}")
            return ""
    
    def is_available(self) -> bool:
        try:
            # Simple test call
            self.client.models.list()
            return True
        except:
            return False


class OllamaProvider(LLMProvider):
    """Ollama local server provider."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.generate_endpoint = f"{self.base_url}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.generate_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Error generating text with Ollama: {e}")
            return ""
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False


class HuggingFaceProvider(LLMProvider):
    """HuggingFace transformers provider."""
    
    def __init__(self, model: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace transformers not available. Install with: pip install transformers torch")
        
        self.model_name = model
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32)
            self.model.to(self.device)
            
            # Set pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {model}: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response (excluding the input prompt)
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating text with HuggingFace: {e}")
            return ""
    
    def is_available(self) -> bool:
        return hasattr(self, 'model') and hasattr(self, 'tokenizer')


class QuestionGenerator:
    """Framework for generating test questions from text chunks using various LLMs."""
    
    def __init__(self, provider: str = "ollama", model: str = "llama3.2", api_key: str = None,
                 ollama_url: str = "http://localhost:11434", device: str = "auto",
                 use_enhanced_targeting: bool = False, central_content_focus: bool = False,
                 generation_temperature: float = 0.7, max_total_questions: int = None):
        """
        Initialize the question generator.
        
        Args:
            provider: LLM provider ("openai", "ollama", "huggingface")
            model: Model name for the provider
            api_key: API key for OpenAI (if using OpenAI)
            ollama_url: URL for Ollama server
            device: Device for HuggingFace models ("auto", "cpu", "cuda", "mps")
            use_enhanced_targeting: Whether to use enhanced chunk-specific targeting
            central_content_focus: Whether to focus on central content of chunks
            generation_temperature: Temperature for LLM generation
            max_total_questions: Maximum total questions to generate
        """
        self.use_enhanced_targeting = use_enhanced_targeting
        self.central_content_focus = central_content_focus
        self.generation_temperature = generation_temperature
        self.max_total_questions = max_total_questions
        
        # Note: Text splitter not needed when working with existing vector stores
        
        # Initialize LLM provider
        if provider == "openai":
            self.llm = OpenAIProvider(model=model, api_key=api_key)
        elif provider == "ollama":
            self.llm = OllamaProvider(model=model, base_url=ollama_url)
        elif provider == "huggingface":
            self.llm = HuggingFaceProvider(model=model, device=device)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.test_questions = []
        self.chunk_map = {}  # Maps chunk_id to chunk text
        self.chunk_sources = {}  # Maps chunk_id to source file
        
        # Statistics tracking
        self.skipped_contextless = 0
        self.skipped_direct_questions = 0
        
        # Vector store related
        self.vectorstore = None
        self.embeddings = None
    
    def load_vector_store(self, store_type: str, **kwargs) -> None:
        """
        Load a vector store and its associated chunks.
        
        Args:
            store_type: Type of vector store ('faiss', 'pinecone', 'chroma')
            **kwargs: Store-specific parameters
        """
        # Initialize embeddings (try to auto-detect for FAISS)
        embedding_model = 'text-embedding-3-small'  # default
        
        if store_type == 'faiss':
            faiss_path = kwargs.get('faiss_path') or os.getenv('FAISS_INDEX_PATH', 'faiss_index')
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS index not found at '{faiss_path}'")
            
            # Try to auto-detect embedding model from metadata
            metadata_file = os.path.join(faiss_path, "index_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        stored_embedding_model = metadata.get('embedding_model')
                        if stored_embedding_model:
                            embedding_model = stored_embedding_model
                except Exception:
                    pass  # Fall back to default
        
        # Initialize embeddings based on detected/default model
        if embedding_model in ["bge-m3", "nomic-embed-text"]:
            # Import OllamaEmbeddings class from ask.py approach
            try:
                import requests
                class OllamaEmbeddings:
                    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
                        self.model = model
                        self.base_url = base_url.rstrip('/')
                    
                    def embed_documents(self, texts):
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
                        response = requests.post(
                            f"{self.base_url}/api/embeddings", 
                            json={"model": self.model, "prompt": text}
                        )
                        response.raise_for_status()
                        return response.json()["embedding"]
                
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            except Exception:
                # Fallback to OpenAI if Ollama not available
                self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        else:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        if store_type == 'faiss':
            self.vectorstore = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
        elif store_type == 'pinecone':
            pinecone_api_key = kwargs.get('pinecone_key') or os.getenv('PINECONE_API_KEY')
            pinecone_index = kwargs.get('pinecone_index') or os.getenv('PINECONE_INDEX')
            
            if not pinecone_api_key or not pinecone_index:
                raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set for Pinecone store")
            
            self.vectorstore = PineconeVectorStore.from_existing_index(
                pinecone_index,
                self.embeddings
            )
            
        elif store_type == 'chroma':
            if not CHROMA_AVAILABLE:
                raise ImportError("Chroma dependencies not installed. Run: pip install langchain-chroma chromadb")
            
            chroma_path = kwargs.get('chroma_path') or os.getenv('CHROMA_PATH', './chroma_db')
            chroma_index = kwargs.get('chroma_index') or os.getenv('CHROMA_INDEX', 'default_index')
            
            if not os.path.exists(chroma_path):
                raise FileNotFoundError(f"Chroma database not found at '{chroma_path}'")
            
            client = chromadb.PersistentClient(path=chroma_path)
            self.vectorstore = Chroma(
                client=client,
                collection_name=chroma_index,
                embedding_function=self.embeddings,
                persist_directory=chroma_path
            )
        
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def get_all_chunks_from_vectorstore(self) -> List[Dict[str, Any]]:
        """
        Get all chunks from the loaded vector store.
        
        Args:
            None
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        chunks = []
        
        if hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
            # FAISS - get all documents by accessing the docstore directly
            total_docs = self.vectorstore.index.ntotal
            limit = total_docs
            
            docs = []
            # Get documents directly from docstore
            for i in range(min(limit, total_docs)):
                try:
                    doc_id = self.vectorstore.index_to_docstore_id[i]
                    doc = self.vectorstore.docstore.search(doc_id)
                    if doc:
                        docs.append(doc)
                except (KeyError, IndexError):
                    continue
            
        elif hasattr(self.vectorstore, '_collection'):
            # Chroma - get all documents
            collection = self.vectorstore._collection
            limit = collection.count()
            
            results = collection.get(limit=limit, include=['documents', 'metadatas'])
            docs = []
            
            for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                from langchain_core.documents import Document
                docs.append(Document(page_content=doc_text, metadata=metadata))
                
        else:
            # Pinecone or other stores - use a broad search
            limit = 1000  # Default limit for Pinecone
            docs = self.vectorstore.similarity_search("", k=limit)
        
        # Convert to our chunk format
        for i, doc in enumerate(docs):
                
            chunk_id = doc.metadata.get('doc_id', f'chunk_{i}')
            source = doc.metadata.get('source', 'unknown')
            
            chunk_info = {
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "source": source,
                "chunk_index": doc.metadata.get('chunk_index', i),
                "total_chunks": len(docs),
                "metadata": doc.metadata
            }
            
            # Store the chunk for later reference
            self.chunk_map[chunk_id] = doc.page_content
            self.chunk_sources[chunk_id] = source
            chunks.append(chunk_info)
        
        return chunks

    def process_transcript(self, file_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process a text file into chunks.
        
        Args:
            file_path: Path to the text file
            verbose: Whether to print verbose output
            
        Returns:
            List of chunk dictionaries with IDs and text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            num_chunks = len(chunks)
            
            if verbose:
                print(f"Split '{Path(file_path).name}' into {num_chunks} chunks")
            
            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                # Generate deterministic ID
                chunk_id = hashlib.sha256(chunk.encode()).hexdigest()
                
                chunk_info = {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": str(Path(file_path).resolve()),
                    "chunk_index": i,
                    "total_chunks": num_chunks,
                }
                
                # Store the chunk for later reference
                self.chunk_map[chunk_id] = chunk
                self.chunk_sources[chunk_id] = str(Path(file_path).resolve())
                processed_chunks.append(chunk_info)
            
            return processed_chunks
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def is_contextless_question(self, question_text: str) -> bool:
        """
        Check if a question lacks necessary context to be answerable.
        
        Args:
            question_text: The question text to check
            
        Returns:
            True if the question lacks context, False otherwise
        """
        # Define patterns that indicate a question lacks context
        contextless_patterns = [
            r'this text',
            r'this passage',
            r'these texts',
            r'this document',
            r'this section',
            r'this chapter',
            r'this story',
            r'this example',
            r'in this context',
            r'in the text',
            r'he say(?:s|ing)?',
            r'she say(?:s|ing)?',
            r'they say(?:s|ing)?',
            r'talking about here',
            r'referring to here',
            r'mentioned here'
        ]
        
        # Check if any pattern is in the question
        for pattern in contextless_patterns:
            if re.search(pattern, question_text, re.IGNORECASE):
                return True
                
        # Check for references to pronouns without clear antecedents
        pronouns = [r'\bhe\b', r'\bshe\b', r'\bthey\b', r'\bhis\b', r'\bher\b', r'\btheir\b']
        
        for pronoun in pronouns:
            if re.search(pronoun, question_text, re.IGNORECASE):
                words = question_text.split()
                if len(words) < 10:  # Short question with a pronoun is likely contextless
                    return True
        
        return False
    
    def has_passage_references(self, question_text: str) -> bool:
        """
        Check if a question references the passage, text, speaker, or similar phrases.
        
        Args:
            question_text: The question text to check
            
        Returns:
            True if the question contains passage references, False otherwise
        """
        # Define patterns that indicate passage references
        reference_patterns = [
            r'the passage',
            r'the text',
            r'this passage',
            r'this text',
            r'the speaker',
            r'the author',
            r'according to',
            r'the document',
            r'this document',
            r'in the passage',
            r'in the text',
            r'the provided context',
            r'this context',
            r'the context',
            r'in the context',
            r'the given context',
            r'described in the',
            r'mentioned in the',
            r'what does .* say',
            r'what does .* think',
            r'what does .* feel',
            r'what aspect does .* feel',
            r'what .* feel is'
        ]
        
        # Check if any pattern is in the question
        for pattern in reference_patterns:
            if re.search(pattern, question_text, re.IGNORECASE):
                return True
                
        return False
    
    def is_vague_question(self, question_text: str) -> bool:
        """
        Check if a question is too vague for accurate retrieval testing.
        
        Args:
            question_text: The question text to check
            
        Returns:
            True if the question is too vague, False otherwise
        """
        # Define patterns that indicate vague questions
        vague_patterns = [
            r'how long has .* been',
            r'how many years .* been',
            r'what does .* do\?$',
            r'who is .*\?$',
            r'what is .* about\?$',
            r'how does .* work\?$',
            r'why does .* matter\?$',
            r'what are .* like\?$',
            r'how can .* help\?$',
            r'what makes .* important\?$',
            r'is mentioned',
            r'are mentioned',
            r'was mentioned',
            r'were mentioned',
            r'mentioned as',
            r'mentioned in',
            r'the person',
            r'the individual',
            r'the company(?! [A-Z])',  # "the company" without a specific company name following
            r'this person',
            r'this individual',
            r'this company(?! [A-Z])',
            r'someone',
            r'they are',
            r'it is',
            r'what are some'
        ]
        
        # Check if any pattern is in the question
        for pattern in vague_patterns:
            if re.search(pattern, question_text, re.IGNORECASE):
                return True
        
        # Check for questions that are too short (likely to be vague)
        word_count = len(question_text.split())
        if word_count < 6:  # Very short questions are often too vague
            return True
        
        # Check if question lacks specific nouns, numbers, or proper names
        has_specific_info = bool(
            re.search(r'[A-Z][a-z]{2,}', question_text) or  # Proper nouns
            re.search(r'\d+', question_text) or  # Numbers
            re.search(r'[a-z]{6,}', question_text)  # Longer specific words
        )
        
        if not has_specific_info:
            return True
            
        return False
        
    def is_direct_question(self, question_text: str, passage: str) -> bool:
        """
        Detect if a question is directly extracted from the text.
        
        Args:
            question_text: The question to check
            passage: The passage from which the question was extracted
            
        Returns:
            True if the question appears to be directly extracted, False otherwise
        """
        # Clean the question for comparison
        clean_question = question_text.strip().rstrip('?').lower()
        
        # Check if the question appears verbatim in the passage
        if clean_question in passage.lower():
            # Look for question indicators
            question_indicators = [
                r'I ask you',
                r'let me ask you',
                r'I want to ask',
                r'the question is',
                r'my question is',
                r'ask yourself',
                r'we must ask',
                r'consider this',
                r'think about'
            ]
            
            # Check for these patterns near the question in the passage
            for indicator in question_indicators:
                indicator_pattern = f"({indicator}.*?{re.escape(clean_question)}|{re.escape(clean_question)}.*?{indicator})"
                if re.search(indicator_pattern, passage.lower(), re.DOTALL):
                    return True
                    
        return False
    
    def extract_significant_passages(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Extract significant passages from a chunk that would make good questions.
        
        Args:
            chunk: The chunk dictionary
            
        Returns:
            List of significant passages
        """
        text = chunk["text"]
        passages = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on various factors
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            score = 0
            
            # Length (longer sentences often contain more content)
            length = len(sentence.split())
            if length > 15:
                score += 2
            elif length > 8:
                score += 1
            
            # Contains proper nouns or names
            proper_nouns = re.findall(r'[A-Z][a-z]+', sentence)
            if len(proper_nouns) > 2:
                score += 2
            elif len(proper_nouns) > 0:
                score += 1
                
            # Contains numbers (often significant)
            if re.search(r'\d+', sentence):
                score += 1
                
            # Contains question words (often indicate key concepts)
            if any(word in sentence.lower() for word in ['what', 'why', 'how', 'who', 'when', 'where']):
                score += 1
                
            # Contains certain key indicators
            if any(word in sentence.lower() for word in ['means', 'signifies', 'represents', 'shows', 'tells', 'explains']):
                score += 1
                
            # Position in chunk (middle sentences often contain key content)
            relative_position = i / len(sentences)
            if 0.25 <= relative_position <= 0.75:
                score += 1
                
            scored_sentences.append((sentence, score, i))
        
        # Sort sentences by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Group into meaningful passages (2-3 sentences)
        created_passages = set()
        
        for high_score_sentence, score, idx in scored_sentences[:min(5, len(scored_sentences))]:
            if idx in created_passages:
                continue
                
            # Find adjacent sentences to create context
            context_before = []
            context_after = []
            
            # Get 1-2 sentences before
            for i in range(idx-1, max(0, idx-3), -1):
                if i not in created_passages and i < len(sentences):
                    context_before.insert(0, sentences[i])
                    created_passages.add(i)
                if len(context_before) >= 1:
                    break
                    
            # Get 1-2 sentences after
            for i in range(idx+1, min(len(sentences), idx+3)):
                if i not in created_passages and i < len(sentences):
                    context_after.append(sentences[i])
                    created_passages.add(i)
                if len(context_after) >= 1:
                    break
            
            # Create the passage with context
            passage_parts = context_before + [high_score_sentence] + context_after
            passage = ' '.join(passage_parts)
            
            # Only add if passage is substantial but not too long
            if 100 <= len(passage) <= 800:
                passages.append(passage)
            created_passages.add(idx)
        
        # If we don't have enough passages, create them from remaining sentences
        if len(passages) < 3:
            current_passage = ""
            for sentence in sentences:
                if sentence in ' '.join(passages):  # Skip sentences already used
                    continue
                    
                if len(current_passage) + len(sentence) < 500:
                    if current_passage:
                        current_passage += " " + sentence
                    else:
                        current_passage = sentence
                else:
                    if current_passage and len(current_passage) > 100:
                        passages.append(current_passage)
                    current_passage = sentence
            
            if current_passage and len(current_passage) > 100:
                passages.append(current_passage)
        
        # Ensure we have at least one passage
        if not passages and sentences:
            passage = ' '.join(sentences[:min(3, len(sentences))])
            if passage:
                passages = [passage]
        
        # Limit to top 3 passages
        if len(passages) > 3:
            passages = passages[:3]
        
        return passages
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by replacing newlines with spaces and normalizing whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace newlines with spaces
        cleaned = re.sub(r'\n+', ' ', text)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def generate_fallback_question(self, passage: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback question generation when LLM is not available.
        Creates rule-based questions from key phrases and topics.
        
        Args:
            passage: The passage to generate a question for
            chunk: The chunk dictionary
            
        Returns:
            A dictionary with the generated question or None
        """
        # Extract topics based on capitalized words
        topics = re.findall(r'\b[A-Z][a-z]{2,}\b', passage)
        
        # Clean up passage
        clean_passage = self.clean_text(passage)
        sentences = re.split(r'(?<=[.!?])\s+', clean_passage)
        
        # Select question type based on content
        generated_question = None
        
        if len(topics) >= 2:
            # Create question about relationship between topics
            topic1 = topics[0]
            topic2 = topics[1]
            generated_question = f"How does this text relate {topic1} to {topic2}?"
        elif topics:
            # Create question about single topic
            topic = topics[0]
            generated_question = f"What does this text say about {topic}?"
        elif sentences:
            # Extract a key statement and ask about it
            key_sentence = max(sentences, key=len)
            generated_question = f"What is the main point about {key_sentence[:30]}...?"
        
        if not generated_question:
            return None
            
        # Create question dict
        question = {
            "question": generated_question,
            "passage": clean_passage,
            "chunk_id": chunk["chunk_id"],
            "sources": [self.chunk_sources.get(chunk["chunk_id"], "")],
            "generated_with": "fallback-rule-based",
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }
        
        return question
        
    def generate_question_with_llm(self, passage: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a test question from a passage using the LLM.
    
        Args:
            passage: The passage to generate a question for
            chunk: The chunk dictionary
            
        Returns:
            A dictionary with the generated question or None if invalid
        """
        # Check if LLM is available
        if not self.llm.is_available():
            print("Warning: LLM is not available. Falling back to rule-based questions.")
            return self.generate_fallback_question(passage, chunk)

        # Extract central sentences if enabled
        sentences = re.split(r'(?<=[.!?])\s+', passage)
        if len(sentences) > 4 and self.central_content_focus:
            start_idx = len(sentences) // 4
            end_idx = start_idx + len(sentences) // 2
            central_sentences = sentences[start_idx:end_idx]
            central_text = " ".join(central_sentences)
        else:
            central_text = passage

        # Capture unique or distinguishing phrases from the passage
        unique_phrases = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 5:  # Only consider substantial sentences
                # Look for names, numbers, or other distinguishing features
                if (re.search(r'[A-Z][a-z]+', sentence) or  # Contains proper names
                    re.search(r'\d+', sentence)):  # Contains numbers
                    unique_phrases.append(' '.join(words[:min(10, len(words))]))

        # Build the prompt for the LLM
        prompt = (
            "Generate a single question about the key information in this text. "
            "The question should:\n"
            "- Include specific details, names, numbers, or concepts from the text\n"
            "- Be precise enough that only someone with access to this specific information could answer it accurately\n"
            "- Contain enough context clues to enable accurate retrieval of the source material\n"
            "- Stand alone as if asking about real-world knowledge\n"
            "- NOT reference 'the passage', 'the text', 'the speaker', 'according to', or 'the author'\n"
            "- Avoid vague pronouns like 'the person', 'the company', 'they' without clear context\n\n"
            "Good examples: 'Why did Uber launch in Atlanta in 2023?' or 'What BLEU score did the transformer model achieve on WMT 2014?' or 'How many cities does Uber operate taxi services in?'\n"
            "Bad examples: 'How long has the person been working?' or 'What task is mentioned in the text?' or 'What tool was mentioned as being innovative?' or 'What does the company do?'\n\n"
        )

        if self.use_enhanced_targeting:
            prompt += "The question should be specific enough to identify this passage and should highlight key information. "
            if unique_phrases:
                prompt += f"Consider these distinctive phrases from the passage: {'; '.join(unique_phrases[:3])}. "
    
        if self.central_content_focus:
            prompt += "Focus the question on the main point or central message, not on peripheral details. "
    
        prompt += "\nText Passage: " + passage + "\nQuestion:"

        # Generate the question using the LLM
        question_text = self.llm.generate(
            prompt, 
            max_tokens=256,
            temperature=self.generation_temperature
        ).strip()

        # Debug: Check if we got an empty or minimal response
        if not question_text or question_text in ['?', '.', '']:
            print(f"Warning: Got empty or minimal response from LLM: '{question_text}'")
            return None

        # Ensure the generated text ends with a question mark
        if not question_text.endswith('?'):
            question_text = question_text.strip() + "?"

        # Clean up newlines and extra whitespace
        question_text = self.clean_text(question_text)
        passage = self.clean_text(passage)
        
        # Check if the question is contextless, vague, has passage references, or is directly extracted
        if self.is_contextless_question(question_text):
            self.skipped_contextless += 1
            return None
            
        if self.is_vague_question(question_text):
            if not hasattr(self, 'skipped_vague'):
                self.skipped_vague = 0
            self.skipped_vague += 1
            return None
            
        if self.has_passage_references(question_text):
            if not hasattr(self, 'skipped_passage_references'):
                self.skipped_passage_references = 0
            self.skipped_passage_references += 1
            return None
            
        if self.is_direct_question(question_text, passage):
            self.skipped_direct_questions += 1
            return None
        
        # Get original source
        original_source = self.chunk_sources.get(chunk["chunk_id"], "")

        # Create the question object
        question = {
            "question": question_text,
            "passage": passage,
            "chunk_id": chunk["chunk_id"],
            "sources": [original_source],
            "generated_with": f"{self.llm.__class__.__name__.lower().replace('provider', '')}-{self.llm.model}",
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }

        return question

    def generate_questions_for_chunk(self, chunk: Dict[str, Any], num_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple test questions for a chunk.
        
        Args:
            chunk: The chunk dictionary
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        # Extract significant passages from the chunk
        passages = self.extract_significant_passages(chunk)
        
        # Generate questions from each passage
        questions = []
        attempts = 0
        max_attempts = min(len(passages) * 2, 10)
        
        while len(questions) < num_questions and attempts < max_attempts and passages:
            passage_index = attempts % len(passages)
            passage = passages[passage_index]
            
            question = self.generate_question_with_llm(passage, chunk)
            
            if question:
                questions.append(question)
                
            attempts += 1
                
        return questions

    def save_questions(self, output_file: str) -> None:
        """
        Save generated questions to a JSON file.
        
        Args:
            output_file: Path to the output file
        """
        if not self.test_questions:
            print("No questions to save.")
            return
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_questions, f, indent=2)
                
            print(f"Saved {len(self.test_questions)} questions to {output_file}")
            
            # Print stats about skipped questions
            if hasattr(self, 'skipped_contextless'):
                print(f"Contextless questions skipped: {self.skipped_contextless}")
            if hasattr(self, 'skipped_vague'):
                print(f"Vague questions skipped: {self.skipped_vague}")
            if hasattr(self, 'skipped_passage_references'):
                print(f"Questions with passage references skipped: {self.skipped_passage_references}")
            if hasattr(self, 'skipped_direct_questions'):
                print(f"Direct questions skipped: {self.skipped_direct_questions}")
                
        except Exception as e:
            print(f"Error saving questions: {e}")
            
    def generate_questions(self, text_files: List[str], questions_per_file: int = 10, 
                         verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Generate test questions for multiple text files.
    
        Args:
            text_files: List of text file paths
            questions_per_file: Number of questions to generate per file
            verbose: Whether to print verbose output
        
        Returns:
            List of question dictionaries
        """
        all_questions = []
        all_chunks = []
    
        # Process all files and collect chunks
        for text_file in text_files:
            if verbose:
                print(f"\nProcessing file: {text_file}")
            
            chunks = self.process_transcript(text_file, verbose=verbose)
            all_chunks.extend(chunks)
    
        # Generate questions for each file
        for text_file in text_files:
            if verbose:
                print(f"\nGenerating questions for: {text_file}")
            
            # Get chunks for this file
            file_path = str(Path(text_file).resolve())
            file_chunks = [c for c in all_chunks if c["source"] == file_path]
        
            if not file_chunks:
                continue
            
            # Determine questions per chunk
            num_chunks = len(file_chunks)
            questions_per_chunk = max(1, questions_per_file // num_chunks)
        
            # Generate questions for each chunk
            file_questions = []
        
            if verbose:
                print(f"Generating questions (targeting {questions_per_file} total)...")
                chunks_iter = tqdm(file_chunks)
            else:
                chunks_iter = file_chunks
            
            for chunk in chunks_iter:
                # Check if we've reached the maximum total questions
                if self.max_total_questions and len(all_questions) + len(file_questions) >= self.max_total_questions:
                    if verbose:
                        print(f"Reached maximum total questions limit ({self.max_total_questions})")
                    break
            
                chunk_questions = self.generate_questions_for_chunk(chunk, num_questions=questions_per_chunk)
                file_questions.extend(chunk_questions)
            
                # Stop if we've generated enough questions for this file
                if len(file_questions) >= questions_per_file:
                    break
        
            # Take only the requested number of questions
            file_questions = file_questions[:questions_per_file]
        
            # Add file source information
            for question in file_questions:
                question["file"] = os.path.basename(text_file)
            
            all_questions.extend(file_questions)
        
            # Check again if we've reached the maximum total questions
            if self.max_total_questions and len(all_questions) >= self.max_total_questions:
                if verbose:
                    print(f"Reached maximum total questions limit ({self.max_total_questions})")
                break
    
        # Keep only up to max_total_questions if specified
        if self.max_total_questions and len(all_questions) > self.max_total_questions:
            all_questions = all_questions[:self.max_total_questions]
                
        self.test_questions = all_questions
        return all_questions
    
    def generate_questions_from_chunks(self, chunks: List[Dict[str, Any]], 
                                     questions_per_chunk: int = 3, questions_per_file: Optional[int] = None,
                                     verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Generate test questions from vector store chunks.
        
        Args:
            chunks: List of chunk dictionaries from vector store
            questions_per_chunk: Number of questions to generate per chunk
            questions_per_file: Maximum number of questions per source file (randomly selects chunks)
            verbose: Whether to print verbose output
            
        Returns:
            List of question dictionaries
        """
        all_questions = []
        
        # Group chunks by source file if questions_per_file is specified
        if questions_per_file:
            from collections import defaultdict
            chunks_by_file = defaultdict(list)
            
            for chunk in chunks:
                source = chunk.get('source', 'unknown')
                chunks_by_file[source].append(chunk)
            
            # Randomly select chunks from each file
            selected_chunks = []
            for source, file_chunks in chunks_by_file.items():
                if verbose:
                    print(f"Source '{os.path.basename(source)}' has {len(file_chunks)} chunks")
                
                # Randomly select up to questions_per_file chunks from this source
                num_to_select = min(questions_per_file, len(file_chunks))
                selected_file_chunks = random.sample(file_chunks, num_to_select)
                selected_chunks.extend(selected_file_chunks)
                
                if verbose:
                    print(f"Selected {num_to_select} chunks from '{os.path.basename(source)}'")
            
            chunks = selected_chunks
            if verbose:
                print(f"Total selected chunks: {len(chunks)}")
        
        if verbose:
            print(f"\nGenerating questions from {len(chunks)} chunks...")
            chunks_iter = tqdm(chunks)
        else:
            chunks_iter = chunks
        
        for chunk in chunks_iter:
            # Check if we've reached the maximum total questions
            if self.max_total_questions and len(all_questions) >= self.max_total_questions:
                if verbose:
                    print(f"Reached maximum total questions limit ({self.max_total_questions})")
                break
            
            # Generate questions for this chunk
            chunk_questions = self.generate_questions_for_chunk(chunk, num_questions=questions_per_chunk)
            
            # Add source file information from metadata
            for question in chunk_questions:
                sources = question.get('sources', [''])
                source_path = sources[0] if sources else ''
                if source_path:
                    question['file'] = os.path.basename(source_path)
                else:
                    question['file'] = 'unknown'
            
            all_questions.extend(chunk_questions)
        
        # Keep only up to max_total_questions if specified
        if self.max_total_questions and len(all_questions) > self.max_total_questions:
            all_questions = all_questions[:self.max_total_questions]
        
        self.test_questions = all_questions
        return all_questions
    
    def generate_questions_random_docs(self, num_docs: int) -> List[Dict[str, Any]]:
        """Generate questions by randomly selecting documents and chunks.
        
        Args:
            num_docs: Number of random documents to select
            
        Returns:
            List of generated questions
        """
        print(f"\nGenerating questions from {num_docs} randomly selected documents...")
        
        # Get all chunks grouped by source file
        all_chunks = self.get_all_chunks_from_vectorstore()
        
        # Group chunks by source file
        chunks_by_file = {}
        for chunk in all_chunks:
            source_path = chunk.get('source_path') or chunk.get('source', 'unknown')
            file_key = os.path.basename(source_path) if source_path != 'unknown' else 'unknown'
            
            if file_key not in chunks_by_file:
                chunks_by_file[file_key] = []
            chunks_by_file[file_key].append(chunk)
        
        available_files = list(chunks_by_file.keys())
        print(f"Found {len(available_files)} unique source documents")
        
        if len(available_files) < num_docs:
            print(f"Warning: Requested {num_docs} documents but only {len(available_files)} available")
            num_docs = len(available_files)
        
        # Randomly select documents
        selected_files = random.sample(available_files, num_docs)
        print(f"Randomly selected {len(selected_files)} documents")
        
        all_questions = []
        
        for i, file_key in enumerate(selected_files):
            file_chunks = chunks_by_file[file_key]
            
            # Randomly select one chunk from this file
            selected_chunk = random.choice(file_chunks)
            
            print(f"\nProcessing document {i+1}/{len(selected_files)}: {file_key}")
            print(f"  Selected 1 random chunk from {len(file_chunks)} available chunks")
            
            # Generate 1 question from this chunk
            try:
                chunk_questions = self.generate_questions_for_chunk(selected_chunk, num_questions=1)
                
                # Add file information
                for question in chunk_questions:
                    question['file'] = file_key
                
                all_questions.extend(chunk_questions)
                print(f"  Generated {len(chunk_questions)} question(s)")
                
            except Exception as e:
                print(f"  Error generating questions: {e}")
                continue
        
        print(f"\nTotal questions generated: {len(all_questions)}")
        self.test_questions = all_questions
        return all_questions


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate test questions from vector stores using various LLMs')
    
    # Vector store options (replaces files argument)
    parser.add_argument('--store', choices=['faiss', 'pinecone', 'chroma'], default='faiss',
                        help='Vector store to use (default: faiss)')
    parser.add_argument('--faiss-path', help='FAISS index directory path (default: FAISS_INDEX_PATH env or "faiss_index")')
    parser.add_argument('--pinecone-key', help='Pinecone API key (default: PINECONE_API_KEY env)')
    parser.add_argument('--pinecone-index', help='Pinecone index name (default: PINECONE_INDEX env)')
    parser.add_argument('--chroma-path', help='Chroma database path (default: CHROMA_PATH env or "./chroma_db")')
    parser.add_argument('--chroma-index', help='Chroma collection name (default: CHROMA_INDEX env or "default_index")')
    
    # Basic options
    parser.add_argument('--output', '-o', default='test_questions.json', 
                        help='Output JSON file for generated questions')
    
    
    # LLM provider options
    parser.add_argument('--provider', choices=['openai', 'ollama', 'huggingface'], default='ollama',
                        help='LLM provider to use')
    parser.add_argument('--model', default='llama3.2', 
                        help='Model name (varies by provider)')
    parser.add_argument('--api-key', 
                        help='API key for OpenAI (if not provided, will use OPENAI_API_KEY environment variable)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', 
                        help='URL for Ollama server')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for HuggingFace models')
    
    # Generation options
    parser.add_argument('--questions-per-chunk', type=int, default=1, 
                        help='Number of questions to generate per chunk (default: 1)')
    parser.add_argument('--questions-per-file', type=int, default=None, 
                        help='Maximum number of questions per source file (randomly selects chunks)')
    parser.add_argument('--total-questions', type=int, default=None, 
                        help='Maximum total questions to generate')
    parser.add_argument('--random-docs', type=int, default=None,
                        help='Randomly select N documents and generate 1 question from 1 random chunk per document')
    parser.add_argument('--temperature', type=float, default=0.7, 
                        help='Temperature for LLM generation (lower = more specific)')
    
    # Enhanced options
    parser.add_argument('--enhance-targeting', action='store_true', 
                        help='Use enhanced chunk-specific targeting for better search performance')
    parser.add_argument('--central-content-focus', action='store_true', 
                        help='Focus on central content rather than peripheral details')
    
    # Utility options
    parser.add_argument('--quiet', '-q', action='store_true', 
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle API key for OpenAI - use command line arg or environment variable
    if args.provider == 'openai':
        if not args.api_key:
            args.api_key = os.getenv('OPENAI_API_KEY')
            if not args.api_key:
                print("Error: OpenAI API key is required. Provide it via --api-key argument or set OPENAI_API_KEY environment variable.")
                sys.exit(1)
    
    # Validate store-specific requirements
    if args.store == 'chroma' and not CHROMA_AVAILABLE:
        print("Error: Chroma dependencies not installed. Run: pip install langchain-chroma chromadb")
        sys.exit(1)
    
    # Validate provider-specific requirements
    if args.provider == 'openai' and not OPENAI_AVAILABLE:
        print("Error: OpenAI provider requires 'openai' package. Install with: pip install openai")
        sys.exit(1)
    
    if args.provider == 'huggingface' and not HUGGINGFACE_AVAILABLE:
        print("Error: HuggingFace provider requires 'transformers' and 'torch' packages.")
        print("Install with: pip install transformers torch")
        sys.exit(1)
    
    # Initialize the question generator
    try:
        generator = QuestionGenerator(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            ollama_url=args.ollama_url,
            device=args.device,
            use_enhanced_targeting=args.enhance_targeting,
            central_content_focus=args.central_content_focus,
            generation_temperature=args.temperature,
            max_total_questions=args.total_questions
        )
    except Exception as e:
        print(f"Error initializing question generator: {e}")
        sys.exit(1)
    
    # Check LLM availability
    if not generator.llm.is_available():
        print("Warning: LLM is not available. Using fallback question generation.")
    
    # Load vector store
    try:
        print(f"Loading {args.store} vector store...")
        generator.load_vector_store(
            args.store,
            faiss_path=args.faiss_path,
            pinecone_key=args.pinecone_key,
            pinecone_index=args.pinecone_index,
            chroma_path=args.chroma_path,
            chroma_index=args.chroma_index
        )
        print("Vector store loaded successfully")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        sys.exit(1)
    
    # Get chunks from vector store
    # Generate questions using the appropriate method
    if args.random_docs:
        # Use random document selection method
        questions = generator.generate_questions_random_docs(num_docs=args.random_docs)
    else:
        # Use traditional chunk-based method
        try:
            print("Retrieving chunks from vector store...")
            chunks = generator.get_all_chunks_from_vectorstore()
            print(f"Retrieved {len(chunks)} chunks from vector store")
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            sys.exit(1)
        
        # Generate questions from chunks
        questions = generator.generate_questions_from_chunks(
            chunks=chunks,
            questions_per_chunk=args.questions_per_chunk,
            questions_per_file=args.questions_per_file,
            verbose=not args.quiet
        )
    
    # Save the generated questions
    generator.save_questions(args.output)
    
    print(f"\nGenerated {len(questions)} questions.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
