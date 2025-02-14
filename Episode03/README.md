# RAG Media Query System

A Python-based Retrieval-Augmented Generation (RAG) system that processes various media files (audio, video, text, PDFs) and web pages, enabling natural language querying of their content using OpenAI's GPT models and embedding technology.

## Changes from Episode02

### Web Page Processing
- **New capability**: Process and index content from web pages via URL

### Enhanced Vector Storage Options
- **ChromaDB integration**: Added as third storage option

### Additional Dependencies
- `langchain-chroma` - ChromaDB vector database integration
- `requests` - HTTP requests for web content fetching  
- `beautifulsoup4` - HTML parsing and text extraction
- `lxml` - XML/HTML parser backend
- `tqdm` - Progress bars for operations
- `chromadb` - ChromaDB database engine
- `langchain_ollama` - Ollama LLM integration
- `langchain_anthropic` - Anthropic Claude integration  
- `sentence-transformers` - Local reranking models

### Document Reranking
- **HuggingFace rerankers**: Integrated reranking using cross-encoder models
- **Popular models supported**: ms-marco-MiniLM, BGE reranker models

### Additional Embedding Models
- **Local embeddings**: Support for local embedding models via Ollama
- **BGE-M3 embeddings**: High-quality multilingual embeddings locally
- **Nomic embeddings**: Efficient local text embeddings
- **Cost-free embeddings**: No API costs for local embedding models

### New Scripts and Features
- `analyze_chroma.py` - ChromaDB analysis and inspection tool
- `rerank_core/` - Dedicated reranking module with HuggingFace integration

## What It Does

This system:
- **Processes multiple media formats**: Audio (MP3, WAV), Video (MP4, AVI, MOV, MKV, WEBM, FLV, M4V, 3GP), TXT, and PDF documents
- **Extracts web content**: Scrapes and extracts text content from web pages
- **Transcribes audio and video**: Uses OpenAI Whisper to convert audio/video files to text transcripts
- **Extracts text**: Extracts readable text from PDFs and web pages 
- **Creates searchable knowledge base**: Splits content into chunks and generates embeddings for semantic search
- **Answers questions**: Uses RAG to find relevant content and generate answers using GPT-3.5-turbo
- **Supports three storage options**: 
  - FAISS local vector store (default, persistent local storage)
  - Pinecone cloud vector database (optional, requires API key)
  - ChromaDB vector database (optional, local persistent storage with advanced features)


## Usage

### Basic Syntax

```bash
# Process media files in a directory (FAISS - default)
python3 rag.py <MediaDirectory>

# Process with specific storage backend
python3 rag.py --store faiss <MediaDirectory>     # FAISS local storage
python3 rag.py --store pinecone <MediaDirectory>  # Pinecone cloud storage  
python3 rag.py --store chroma <MediaDirectory>    # ChromaDB local storage

# Process a single web page
python3 rag.py <URL>
```

### Examples

#### Processing a Web Page

```bash
# Process a web page and create/update vector index
python3 rag.py https://en.wikipedia.org/wiki/Artificial_intelligence

*** Processing URL: https://en.wikipedia.org/wiki/Artificial_intelligence ***
*** Fetching web page: https://en.wikipedia.org/wiki/Artificial_intelligence ***
*** Extracted 45234 characters from web page ***
*** Split web page into 67 documents ***
```

#### Processing Media Files Directory

```bash
# This is the first time the media files have been seen, so the PDF's and audio
# files need to be processed and the results will be saved to TRANSCRIPT_DIR to
# avoid reprocessing

python3 rag.py ./files

*** No Pinecone configuration - using FAISS vector store ***

*** Processing file: NIPS-2017-attention-is-all-you-need-Paper.pdf ***
*** Transcript file: ./transcripts/NIPS-2017-attention-is-all-you-need-Paper.txt ***
*** Split transcript into 53 documents ***

*** Processing file: Adams.txt ***
*** Transcript file: ./transcripts/Adams.txt ***
*** Split transcript into 143 documents ***

*** Processing file: Uber.mp3 ***
*** Transcript file: ./transcripts/Uber.txt ***
*** Transcribing audio: files/Uber.mp3 ***
*** Transcribed in 202.18 seconds; 80459 characters
*** Split transcript into 134 documents ***
```


#### Processing with ChromaDB

```bash
# First time setup with ChromaDB
python3 rag.py --store chroma ./files

*** Using ChromaDB vector store ***
*** Created new Chroma collection: default_index ***

*** Processing file: NIPS-2017-attention-is-all-you-need-Paper.pdf ***
*** Added documents to ChromaDB collection ***

*** Answer using ChromaDB: ***
A transformer is a model architecture that relies entirely on an attention mechanism...

# Subsequent runs with ChromaDB (uses cached transcripts)
python3 rag.py --store chroma ./files "what is attention mechanism"

*** Using ChromaDB vector store ***
*** Found existing Chroma collection: default_index ***
*** Skipping extraction/transcription; text file already exists ***
*** Added documents to ChromaDB collection ***

*** Answer using ChromaDB: ***
The attention mechanism allows the model to focus on different parts of the input sequence...
```

### Parameters

- **`MediaDirectory`**: Path to directory containing your media files (supports .mp3, .wav, .txt, .pdf, video files)
- **`URL`**: Web page URL to process and add to knowledge base
- **`--store`**: Storage backend (faiss, pinecone, chroma)
- **`--model`**: Embedding model (ada002, 3-small, 3-large, bge-m3, nomic-embed-text)
- **`--chroma-path`**: Override ChromaDB database path
- **`--chroma-index`**: Override ChromaDB collection name
- **`--chunk-size`**: Text chunk size (default: 1000)
- **`--chunk-overlap`**: Text chunk overlap (default: 400)

## Storage Options

### FAISS Vector Store (Default)
- **Pros**: Local persistent storage, no API keys required, fast retrieval, free
- **Cons**: Single-machine only, limited scalability
- **Use case**: Personal use, development, medium datasets, offline usage
- **Setup**: No additional configuration needed

### Pinecone Vector Store (Optional)
- **Pros**: Cloud-based, highly scalable, managed service, fast retrieval
- **Cons**: Requires API key and setup, usage costs, internet dependency
- **Use case**: Production deployments, large datasets, team collaboration
- **Setup**: Requires `PINECONE_API_KEY` and `PINECONE_INDEX`

### ChromaDB Vector Store (Optional)
- **Pros**: Local persistent storage, advanced metadata filtering, SQL-like queries, free
- **Cons**: Additional dependencies, single-machine storage
- **Use case**: Advanced local deployments, complex metadata queries, development
- **Setup**: Requires `langchain-chroma` and `chromadb` packages (included in requirements.txt)

### Querying Existing Vector Stores

If you have already processed files and created embeddings with any storage backend, use the `ask.py` script for additional queries without reprocessing:

```bash
# Query existing FAISS index
python3 ask.py "your question here"

# Query existing Pinecone index  
python3 ask.py --store pinecone "your question here"

# Query existing ChromaDB collection
python3 ask.py --store chroma "your question here"
```

The `rag.py` script handles content extraction and embedding creation, while `ask.py` only queries existing vector stores. For Pinecone users, this avoids unnecessary embedding costs when asking new questions.

**Note**: Web page content is cached to the transcript directory just like media files, so re-running the same URL will use the cached content unless manually deleted.

## ChromaDB Usage

ChromaDB offers additional features beyond basic storage:

### Custom Collection and Path
```bash
# Use custom ChromaDB path and collection name
python3 rag.py --store chroma --chroma-path ./my_knowledge_base --chroma-index research_papers ./documents/

# Override environment variables temporarily  
python3 rag.py --store chroma --chroma-path /tmp/chroma_test --chroma-index test_collection ./files/
```

### Chunk Size Optimization
```bash  
# Use smaller chunks for better precision
python3 rag.py --store chroma --chunk-size 500 --chunk-overlap 100 ./documents/

# Use larger chunks for more context
python3 rag.py --store chroma --chunk-size 2000 --chunk-overlap 500 ./documents/
```

### Querying ChromaDB
```bash
# Query with specific ChromaDB settings
python3 ask.py --store chroma --chroma-path ./my_knowledge_base --chroma-index research_papers "What is machine learning?"

# Use environment variables for repeated queries
export CHROMA_PATH=./my_knowledge_base
export CHROMA_INDEX=research_papers
python3 ask.py --store chroma "Explain neural networks"
```

## Local Embedding Model Support

Episode03 introduces support for local embedding models via Ollama, eliminating API costs and improving privacy for the embedding generation phase.

### Local Embedding Models

#### Available Models
- **`bge-m3`**: High-quality multilingual embeddings (1024 dimensions)
- **`nomic-embed-text`**: Efficient English text embeddings (768 dimensions)

#### Setup Requirements

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)

2. **Pull embedding models**:
```bash
# For BGE-M3 (multilingual, high quality)
ollama pull bge-m3

# For Nomic (English, efficient)  
ollama pull nomic-embed-text
```

#### Usage Examples

```bash
# Use BGE-M3 local embeddings
python3 rag.py --model bge-m3 --store faiss ./documents/

*** Using local BGE-M3 embeddings via Ollama ***
*** Processing with local model - no API costs ***

# Use Nomic embeddings for English content
python3 rag.py --model nomic-embed-text --store chroma ./english_docs/

*** Using local Nomic embeddings via Ollama ***
*** Processing file: report.pdf ***
```

#### Embedding Model Comparison

| Model | Type | Dimensions | Languages | Use Case |
|-------|------|------------|-----------|----------|
| 3-small | OpenAI API | 1536 | Multi | Cost-efficient API option |
| 3-large | OpenAI API | 3072 | Multi | Highest quality API option |
| bge-m3 | Local/Ollama | 1024 | Multi | Local multilingual, free |
| nomic-embed-text | Local/Ollama | 768 | English | Local English, efficient |

**Note**: While embedding generation can be done locally, answer generation currently still uses OpenAI's GPT models and requires an API key.

## Document Reranking

Reranking can improve search accuracy by reordering initial retrieval results using more sophisticated models. Note that in some cases, reranking can reduce accuracy (so test with different reranker models and configurations)

### Available Reranker Models

The `rerank_core` module provides HuggingFace-based reranking:

#### Model Presets
- **`fast`**: `cross-encoder/ms-marco-MiniLM-L-6-v2` - Quick reranking
- **`balanced`**: `cross-encoder/ms-marco-MiniLM-L-12-v2` - Good speed/quality balance  
- **`quality`**: `BAAI/bge-reranker-base` - High quality, supports Chinese/English
- **`best`**: `BAAI/bge-reranker-large` - Highest quality but slower

### Reranking Usage

```python
# Example usage in custom scripts
from rerank_core.reranker import HuggingFaceReranker

# Initialize reranker
reranker = HuggingFaceReranker(model_name="quality")

# Rerank search results  
results = reranker.rerank_simple(
    query="machine learning algorithms",
    docs_with_scores=initial_search_results,
    top_k=5
)
```

### Testing Reranker

```python
from rerank_core.reranker import test_reranker

# Test if reranker works
if test_reranker("fast"):
    print("Reranker is working correctly")
```

## Output

The system provides a single response using your configured storage backend:
- **FAISS**: Local persistent vector store (default)
- **Pinecone**: Cloud vector database (if API credentials provided)
- **ChromaDB**: Local persistent vector database with advanced features

If no specific storage is configured, FAISS is used as the default.

## File Support

| Format | Extension/Input | Processing Method |
|--------|-----------|-------------------|
| Audio | .mp3, .wav | OpenAI Whisper transcription |
| Video | .mp4, .avi, .mov, .mkv, .webm, .flv, .m4v, .3gp | OpenAI Whisper transcription (audio extraction) |
| Text | .txt | Direct reading |
| PDF | .pdf | PyPDF text extraction |
| Web Pages | HTTP/HTTPS URLs | BeautifulSoup web scraping and text extraction |

## Performance Notes

- **First run**: Audio and video transcription may take time depending on file size
- **Subsequent runs**: Cached transcripts are reused for faster processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Brad McMillen/CapyAI]
