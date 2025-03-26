# Advanced RAG Media Query System

A Python-based Retrieval-Augmented Generation (RAG) system that processes various media files (audio, video, text, PDFs) and web pages, enabling sophisticated natural language querying with reranking, multiple collection support, and rich output formatting.

## What It Does

This system provides a comprehensive RAG solution with:
- **Multi-format media processing**: Audio (MP3, WAV), Video (MP4, AVI, MOV, MKV, WEBM, FLV, M4V, 3GP), TXT, and PDF documents
- **Web content extraction**: Intelligent web scraping with content detection
- **Multiple vector stores**: FAISS (local), Pinecone (cloud), ChromaDB (local persistent)
- **Advanced reranking**: HuggingFace and Cohere rerankers for improved relevance
- **Context expansion**: Intelligent chunk expansion around matches
- **Multiple collections**: Query across multiple document collections simultaneously 
- **Rich output formats**: Standard text, rich formatted output, structured JSON
- **Model comparisons**: Compare answers across different LLMs and configurations
- **Auto-detection**: Automatic embedding model detection from vector store metadata

## Quick Start

### 1. Process Media Files
```bash
# Create searchable knowledge base from media files
python3 rag.py ./media_directory/

# Or with specific vector store
python3 rag.py --store chroma ./media_directory/
```

### 2. Query Your Knowledge Base
```bash
# Basic query
python3 ask.py "What is machine learning?"

# Advanced query with rich formatting
python3 ask.py --rich -v "What is machine learning?"

# Query with reranking for better results
python3 ask.py --rerank -k 10 -kk 4 "What is machine learning?"
```

## Advanced Usage

### Multiple Collection Support
```bash
# Query across multiple collections
python3 ask.py --collections "podcasts,docs,research" "What does the research say?"

# With detailed reranking visualization
python3 ask.py -c "podcasts,docs" -k 20 -kk 5 --rerank --show-rerank-results "AI trends"
```

### Model and Configuration Comparisons
```bash
# Compare different LLM models
python3 ask.py --compare-models "gpt-4o,gpt-3.5-turbo" "What is AI?"

# Compare with/without reranking
python3 ask.py --compare-reranking --rerank "What is AI?"

# Compare different retrieval parameters
python3 ask.py --compare-k "k=5,k=10,k=20" "What is AI?"
```

### Output Formats

#### Rich Formatted Output
```bash
python3 ask.py --rich "What is machine learning?"
```
Shows beautifully formatted answers with colored panels and enhanced readability.

#### JSON Output
```bash
python3 ask.py --json "What is machine learning?"
```
Returns structured JSON perfect for API integration:
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "references": [...],
  "metadata": {...}
}
```

### Context Expansion and Verbose Modes
```bash
# Verbose mode with context expansion details
python3 ask.py -v --expand-context 3 "What is AI?"

# Extra verbose mode showing detailed processing
python3 ask.py -vv "What is AI?"
```

The `-vv` mode shows detailed progress similar to:
```
- Processing collection: research_papers
  - Found 2 relevant chunks at positions: [45, 67]  
  - Expanding to include chunks: [43, 44, 45, 46, 47, 65, 66, 67, 68, 69]
  - Retrieved 10 contextual chunks (was getting 1 chunk without expansion)
  - Context size: 8,451 chars (vs ~845 chars for single chunk)
```

### Reranking with Detailed Results
```bash
python3 ask.py --rerank --show-rerank-results -k 20 -kk 4 "What is machine learning?"
```

Shows comprehensive reranking analysis:
```
*** Cross-Collection Reranking Results (20 → 4) ***

Pre-rerank (top 10 by similarity):
  1. 0.8456 - [docs] Machine learning fundamentals... (chunk 23/156)
  2. 0.8234 - [research] AI and ML overview... (chunk 59/100) 
  ...

Post-rerank (final 4 sent to LLM):
  1. 0.9234 ↗️ [research] AI and ML overview... (chunk 59/100) [was #2]
  2. 0.9156 ↗️ [docs] Machine learning fundamentals... (chunk 23/156) [was #1]
  ...

Collection Distribution in Final Results:
- research: 2 documents (50%)
- docs: 2 documents (50%)
```

## LLM and Embedding Support

### LLM Providers
- **OpenAI**: GPT-4o, GPT-3.5-turbo, GPT-4-turbo
- **HuggingFace**: Various open-source models
- **Ollama**: Local models (llama2, mistral, etc.)

### Embedding Models
- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **HuggingFace**: BGE-M3, all-MiniLM-L6-v2
- **Nomic**: nomic-embed-text

### Rerankers
- **HuggingFace**: ms-marco-MiniLM-L-6-v2, cross-encoder models
- **Cohere**: rerank-english-v2.0, rerank-multilingual-v2.0

## Command Reference

### Basic Commands
```bash
# Query existing knowledge base
python3 ask.py "your question"

# Verbose output with system info
python3 ask.py -v "your question"

# Extra verbose with detailed processing
python3 ask.py -vv "your question"

# Rich formatted output  
python3 ask.py --rich "your question"

# JSON output for APIs
python3 ask.py --json "your question"
```

### Vector Store Options
```bash
# FAISS (local, default)
python3 ask.py --store faiss "your question"

# Pinecone (cloud)
python3 ask.py --store pinecone "your question"

# ChromaDB (local persistent) 
python3 ask.py --store chroma "your question"
```

### Retrieval Configuration
```bash
# Basic retrieval parameters
python3 ask.py -k 10 "your question"              # Get 10 documents

# With reranking
python3 ask.py -k 20 -kk 5 --rerank "your question"  # Get 20, rerank to top 5

# Context expansion
python3 ask.py --expand-context 3 "your question"  # Expand ±3 chunks around matches

# Show reranking details
python3 ask.py --rerank --show-rerank-results "your question"
```

### Multiple Collections
```bash
# Query multiple collections
python3 ask.py -c "collection1,collection2,collection3" "your question"

# With reranking across collections
python3 ask.py -c "docs,research" -k 15 -kk 4 --rerank "your question"
```

### Model Comparisons
```bash
# Compare LLM models
python3 ask.py --compare-models "gpt-4o,gpt-3.5-turbo,claude-3.5" "your question"

# Compare reranking effectiveness  
python3 ask.py --compare-reranking --rerank "your question"

# Compare retrieval parameters
python3 ask.py --compare-k "k=5,k=10,k=20" "your question"
python3 ask.py --compare-k "k=20:kk=3,k=20:kk=5,k=20:kk=10" "your question"
```

## Vector Store Comparison

| Feature | FAISS | Pinecone | ChromaDB |
|---------|-------|----------|----------|
| **Storage** | Local persistent | Cloud managed | Local persistent |
| **Cost** | Free | Usage-based pricing | Free |
| **Scalability** | Single machine | Highly scalable | Single machine |
| **Setup** | No config needed | API key required | Local setup |
| **Collections** | Single collection | Multiple indexes | Multiple collections |
| **Metadata** | Basic | Advanced filtering | Advanced SQL-like |
| **Internet** | Offline capable | Requires connection | Offline capable |

## Performance Features

### Caching and Optimization
- **Transcript caching**: Processed content cached to avoid re-extraction
- **Incremental processing**: Only processes new/changed files
- **Embedding model auto-detection**: Automatically uses compatible embeddings
- **Context expansion**: Retrieves surrounding chunks for better context

### Intelligent Processing
- **Smart web scraping**: Detects main content vs navigation/ads
- **File change detection**: Tracks file modifications to avoid reprocessing  
- **Chunk overlap**: Preserves context across chunk boundaries
- **Document metadata**: Tracks source files, chunk positions, timestamps

## Integration Examples

### API Integration (JSON Mode)
```python
import subprocess
import json

result = subprocess.run([
    'python3', 'ask.py', '--json', '--store', 'chroma', 
    'What is machine learning?'
], capture_output=True, text=True)

data = json.loads(result.stdout)
answer = data['answer']
references = data['references'] 
```

### Batch Processing
```bash
# Process multiple queries
while IFS= read -r query; do
    echo "Query: $query"
    python3 ask.py --json "$query" | jq '.answer'
    echo "---"
done < queries.txt
```

### Web Interface Integration
The JSON output format makes it easy to integrate with web applications, APIs, and other services.

## Development

### Modular Architecture
- `src/ask_core/` - Core RAG system components
- `src/ask_core/qa_system.py` - Main question-answering logic
- `src/ask_core/embeddings.py` - Embedding model management
- `src/ask_core/vector_stores.py` - Vector store abstractions  
- `src/ask_core/rerankers.py` - Reranking implementations
- `src/ask_core/llm_managers.py` - LLM provider integrations

### Extending the System
The modular design makes it easy to:
- Add new vector stores
- Integrate additional LLM providers
- Implement custom rerankers
- Add new output formats

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Brad McMillen/CapyAI]
