# RAG Media Query System

A Python-based Retrieval-Augmented Generation (RAG) system that processes various media files (audio, text, PDFs) and enables natural language querying of their content using OpenAI's GPT models and embedding technology.

## What It Does

This system:
- **Processes multiple media formats**: MP3, WAV, TXT, and PDF documents
- **Transcribes audio**: Uses OpenAI Whisper to convert audio files to text transcripts
- **Extracts text**: Extracts readable text from PDFs 
- **Creates searchable knowledge base**: Splits content into chunks and generates embeddings for semantic search
- **Answers questions**: Uses RAG to find relevant content and generate answers using GPT-3.5-turbo
- **Supports two storage options**: 
  - In-memory vector store (default, no additional setup required)
  - Pinecone cloud vector database (optional, requires API key)

## Prerequisites

### 1. Setup Python Environment

Create and activate a new Python 3 virtual environment:

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 2. Upgrade pip (Optional but recommended)

```bash
pip install --upgrade pip
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies:
- `python-dotenv` - Environment variable management
- `openai-whisper` - Audio transcription
- `openai` - OpenAI API client
- `langchain` - LangChain framework and components
- `pypdf` - PDF text extraction
- `docarray` - Document array operations


## Environment Setup

Create a `.env` file in your project directory with the following variables:

### Required Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
TRANSCRIPT_DIR=/path/to/transcript/directory
```

### Optional Variables (for Pinecone integration)

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_index_name
```

**Note**: If `PINECONE_API_KEY` and `PINECONE_INDEX` are not provided, the system will automatically use an in-memory vector store instead.

### Environment Variable Details

- **`OPENAI_API_KEY`**: Your OpenAI API key for GPT and embedding models
- **`TRANSCRIPT_DIR`**: Directory where transcript files will be stored/cached
- **`PINECONE_API_KEY`** *(Optional)*: Your Pinecone API key for cloud vector storage
- **`PINECONE_INDEX`** *(Optional)*: Name of your Pinecone index

## Usage

### Basic Syntax

```bash
python3 rag.py <MediaDirectory> <query>
```

### Examples

```bash
# This is the first time the media files have been seen, so the PDF's and audio
# files need to be processed and the results will be saved to TRANSCRIPT_DIR to
# avoid reprocessing

python3 rag.py ./files what is a rnn

*** No Pinecone configuration - using in-memory vector store ***

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

*** Answer using in-memory vector store: ***
A recurrent neural network (RNN) is a type of neural network that is designed to recognize patterns in sequences of data, such as text or speech.
```


```bash
# Running the script again will avoid the time consuming extraction of the text
# from audio files, and use the cached transcript in TRANSCRIPT_DIR

python3 rag.py ./files what is a transformer 

*** No Pinecone configuration - using in-memory vector store ***

*** Processing file: NIPS-2017-attention-is-all-you-need-Paper.pdf ***
*** Transcript file: ./transcripts/NIPS-2017-attention-is-all-you-need-Paper.txt ***
*** Skipping extraction/transcription; text file already exists ***
*** Split transcript into 55 documents ***

*** Processing file: Adams.txt ***
*** Transcript file: ./transcripts/Adams.txt ***
*** Skipping extraction/transcription; text file already exists ***
*** Split transcript into 143 documents ***

*** Processing file: Uber.mp3 ***
*** Transcript file: ./transcripts/Uber.txt ***
*** Skipping extraction/transcription; text file already exists ***
*** Split transcript into 134 documents ***

*** Answer using in-memory vector store: ***
A transformer is a model architecture that relies entirely on an attention mechanism to draw global dependencies between input and output, eschewing recurrence typically used in traditional sequence transduction models.

```

### Parameters

- **`MediaDirectory`**: Path to directory containing your media files (supports .mp3, .wav, .txt, .pdf)
- **`query`**: Your question or search query (can be multiple words)

## How It Works

1. **File Processing**: The system scans the specified directory for supported media files
2. **Content Extraction**:
   - Audio files (.mp3, .wav): Transcribed using OpenAI Whisper
   - PDF files: Text extracted using PyPDF
   - Text files: Content read directly
3. **Transcript Caching**: All extracted/transcribed content is saved to the transcript directory to avoid reprocessing
4. **Text Chunking**: Content is split into overlapping chunks for optimal retrieval
5. **Embedding Generation**: Each chunk is converted to vector embeddings using OpenAI's embedding model
6. **Vector Storage**: Embeddings stored in either:
   - DocArrayInMemorySearch (default)
   - Pinecone cloud database (if credentials are provided)
7. **Query Processing**: Your question is embedded and matched against stored content
8. **Answer Generation**: Relevant chunks are retrieved and used as context for GPT to generate an answer

## Storage Options

### In-Memory Vector Store (Default)
- **Pros**: No additional setup, works immediately, free
- **Cons**: Data lost when program ends, limited to available RAM
- **Use case**: Quick queries, testing, small datasets

### Pinecone Vector Store (Optional)
- **Pros**: Persistent storage, scalable, fast retrieval
- **Cons**: Requires API key and setup, may have usage costs
- **Use case**: Production use, large datasets, persistent knowledge base

If you are using Pinecone and have already stored your embeddings by running the `rag.py` script, 
but you wish to ask additional questions without having to re-create the embeddings, use the `ask.py` 
script. The `rag.py` script is for extracting text from media files and creating the embeddings 
and adding them to the vector store. There is no harm to reprocessing the same media files again
with `rag.py`, but it will incur unnecessary costs for the embeddings. To avoid that,
place new content in a different directory and provide that directory name to `rag.py`

## Output

The system provides two responses when both storage options are available:
1. Answer using in-memory vector store
2. Answer using Pinecone (if configured)

If Pinecone credentials are not provided, only the in-memory response is shown.

## File Support

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| Audio | .mp3, .wav | OpenAI Whisper transcription |
| Text | .txt | Direct reading |
| PDF | .pdf | PyPDF text extraction |

## Performance Notes

- **First run**: Audio transcription may take time depending on file size
- **Subsequent runs**: Cached transcripts are reused for faster processing
- **Whisper model**: Uses 'base' model for balance of speed and accuracy
- **Chunk overlap**: 400 characters overlap ensures context preservation

## Troubleshooting

### Common Issues

1. **Missing API keys**: Ensure `OPENAI_API_KEY` is set in your `.env` file
2. **Transcript directory**: Make sure the `TRANSCRIPT_DIR` exists and is writable
3. **Audio processing**: First-time audio transcription requires internet connection for Whisper model download
4. **Pinecone errors**: If Pinecone fails, the system will fall back to in-memory storage

### Error Messages

- `"Error: OPENAI_API_KEY and TRANSCRIPT_DIR must be set"`: Check your `.env` file
- `"Skipping Pinecone-based response (missing PINECONE_API_KEY)"`: Normal when Pinecone is not configured
- `"Index did not become available within 10 seconds"`: Pinecone index creation timeout (usually resolves on retry)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Brad McMillen/CapyAI]
