#!/usr/bin/env python3

import os

# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import json
import whisper
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore

# Validate arguments
if len(sys.argv) != 2:
    print(f"Syntax: {sys.argv[0]} MediaDirectory")
    sys.exit(1)

MEDIA_DIR = sys.argv[1]

# Validate media directory
if not os.path.exists(MEDIA_DIR):
    print(f"Error: Directory '{MEDIA_DIR}' does not exist.")
    sys.exit(1)

if not os.path.isdir(MEDIA_DIR):
    print(f"Error: '{MEDIA_DIR}' is not a directory.")
    sys.exit(1)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR", "./transcripts")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# New configurable model parameter with default
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# FAISS index persistence
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
PROCESSED_FILES_PATH = os.path.join(os.path.dirname(FAISS_INDEX_PATH), "processed_files.json")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY must be set in the environment.")
    sys.exit(1)

# Create transcript directory if it doesn't exist
if not os.path.exists(TRANSCRIPT_DIR):
    print(f"*** Creating transcript directory: {TRANSCRIPT_DIR} ***")
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

print(f"*** Using Whisper model: {WHISPER_MODEL} ***")
print(f"*** Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL} ***")

def load_processed_files():
    """Load the list of already processed files from disk"""
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"*** Warning: Could not load processed files metadata ({e}). Treating all files as new. ***")
            return {}
    return {}

def save_processed_files(processed_files):
    """Save the list of processed files to disk"""
    try:
        dirname = os.path.dirname(PROCESSED_FILES_PATH)
        if dirname:  # Only create directory if it's not empty (not current dir)
            os.makedirs(dirname, exist_ok=True)
        
        with open(PROCESSED_FILES_PATH, 'w') as f:
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

whisper_model = None
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# Determine which vector store to use
use_pinecone = bool(PINECONE_API_KEY and PINECONE_INDEX)

if use_pinecone:
    print(f"\n*** Using Pinecone vector store: {PINECONE_INDEX} ***")
else:
    print(f"\n*** Using local FAISS vector store: {FAISS_INDEX_PATH} ***")

# Load existing index and processed files metadata
processed_files = load_processed_files()
vectorstore1 = None

if not use_pinecone:
    # FAISS setup
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"*** Loading existing FAISS index from: {FAISS_INDEX_PATH} ***")
        vectorstore1 = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"*** Loaded existing index with {vectorstore1.index.ntotal} documents ***")
    else:
        print(f"*** No existing FAISS index found. Will create new index ***")

all_documents = []
new_files_processed = 0
processed_in_this_run = set()  # Track files processed in current run to prevent duplicates

# Process media files
for filename in os.listdir(MEDIA_DIR):
    media_path = os.path.join(MEDIA_DIR, filename)
    if not os.path.isfile(media_path):
        continue

    if not filename.lower().endswith(('.mp3', '.wav', '.txt', '.pdf', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp')):
        continue

    # Prevent processing the same file twice in one run
    if filename in processed_in_this_run:
        print(f"\n*** Skipping {filename} (already processed in this run) ***")
        continue

    # Check if file has changed or is new
    if not file_has_changed(media_path, processed_files):
        print(f"\n*** Skipping {filename} (already processed and unchanged) ***")
        continue

    base_name = os.path.splitext(filename)[0]
    transcript_filename = os.path.join(TRANSCRIPT_DIR, base_name + ".txt")

    print(f"\n*** Processing file: {filename} ***")
    print(f"*** Transcript file: {transcript_filename} ***")

    if os.path.exists(transcript_filename):
        print("*** Skipping extraction/transcription; text file already exists ***")
        loader = TextLoader(transcript_filename)
    elif filename.lower().endswith('.txt'):
        with open(media_path) as file:
            tokens = file.read()
        with open(transcript_filename, "w") as file:
            file.write(tokens)
        loader = TextLoader(transcript_filename)
    elif filename.lower().endswith('.pdf'):
        loader = PyPDFLoader(media_path)
        tokens = "\n".join([doc.page_content for doc in loader.load()])
        with open(transcript_filename, "w") as file:
            file.write(tokens)
    else:
        # Handle audio and video files with Whisper transcription
        file_type = "audio" if filename.lower().endswith(('.mp3', '.wav')) else "video"
        print(f"*** Transcribing {file_type}: {media_path} ***")
        if whisper_model is None:
            print(f"*** Loading Whisper model: {WHISPER_MODEL} ***")
            whisper_model = whisper.load_model(WHISPER_MODEL)
        start = time.time()
        transcription = whisper_model.transcribe(media_path, fp16=False)
        end = time.time()
        tokens = transcription["text"].strip()
        print(f"*** Transcribed in {end - start:.2f} seconds; {len(tokens)} characters")
        with open(transcript_filename, "w") as file:
            file.write(tokens)
        loader = TextLoader(transcript_filename)

    raw_documents = loader.load()
    split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400).split_documents(raw_documents)

    # Assign document IDs
    for idx, doc in enumerate(split_docs):
        doc.metadata["doc_id"] = f"{base_name}_{idx}"
        doc.metadata["source_file"] = filename

    print(f"*** Split transcript into {len(split_docs)} documents ***")
    all_documents.extend(split_docs)
    
    # Mark file as processed
    processed_files[filename] = get_file_info(media_path)
    processed_in_this_run.add(filename)
    new_files_processed += 1


# Create or update vector store
if new_files_processed > 0 and len(all_documents) > 0:
    print(f"\n*** Processing {new_files_processed} new/changed files with {len(all_documents)} document chunks ***")
    
    if use_pinecone:
        # Pinecone setup and processing
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index_list = pc.list_indexes().names()

            if PINECONE_INDEX not in index_list:
                # Create index and upload documents
                print(f"*** Creating Pinecone index: {PINECONE_INDEX} ***")
                start_time = time.time()
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,  # For text-embedding-3-small
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )

                # Wait for index to become available
                for _ in range(10):
                    time.sleep(1)
                    if PINECONE_INDEX in pc.list_indexes().names():
                        break
                else:
                    print("Error: Index did not become available within 10 seconds.")
                    sys.exit(1)

                elapsed = time.time() - start_time
                print(f"*** Index created and confirmed in {elapsed:.2f} seconds ***")

            # Upload documents to Pinecone
            print(f"*** Uploading documents to Pinecone index '{PINECONE_INDEX}' ***")
            start_upload = time.time()

            # Convert documents into (id, content, metadata) tuples
            docs_with_ids = [
                {
                    "id": doc.metadata["doc_id"],
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in all_documents
            ]

            vectorstore1 = PineconeVectorStore.from_texts(
                texts=[doc["page_content"] for doc in docs_with_ids],
                embedding=embeddings,
                metadatas=[doc["metadata"] for doc in docs_with_ids],
                ids=[doc["id"] for doc in docs_with_ids],
                index_name=PINECONE_INDEX
            )

            upload_time = time.time() - start_upload
            print(f"*** Uploaded {len(all_documents)} documents in {upload_time:.2f} seconds ***")
            print(f"*** Successfully processed {new_files_processed} files ***")

        except Exception as e:
            print(f"Error with Pinecone operation: {e}")
            sys.exit(1)
    
    else:
        # FAISS processing
        if vectorstore1 is None:
            # Create new FAISS index
            print(f"*** Creating new FAISS index ***")
            vectorstore1 = FAISS.from_documents(all_documents, embedding=embeddings)
        else:
            # Add new documents to existing index
            print(f"*** Adding new documents to existing FAISS index ***")
            print(f"*** Index had {vectorstore1.index.ntotal} documents before merge ***")
            new_vectorstore = FAISS.from_documents(all_documents, embedding=embeddings)
            vectorstore1.merge_from(new_vectorstore)
        
        # Save updated index to disk
        print(f"*** Saving FAISS index to: {FAISS_INDEX_PATH} ***")
        vectorstore1.save_local(FAISS_INDEX_PATH)
        
        total_docs = vectorstore1.index.ntotal
        print(f"*** FAISS index now contains {total_docs} total documents ***")
        print(f"*** Successfully processed {new_files_processed} files ***")
    
    # Save processed files metadata
    save_processed_files(processed_files)
    
elif new_files_processed > 0 and len(all_documents) == 0:
    print(f"\n*** Warning: {new_files_processed} files were processed but generated no document chunks ***")
    # Still save the processed files metadata to avoid reprocessing
    save_processed_files(processed_files)
    
else:
    if use_pinecone:
        print(f"\n*** No new files to process. ***")
    else:
        if vectorstore1 is not None:
            total_docs = vectorstore1.index.ntotal
            print(f"\n*** No new files to process. FAISS index contains {total_docs} documents ***")
        else:
            print(f"\n*** No files found to process in {MEDIA_DIR} ***")

print("*** Processing complete ***")
