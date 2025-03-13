"""Utility functions and helpers for the ask_core system."""

import os
from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from collections import defaultdict


def format_documents(docs: List[Document]) -> str:
    """Format documents into a context string."""
    return "\n\n".join([doc.page_content for doc in docs])


def expand_context_chunks(docs_with_scores: List[Tuple[Document, float]], 
                         vectorstore, expand_window: int = 2) -> List[Tuple[Document, float]]:
    """
    Expand retrieved chunks with surrounding context and intelligently merge overlapping ranges.
    
    Args:
        docs_with_scores: List of (Document, score) tuples from vector search
        vectorstore: The vector store to retrieve additional chunks from
        expand_window: Number of chunks to expand before/after each match
        
    Returns:
        List of expanded and merged documents with their scores
    """
    if expand_window == 0:
        return docs_with_scores
    
    # Group documents by source file and collect chunk information
    docs_by_source = defaultdict(list)
    
    for doc, score in docs_with_scores:
        source = doc.metadata.get('source', 'unknown')
        chunk_index = doc.metadata.get('chunk_index')
        total_chunks = doc.metadata.get('total_chunks')
        
        if chunk_index is not None and total_chunks is not None:
            docs_by_source[source].append({
                'doc': doc,
                'score': score,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks
            })
        else:
            # If no chunk info, keep as-is
            docs_by_source[source].append({
                'doc': doc,
                'score': score,
                'chunk_index': None,
                'total_chunks': None
            })
    
    expanded_docs = []
    
    # Process each source file
    for source, source_docs in docs_by_source.items():
        # Separate docs with and without chunk info
        chunked_docs = [d for d in source_docs if d['chunk_index'] is not None]
        non_chunked_docs = [d for d in source_docs if d['chunk_index'] is None]
        
        # Add non-chunked docs as-is
        for doc_info in non_chunked_docs:
            expanded_docs.append((doc_info['doc'], doc_info['score']))
        
        if not chunked_docs:
            continue
            
        # Sort chunked docs by chunk index
        chunked_docs.sort(key=lambda x: x['chunk_index'])
        
        # Create expansion ranges and merge overlapping ones
        ranges = []
        for doc_info in chunked_docs:
            chunk_idx = doc_info['chunk_index']
            total_chunks = doc_info['total_chunks']
            
            # Calculate expansion range
            start_idx = max(0, chunk_idx - expand_window)
            end_idx = min(total_chunks - 1, chunk_idx + expand_window)
            
            ranges.append({
                'start': start_idx,
                'end': end_idx,
                'original_doc': doc_info['doc'],
                'score': doc_info['score'],
                'chunk_index': chunk_idx
            })
        
        # Merge overlapping ranges
        merged_ranges = merge_overlapping_ranges(ranges)
        
        # Create expanded documents for each merged range
        for range_info in merged_ranges:
            try:
                expanded_doc = create_expanded_document(
                    vectorstore, source, range_info, chunked_docs[0]['total_chunks']
                )
                if expanded_doc:
                    expanded_docs.append((expanded_doc, range_info['score']))
            except Exception as e:
                # If expansion fails, fall back to original document
                print(f"Warning: Could not expand context for {source}: {e}")
                expanded_docs.append((range_info['original_doc'], range_info['score']))
    
    return expanded_docs


def merge_overlapping_ranges(ranges: List[Dict]) -> List[Dict]:
    """
    Merge overlapping chunk ranges intelligently.
    
    Example: If chunk #4 and #6 are retrieved with expand_window=2:
    - Range 1: chunks #2-6 (4±2) 
    - Range 2: chunks #4-8 (6±2)
    - Merged: chunks #2-8 (one super-chunk)
    """
    if not ranges:
        return ranges
        
    # Sort ranges by start index
    sorted_ranges = sorted(ranges, key=lambda x: x['start'])
    
    merged = []
    current_range = sorted_ranges[0].copy()
    
    for next_range in sorted_ranges[1:]:
        # Check if ranges overlap or are adjacent
        if next_range['start'] <= current_range['end'] + 1:
            # Merge ranges
            current_range['end'] = max(current_range['end'], next_range['end'])
            # Keep the best score (assuming lower is better for distances, higher for similarities)
            # We'll use the score from the first document in the range
            if next_range['score'] < current_range['score']:  # Assuming distance scores
                current_range['score'] = next_range['score']
        else:
            # No overlap, start a new range
            merged.append(current_range)
            current_range = next_range.copy()
    
    # Add the last range
    merged.append(current_range)
    
    return merged


def create_expanded_document(vectorstore, source: str, range_info: Dict, total_chunks: int) -> Optional[Document]:
    """
    Create an expanded document by retrieving and combining chunks in the specified range.
    """
    try:
        start_idx = range_info['start']
        end_idx = range_info['end']
        original_doc = range_info['original_doc']
        
        # Retrieve chunks in the specified range
        range_chunks = vectorstore.get_chunks_by_source_and_range(source, start_idx, end_idx)
        
        if not range_chunks:
            # Fallback to original document if we can't retrieve the range
            return original_doc
        
        # Combine the chunks into a single expanded document
        combined_content = "\n\n".join([chunk.page_content for chunk in range_chunks])
        
        # Create metadata for the expanded document
        expanded_metadata = original_doc.metadata.copy()
        expanded_metadata['expanded_range'] = f"{start_idx}-{end_idx}"
        expanded_metadata['expansion_window'] = end_idx - start_idx + 1
        expanded_metadata['chunk_count'] = len(range_chunks)
        
        # Use the chunk index from the first chunk in the range
        if range_chunks:
            expanded_metadata['chunk_index'] = range_chunks[0].metadata.get('chunk_index', start_idx)
        
        return Document(
            page_content=combined_content,
            metadata=expanded_metadata
        )
        
    except Exception as e:
        print(f"Error creating expanded document: {e}")
        return original_doc  # Fallback to original


def display_results(query: str, docs_with_scores: List[Tuple[Document, float]], 
                   answer: str, store_info: dict, llm_info: dict, 
                   preview_bytes: int = 0, verbose: bool = False, 
                   reranking_enabled: bool = False, show_all_references: bool = False):
    """Display search results in the original ask.py format."""
    # Main answer output (like original ask.py)
    print(f"\nAnswer: {answer}")
    
    # Filter references to only contributing ones unless --show-all-references is used
    display_docs = docs_with_scores
    if not show_all_references and docs_with_scores:
        from .reference_filter import ReferenceFilter
        filter = ReferenceFilter()
        display_docs = filter.filter_contributing_references(answer, docs_with_scores, verbose=verbose)
    
    # Show source documents with similarity scores (like original ask.py)
    if display_docs:
        if show_all_references:
            print(f"\nReferences: {len(display_docs)} documents (all retrieved)")
        else:
            filtered_count = len(docs_with_scores) - len(display_docs)
            if filtered_count > 0:
                print(f"\nReferences: {len(display_docs)} documents (filtered out {filtered_count} non-contributing)")
            else:
                print(f"\nReferences: {len(display_docs)} documents")
        
        for i, (doc, score) in enumerate(display_docs, 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            # Build source info with chunk information
            chunk_index = doc.metadata.get('chunk_index')
            total_chunks = doc.metadata.get('total_chunks')
            if chunk_index is not None and total_chunks is not None:
                source_info = f"from {os.path.basename(source)}, chunk {chunk_index + 1} of {total_chunks}"
            else:
                source_info = f"from {os.path.basename(source)}"
            
            # Handle content preview based on --preview-bytes
            if preview_bytes > 0:
                content = doc.page_content[:preview_bytes].replace('\n', ' ')
                if len(doc.page_content) > preview_bytes:
                    content += "..."
                preview_text = f": {content}"
            else:
                preview_text = ""
            
            # Format score based on store type and reranking
            store_type = store_info.get('store_type', 'faiss')
            if reranking_enabled:
                # Rerank scores are typically between -10 to 10 (higher is better)
                print(f"  {i}. {doc_id} ({source_info}) [rerank: {score:.4f}]{preview_text}")
            elif store_type == "pinecone":
                # Pinecone scores are typically between 0-1 (higher is better)
                print(f"  {i}. {doc_id} ({source_info}) [similarity: {score:.4f}]{preview_text}")
            elif store_type in ["faiss", "chroma"]:
                # FAISS/Chroma scores are distances (lower is better), convert to similarity
                similarity = 1 / (1 + score)  # Convert distance to similarity-like score
                print(f"  {i}. {doc_id} ({source_info}) [similarity: {similarity:.4f}]{preview_text}")
            
            # Add newline between references for better readability (only when showing content)
            if preview_bytes > 0 and i < len(docs_with_scores):
                print()


def print_system_info(store_info: dict, llm_info: dict, embedding_model: str, 
                     reranker_info: Optional[dict] = None, verbose: bool = False):
    """Print system information if verbose mode is enabled."""
    if not verbose:
        return
    
    print(f"*** SYSTEM INFO ***")
    print(f"Store Type: {store_info.get('store_type')}")
    print(f"Store Path: {store_info.get('store_path')}")
    if store_info.get('document_count', -1) >= 0:
        print(f"Documents: {store_info.get('document_count')}")
    
    print(f"Embedding Model: {embedding_model}")
    print(f"LLM Type: {llm_info.get('llm_type')}")
    print(f"LLM Model: {llm_info.get('model_name')}")
    
    if reranker_info and reranker_info.get('available'):
        print(f"Reranker: {reranker_info.get('reranker_type')} ({reranker_info.get('model_name')})")


def validate_environment_variables() -> dict:
    """Validate required environment variables."""
    errors = []
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        errors.append("OPENAI_API_KEY must be set in the environment")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "openai_key_set": bool(openai_key)
    }


def get_store_config(args) -> dict:
    """Get vector store configuration from arguments."""
    if args.store == "faiss":
        return {
            "faiss_path": args.faiss_path
        }
    elif args.store == "pinecone":
        return {
            "pinecone_key": args.pinecone_key,
            "pinecone_index": args.pinecone_index
        }
    elif args.store == "chroma":
        return {
            "chroma_path": args.chroma_path,
            "chroma_index": args.chroma_index,
            "collection_name": args.chroma_index  # Pass collection name for metadata detection
        }
    else:
        raise ValueError(f"Unsupported store type: {args.store}")


def print_usage_examples():
    """Print usage examples for the ask system."""
    examples = """
Examples:
  # Query FAISS index (default)
  ask.py "What is artificial intelligence?"
  
  # Query Pinecone index
  ask.py --store pinecone "What is machine learning?"
  
  # Query Chroma index
  ask.py --store chroma --chroma-path ./my_chroma "Explain neural networks"
  
  # Query with reranking (any store)
  ask.py --rerank "What is deep learning?"
  ask.py --store pinecone --rerank --rerank-type huggingface --rerank-model quality "Explain transformers"
  
  # Show detailed results
  ask.py --rerank --show-rerank-results -v "What is AI?"
  
  # Use different LLM models  
  ask.py --llm-type openai --llm-model gpt-4 "What is AI?"
  ask.py --llm-type huggingface --llm-model gemma-3-1b "What is AI?"
    """
    print(examples)