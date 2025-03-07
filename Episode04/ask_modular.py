#!/usr/bin/env python3

"""Modular RAG system with FAISS, Pinecone, and Chroma vector stores.

This is a refactored version of ask.py that uses the ask_core modular architecture
for better organization and maintainability.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Fix HuggingFace tokenizers warning when using reranking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ask_core.qa_system import QASystem
from ask_core.utils import (
    display_results, 
    validate_environment_variables, 
    get_store_config,
    print_usage_examples
)
from ask_core.rerankers import check_reranking_dependencies

# Optional rich formatting imports
try:
    from cask_core.utils import VerboseConsole, display_results as rich_display_results
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Query RAG system with FAISS, Pinecone, or Chroma vector stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query FAISS index (default)
  %(prog)s "What is artificial intelligence?"
  
  # Query Pinecone index
  %(prog)s --store pinecone "What is machine learning?"
  
  # Query Chroma index
  %(prog)s --store chroma --chroma-path ./my_chroma "Explain neural networks"
  
  # Query with reranking (any store)
  %(prog)s --rerank "What is deep learning?"
  %(prog)s --store pinecone --rerank --rerank-type huggingface --rerank-model quality "Explain transformers"
  %(prog)s --rerank --rerank-type qwen3 --rerank-model qwen3-8b "Machine learning algorithms"
  %(prog)s --rerank --rerank-type mlx-qwen3 --rerank-model mlx-qwen3-8b "Fast Apple Silicon reranking"
  %(prog)s --rerank -kk 3 "Machine learning algorithms"
  
  # Show reranking results (requires verbose mode)
  %(prog)s --rerank --show-rerank-results -v "What is AI?"
  
  # Control content preview in references
  %(prog)s --preview-bytes 200 "What is machine learning?"  # Show 200 bytes
  %(prog)s --preview-bytes 0 "What is deep learning?"       # No content (default)
  
  # Use different LLM models  
  %(prog)s --llm-type openai --llm-model gpt-4 "What is AI?"
  %(prog)s --llm-type huggingface --llm-model gemma-3-1b "What is AI?"
  %(prog)s --llm-type huggingface --llm-model gemma-3-1b --device mps "Explain transformers"
        """
    )
    
    # Query arguments
    parser.add_argument("query", nargs="+", help="Query text to search for")
    
    # Vector store arguments
    parser.add_argument("--store", choices=["faiss", "pinecone", "chroma"], default="faiss",
                       help="Vector store to use (default: faiss)")
    parser.add_argument("--faiss-path", help="FAISS index directory path (default: FAISS_INDEX_PATH env or 'faiss_index')")
    parser.add_argument("--pinecone-key", help="Pinecone API key")
    parser.add_argument("--pinecone-index", help="Pinecone index name")
    parser.add_argument("--chroma-path", help="Chroma database path (default: CHROMA_PATH env or './chroma_db')")
    parser.add_argument("--chroma-index", help="Chroma collection name (default: CHROMA_INDEX env or 'default_index')")
    
    # Search arguments
    parser.add_argument("-k", "--top-k", type=int, default=4, help="Number of similar documents to retrieve (default: 4)")
    
    # Reranking arguments
    parser.add_argument("--rerank", action="store_true", help="Enable reranking with HuggingFace or Qwen3 models")
    parser.add_argument("--rerank-type", choices=["huggingface", "qwen3", "mlx-qwen3"], default="huggingface",
                       help="Type of reranker to use (default: huggingface). Use mlx-qwen3 for Apple Silicon optimization")
    parser.add_argument("--rerank-model", default="quality",
                       help="Reranking model name or preset (default: quality)")
    parser.add_argument("--rerank-top-k", "-kk", type=int, help="Number of documents to return after reranking (default: same as --top-k)")
    parser.add_argument("--show-rerank-results", action="store_true", help="Show detailed reranking results (requires --verbose)")
    
    # Model arguments
    parser.add_argument("--embedding-model", choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002", "bge-m3", "nomic-embed-text"], 
                       default="text-embedding-3-small", help="Embedding model to use (default: text-embedding-3-small, auto-detected from vector store if available)")
    parser.add_argument("--llm-type", choices=["openai", "huggingface", "ollama"], default="openai",
                       help="Type of LLM to use (default: openai)")
    parser.add_argument("--llm-model", default="gpt-3.5",
                       help="LLM model name or preset (default: gpt-3.5)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                       help="Device for HuggingFace models (default: auto)")
    
    # Document handling arguments
    parser.add_argument("--expand-context", type=int, default=2, help="Number of chunks to expand before/after each match for context (default: 2)")
    parser.add_argument("--use-chunks-only", action="store_true", help="Use individual chunks only without context expansion")
    
    # Output arguments
    parser.add_argument("--preview-bytes", type=int, default=0, help="Number of bytes to show from each document in references (default: 0, no content)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-vv", "--extra-verbose", action="store_true", help="Extra verbose output showing detailed processing steps (implies -v)")
    parser.add_argument("--rich", action="store_true", help="Enable rich console formatting similar to cask.py")
    
    return parser


def _display_results_rich(console, query, docs_with_scores, answer, system_info, 
                         preview_bytes=0, verbose=False, reranking_enabled=False):
    """Display results using rich formatting similar to cask.py."""
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    # Display the answer in a panel
    answer_panel = Panel(
        answer,
        title=f"Answer from {system_info['store_info']['store_type']}",
        title_align="left",
        border_style="green"
    )
    console.print(answer_panel)
    
    # Display references
    if docs_with_scores:
        console.print(f"\n[bold]References:[/bold] {len(docs_with_scores)} documents")
        
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')[:16] + '...'
            source = doc.metadata.get('source', 'unknown')
            
            # Build source info with chunk information
            chunk_index = doc.metadata.get('chunk_index')
            total_chunks = doc.metadata.get('total_chunks')
            if chunk_index is not None and total_chunks is not None:
                source_info = f"from {source}, chunk {chunk_index + 1} of {total_chunks}"
            else:
                source_info = f"from {source}"
            
            # Format score display
            if reranking_enabled and hasattr(doc, 'rerank_score'):
                score_info = f"[rerank: {doc.rerank_score:.4f}]"
            else:
                score_info = f"[similarity: {score:.4f}]"
            
            console.print(f"  {i}. [cyan]{doc_id}[/cyan] ({source_info}) {score_info}")
            
            # Show content preview if requested
            if preview_bytes > 0:
                content_preview = doc.page_content[:preview_bytes]
                if len(doc.page_content) > preview_bytes:
                    content_preview += "..."
                console.print(f"     [dim]{content_preview}[/dim]")
                console.print()
    
    # Display system info if verbose
    if verbose:
        store_info = system_info["store_info"]
        llm_info = system_info["llm_info"]
        embedding_model = system_info["embedding_model"]
        
        system_panel = Panel(
            f"Store Type: {store_info['store_type']}\n"
            f"Store Path: {store_info.get('store_path', 'N/A')}\n"
            f"Documents: {store_info.get('total_documents', 'N/A')}\n"
            f"Embedding Model: {embedding_model}\n"
            f"LLM Type: {llm_info['llm_type']}\n"
            f"LLM Model: {llm_info.get('model', llm_info.get('llm_model', 'N/A'))}\n" +
            (f"Reranker: {system_info.get('reranker_info', {}).get('reranker_type', 'N/A')}" if reranking_enabled else ""),
            title="System Info",
            title_align="left",
            border_style="blue"
        )
        console.print(system_panel)


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Join query arguments
    query = " ".join(args.query)
    
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    env_validation = validate_environment_variables()
    if not env_validation["valid"]:
        for error in env_validation["errors"]:
            print(f"Error: {error}")
        sys.exit(1)
    
    # Validate reranking arguments
    if args.rerank:
        rerank_deps = check_reranking_dependencies()
        if not rerank_deps["reranking_core_available"]:
            print(f"Error: {rerank_deps['error_message']}")
            sys.exit(1)
    
    # Set rerank_top_k default
    if args.rerank_top_k is None:
        args.rerank_top_k = args.top_k
    
    # Set verbose if extra-verbose is enabled
    if args.extra_verbose:
        args.verbose = True
    
    # Validate rich formatting availability
    if args.rich and not RICH_AVAILABLE:
        print("Error: Rich formatting requested but cask_core not available. Install cask_core or remove --rich flag.")
        sys.exit(1)
    
    try:
        # Get store configuration
        store_config = get_store_config(args)
        
        # Initialize QA system
        qa_system = QASystem(
            store_type=args.store,
            embedding_model=args.embedding_model,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            enable_reranking=args.rerank,
            reranker_type=args.rerank_type,
            reranker_model=args.rerank_model,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            **store_config
        )
        
        if args.verbose:
            print(f"*** Using {args.llm_type} LLM: {args.llm_model}")
            print(f"*** Using embedding model: {qa_system.embedding_model}")
            if args.rerank:
                print(f"*** Reranking enabled with {args.rerank_type} model: {args.rerank_model}")
            if not args.use_chunks_only and args.expand_context > 0:
                print(f"*** Context expansion enabled with window ±{args.expand_context} chunks")
        
        # Validate setup
        validation = qa_system.validate_setup()
        if not validation["valid"]:
            for error in validation["errors"]:
                print(f"Error: {error}")
            sys.exit(1)
        
        if validation["warnings"] and args.verbose:
            for warning in validation["warnings"]:
                print(f"Warning: {warning}")
        
        # Query the system
        results = qa_system.query(
            question=query,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k if args.rerank else None,
            verbose=args.verbose,
            extra_verbose=args.extra_verbose,
            show_rerank_results=args.show_rerank_results
        )
        
        # Display results
        system_info = qa_system.get_system_info()
        
        if args.rich:
            # Use rich formatting with VerboseConsole
            console = VerboseConsole(is_verbose=args.verbose)
            _display_results_rich(
                console=console,
                query=results["question"],
                docs_with_scores=results["docs_with_scores"],
                answer=results["answer"],
                system_info=system_info,
                preview_bytes=args.preview_bytes,
                verbose=args.verbose,
                reranking_enabled=results["metadata"]["reranking_enabled"]
            )
        else:
            # Use standard formatting
            display_results(
                query=results["question"],
                docs_with_scores=results["docs_with_scores"],
                answer=results["answer"],
                store_info=system_info["store_info"],
                llm_info=system_info["llm_info"],
                preview_bytes=args.preview_bytes,
                verbose=args.verbose,
                reranking_enabled=results["metadata"]["reranking_enabled"]
            )
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()