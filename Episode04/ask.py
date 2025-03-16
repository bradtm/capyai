#!/usr/bin/env python3

"""Modular RAG system with FAISS, Pinecone, and Chroma vector stores.

This is a refactored version of ask.py that uses the ask_core modular architecture
for better organization and maintainability.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Fix HuggingFace tokenizers warning when using reranking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ask_core.qa_system import QASystem
from ask_core.utils import (
    display_results, 
    validate_environment_variables, 
    get_store_config
)
from ask_core.rerankers import check_reranking_dependencies

# Optional rich formatting imports
try:
    from cask_core.utils import VerboseConsole
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
    parser.add_argument("--collections", "-c", help="Multiple collection names (comma-separated, overrides --chroma-index)")
    
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
    
    # Answer validation arguments
    parser.add_argument("--no-validation", action="store_true", help="Disable answer validation against context (validation enabled by default)")
    parser.add_argument("--validation-verbose", action="store_true", help="Show detailed validation process (requires --verbose)")
    
    # Output arguments
    parser.add_argument("--preview-bytes", type=int, default=0, help="Number of bytes to show from each document in references (default: 0, no content)")
    parser.add_argument("--show-all-references", action="store_true", help="Show all retrieved references instead of only those that contributed to the answer")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-vv", "--extra-verbose", action="store_true", help="Extra verbose output showing detailed processing steps (implies -v)")
    parser.add_argument("--rich", action="store_true", help="Enable rich console formatting similar to cask.py")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    # Comparison arguments
    parser.add_argument("--compare-models", help="Compare answers from multiple LLM models (comma-separated, e.g., 'gpt-4o,gpt-3.5-turbo')")
    parser.add_argument("--compare-reranking", action="store_true", help="Compare results with and without reranking")
    parser.add_argument("--compare-k", help="Compare different retrieval parameters (e.g., 'k=3,k=5,k=10' or 'k=5:kk=3,k=5:kk=5')")
    
    return parser


def _display_results_rich(console, query, docs_with_scores, answer, system_info, 
                         preview_bytes=0, verbose=False, reranking_enabled=False, show_all_references=False):
    """Display results using rich formatting similar to cask.py."""
    from rich.panel import Panel
    
    # Display the answer in a panel
    answer_panel = Panel(
        answer,
        title=f"Answer from {system_info['store_info']['store_type']}",
        title_align="left",
        border_style="green"
    )
    console.print(answer_panel)
    
    # Filter references to only contributing ones unless --show-all-references is used
    display_docs = docs_with_scores
    if not show_all_references and docs_with_scores:
        from ask_core.reference_filter import ReferenceFilter
        filter = ReferenceFilter()
        display_docs = filter.filter_contributing_references(answer, docs_with_scores, verbose=verbose)
    
    # Display references
    if display_docs:
        if show_all_references:
            console.print(f"\n[bold]References:[/bold] {len(display_docs)} documents (all retrieved)")
        else:
            filtered_count = len(docs_with_scores) - len(display_docs)
            if filtered_count > 0:
                console.print(f"\n[bold]References:[/bold] {len(display_docs)} documents (filtered out {filtered_count} non-contributing)")
            else:
                console.print(f"\n[bold]References:[/bold] {len(display_docs)} documents")
        
        for i, (doc, score) in enumerate(display_docs, 1):
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
        
        # Build reranker info with device information
        reranker_text = ""
        if reranking_enabled and "reranker_info" in system_info:
            reranker_info = system_info["reranker_info"]
            reranker_type = reranker_info.get('reranker_type', 'N/A')
            device = reranker_info.get('device', 'N/A')
            reranker_text = f"Reranker: {reranker_type} (device: {device})\n"
        
        system_panel = Panel(
            f"Store Type: {store_info['store_type']}\n"
            f"Store Path: {store_info.get('store_path', 'N/A')}\n"
            f"Documents: {store_info.get('document_count', store_info.get('total_documents', 'N/A'))}\n"
            f"Embedding Model: {embedding_model}\n"
            f"LLM Type: {llm_info['llm_type']}\n"
            f"LLM Model: {llm_info.get('model_name', llm_info.get('model', llm_info.get('llm_model', 'N/A')))}\n" +
            reranker_text,
            title="System Info",
            title_align="left",
            border_style="blue"
        )
        console.print(system_panel)


def _display_results_json(query, docs_with_scores, answer, system_info, 
                         preview_bytes=0, verbose=False, reranking_enabled=False, show_all_references=False):
    """Display results in JSON format similar to cask.py."""
    
    # Filter references to only contributing ones unless --show-all-references is used
    display_docs = docs_with_scores
    if not show_all_references and docs_with_scores:
        from ask_core.reference_filter import ReferenceFilter
        filter = ReferenceFilter()
        display_docs = filter.filter_contributing_references(answer, docs_with_scores, verbose=verbose)
    
    # Build references list
    references = []
    for i, (doc, score) in enumerate(display_docs):
        doc_id = doc.metadata.get('doc_id', 'unknown')
        source = doc.metadata.get('source', 'unknown')
        chunk_index = doc.metadata.get('chunk_index')
        total_chunks = doc.metadata.get('total_chunks')
        
        ref = {
            "rank": i + 1,
            "doc_id": doc_id,
            "source": source,
            "similarity_score": float(score)
        }
        
        if chunk_index is not None and total_chunks is not None:
            ref["chunk_index"] = chunk_index + 1  # 1-based for display
            ref["total_chunks"] = total_chunks
        
        if reranking_enabled and hasattr(doc, 'rerank_score'):
            ref["rerank_score"] = float(doc.rerank_score)
        
        if preview_bytes > 0:
            content_preview = doc.page_content[:preview_bytes]
            if len(doc.page_content) > preview_bytes:
                content_preview += "..."
            ref["content_preview"] = content_preview
            
        references.append(ref)
    
    # Build the complete result
    result = {
        "query": query,
        "answer": answer,
        "references": references,
        "metadata": {
            "num_documents": len(display_docs),
            "total_retrieved": len(docs_with_scores),
            "filtered_count": len(docs_with_scores) - len(display_docs),
            "references_filtered": not show_all_references,
            "reranking_enabled": reranking_enabled
        }
    }
    
    # Add system info if verbose
    if verbose:
        result["system_info"] = {
            "store_type": system_info["store_info"]["store_type"],
            "store_path": system_info["store_info"].get("store_path", "N/A"),
            "total_documents": system_info["store_info"].get("document_count", system_info["store_info"].get("total_documents", "N/A")),
            "embedding_model": system_info["embedding_model"],
            "llm_type": system_info["llm_info"]["llm_type"],
            "llm_model": system_info["llm_info"].get("model_name", system_info["llm_info"].get("model", system_info["llm_info"].get("llm_model", "N/A")))
        }
        
        if reranking_enabled and "reranker_info" in system_info:
            result["system_info"]["reranker_type"] = system_info["reranker_info"].get("reranker_type", "N/A")
            result["system_info"]["reranker_device"] = system_info["reranker_info"].get("device", "N/A")
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _run_comparison(args, query):
    """Run comparison queries and display results."""
    if args.compare_models:
        return _compare_models(args, query)
    elif args.compare_reranking:
        return _compare_reranking(args, query)
    elif args.compare_k:
        return _compare_k_values(args, query)


def _compare_models(args, query):
    """Compare answers from different LLM models."""
    models = [m.strip() for m in args.compare_models.split(',')]
    
    print(f"Comparing models: {', '.join(models)}")
    print(f"Query: {query}\n")
    
    results = []
    
    for model in models:
        print(f"Running query with {model}...")
        
        # Get store configuration
        store_config = get_store_config(args)
        
        # Create QA system with this model
        qa_system = QASystem(
            store_type=args.store,
            embedding_model=args.embedding_model,
            llm_type=args.llm_type,
            llm_model=model,
            enable_reranking=args.rerank,
            reranker_type=args.rerank_type,
            reranker_model=args.rerank_model,
            reranker_device=None if args.device == "auto" else args.device,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            enable_answer_validation=not args.no_validation,
            **store_config
        )
        
        # Run query
        result = qa_system.query(
            question=query,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k if args.rerank else None,
            verbose=False,  # Suppress verbose output during comparison
            extra_verbose=False,
            show_rerank_results=False
        )
        
        results.append({
            'model': model,
            'result': result,
            'system_info': qa_system.get_system_info()
        })
    
    _display_comparison_results(results, "Model Comparison", args.rich)
    return True


def _compare_reranking(args, query):
    """Compare results with and without reranking."""
    print("Comparing with and without reranking")
    print(f"Query: {query}\n")
    
    results = []
    configs = [
        ('Without Reranking', False),
        ('With Reranking', True)
    ]
    
    for config_name, enable_rerank in configs:
        print(f"Running query {config_name.lower()}...")
        
        store_config = get_store_config(args)
        
        qa_system = QASystem(
            store_type=args.store,
            embedding_model=args.embedding_model,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            enable_reranking=enable_rerank,
            reranker_type=args.rerank_type,
            reranker_model=args.rerank_model,
            reranker_device=None if args.device == "auto" else args.device,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            enable_answer_validation=not args.no_validation,
            **store_config
        )
        
        result = qa_system.query(
            question=query,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k if enable_rerank else None,
            verbose=False,
            extra_verbose=False,
            show_rerank_results=False
        )
        
        results.append({
            'model': config_name,
            'result': result,
            'system_info': qa_system.get_system_info()
        })
    
    _display_comparison_results(results, "Reranking Comparison", args.rich)
    return True


def _compare_k_values(args, query):
    """Compare different k and kk parameter values."""
    # Parse k values like "k=3,k=5,k=10" or "k=5:kk=3,k=5:kk=5"
    k_configs = []
    for config in args.compare_k.split(','):
        config = config.strip()
        if ':' in config:
            # Format: k=5:kk=3
            parts = config.split(':')
            k_part = parts[0].strip()
            kk_part = parts[1].strip()
            k_val = int(k_part.split('=')[1])
            kk_val = int(kk_part.split('=')[1])
            k_configs.append((f"k={k_val}, kk={kk_val}", k_val, kk_val))
        else:
            # Format: k=5
            k_val = int(config.split('=')[1])
            k_configs.append((f"k={k_val}", k_val, k_val))
    
    print(f"Comparing retrieval parameters: {', '.join([c[0] for c in k_configs])}")
    print(f"Query: {query}\n")
    
    results = []
    
    for config_name, k_val, kk_val in k_configs:
        print(f"Running query with {config_name}...")
        
        store_config = get_store_config(args)
        
        qa_system = QASystem(
            store_type=args.store,
            embedding_model=args.embedding_model,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            enable_reranking=args.rerank and kk_val != k_val,  # Enable reranking if kk specified
            reranker_type=args.rerank_type,
            reranker_model=args.rerank_model,
            reranker_device=None if args.device == "auto" else args.device,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            enable_answer_validation=not args.no_validation,
            **store_config
        )
        
        result = qa_system.query(
            question=query,
            top_k=k_val,
            rerank_top_k=kk_val if args.rerank and kk_val != k_val else None,
            verbose=False,
            extra_verbose=False,
            show_rerank_results=False
        )
        
        results.append({
            'model': config_name,
            'result': result,
            'system_info': qa_system.get_system_info()
        })
    
    _display_comparison_results(results, "Retrieval Parameter Comparison", args.rich)
    return True


def _display_comparison_results(results, title, use_rich=False):
    """Display comparison results."""
    if use_rich and RICH_AVAILABLE:
        _display_comparison_results_rich(results, title)
    else:
        _display_comparison_results_standard(results, title)


def _display_comparison_results_standard(results, title):
    """Display comparison results in standard format."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}\n")
    
    for i, result_data in enumerate(results, 1):
        model_name = result_data['model']
        result = result_data['result']
        
        print(f"{i}. {model_name}")
        print("-" * 40)
        print(f"Answer: {result['answer']}")
        print(f"Documents used: {len(result['docs_with_scores'])}")
        
        if result['docs_with_scores']:
            avg_score = sum(score for _, score in result['docs_with_scores']) / len(result['docs_with_scores'])
            print(f"Average similarity: {avg_score:.4f}")
        
        print()


def _display_comparison_results_rich(results, title):
    """Display comparison results with rich formatting."""
    from rich.panel import Panel
    from rich.console import Console
    
    console = Console()
    
    # Create main comparison panel
    console.print(f"\n[bold blue]{title}[/bold blue]\n")
    
    for i, result_data in enumerate(results, 1):
        model_name = result_data['model']
        result = result_data['result']
        
        # Create answer panel for each model
        answer_panel = Panel(
            result['answer'],
            title=f"{i}. {model_name}",
            title_align="left",
            border_style="green" if i == 1 else "yellow"
        )
        console.print(answer_panel)
        
        # Show brief stats
        stats = f"Documents: {len(result['docs_with_scores'])}"
        if result['docs_with_scores']:
            avg_score = sum(score for _, score in result['docs_with_scores']) / len(result['docs_with_scores'])
            stats += f" | Avg similarity: {avg_score:.4f}"
        
        console.print(f"[dim]{stats}[/dim]\n")


def _run_multi_collection_query(args, query):
    """Run query across multiple collections."""
    if args.verbose:
        print(f"*** Querying multiple collections: {', '.join(args.collection_names)} ***")
    
    # Validate that all collections exist and are compatible
    if args.store != "chroma":
        print("Error: Multiple collections only supported with Chroma store.")
        sys.exit(1)
    
    # Collect results from each collection
    collection_results = []
    all_docs_with_scores = []
    
    for collection_name in args.collection_names:
        if args.verbose:
            print(f"*** Processing collection: {collection_name} ***")
        
        # Create temporary args for this collection
        temp_args = args.__class__(**vars(args))
        temp_args.chroma_index = collection_name
        temp_args.collection_names = None  # Prevent recursion
        
        try:
            # Get store configuration for this collection
            store_config = get_store_config(temp_args)
            
            # Create QA system for this collection
            qa_system = QASystem(
                store_type=args.store,
                embedding_model=args.embedding_model,
                llm_type=args.llm_type,
                llm_model=args.llm_model,
                enable_reranking=False,  # Disable per-collection reranking
                reranker_type=args.rerank_type,
                reranker_model=args.rerank_model,
                reranker_device=None if args.device == "auto" else args.device,
                expand_context=args.expand_context,
                use_chunks_only=args.use_chunks_only,
                enable_answer_validation=not args.no_validation,
                **store_config
            )
            
            # Perform similarity search only (no reranking yet)
            docs_with_scores = qa_system.vector_store.similarity_search_with_score(query, k=args.top_k)
            
            # Add collection metadata to each document
            for doc, score in docs_with_scores:
                doc.metadata['collection_name'] = collection_name
            
            collection_results.append({
                'collection': collection_name,
                'docs_with_scores': docs_with_scores,
                'qa_system': qa_system
            })
            
            all_docs_with_scores.extend(docs_with_scores)
            
        except Exception as e:
            print(f"Warning: Could not query collection '{collection_name}': {e}")
            continue
    
    if not all_docs_with_scores:
        print("Error: No documents retrieved from any collection.")
        sys.exit(1)
    
    # Apply cross-collection reranking if enabled
    final_docs_with_scores = all_docs_with_scores
    
    if args.rerank:
        if args.verbose:
            print(f"*** Cross-collection reranking: {len(all_docs_with_scores)} → {args.rerank_top_k} docs ***")
        
        # Create a new QA system with reranking enabled to get the reranker
        first_collection = args.collection_names[0]
        temp_args = args.__class__(**vars(args))
        temp_args.chroma_index = first_collection
        temp_args.collection_names = None
        
        store_config = get_store_config(temp_args)
        
        reranker_qa_system = QASystem(
            store_type=args.store,
            embedding_model=args.embedding_model,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            enable_reranking=True,  # Enable reranking
            reranker_type=args.rerank_type,
            reranker_model=args.rerank_model,
            reranker_device=None if args.device == "auto" else args.device,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            enable_answer_validation=not args.no_validation,
            **store_config
        )
        
        reranker = reranker_qa_system.reranker
        
        if reranker:
            # Show detailed reranking if requested
            if args.show_rerank_results:
                _display_multi_collection_rerank_details(collection_results, all_docs_with_scores, query, args.rerank_top_k, reranker)
            
            # Extract documents for reranking
            docs = [doc for doc, score in all_docs_with_scores]
            
            # Rerank documents
            final_docs_with_scores = reranker.rerank_documents(query, all_docs_with_scores, args.rerank_top_k, False)
        else:
            # No reranker available, just sort by similarity and take top k
            sorted_docs = sorted(all_docs_with_scores, key=lambda x: x[1], reverse=True)
            final_docs_with_scores = sorted_docs[:args.rerank_top_k]
    
    # Generate final answer using the first QA system's LLM
    first_qa_system = collection_results[0]['qa_system']
    from ask_core.utils import format_documents
    
    docs = [doc for doc, score in final_docs_with_scores]
    context = format_documents(docs)
    answer = first_qa_system.llm_manager.generate_response(context, query)
    
    # Display results
    results = {
        "question": query,
        "answer": answer,
        "docs_with_scores": final_docs_with_scores,
        "metadata": {
            "reranking_enabled": args.rerank,
            "collections": args.collection_names,
            "total_collections_queried": len(collection_results)
        }
    }
    
    system_info = first_qa_system.get_system_info()
    system_info["store_info"]["collections"] = args.collection_names
    
    # Use appropriate display format
    if args.json:
        _display_results_json(
            query=results["question"],
            docs_with_scores=results["docs_with_scores"],
            answer=results["answer"],
            system_info=system_info,
            preview_bytes=args.preview_bytes,
            verbose=args.verbose,
            reranking_enabled=results["metadata"]["reranking_enabled"],
            show_all_references=args.show_all_references
        )
    elif args.rich:
        console = VerboseConsole(is_verbose=args.verbose)
        _display_results_rich(
            console=console,
            query=results["question"],
            docs_with_scores=results["docs_with_scores"],
            answer=results["answer"],
            system_info=system_info,
            preview_bytes=args.preview_bytes,
            verbose=args.verbose,
            reranking_enabled=results["metadata"]["reranking_enabled"],
            show_all_references=args.show_all_references
        )
    else:
        display_results(
            query=results["question"],
            docs_with_scores=results["docs_with_scores"],
            answer=results["answer"],
            store_info=system_info["store_info"],
            llm_info=system_info["llm_info"],
            preview_bytes=args.preview_bytes,
            verbose=args.verbose,
            reranking_enabled=results["metadata"]["reranking_enabled"],
            show_all_references=args.show_all_references
        )


def _display_multi_collection_rerank_details(collection_results, all_docs_with_scores, query, rerank_top_k, reranker):
    """Display detailed reranking information for multiple collections."""
    print("\n*** Initial Retrieval Results ***\n")
    
    # Show per-collection results
    for result in collection_results:
        collection_name = result['collection']
        docs_with_scores = result['docs_with_scores']
        
        print(f"Collection: {collection_name} ({len(docs_with_scores)} documents)")
        for i, (doc, score) in enumerate(docs_with_scores[:10], 1):  # Show top 10
            source = doc.metadata.get('source', 'unknown')
            chunk_info = ""
            if doc.metadata.get('chunk_index') is not None and doc.metadata.get('total_chunks') is not None:
                chunk_info = f" (chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']})"
            
            # Truncate source name for display
            source_display = source if len(source) <= 50 else source[:47] + "..."
            print(f"  {i:2d}. {score:.4f} - {source_display}{chunk_info}")
        
        if len(docs_with_scores) > 10:
            print(f"  ... and {len(docs_with_scores) - 10} more")
        print()
    
    # Show top combined results before reranking
    print(f"*** Cross-Collection Reranking Results ({len(all_docs_with_scores)} → {rerank_top_k}) ***\n")
    
    # Sort by similarity score
    sorted_docs = sorted(all_docs_with_scores, key=lambda x: x[1], reverse=True)
    
    print("Pre-rerank (top 10 by similarity):")
    for i, (doc, score) in enumerate(sorted_docs[:10], 1):
        collection = doc.metadata.get('collection_name', 'unknown')
        source = doc.metadata.get('source', 'unknown')
        chunk_info = ""
        if doc.metadata.get('chunk_index') is not None:
            chunk_info = f" (chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata.get('total_chunks', '?')})"
        
        source_display = source if len(source) <= 40 else source[:37] + "..."
        print(f"  {i:2d}. {score:.4f} - [{collection}] {source_display}{chunk_info}")
    
    # Apply reranking
    reranked_results = reranker.rerank_documents(query, all_docs_with_scores, rerank_top_k, False)
    reranked_docs = [doc for doc, score in reranked_results]
    
    # Show post-rerank results
    print(f"\nPost-rerank (final {len(reranked_docs)} sent to LLM):")
    
    # Count distribution
    collection_count = {}
    for i, doc in enumerate(reranked_docs, 1):
        collection = doc.metadata.get('collection_name', 'unknown')
        collection_count[collection] = collection_count.get(collection, 0) + 1
        
        source = doc.metadata.get('source', 'unknown')
        source_display = source if len(source) <= 40 else source[:37] + "..."
        
        chunk_info = ""
        if doc.metadata.get('chunk_index') is not None:
            chunk_info = f" (chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata.get('total_chunks', '?')})"
        
        # Find original position
        original_pos = None
        for j, (orig_doc, _) in enumerate(sorted_docs):
            if orig_doc.metadata.get('doc_id') == doc.metadata.get('doc_id'):
                original_pos = j + 1
                break
        
        position_info = f" [was #{original_pos}]" if original_pos else ""
        
        # Get rerank score from the reranked results
        rerank_score = 'N/A'
        if i <= len(reranked_results):
            _, score = reranked_results[i-1]  # Get score from tuple
            rerank_score = score
        
        if isinstance(rerank_score, float):
            print(f"  {i:2d}. {rerank_score:.4f} ↗️ [{collection}] {source_display}{chunk_info}{position_info}")
        else:
            print(f"  {i:2d}. N/A ↗️ [{collection}] {source_display}{chunk_info}{position_info}")
    
    # Show distribution
    print("\nCollection Distribution in Final Results:")
    for collection, count in collection_count.items():
        percentage = (count / len(reranked_docs)) * 100
        print(f"- {collection}: {count} documents ({percentage:.0f}%)")
    
    print(f"\n*** Generating answer from {len(reranked_docs)} cross-collection documents ***")


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
    
    # Process multiple collections
    if args.collections:
        args.collection_names = [name.strip() for name in args.collections.split(',')]
        # Override chroma-index if collections specified
        if args.store == "chroma":
            args.chroma_index = args.collection_names[0]  # For compatibility, use first collection
    else:
        args.collection_names = None
    
    # Set verbose if extra-verbose is enabled
    if args.extra_verbose:
        args.verbose = True
    
    # Validate rich formatting availability
    if args.rich and not RICH_AVAILABLE:
        print("Error: Rich formatting requested but cask_core not available. Install cask_core or remove --rich flag.")
        sys.exit(1)
    
    # Validate output mode conflicts
    if args.json and args.rich:
        print("Error: Cannot use both --json and --rich flags simultaneously. Choose one output format.")
        sys.exit(1)
    
    # Check if any comparison mode is enabled
    comparison_mode = bool(args.compare_models or args.compare_reranking or args.compare_k)
    
    # Validate comparison mode conflicts
    if comparison_mode and args.json:
        print("Error: JSON output not supported with comparison modes. Use --rich or standard output.")
        sys.exit(1)
    
    # Store original verbose settings for JSON output
    original_verbose = args.verbose
    original_extra_verbose = args.extra_verbose
    
    # JSON mode disables verbose printing during processing
    if args.json:
        args.verbose = False
        args.extra_verbose = False
    
    try:
        # Handle comparison modes
        if comparison_mode:
            _run_comparison(args, query)
            return
        
        # Handle multiple collections mode
        if args.collection_names and len(args.collection_names) > 1:
            _run_multi_collection_query(args, query)
            return
        
        # Regular single-query mode
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
            reranker_device=None if args.device == "auto" else args.device,
            expand_context=args.expand_context,
            use_chunks_only=args.use_chunks_only,
            enable_answer_validation=not args.no_validation,
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
        
        if args.json:
            # Use JSON formatting
            _display_results_json(
                query=results["question"],
                docs_with_scores=results["docs_with_scores"],
                answer=results["answer"],
                system_info=system_info,
                preview_bytes=args.preview_bytes,
                verbose=original_verbose,  # Use original verbose setting
                reranking_enabled=results["metadata"]["reranking_enabled"],
                show_all_references=args.show_all_references
            )
        elif args.rich:
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
                reranking_enabled=results["metadata"]["reranking_enabled"],
                show_all_references=args.show_all_references
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
                reranking_enabled=results["metadata"]["reranking_enabled"],
                show_all_references=args.show_all_references
            )
        
    except Exception as e:
        if args.json:
            # Output error in JSON format
            error_result = {
                "error": str(e),
                "query": query if 'query' in locals() else args.query[0] if args.query else "unknown"
            }
            if original_verbose:
                import traceback
                error_result["traceback"] = traceback.format_exc()
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()