#!/usr/bin/env python3

"""Unit tests for ask.py CLI interface and options."""

import unittest
import sys
import os

# Add src directory to the path so we can import ask
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import ask


class TestAskArgumentParser(unittest.TestCase):
    """Test argument parsing for ask.py CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ask.create_argument_parser()
    
    def test_basic_query_parsing(self):
        """Test basic query argument parsing."""
        args = self.parser.parse_args(["what is AI"])
        self.assertEqual(args.query, ["what is AI"])
        self.assertEqual(args.store, "faiss")  # default
        self.assertEqual(args.top_k, 4)  # default
        self.assertFalse(args.verbose)
        self.assertFalse(args.rerank)
    
    def test_multi_word_query(self):
        """Test multi-word query parsing."""
        args = self.parser.parse_args(["what", "is", "artificial", "intelligence"])
        self.assertEqual(args.query, ["what", "is", "artificial", "intelligence"])
    
    def test_vector_store_options(self):
        """Test vector store selection options."""
        # FAISS (default)
        args = self.parser.parse_args(["test query"])
        self.assertEqual(args.store, "faiss")
        
        # Pinecone
        args = self.parser.parse_args(["--store", "pinecone", "test query"])
        self.assertEqual(args.store, "pinecone")
        
        # Chroma
        args = self.parser.parse_args(["--store", "chroma", "test query"])
        self.assertEqual(args.store, "chroma")
    
    def test_faiss_options(self):
        """Test FAISS-specific options."""
        args = self.parser.parse_args(["--faiss-path", "/custom/path", "test query"])
        self.assertEqual(args.faiss_path, "/custom/path")
    
    def test_pinecone_options(self):
        """Test Pinecone-specific options."""
        args = self.parser.parse_args([
            "--store", "pinecone", 
            "--pinecone-key", "test-key",
            "--pinecone-index", "test-index",
            "test query"
        ])
        self.assertEqual(args.store, "pinecone")
        self.assertEqual(args.pinecone_key, "test-key")
        self.assertEqual(args.pinecone_index, "test-index")
    
    def test_chroma_options(self):
        """Test Chroma-specific options."""
        args = self.parser.parse_args([
            "--store", "chroma",
            "--chroma-path", "/custom/chroma",
            "--chroma-index", "my-collection",
            "test query"
        ])
        self.assertEqual(args.store, "chroma")
        self.assertEqual(args.chroma_path, "/custom/chroma")
        self.assertEqual(args.chroma_index, "my-collection")
    
    def test_multiple_collections(self):
        """Test multiple collections option."""
        args = self.parser.parse_args([
            "--store", "chroma",
            "--collections", "col1,col2,col3",
            "test query"
        ])
        self.assertEqual(args.collections, "col1,col2,col3")
    
    def test_search_options(self):
        """Test search-related options."""
        args = self.parser.parse_args(["-k", "10", "test query"])
        self.assertEqual(args.top_k, 10)
        
        args = self.parser.parse_args(["--top-k", "5", "test query"])
        self.assertEqual(args.top_k, 5)
    
    def test_reranking_options(self):
        """Test reranking options."""
        # Basic reranking
        args = self.parser.parse_args(["--rerank", "test query"])
        self.assertTrue(args.rerank)
        self.assertEqual(args.rerank_type, "huggingface")  # default
        self.assertEqual(args.rerank_model, "quality")  # default
        
        # Custom reranking
        args = self.parser.parse_args([
            "--rerank",
            "--rerank-type", "qwen3",
            "--rerank-model", "qwen3-8b",
            "--rerank-top-k", "3",
            "test query"
        ])
        self.assertTrue(args.rerank)
        self.assertEqual(args.rerank_type, "qwen3")
        self.assertEqual(args.rerank_model, "qwen3-8b")
        self.assertEqual(args.rerank_top_k, 3)
        
        # MLX reranking for Apple Silicon
        args = self.parser.parse_args([
            "--rerank",
            "--rerank-type", "mlx-qwen3",
            "--rerank-model", "mlx-qwen3-8b",
            "test query"
        ])
        self.assertEqual(args.rerank_type, "mlx-qwen3")
        self.assertEqual(args.rerank_model, "mlx-qwen3-8b")
    
    def test_show_rerank_results(self):
        """Test show reranking results option."""
        args = self.parser.parse_args(["--show-rerank-results", "test query"])
        self.assertTrue(args.show_rerank_results)
    
    def test_llm_options(self):
        """Test LLM model options."""
        # OpenAI (default)
        args = self.parser.parse_args(["test query"])
        self.assertEqual(args.llm_type, "openai")
        self.assertEqual(args.llm_model, "gpt-3.5")
        
        # Custom OpenAI model
        args = self.parser.parse_args([
            "--llm-type", "openai",
            "--llm-model", "gpt-4",
            "test query"
        ])
        self.assertEqual(args.llm_type, "openai")
        self.assertEqual(args.llm_model, "gpt-4")
        
        # HuggingFace
        args = self.parser.parse_args([
            "--llm-type", "huggingface",
            "--llm-model", "gemma-3-1b",
            "--device", "cuda",
            "test query"
        ])
        self.assertEqual(args.llm_type, "huggingface")
        self.assertEqual(args.llm_model, "gemma-3-1b")
        self.assertEqual(args.device, "cuda")
        
        # Ollama
        args = self.parser.parse_args([
            "--llm-type", "ollama",
            "--llm-model", "llama2",
            "test query"
        ])
        self.assertEqual(args.llm_type, "ollama")
        self.assertEqual(args.llm_model, "llama2")
    
    def test_embedding_model_options(self):
        """Test embedding model options."""
        # Default
        args = self.parser.parse_args(["test query"])
        self.assertEqual(args.embedding_model, "text-embedding-3-small")
        
        # Custom embedding models
        for model in ["text-embedding-3-large", "text-embedding-ada-002", "bge-m3", "nomic-embed-text"]:
            args = self.parser.parse_args(["--embedding-model", model, "test query"])
            self.assertEqual(args.embedding_model, model)
    
    def test_device_options(self):
        """Test device selection options."""
        for device in ["auto", "cpu", "mps", "cuda"]:
            args = self.parser.parse_args(["--device", device, "test query"])
            self.assertEqual(args.device, device)
    
    def test_document_handling_options(self):
        """Test document handling options."""
        # Context expansion
        args = self.parser.parse_args(["--expand-context", "5", "test query"])
        self.assertEqual(args.expand_context, 5)
        
        # Use chunks only
        args = self.parser.parse_args(["--use-chunks-only", "test query"])
        self.assertTrue(args.use_chunks_only)
    
    def test_validation_options(self):
        """Test answer validation options."""
        # No validation
        args = self.parser.parse_args(["--no-validation", "test query"])
        self.assertTrue(args.no_validation)
        
        # Validation verbose
        args = self.parser.parse_args(["--validation-verbose", "test query"])
        self.assertTrue(args.validation_verbose)
    
    def test_output_options(self):
        """Test output formatting options."""
        # Preview bytes
        args = self.parser.parse_args(["--preview-bytes", "200", "test query"])
        self.assertEqual(args.preview_bytes, 200)
        
        # Show all references
        args = self.parser.parse_args(["--show-all-references", "test query"])
        self.assertTrue(args.show_all_references)
        
        # Verbose levels
        args = self.parser.parse_args(["-v", "test query"])
        self.assertTrue(args.verbose)
        
        args = self.parser.parse_args(["-vv", "test query"])
        self.assertTrue(args.extra_verbose)
        
        # Rich formatting
        args = self.parser.parse_args(["--rich", "test query"])
        self.assertTrue(args.rich)
        
        # JSON output
        args = self.parser.parse_args(["--json", "test query"])
        self.assertTrue(args.json)
    
    def test_comparison_options(self):
        """Test model and configuration comparison options."""
        # Compare models
        args = self.parser.parse_args([
            "--compare-models", "gpt-4o,gpt-3.5-turbo",
            "test query"
        ])
        self.assertEqual(args.compare_models, "gpt-4o,gpt-3.5-turbo")
        
        # Compare reranking
        args = self.parser.parse_args(["--compare-reranking", "test query"])
        self.assertTrue(args.compare_reranking)
        
        # Compare k values
        args = self.parser.parse_args([
            "--compare-k", "k=3,k=5,k=10",
            "test query"
        ])
        self.assertEqual(args.compare_k, "k=3,k=5,k=10")
    
    def test_short_options(self):
        """Test short option aliases."""
        # -k for --top-k
        args = self.parser.parse_args(["-k", "8", "test query"])
        self.assertEqual(args.top_k, 8)
        
        # -kk for --rerank-top-k
        args = self.parser.parse_args(["--rerank", "-kk", "2", "test query"])
        self.assertEqual(args.rerank_top_k, 2)
        
        # -c for --collections
        args = self.parser.parse_args(["-c", "col1,col2", "test query"])
        self.assertEqual(args.collections, "col1,col2")
        
        # -v for --verbose
        args = self.parser.parse_args(["-v", "test query"])
        self.assertTrue(args.verbose)
        
        # -vv for --extra-verbose
        args = self.parser.parse_args(["-vv", "test query"])
        self.assertTrue(args.extra_verbose)


class TestAskComplexCombinations(unittest.TestCase):
    """Test complex option combinations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ask.create_argument_parser()
    
    def test_full_reranking_setup(self):
        """Test complete reranking configuration."""
        args = self.parser.parse_args([
            "--store", "pinecone",
            "--pinecone-key", "test-key",
            "--pinecone-index", "test-index",
            "--rerank",
            "--rerank-type", "huggingface",
            "--rerank-model", "quality",
            "--rerank-top-k", "3",
            "--show-rerank-results",
            "-k", "10",
            "-v",
            "what is machine learning"
        ])
        
        self.assertEqual(args.store, "pinecone")
        self.assertEqual(args.pinecone_key, "test-key")
        self.assertEqual(args.pinecone_index, "test-index")
        self.assertTrue(args.rerank)
        self.assertEqual(args.rerank_type, "huggingface")
        self.assertEqual(args.rerank_model, "quality")
        self.assertEqual(args.rerank_top_k, 3)
        self.assertTrue(args.show_rerank_results)
        self.assertEqual(args.top_k, 10)
        self.assertTrue(args.verbose)
    
    def test_huggingface_with_device(self):
        """Test HuggingFace LLM with specific device."""
        args = self.parser.parse_args([
            "--llm-type", "huggingface",
            "--llm-model", "gemma-3-1b",
            "--device", "mps",
            "--embedding-model", "bge-m3",
            "explain transformers"
        ])
        
        self.assertEqual(args.llm_type, "huggingface")
        self.assertEqual(args.llm_model, "gemma-3-1b")
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.embedding_model, "bge-m3")
    
    def test_chroma_with_multiple_collections(self):
        """Test Chroma with multiple collections and context expansion."""
        args = self.parser.parse_args([
            "--store", "chroma",
            "--chroma-path", "/custom/chroma",
            "--collections", "docs,papers,notes",
            "--expand-context", "3",
            "--preview-bytes", "150",
            "research question"
        ])
        
        self.assertEqual(args.store, "chroma")
        self.assertEqual(args.chroma_path, "/custom/chroma")
        self.assertEqual(args.collections, "docs,papers,notes")
        self.assertEqual(args.expand_context, 3)
        self.assertEqual(args.preview_bytes, 150)
    
    def test_comparison_modes(self):
        """Test various comparison configurations."""
        # Model comparison
        args = self.parser.parse_args([
            "--compare-models", "gpt-4,gpt-3.5-turbo,gemma-3-1b",
            "--json",
            "compare these models"
        ])
        self.assertEqual(args.compare_models, "gpt-4,gpt-3.5-turbo,gemma-3-1b")
        self.assertTrue(args.json)
        
        # K-value comparison with reranking
        args = self.parser.parse_args([
            "--compare-k", "k=5:kk=3,k=10:kk=5",
            "--rerank",
            "test retrieval"
        ])
        self.assertEqual(args.compare_k, "k=5:kk=3,k=10:kk=5")
        self.assertTrue(args.rerank)
    
    def test_validation_and_output_combo(self):
        """Test validation with output options."""
        args = self.parser.parse_args([
            "--no-validation",
            "--validation-verbose",
            "--show-all-references",
            "--rich",
            "-vv",
            "test validation output"
        ])
        
        self.assertTrue(args.no_validation)
        self.assertTrue(args.validation_verbose)
        self.assertTrue(args.show_all_references)
        self.assertTrue(args.rich)
        self.assertTrue(args.extra_verbose)


class TestAskErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ask.create_argument_parser()
    
    def test_missing_query_fails(self):
        """Test that missing query argument fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])
    
    def test_invalid_store_choice_fails(self):
        """Test that invalid store choice fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--store", "invalid", "test query"])
    
    def test_invalid_rerank_type_fails(self):
        """Test that invalid reranking type fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--rerank-type", "invalid", "test query"])
    
    def test_invalid_llm_type_fails(self):
        """Test that invalid LLM type fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--llm-type", "invalid", "test query"])
    
    def test_invalid_device_fails(self):
        """Test that invalid device fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--device", "invalid", "test query"])
    
    def test_invalid_embedding_model_fails(self):
        """Test that invalid embedding model fails."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--embedding-model", "invalid", "test query"])
    
    def test_negative_numbers_parsing(self):
        """Test parsing of negative numbers (argparse allows them by default)."""
        # Note: argparse allows negative integers by default, so these won't fail
        # This test documents the current behavior
        args = self.parser.parse_args(["-k", "-1", "test query"])
        self.assertEqual(args.top_k, -1)
        
        args = self.parser.parse_args(["--rerank-top-k", "-5", "test query"])
        self.assertEqual(args.rerank_top_k, -5)
        
        args = self.parser.parse_args(["--expand-context", "-2", "test query"])
        self.assertEqual(args.expand_context, -2)
        
        args = self.parser.parse_args(["--preview-bytes", "-10", "test query"])
        self.assertEqual(args.preview_bytes, -10)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
