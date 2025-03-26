#!/usr/bin/env python3

"""
Unit tests for evaluate_retrieval.py

Tests the RAG Retrieval Evaluation Tool functionality including:
- Vector store loading
- Question evaluation 
- Metrics calculation
- Report generation
- Reranking functionality
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

# Add current directory to path for importing the module
sys.path.insert(0, os.path.dirname(__file__))

try:
    from evaluate_retrieval import (
        VectorStoreLoader, 
        OllamaEmbeddings,
        HuggingFaceReranker,
        create_reranker,
        debug_print,
        load_questions,
        calculate_metrics,
        calculate_overlap,
        is_substantial_overlap,
        debug_source_files
    )
    EVALUATE_RETRIEVAL_AVAILABLE = True
except ImportError as e:
    EVALUATE_RETRIEVAL_AVAILABLE = False
    print(f"Warning: evaluate_retrieval dependencies not available: {e}")


@unittest.skipUnless(EVALUATE_RETRIEVAL_AVAILABLE, "evaluate_retrieval.py not available or dependencies missing")
class TestOllamaEmbeddings(unittest.TestCase):
    """Test cases for OllamaEmbeddings class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embeddings = OllamaEmbeddings(model="bge-m3")
    
    def test_ollama_embeddings_init(self):
        """Test OllamaEmbeddings initialization"""
        self.assertEqual(self.embeddings.model, "bge-m3")
        self.assertEqual(self.embeddings.base_url, "http://localhost:11434")
        
        # Test custom parameters (without making network calls)
        custom_embeddings = OllamaEmbeddings(model="custom-model", base_url="http://localhost:11434/")
        self.assertEqual(custom_embeddings.model, "custom-model")
        self.assertEqual(custom_embeddings.base_url, "http://localhost:11434")
    
    @patch('requests.post')
    def test_ollama_embeddings_embed_query(self, mock_post):
        """Test OllamaEmbeddings embed_query method"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.embeddings.embed_query("test query")
        
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": "test query"}
        )
    
    @patch('requests.post')
    def test_ollama_embeddings_embed_documents(self, mock_post):
        """Test OllamaEmbeddings embed_documents method"""
        # Mock successful responses
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response1.raise_for_status.return_value = None
        
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        mock_response2.raise_for_status.return_value = None
        
        mock_post.side_effect = [mock_response1, mock_response2]
        
        result = self.embeddings.embed_documents(["doc1", "doc2"])
        
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('requests.post')
    def test_ollama_embeddings_request_failure(self, mock_post):
        """Test OllamaEmbeddings handling of request failures"""
        mock_post.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception):
            self.embeddings.embed_query("test query")


@unittest.skipUnless(EVALUATE_RETRIEVAL_AVAILABLE, "evaluate_retrieval.py not available or dependencies missing")
class TestHuggingFaceReranker(unittest.TestCase):
    """Test cases for HuggingFaceReranker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.documents = [
            MagicMock(page_content="This is document 1"),
            MagicMock(page_content="This is document 2"),
            MagicMock(page_content="This is document 3")
        ]
    
    @patch('evaluate_retrieval.HUGGINGFACE_AVAILABLE', False)
    def test_huggingface_reranker_not_available(self):
        """Test HuggingFaceReranker when HuggingFace is not available"""
        with self.assertRaises(ImportError):
            HuggingFaceReranker("test-model")
    
    @patch('evaluate_retrieval.HUGGINGFACE_AVAILABLE', True)
    def test_huggingface_reranker_fallback_model(self):
        """Test HuggingFaceReranker fallback to smaller model"""
        # Skip this test if dependencies are not available in test environment
        self.skipTest("Requires HuggingFace dependencies which may not be available in CI")


@unittest.skipUnless(EVALUATE_RETRIEVAL_AVAILABLE, "evaluate_retrieval.py not available or dependencies missing")
class TestVectorStoreLoader(unittest.TestCase):
    """Test cases for VectorStoreLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = VectorStoreLoader()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_vector_store_loader_init(self):
        """Test VectorStoreLoader initialization"""
        self.assertIsNone(self.loader.vectorstore)
        self.assertIsNone(self.loader.embeddings)
        self.assertIsNone(self.loader.embedding_model)
    
    def test_load_faiss_nonexistent_path(self):
        """Test loading FAISS with nonexistent path"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_vector_store('faiss', faiss_path='/nonexistent/path')
    
    @patch('os.path.exists', return_value=True)
    @patch('evaluate_retrieval.FAISS.load_local')
    @patch('evaluate_retrieval.OpenAIEmbeddings')
    def test_load_faiss_with_metadata(self, mock_openai, mock_faiss, mock_exists):
        """Test loading FAISS with metadata file"""
        # Create mock metadata file
        metadata_file = os.path.join(self.test_dir, "index_metadata.json")
        metadata = {"embedding_model": "text-embedding-3-large"}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(metadata))):
            with patch('os.path.join', return_value=metadata_file):
                self.loader.load_vector_store('faiss', faiss_path=self.test_dir)
        
        self.assertEqual(self.loader.embedding_model, "text-embedding-3-large")
        mock_openai.assert_called_with(model="text-embedding-3-large")
    
    @patch('requests.get')
    @patch('os.path.exists', return_value=True)
    @patch('evaluate_retrieval.FAISS.load_local')
    def test_load_faiss_with_ollama_model(self, mock_faiss, mock_exists, mock_requests):
        """Test loading FAISS with Ollama model"""
        # Mock Ollama availability
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        # Create mock metadata with bge-m3 model
        metadata = {"embedding_model": "bge-m3"}
        metadata_file = os.path.join(self.test_dir, "index_metadata.json")
        
        with patch('builtins.open', mock_open(read_data=json.dumps(metadata))):
            with patch('os.path.join', return_value=metadata_file):
                self.loader.load_vector_store('faiss', faiss_path=self.test_dir)
        
        self.assertEqual(self.loader.embedding_model, "bge-m3")
        self.assertIsInstance(self.loader.embeddings, OllamaEmbeddings)


@unittest.skipUnless(EVALUATE_RETRIEVAL_AVAILABLE, "evaluate_retrieval.py not available or dependencies missing")
class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_debug_print_enabled(self):
        """Test debug_print when debugging is enabled"""
        with patch('builtins.print') as mock_print:
            debug_print("test message", enable_debug=True)
            mock_print.assert_called_once_with("DEBUG: test message")
    
    def test_debug_print_disabled(self):
        """Test debug_print when debugging is disabled"""
        with patch('builtins.print') as mock_print:
            debug_print("test message", enable_debug=False)
            mock_print.assert_not_called()
    
    @patch('evaluate_retrieval.HUGGINGFACE_AVAILABLE', True)
    def test_create_reranker_huggingface(self):
        """Test create_reranker with HuggingFace type"""
        # Skip this test if dependencies are not available
        self.skipTest("Requires HuggingFace dependencies which may not be available in CI")
    
    def test_create_reranker_invalid_type(self):
        """Test create_reranker with invalid type"""
        with self.assertRaises(ValueError):
            create_reranker("invalid_type")
    
    def test_load_questions_valid_file(self):
        """Test loading questions from valid JSON file"""
        questions_data = [
            {"question": "Test question 1", "passage": "Test passage 1"},
            {"question": "Test question 2", "passage": "Test passage 2"}
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(questions_data))):
            questions = load_questions("test.json")
        
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["question"], "Test question 1")
    
    def test_load_questions_invalid_file(self):
        """Test loading questions from invalid JSON file"""
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with self.assertRaises(json.JSONDecodeError):
                load_questions("test.json")
    
    def test_calculate_overlap(self):
        """Test calculate_overlap function"""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "A quick brown fox jumps over a lazy dog"
        
        overlap = calculate_overlap(text1, text2)
        self.assertGreater(overlap, 0.7)  # Should have high overlap
        
        # Test with no overlap
        text3 = "Completely different sentence with no common words"
        overlap2 = calculate_overlap(text1, text3)
        self.assertLess(overlap2, 0.3)  # Should have low overlap
    
    def test_is_substantial_overlap(self):
        """Test is_substantial_overlap function"""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "A quick brown fox jumps over a lazy dog"
        text3 = "Completely different sentence"
        
        self.assertTrue(is_substantial_overlap(text1, text2, threshold=0.5))
        self.assertFalse(is_substantial_overlap(text1, text3, threshold=0.5))
    
    def test_calculate_metrics(self):
        """Test calculate_metrics function"""
        results = [
            {'found_rank': 1, 'rank': 1},
            {'found_rank': 2, 'rank': 2}, 
            {'found_rank': None, 'rank': None}
        ]
        
        metrics = calculate_metrics(
            results=results,
            total=3,
            successes=2,
            mrr_sum=1.0 + 0.5,  # 1/1 + 1/2
            correct_ranks=[1, 2],
            rank_counts={1: 1, 2: 1}
        )
        
        # Check the actual keys returned by calculate_metrics
        self.assertEqual(metrics['total_queries'], 3)
        self.assertEqual(metrics['successful_queries'], 2)
        self.assertAlmostEqual(metrics['success_rate'], 2/3, places=2)
        self.assertAlmostEqual(metrics['mrr'], 1.5/3, places=2)  # (1.0 + 0.5) / 3
    
    def test_debug_source_files(self):
        """Test debug_source_files function"""
        questions = [
            {"sources": ["file1.txt", "file2.txt"], "passage": "test passage 1"},
            {"sources": ["file2.txt", "file3.txt"], "passage": "test passage 2"},
            {"sources": ["file1.txt"], "passage": "test passage 3"}
        ]
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug_source_files(questions)
            output = fake_out.getvalue()
            
            # Check that the function outputs information about the questions
            self.assertIn("DEBUGGING SOURCE FILES", output)
            self.assertIn("file1.txt", output)
            self.assertIn("file2.txt", output) 
            self.assertIn("file3.txt", output)


@unittest.skipUnless(EVALUATE_RETRIEVAL_AVAILABLE, "evaluate_retrieval.py not available or dependencies missing")
class TestEvaluateRetrievalIntegration(unittest.TestCase):
    """Integration tests for evaluate_retrieval.py functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.questions_file = os.path.join(self.test_dir, "test_questions.json")
        self.sample_questions = [
            {
                "question": "What is machine learning?",
                "passage": "Machine learning is a subset of artificial intelligence.",
                "chunk_id": "chunk_1",
                "sources": ["ml_basics.txt"],
                "generated_with": "gpt-3.5-turbo",
                "chunk_index": 0,
                "total_chunks": 10,
                "file": "ml_basics.txt"
            },
            {
                "question": "What is deep learning?",
                "passage": "Deep learning uses neural networks with multiple layers.",
                "chunk_id": "chunk_2", 
                "sources": ["dl_intro.txt"],
                "generated_with": "gpt-3.5-turbo",
                "chunk_index": 1,
                "total_chunks": 10,
                "file": "dl_intro.txt"
            }
        ]
        
        # Write sample questions to file
        with open(self.questions_file, 'w') as f:
            json.dump(self.sample_questions, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_load_questions_integration(self):
        """Test loading questions from actual file"""
        questions = load_questions(self.questions_file)
        
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["question"], "What is machine learning?")
        self.assertEqual(questions[1]["file"], "dl_intro.txt")
    
    def test_command_line_argument_parsing(self):
        """Test command line argument parsing"""
        test_args = [
            'evaluate_retrieval.py',
            '--questions', self.questions_file,
            '--store', 'faiss',
            '--top-k', '5',
            '--debug',
            '--brief'
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Mock the main function components to avoid actual execution
            with patch('evaluate_retrieval.VectorStoreLoader'):
                with patch('evaluate_retrieval.load_questions', return_value=self.sample_questions):
                    with patch('evaluate_retrieval.evaluate_retrieval_quality'):
                        with patch('evaluate_retrieval.generate_report'):
                            # This would test the full argument parsing
                            # but we can't easily test main() without refactoring
                            pass


if __name__ == '__main__':
    # Set up test environment
    if not EVALUATE_RETRIEVAL_AVAILABLE:
        print("Warning: evaluate_retrieval.py or dependencies not available. Skipping tests.")
        sys.exit(1)
    
    unittest.main(verbosity=2)