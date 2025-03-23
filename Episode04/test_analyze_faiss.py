#!/usr/bin/env python3

"""Unit tests for analyze_faiss.py"""

import unittest
import tempfile
import shutil
import os
import sys
import json
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the project root to the path so we can import analyze_faiss
sys.path.insert(0, os.path.dirname(__file__))

try:
    from analyze_faiss import analyze_faiss_index, test_search
    ANALYZE_FAISS_AVAILABLE = True
except ImportError:
    ANALYZE_FAISS_AVAILABLE = False


@unittest.skipUnless(ANALYZE_FAISS_AVAILABLE, "analyze_faiss.py not available or dependencies missing")
class TestAnalyzeFaiss(unittest.TestCase):
    """Test cases for analyze_faiss.py functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix="test_analyze_faiss_")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_analyze_faiss_index_nonexistent_path(self):
        """Test analyze_faiss_index with nonexistent path"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with patch('sys.exit') as mock_exit:
                analyze_faiss_index("/nonexistent/path")
                mock_exit.assert_called_once()
                self.assertIn("does not exist", fake_out.getvalue())
    
    def test_analyze_faiss_index_not_directory(self):
        """Test analyze_faiss_index with file instead of directory"""
        # Create a file instead of directory
        test_file = os.path.join(self.test_dir, "test_file.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_file)
            self.assertIn("is not a directory", fake_out.getvalue())
    
    def test_analyze_faiss_index_missing_required_files(self):
        """Test analyze_faiss_index with missing required FAISS files"""
        # Create directory but don't add required files
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            # Mock the FAISS loading to fail
            with patch('analyze_faiss.FAISS.load_local', side_effect=Exception("Missing files")):
                analyze_faiss_index(test_faiss_dir)
                output = fake_out.getvalue()
                self.assertIn("index.faiss: Missing", output)
                self.assertIn("index.pkl: Missing", output)
                self.assertIn("Error loading FAISS index", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_empty_index(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with empty index"""
        # Mock empty FAISS index
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 0
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory with required files
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir)
            output = fake_out.getvalue()
            self.assertIn("Document store contains 0 chunks", output)
            self.assertIn("Index is empty", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_with_metadata(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with metadata file"""
        # Mock FAISS index with documents
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'source': 'test1.pdf'}
        mock_doc1.page_content = "Test document content 1"
        
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'source': 'test2.pdf'}
        mock_doc2.page_content = "Test document content 2"
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {'doc1': mock_doc1, 'doc2': mock_doc2}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 2
        mock_vectorstore.docstore = mock_docstore
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory with required files
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        # Create metadata file
        metadata = {
            "embedding_model": "text-embedding-3-small",
            "created_date": "2024-01-01",
            "total_documents": 2
        }
        metadata_path = os.path.join(test_faiss_dir, "index_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir, num_entries=2)
            output = fake_out.getvalue()
            
            self.assertIn("Document store contains 2 chunks", output)
            self.assertIn("INDEX METADATA", output)
            self.assertIn("embedding_model: text-embedding-3-small", output)
            self.assertIn("SOURCE FILES (2 unique files)", output)
            self.assertIn("test1.pdf: 1 chunks", output)
            self.assertIn("test2.pdf: 1 chunks", output)
            self.assertIn("FIRST 2 ENTRIES", output)
            self.assertIn("Key: doc1", output)
            self.assertIn("Key: doc2", output)
            self.assertIn("Test document content 1", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_keys_only(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with keys_only=True"""
        # Mock FAISS index with documents
        mock_doc = MagicMock()
        mock_doc.metadata = {'source': 'test.pdf'}
        mock_doc.page_content = "Test content"
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {'doc1': mock_doc, 'doc2': mock_doc}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 2
        mock_vectorstore.docstore = mock_docstore
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir, num_entries=2, keys_only=True)
            output = fake_out.getvalue()
            
            self.assertIn("FIRST 2 KEYS", output)
            self.assertIn("Key 1: doc1", output)
            self.assertIn("Key 2: doc2", output)
            # Should not contain content or metadata sections
            self.assertNotIn("Content:", output)
            self.assertNotIn("Metadata:", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_truncated_content(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with content truncation"""
        # Mock FAISS index with long content
        long_content = "This is a very long document content that should be truncated when displayed. " * 5
        mock_doc = MagicMock()
        mock_doc.metadata = {'source': 'test.pdf'}
        mock_doc.page_content = long_content
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {'doc1': mock_doc}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 1
        mock_vectorstore.docstore = mock_docstore
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir, num_entries=1, truncate_content=True)
            output = fake_out.getvalue()
            
            # Should contain truncated content with "..."
            self.assertIn("Content:", output)
            self.assertIn("...", output)
            # Should not contain the full long content
            self.assertNotIn(long_content, output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_no_truncation(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with no content truncation"""
        long_content = "This is a long document content that should not be truncated. " * 3
        mock_doc = MagicMock()
        mock_doc.metadata = {'source': 'test.pdf'}
        mock_doc.page_content = long_content
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {'doc1': mock_doc}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 1
        mock_vectorstore.docstore = mock_docstore
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir, num_entries=1, truncate_content=False)
            output = fake_out.getvalue()
            
            # Should contain full content without truncation
            self.assertIn(long_content, output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_analyze_faiss_index_no_docstore(self, mock_embeddings, mock_faiss_load):
        """Test analyze_faiss_index with missing docstore"""
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 5
        # No docstore attribute
        delattr(mock_vectorstore, 'docstore')
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_faiss_index(test_faiss_dir)
            output = fake_out.getvalue()
            
            self.assertIn("Document store contains 5 chunks", output)
            self.assertIn("Document store is empty or inaccessible", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_test_search_success(self, mock_embeddings, mock_faiss_load):
        """Test successful search functionality"""
        # Mock document result
        mock_doc = MagicMock()
        mock_doc.metadata = {'source': 'test.pdf'}
        mock_doc.page_content = 'Test document content for search'
        
        # Mock vectorstore search results
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
        mock_faiss_load.return_value = mock_vectorstore
        
        # Create test directory
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = test_search(test_faiss_dir, "test query")
            
            self.assertTrue(result)
            output = fake_out.getvalue()
            self.assertIn("SEARCH TEST", output)
            self.assertIn("Result 1 (score: 0.8000)", output)
            self.assertIn("Source: test.pdf", output)
            self.assertIn("Content: Test document content for search", output)
    
    @patch('analyze_faiss.FAISS.load_local')
    @patch('analyze_faiss.OpenAIEmbeddings')
    def test_test_search_failure(self, mock_embeddings, mock_faiss_load):
        """Test search functionality failure"""
        mock_faiss_load.side_effect = Exception("Failed to load index")
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = test_search(self.test_dir, "test query")
            
            self.assertFalse(result)
            self.assertIn("Search test failed", fake_out.getvalue())
    
    def test_metadata_file_load_error(self):
        """Test handling of corrupted metadata file"""
        # Create test directory with corrupted metadata
        test_faiss_dir = os.path.join(self.test_dir, "faiss_test")
        os.makedirs(test_faiss_dir)
        open(os.path.join(test_faiss_dir, "index.faiss"), 'w').close()
        open(os.path.join(test_faiss_dir, "index.pkl"), 'w').close()
        
        # Create corrupted metadata file
        metadata_path = os.path.join(test_faiss_dir, "index_metadata.json")
        with open(metadata_path, 'w') as f:
            f.write("invalid json {")
        
        with patch('analyze_faiss.FAISS.load_local') as mock_faiss_load:
            mock_vectorstore = MagicMock()
            mock_vectorstore.index.ntotal = 1
            mock_docstore = MagicMock()
            mock_docstore._dict = {}
            mock_vectorstore.docstore = mock_docstore
            mock_faiss_load.return_value = mock_vectorstore
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                analyze_faiss_index(test_faiss_dir)
                output = fake_out.getvalue()
                self.assertIn("Warning: Could not load index metadata", output)


class TestAnalyzeFaissIntegration(unittest.TestCase):
    """Integration test helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix="test_faiss_integration_")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def validate_faiss_index(self, faiss_path, expected_docs=None, expected_sources=None):
        """
        Validate a FAISS index has expected properties.
        
        Args:
            faiss_path: Path to FAISS index directory
            expected_docs: Expected number of documents (None to skip check)
            expected_sources: Expected source files list (None to skip check)
            
        Returns:
            dict: Validation results with keys 'valid', 'doc_count', 'sources', 'errors'
        """
        if not ANALYZE_FAISS_AVAILABLE:
            return {
                'valid': False,
                'doc_count': 0,
                'sources': [],
                'errors': ['analyze_faiss dependencies not available']
            }
        
        results = {
            'valid': True,
            'doc_count': 0,
            'sources': [],
            'errors': []
        }
        
        try:
            # Capture stdout to suppress output during validation
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                from langchain_openai.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import FAISS
                
                # Check if directory and required files exist
                if not os.path.exists(faiss_path):
                    results['valid'] = False
                    results['errors'].append(f"FAISS directory '{faiss_path}' does not exist")
                    return results
                
                required_files = ['index.faiss', 'index.pkl']
                for file in required_files:
                    if not os.path.exists(os.path.join(faiss_path, file)):
                        results['valid'] = False
                        results['errors'].append(f"Missing required file: {file}")
                
                if not results['valid']:
                    return results
                
                # Load the FAISS index
                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                embeddings = OpenAIEmbeddings(model=embedding_model)
                vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                
                results['doc_count'] = vectorstore.index.ntotal
                
                # Check document count
                if expected_docs is not None and results['doc_count'] != expected_docs:
                    results['valid'] = False
                    results['errors'].append(f"Expected {expected_docs} docs, found {results['doc_count']}")
                
                # Get source files if we have documents and docstore
                if (results['doc_count'] > 0 and 
                    hasattr(vectorstore, 'docstore') and 
                    hasattr(vectorstore.docstore, '_dict')):
                    
                    unique_sources = set()
                    for doc in vectorstore.docstore._dict.values():
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            # Extract just the filename for comparison
                            source = os.path.basename(doc.metadata['source'])
                            unique_sources.add(source)
                    
                    results['sources'] = sorted(list(unique_sources))
                    
                    # Check expected sources
                    if expected_sources is not None:
                        expected_basenames = [os.path.basename(src) for src in expected_sources]
                        if set(results['sources']) != set(expected_basenames):
                            results['valid'] = False
                            results['errors'].append(f"Expected sources {expected_basenames}, found {results['sources']}")
                
            finally:
                # Restore stdout
                sys.stdout = original_stdout
                
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation error: {str(e)}")
        
        return results


if __name__ == '__main__':
    # Set up test environment
    if not ANALYZE_FAISS_AVAILABLE:
        print("Warning: analyze_faiss.py or dependencies not available. Skipping tests.")
    
    unittest.main(verbosity=2)