#!/usr/bin/env python3

"""Unit tests for analyze_chroma.py"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

# Add Episode03 to the path so we can import analyze_chroma
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Episode03'))

try:
    import analyze_chroma
    from analyze_chroma import analyze_chroma_index, test_search, get_collection_embedding_model, OllamaEmbeddings
    ANALYZE_CHROMA_AVAILABLE = True
except ImportError:
    ANALYZE_CHROMA_AVAILABLE = False


@unittest.skipUnless(ANALYZE_CHROMA_AVAILABLE, "analyze_chroma.py not available or dependencies missing")
class TestAnalyzeChroma(unittest.TestCase):
    """Test cases for analyze_chroma.py functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix="test_analyze_chroma_")
        self.test_collection = "test_collection"
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_ollama_embeddings_init(self):
        """Test OllamaEmbeddings initialization"""
        embeddings = OllamaEmbeddings()
        self.assertEqual(embeddings.model, "bge-m3")
        self.assertEqual(embeddings.base_url, "http://localhost:11434")
        
        # Test custom parameters
        custom_embeddings = OllamaEmbeddings(model="custom-model", base_url="http://custom:8080/")
        self.assertEqual(custom_embeddings.model, "custom-model")
        self.assertEqual(custom_embeddings.base_url, "http://custom:8080")
    
    @patch('requests.post')
    def test_ollama_embeddings_embed_query(self, mock_post):
        """Test OllamaEmbeddings embed_query method"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        embeddings = OllamaEmbeddings()
        result = embeddings.embed_query("test query")
        
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
        
        embeddings = OllamaEmbeddings()
        result = embeddings.embed_documents(["doc1", "doc2"])
        
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(mock_post.call_count, 2)
    
    def test_analyze_chroma_index_nonexistent_path(self):
        """Test analyze_chroma_index with nonexistent path"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with patch('sys.exit') as mock_exit:
                analyze_chroma_index("/nonexistent/path", "test_collection")
                mock_exit.assert_called_once_with(1)
                self.assertIn("does not exist", fake_out.getvalue())
    
    def test_analyze_chroma_index_chroma_not_available(self):
        """Test analyze_chroma_index when Chroma is not available"""
        with patch.object(analyze_chroma, 'CHROMA_AVAILABLE', False):
            with patch('sys.stdout', new=StringIO()) as fake_out:
                with patch('sys.exit') as mock_exit:
                    analyze_chroma_index(self.test_dir, "test_collection")
                    mock_exit.assert_called_once_with(1)
                    self.assertIn("Chroma dependencies not installed", fake_out.getvalue())
    
    @patch('chromadb.PersistentClient')
    def test_analyze_chroma_index_collection_not_found(self, mock_client_class):
        """Test analyze_chroma_index with collection not found"""
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_client_class.return_value = mock_client
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_chroma_index(self.test_dir, "nonexistent_collection")
            self.assertIn("Collection 'nonexistent_collection' not found", fake_out.getvalue())
    
    @patch('chromadb.PersistentClient')
    def test_analyze_chroma_index_empty_collection(self, mock_client_class):
        """Test analyze_chroma_index with empty collection"""
        mock_collection = MagicMock()
        mock_collection.name = self.test_collection
        mock_collection.count.return_value = 0
        mock_collection.metadata = {}
        
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_chroma_index(self.test_dir, self.test_collection)
            output = fake_out.getvalue()
            self.assertIn("Collection is empty", output)
            self.assertIn("Document store contains 0 chunks", output)
    
    @patch('chromadb.PersistentClient')
    def test_analyze_chroma_index_with_documents(self, mock_client_class):
        """Test analyze_chroma_index with documents"""
        mock_collection = MagicMock()
        mock_collection.name = self.test_collection
        mock_collection.count.return_value = 2
        mock_collection.metadata = {"embedding_model": "bge-m3", "created": "2024-01-01"}
        mock_collection.get.return_value = {
            'documents': ['Document 1 content', 'Document 2 content'],
            'metadatas': [{'source': 'file1.pdf'}, {'source': 'file2.pdf'}],
            'ids': ['doc1', 'doc2']
        }
        
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_chroma_index(self.test_dir, self.test_collection, num_entries=2)
            output = fake_out.getvalue()
            
            self.assertIn("Document store contains 2 chunks", output)
            self.assertIn("COLLECTION METADATA", output)
            self.assertIn("embedding_model: bge-m3", output)
            self.assertIn("SOURCE FILES (2 unique files)", output)
            self.assertIn("file1.pdf: 1 chunks", output)
            self.assertIn("file2.pdf: 1 chunks", output)
            self.assertIn("FIRST 2 ENTRIES", output)
            self.assertIn("Key: doc1", output)
            self.assertIn("Key: doc2", output)
    
    @patch('chromadb.PersistentClient')
    def test_analyze_chroma_index_keys_only(self, mock_client_class):
        """Test analyze_chroma_index with keys_only=True"""
        mock_collection = MagicMock()
        mock_collection.name = self.test_collection
        mock_collection.count.return_value = 3
        mock_collection.metadata = {}
        mock_collection.get.return_value = {
            'documents': ['Doc 1', 'Doc 2', 'Doc 3'],
            'metadatas': [{'source': 'file1'}, {'source': 'file2'}, {'source': 'file3'}],
            'ids': ['key1', 'key2', 'key3']
        }
        
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            analyze_chroma_index(self.test_dir, self.test_collection, num_entries=2, keys_only=True)
            output = fake_out.getvalue()
            
            self.assertIn("FIRST 2 KEYS", output)
            self.assertIn("Key 1: key1", output)
            self.assertIn("Key 2: key2", output)
            # Should not contain content or metadata sections
            self.assertNotIn("Content:", output)
            self.assertNotIn("Metadata:", output)
    
    @patch('chromadb.PersistentClient')
    def test_get_collection_embedding_model(self, mock_client_class):
        """Test get_collection_embedding_model function"""
        mock_collection = MagicMock()
        mock_collection.metadata = {"embedding_model": "test-model"}
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        result = get_collection_embedding_model(self.test_dir, self.test_collection)
        self.assertEqual(result, "test-model")
    
    @patch('chromadb.PersistentClient')
    def test_get_collection_embedding_model_exception(self, mock_client_class):
        """Test get_collection_embedding_model with exception"""
        mock_client_class.side_effect = Exception("Connection error")
        
        result = get_collection_embedding_model(self.test_dir, self.test_collection)
        self.assertIsNone(result)
    
    @patch('analyze_chroma.OllamaEmbeddings')
    @patch('chromadb.PersistentClient')
    @patch('analyze_chroma.Chroma')
    def test_search_success(self, mock_chroma_class, mock_client_class, mock_ollama_class):
        """Test successful search functionality"""
        # Mock document result
        mock_doc = MagicMock()
        mock_doc.metadata = {'source': 'test.pdf'}
        mock_doc.page_content = 'Test document content'
        
        # Mock Chroma vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        mock_chroma_class.return_value = mock_vectorstore
        
        # Mock client and collection
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_ollama_class.return_value = mock_embeddings
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with patch('analyze_chroma.get_collection_embedding_model', return_value="bge-m3"):
                result = test_search(self.test_dir, self.test_collection, "test query")
                
                self.assertTrue(result)
                output = fake_out.getvalue()
                self.assertIn("SEARCH TEST", output)
                self.assertIn("Result 1", output)
                self.assertIn("Source: test.pdf", output)
                self.assertIn("Content: Test document content", output)
    
    @patch('chromadb.PersistentClient')
    def test_search_failure(self, mock_client_class):
        """Test search functionality failure"""
        mock_client_class.side_effect = Exception("Connection failed")
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = test_search(self.test_dir, self.test_collection, "test query")
            
            self.assertFalse(result)
            self.assertIn("Search test failed", fake_out.getvalue())


class TestAnalyzeChromaIntegration(unittest.TestCase):
    """Integration test helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix="test_chroma_integration_")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def validate_chroma_collection(self, chroma_path, collection_name, expected_docs=None, expected_sources=None):
        """
        Validate a Chroma collection has expected properties.
        
        Args:
            chroma_path: Path to Chroma database
            collection_name: Name of collection to validate
            expected_docs: Expected number of documents (None to skip check)
            expected_sources: Expected source files list (None to skip check)
            
        Returns:
            dict: Validation results with keys 'valid', 'doc_count', 'sources', 'errors'
        """
        if not ANALYZE_CHROMA_AVAILABLE:
            return {
                'valid': False,
                'doc_count': 0,
                'sources': [],
                'errors': ['analyze_chroma dependencies not available']
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
                import chromadb
                client = chromadb.PersistentClient(path=chroma_path)
                
                # Check if collection exists
                collections = client.list_collections()
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    results['valid'] = False
                    results['errors'].append(f"Collection '{collection_name}' not found")
                    return results
                
                collection = client.get_collection(name=collection_name)
                results['doc_count'] = collection.count()
                
                # Check document count
                if expected_docs is not None and results['doc_count'] != expected_docs:
                    results['valid'] = False
                    results['errors'].append(f"Expected {expected_docs} docs, found {results['doc_count']}")
                
                # Get source files if we have documents
                if results['doc_count'] > 0:
                    collection_results = collection.get(
                        include=['metadatas'],
                        limit=min(results['doc_count'], 10000)
                    )
                    
                    metadatas = collection_results.get('metadatas', [])
                    unique_sources = set()
                    
                    for metadata in metadatas:
                        if metadata and 'source' in metadata:
                            # Extract just the filename for comparison
                            source = os.path.basename(metadata['source'])
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
    if not ANALYZE_CHROMA_AVAILABLE:
        print("Warning: analyze_chroma.py or dependencies not available. Skipping tests.")
    
    unittest.main(verbosity=2)