#!/usr/bin/env python3

"""Unit tests for rag.py RAG processor."""

import unittest
import os
import sys
import json
import tempfile
import shutil
import requests
from unittest.mock import patch, MagicMock

# Add src directory to the path so we can import rag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rag


class TestEmbeddingModels(unittest.TestCase):
    """Test embedding model classes and cost estimation."""
    
    def test_openai_ada002_properties(self):
        """Test OpenAI Ada-002 model properties."""
        self.assertEqual(rag.OpenAIAda002.name, "text-embedding-ada-002")
        self.assertEqual(rag.OpenAIAda002.cost_per_1k_tokens, 0.0001)
        self.assertEqual(rag.OpenAIAda002.max_tokens, 8191)
        self.assertEqual(rag.OpenAIAda002.dimension, 1536)
    
    def test_openai_3_small_properties(self):
        """Test OpenAI 3-Small model properties."""
        self.assertEqual(rag.OpenAI3Small.name, "text-embedding-3-small")
        self.assertEqual(rag.OpenAI3Small.cost_per_1k_tokens, 0.00002)
        self.assertEqual(rag.OpenAI3Small.max_tokens, 8191)
        self.assertEqual(rag.OpenAI3Small.dimension, 1536)
    
    def test_openai_3_large_properties(self):
        """Test OpenAI 3-Large model properties."""
        self.assertEqual(rag.OpenAI3Large.name, "text-embedding-3-large")
        self.assertEqual(rag.OpenAI3Large.cost_per_1k_tokens, 0.00013)
        self.assertEqual(rag.OpenAI3Large.max_tokens, 8191)
        self.assertEqual(rag.OpenAI3Large.dimension, 3072)
    
    def test_cost_estimation(self):
        """Test cost estimation calculations."""
        # Test with OpenAI Ada-002
        cost = rag.OpenAIAda002.estimate_cost(1000)
        self.assertAlmostEqual(cost, 0.0001, places=6)
        
        cost = rag.OpenAIAda002.estimate_cost(5000)
        self.assertAlmostEqual(cost, 0.0005, places=6)
        
        # Test with OpenAI 3-Small
        cost = rag.OpenAI3Small.estimate_cost(1000)
        self.assertAlmostEqual(cost, 0.00002, places=8)
        
        # Test with zero tokens
        cost = rag.OpenAI3Small.estimate_cost(0)
        self.assertEqual(cost, 0.0)


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_get_file_info(self):
        """Test getting file modification time and size."""
        info = rag.get_file_info(self.test_file)
        self.assertIn('mtime', info)
        self.assertIn('size', info)
        self.assertEqual(info['size'], 12)  # "test content" is 12 characters
        self.assertIsInstance(info['mtime'], float)
    
    def test_file_has_changed_new_file(self):
        """Test file change detection for new files."""
        processed_files = {}
        self.assertTrue(rag.file_has_changed(self.test_file, processed_files))
    
    def test_file_has_changed_existing_file_unchanged(self):
        """Test file change detection for unchanged files."""
        info = rag.get_file_info(self.test_file)
        processed_files = {"test.txt": info}
        self.assertFalse(rag.file_has_changed(self.test_file, processed_files))
    
    def test_file_has_changed_existing_file_modified(self):
        """Test file change detection for modified files."""
        info = rag.get_file_info(self.test_file)
        processed_files = {"test.txt": info}
        
        # Modify the file
        with open(self.test_file, "a") as f:
            f.write(" more content")
        
        self.assertTrue(rag.file_has_changed(self.test_file, processed_files))
    
    def test_load_processed_files_existing(self):
        """Test loading existing processed files."""
        processed_file_path = os.path.join(self.test_dir, "processed.json")
        test_data = {"file1.txt": {"mtime": 123.456, "size": 100}}
        
        with open(processed_file_path, "w") as f:
            json.dump(test_data, f)
        
        result = rag.load_processed_files(processed_file_path)
        self.assertEqual(result, test_data)
    
    def test_load_processed_files_nonexistent(self):
        """Test loading processed files when file doesn't exist."""
        result = rag.load_processed_files("/nonexistent/path.json")
        self.assertEqual(result, {})
    
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('builtins.print')
    def test_load_processed_files_error(self, mock_print, mock_open, mock_exists):
        """Test loading processed files with read error."""
        result = rag.load_processed_files("test.json")
        self.assertEqual(result, {})
        mock_print.assert_called_once()
    
    def test_save_processed_files(self):
        """Test saving processed files."""
        processed_file_path = os.path.join(self.test_dir, "processed.json")
        test_data = {"file1.txt": {"mtime": 123.456, "size": 100}}
        
        rag.save_processed_files(test_data, processed_file_path)
        
        with open(processed_file_path, "r") as f:
            result = json.load(f)
        
        self.assertEqual(result, test_data)
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('builtins.print')
    def test_save_processed_files_error(self, mock_print, mock_open):
        """Test saving processed files with write error."""
        test_data = {"file1.txt": {"mtime": 123.456, "size": 100}}
        rag.save_processed_files(test_data, "test.json")
        mock_print.assert_called_once()


class TestUrlUtils(unittest.TestCase):
    """Test URL utility functions."""
    
    def test_is_url_valid_http(self):
        """Test URL detection with valid HTTP URLs."""
        self.assertTrue(rag.is_url("http://example.com"))
        self.assertTrue(rag.is_url("https://www.google.com"))
        self.assertTrue(rag.is_url("https://example.com/path/to/page"))
    
    def test_is_url_invalid(self):
        """Test URL detection with invalid strings."""
        self.assertFalse(rag.is_url("not-a-url"))
        self.assertFalse(rag.is_url("ftp://example.com"))  # No scheme/netloc
        self.assertFalse(rag.is_url(""))
        self.assertFalse(rag.is_url("example.com"))  # Missing scheme
    
    def test_get_url_filename_basic(self):
        """Test URL filename generation."""
        filename = rag.get_url_filename("https://example.com/path/to/page")
        self.assertEqual(filename, "example.com_path_to_page")
    
    def test_get_url_filename_with_port(self):
        """Test URL filename generation with port."""
        filename = rag.get_url_filename("https://example.com:8080/path")
        self.assertEqual(filename, "example.com_8080_path")
    
    def test_get_url_filename_special_chars(self):
        """Test URL filename generation with special characters."""
        filename = rag.get_url_filename("https://example.com/path?query=value&other=test")
        # Special characters should be replaced with underscores
        self.assertNotIn("?", filename)
        self.assertNotIn("&", filename)
        self.assertNotIn("=", filename)
    
    def test_get_url_filename_long_url(self):
        """Test URL filename generation with very long URLs."""
        long_path = "a" * 200
        filename = rag.get_url_filename(f"https://example.com/{long_path}")
        self.assertLessEqual(len(filename), 100)
    
    def test_get_url_filename_empty_path(self):
        """Test URL filename generation with empty path."""
        filename = rag.get_url_filename("https://example.com")
        self.assertEqual(filename, "example.com")
    
    def test_get_url_filename_fallback(self):
        """Test URL filename generation fallback."""
        filename = rag.get_url_filename("https://")
        self.assertEqual(filename, "webpage")


class TestWebContentExtraction(unittest.TestCase):
    """Test web content extraction functionality."""
    
    @patch('requests.get')
    def test_extract_web_content_success(self, mock_get):
        """Test successful web content extraction."""
        # Mock HTML response
        mock_response = MagicMock()
        mock_response.content = b"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content of the article.</p>
                    <p>This is another paragraph with useful information.</p>
                </article>
                <script>console.log('should be removed');</script>
                <nav>Navigation menu</nav>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        content, title = rag.extract_web_content("https://example.com")
        
        self.assertIsNotNone(content)
        self.assertEqual(title, "Test Page")
        self.assertIn("Main Article", content)
        self.assertIn("main content", content)
        self.assertNotIn("console.log", content)
        self.assertNotIn("Navigation menu", content)
    
    @patch('requests.get')
    def test_extract_web_content_request_error(self, mock_get):
        """Test web content extraction with request error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        content, title = rag.extract_web_content("https://example.com")
        
        self.assertIsNone(content)
        self.assertIsNone(title)
    
    @patch('requests.get')
    def test_extract_web_content_no_title(self, mock_get):
        """Test web content extraction with no title."""
        mock_response = MagicMock()
        mock_response.content = b"""
        <html>
            <body>
                <article>
                    <p>Content without title</p>
                </article>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        content, title = rag.extract_web_content("https://example.com")
        
        self.assertIsNotNone(content)
        self.assertEqual(title, "Untitled")
    
    @patch('requests.get')
    def test_extract_web_content_fallback_extraction(self, mock_get):
        """Test web content extraction with fallback method."""
        mock_response = MagicMock()
        mock_response.content = b"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <div>
                    <p>This is a long paragraph with substantial content that should be extracted even without semantic tags.</p>
                </div>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        content, title = rag.extract_web_content("https://example.com")
        
        self.assertIsNotNone(content)
        self.assertEqual(title, "Test Page")
        self.assertIn("substantial content", content)


if __name__ == '__main__':
    unittest.main()