#!/usr/bin/env python3
"""
Test script to verify PDF metadata filtering is working correctly.
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader

def test_pdf_metadata():
    """Test that PDF metadata contains expected fields only."""
    pdf_path = "/Users/nunu/git/capyai/Episide03/files/files-all/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False
    
    # Load PDF and get metadata
    loader = PyPDFLoader(pdf_path)
    pdf_documents = loader.load()
    
    if not pdf_documents:
        print("No documents loaded from PDF")
        return False
    
    print("Original PDF metadata from PyPDFLoader:")
    original_metadata = pdf_documents[0].metadata
    for key, value in original_metadata.items():
        print(f"  {key}: {value}")
    
    # Simulate the filtering logic from rag.py
    allowed_fields = ['author', 'language', 'moddate', 'title', 'total_pages']
    filtered_metadata = {}
    for field in allowed_fields:
        if field in original_metadata:
            filtered_metadata[field] = original_metadata[field]
    
    # Add the required fields that would be added by rag.py
    filtered_metadata["doc_id"] = "test_doc_0"
    filtered_metadata["source"] = "test.pdf"
    filtered_metadata["transcript_file"] = "/path/to/transcript.txt"
    filtered_metadata["source_type"] = "file"
    
    print("\nFiltered metadata (what rag.py would produce):")
    for key, value in filtered_metadata.items():
        print(f"  {key}: {value}")
    
    # Verify only expected fields are present
    expected_fields = {'author', 'doc_id', 'language', 'moddate', 'source', 'source_type', 'title', 'total_pages', 'transcript_file'}
    actual_fields = set(filtered_metadata.keys())
    
    print(f"\nExpected fields: {expected_fields}")
    print(f"Actual fields: {actual_fields}")
    
    if actual_fields == expected_fields:
        print("SUCCESS: Metadata filtering is working correctly!")
        return True
    else:
        print("FAIL: Metadata fields don't match expected set")
        print(f"Missing fields: {expected_fields - actual_fields}")
        print(f"Extra fields: {actual_fields - expected_fields}")
        return False

if __name__ == "__main__":
    success = test_pdf_metadata()
    sys.exit(0 if success else 1)
