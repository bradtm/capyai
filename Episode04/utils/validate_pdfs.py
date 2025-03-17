#!/usr/bin/env python3

"""
Validate and repair PDF collection for RAG testing.

This script checks all downloaded PDFs for corruption and replaces
any problematic files with new downloads from arXiv.
"""

import os
from pathlib import Path
import requests
import time
import re

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    try:
        from pypdf import PdfReader
        PYPDF2_AVAILABLE = True
    except ImportError:
        PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def check_pdf_with_pypdf(pdf_path):
    """Check PDF using PyPDF2/pypdf library."""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        if num_pages == 0:
            return False, "No pages found"
        
        # Try to read first page text
        try:
            first_page = reader.pages[0]
            text = first_page.extract_text()
            if len(text.strip()) < 10:
                return False, "No readable text on first page"
        except Exception as e:
            return False, f"Cannot extract text: {e}"
        
        return True, f"OK ({num_pages} pages)"
    
    except Exception as e:
        return False, f"PyPDF error: {e}"


def check_pdf_with_pymupdf(pdf_path):
    """Check PDF using PyMuPDF library."""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        
        if num_pages == 0:
            return False, "No pages found"
        
        # Try to read first page text
        try:
            first_page = doc[0]
            text = first_page.get_text()
            if len(text.strip()) < 10:
                return False, "No readable text on first page"
        except Exception as e:
            return False, f"Cannot extract text: {e}"
        
        doc.close()
        return True, f"OK ({num_pages} pages)"
    
    except Exception as e:
        return False, f"PyMuPDF error: {e}"


def check_pdf_basic(pdf_path):
    """Basic PDF check using file size and magic number."""
    try:
        file_size = os.path.getsize(pdf_path)
        
        if file_size < 1000:  # Less than 1KB is suspicious
            return False, f"File too small ({file_size} bytes)"
        
        # Check PDF magic number
        with open(pdf_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                return False, "Invalid PDF header"
        
        return True, f"Basic check OK ({file_size} bytes)"
    
    except Exception as e:
        return False, f"Basic check error: {e}"


def validate_pdf(pdf_path):
    """Validate a PDF file using available libraries."""
    print(f"Checking {os.path.basename(pdf_path)}...")
    
    # Try PyMuPDF first (more robust)
    if PYMUPDF_AVAILABLE:
        is_valid, message = check_pdf_with_pymupdf(pdf_path)
        print(f"  PyMuPDF: {message}")
        if is_valid:
            return True, message
    
    # Try PyPDF2 second
    if PYPDF2_AVAILABLE:
        is_valid, message = check_pdf_with_pypdf(pdf_path)
        print(f"  PyPDF: {message}")
        if is_valid:
            return True, message
    
    # Fall back to basic check
    is_valid, message = check_pdf_basic(pdf_path)
    print(f"  Basic: {message}")
    return is_valid, message


def extract_arxiv_id(filename):
    """Extract arXiv ID from filename."""
    # Pattern: 2509.03462v1_title.pdf
    match = re.match(r'^(\d{4}\.\d{5}v\d+)_', filename)
    if match:
        return match.group(1)
    return None


def download_pdf_replacement(arxiv_id, output_path):
    """Download a replacement PDF from arXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    print(f"  Downloading replacement from {pdf_url}...")
    
    try:
        headers = {
            'User-Agent': 'RAG-PDF-Validator/1.0 (Educational Research)'
        }
        
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"  Downloaded {len(response.content)} bytes")
        return True
        
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    """Main function to validate and repair PDFs."""
    print("PDF Validation and Repair Tool")
    print("=" * 40)
    
    pdf_dir = Path("test_pdfs")
    
    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to validate")
    
    if not (PYPDF2_AVAILABLE or PYMUPDF_AVAILABLE):
        print("Warning: Neither PyPDF2 nor PyMuPDF available. Using basic validation only.")
        print("   Install with: pip install PyPDF2 pymupdf")
    
    print("\n" + "=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)
    
    valid_pdfs = []
    corrupt_pdfs = []
    
    for pdf_path in pdf_files:
        is_valid, message = validate_pdf(pdf_path)
        
        if is_valid:
            valid_pdfs.append(pdf_path)
            print(f"  VALID: {pdf_path.name}")
        else:
            corrupt_pdfs.append((pdf_path, message))
            print(f"  CORRUPT: {pdf_path.name}: {message}")
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Valid PDFs: {len(valid_pdfs)}")
    print(f"Corrupt PDFs: {len(corrupt_pdfs)}")
    
    if corrupt_pdfs:
        print("\nCorrupt PDF details:")
        for pdf_path, message in corrupt_pdfs:
            print(f"  - {pdf_path.name}: {message}")
        
        print("\nAttempting to repair corrupt PDFs...")
        
        repaired = 0
        failed_repairs = []
        
        for pdf_path, message in corrupt_pdfs:
            print(f"\nRepairing {pdf_path.name}...")
            
            # Extract arXiv ID
            arxiv_id = extract_arxiv_id(pdf_path.name)
            if not arxiv_id:
                print("  Cannot extract arXiv ID from filename")
                failed_repairs.append(pdf_path.name)
                continue
            
            # Backup original file
            backup_path = pdf_path.with_suffix('.pdf.backup')
            pdf_path.rename(backup_path)
            print(f"  Backed up to {backup_path.name}")
            
            # Download replacement
            if download_pdf_replacement(arxiv_id, pdf_path):
                # Validate the new download
                is_valid, validation_message = validate_pdf(pdf_path)
                if is_valid:
                    print(f"  Repair successful: {validation_message}")
                    backup_path.unlink()  # Remove backup
                    repaired += 1
                else:
                    print(f"  Downloaded file still corrupt: {validation_message}")
                    # Restore backup
                    pdf_path.unlink()
                    backup_path.rename(pdf_path)
                    failed_repairs.append(pdf_path.name)
            else:
                # Restore backup
                backup_path.rename(pdf_path)
                failed_repairs.append(pdf_path.name)
            
            # Rate limiting - be respectful to arXiv
            time.sleep(2)
        
        print("\n" + "=" * 40)
        print("REPAIR SUMMARY")
        print("=" * 40)
        print(f"Successfully repaired: {repaired}")
        print(f"Failed to repair: {len(failed_repairs)}")
        
        if failed_repairs:
            print("\nStill problematic:")
            for filename in failed_repairs:
                print(f"  - {filename}")
    else:
        print("\nAll PDFs are valid!")
    
    # Final count
    final_pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"\nFinal PDF count: {len(final_pdf_files)}")


if __name__ == "__main__":
    main()