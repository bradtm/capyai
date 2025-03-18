#!/usr/bin/env python3

"""
Download AI research papers from arXiv for RAG testing.

This script downloads 250 publicly available AI research papers from arXiv
that can be legally used for testing and research purposes.
"""

import os
import sys
import time
import requests
import feedparser
from urllib.parse import urljoin
from pathlib import Path


def download_pdf(url, filename, max_retries=3):
    """Download a PDF from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading {filename}... (attempt {attempt + 1})")
            
            headers = {
                'User-Agent': 'RAG-Test-Downloader/1.0 (Educational Research)'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"  Downloaded {filename} ({len(response.content)} bytes)")
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            
    return False


def get_arxiv_ai_papers(max_papers=250):
    """Get AI paper metadata from arXiv API."""
    print("Fetching AI papers from arXiv...")
    
    # arXiv API query for AI papers
    # cs.AI = Artificial Intelligence, cs.LG = Machine Learning, cs.CL = Computation and Language
    base_url = "http://export.arxiv.org/api/query"
    
    categories = ['cs.AI', 'cs.LG', 'cs.CL']
    papers = []
    
    for category in categories:
        if len(papers) >= max_papers:
            break
            
        print(f"  Fetching from category: {category}")
        
        # Query parameters
        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': min(250, max_papers - len(papers)),  # Get up to 250 per category
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        query_url = f"{base_url}?search_query={params['search_query']}&start={params['start']}&max_results={params['max_results']}&sortBy={params['sortBy']}&sortOrder={params['sortOrder']}"
        
        try:
            print(f"    Querying: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()
            
            # Parse the Atom feed
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                if len(papers) >= max_papers:
                    break
                    
                # Extract paper info
                paper_id = entry.id.split('/')[-1]  # Get ID from URL
                title = entry.title.replace('\n', ' ').replace('\r', ' ').strip()
                authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else ['Unknown']
                
                # Create safe filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title[:250]  # Limit length
                filename = f"{paper_id}_{safe_title}.pdf"
                
                # PDF download URL
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'authors': authors,
                    'filename': filename,
                    'pdf_url': pdf_url,
                    'category': category
                })
            
            print(f"    Found {len(feed.entries)} papers in {category}")
            
        except Exception as e:
            print(f"    Error fetching from {category}: {e}")
        
        # Rate limiting - arXiv requests 3 second delays
        time.sleep(3)
    
    return papers[:max_papers]


def main():
    """Main function to download AI papers."""
    print("arXiv AI Papers Downloader")
    print("=" * 50)
    
    # Create test_pdfs directory
    pdf_dir = Path("test_pdfs")
    pdf_dir.mkdir(exist_ok=True)
    
    print(f"Download directory: {pdf_dir.absolute()}")
    
    # Get paper list
    papers = get_arxiv_ai_papers(250)
    
    if not papers:
        print("No papers found. Exiting.")
        return
    
    print(f"\nFound {len(papers)} papers to download")
    print("=" * 250)
    
    # Download papers
    successful_downloads = 0
    failed_downloads = []
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}/{len(papers)}] {paper['title'][:60]}...")
        print(f"  Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"  Category: {paper['category']}")
        print(f"  arXiv ID: {paper['id']}")
        
        pdf_path = pdf_dir / paper['filename']
        
        # Skip if already exists
        if pdf_path.exists():
            print(f"  Already exists: {pdf_path}")
            successful_downloads += 1
            continue
        
        # Download PDF
        if download_pdf(paper['pdf_url'], pdf_path):
            successful_downloads += 1
        else:
            failed_downloads.append(paper)
        
        # Rate limiting - be respectful to arXiv
        if i % 10 == 0:  # Every 10 downloads, longer pause
            print("  Longer pause to be respectful to arXiv servers...")
            time.sleep(5)
        else:
            time.sleep(1)  # Short pause between downloads
    
    # Summary
    print("\n" + "=" * 250)
    print("DOWNLOAD SUMMARY")
    print("=" * 250)
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print("\nFailed papers:")
        for paper in failed_downloads:
            print(f"  - {paper['id']}: {paper['title'][:250]}...")
    
    print(f"\nPDFs saved to: {pdf_dir.absolute()}")
    print("\nDownload complete!")


if __name__ == "__main__":
    main()
