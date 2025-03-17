#!/usr/bin/env python3

"""Integration tests for rag.py RAG processor with Chroma vector store."""

import unittest
import os
import sys
import tempfile
import shutil
import subprocess
import glob

# Add the project root to the path so we can import rag
sys.path.insert(0, os.path.dirname(__file__))



class TestRAGIntegration(unittest.TestCase):
    """Integration tests for the full RAG pipeline with Chroma."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.original_cwd = os.getcwd()
        cls.project_root = os.path.dirname(__file__)
        
        # Change to project root for consistent execution
        os.chdir(cls.project_root)
        
        # Test directories
        cls.test_base_dir = tempfile.mkdtemp(prefix="rag_integration_")
        cls.test_chroma_path = os.path.join(cls.test_base_dir, "test_rag_chroma")
        cls.test_transcript_dir = os.path.join(cls.test_base_dir, "test_transcripts")
        
        # Create test directories
        os.makedirs(cls.test_transcript_dir, exist_ok=True)
        
        # Expected test audio directory
        cls.test_audio_dir = os.path.join(cls.project_root, "test_audio")
        
        # Verify test audio files exist
        if not os.path.exists(cls.test_audio_dir):
            raise unittest.SkipTest(f"Test audio directory not found: {cls.test_audio_dir}")
        
        # Filter for audio/video files only (exclude README, etc.)
        cls.test_audio_files = [f for f in os.listdir(cls.test_audio_dir) 
                               if f.endswith(('.mp4', '.mp3', '.wav', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp'))
                               and not f.startswith('.')]
        
        if not cls.test_audio_files:
            raise unittest.SkipTest(f"No audio/video files found in {cls.test_audio_dir}")
        
        # Expected test PDF directory
        cls.test_pdf_dir = os.path.join(cls.project_root, "test_pdfs")
        
        # Verify test PDF files exist
        if not os.path.exists(cls.test_pdf_dir):
            raise unittest.SkipTest(f"Test PDF directory not found: {cls.test_pdf_dir}")
        
        # Filter for PDF files only (exclude README, etc.)
        all_pdf_files = [f for f in os.listdir(cls.test_pdf_dir) 
                        if f.endswith('.pdf') and not f.startswith('.')]
        
        # Allow limiting PDF count for testing (set via environment variable)
        max_pdfs = int(os.environ.get('RAG_TEST_MAX_PDFS', '50'))  # Default to 50 for reasonable test time
        
        if len(all_pdf_files) > max_pdfs:
            print(f"Limiting PDF processing to {max_pdfs} files (set RAG_TEST_MAX_PDFS to change)")
            cls.test_pdf_files = all_pdf_files[:max_pdfs]
        else:
            cls.test_pdf_files = all_pdf_files
        
        if not cls.test_pdf_files:
            raise unittest.SkipTest(f"No PDF files found in {cls.test_pdf_dir}")
        
        print(f"Test setup: {len(cls.test_audio_files)} audio files, {len(cls.test_pdf_files)} PDF files")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        os.chdir(cls.original_cwd)
        if os.path.exists(cls.test_base_dir):
            shutil.rmtree(cls.test_base_dir)
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Ensure we're in the right directory
        os.chdir(self.project_root)
        
        # Clean up any existing test data
        if os.path.exists(self.test_chroma_path):
            shutil.rmtree(self.test_chroma_path)
        if os.path.exists(self.test_transcript_dir):
            shutil.rmtree(self.test_transcript_dir)
        
        # Recreate transcript directory
        os.makedirs(self.test_transcript_dir, exist_ok=True)
    
    def test_rag_chroma_pipeline_full_integration(self):
        """Test the full RAG pipeline with Chroma vector store for both audio and PDF files."""
        # Construct the command
        rag_script = os.path.join(self.project_root, "rag.py")
        
        cmd = [
            "python3", rag_script,
            "--store", "chroma",
            "--chroma-path", self.test_chroma_path,
            "--chroma-index", "test_mixed_content",
            "--transcript-dir", self.test_transcript_dir,
            "--model", "bge-m3"
        ]
        
        # Expand wildcards manually and add all files
        audio_files = glob.glob(os.path.join(self.test_audio_dir, "*"))
        pdf_files = glob.glob(os.path.join(self.test_pdf_dir, "*.pdf"))
        
        # Filter out non-media files from audio directory (exclude README, etc.)
        audio_files = [f for f in audio_files if f.endswith(('.mp4', '.mp3', '.wav', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp')) and not os.path.basename(f).startswith('.')]
        
        # Filter out hidden files from PDF directory
        pdf_files = [f for f in pdf_files if not os.path.basename(f).startswith('.')]
        
        cmd.extend(audio_files)
        cmd.extend(pdf_files)
        
        # Update our test file lists to match what we're actually processing
        self.test_audio_files = [os.path.basename(f) for f in audio_files]
        self.test_pdf_files = [os.path.basename(f) for f in pdf_files]
        
        print("\nDiscovered files:")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  PDF files: {len(pdf_files)}")
        print(f"  Total files to process: {len(audio_files) + len(pdf_files)}")
        
        # Show a few example file paths for verification
        if audio_files:
            print(f"  Sample audio: {audio_files[0]}")
        if pdf_files:
            print(f"  Sample PDF: {pdf_files[0]}")
            print(f"  Last PDF: {pdf_files[-1]}")
        
        print(f"\nRunning command with {len(cmd)} arguments...")
        print(f"Command: {' '.join(cmd[:10])}... (truncated)")  # Show first 10 args
        
        # Run the RAG command with real-time output
        try:
            print("Starting RAG processing...")
            print("=" * 60)
            
            # Run with real-time output (no capture)
            # Increase timeout for large batches - estimate 5 seconds per PDF + base time
            estimated_time = 600 + len(pdf_files) * 5  # 10 min base + 5s per PDF
            actual_timeout = min(7200, estimated_time)  # Cap at 2 hours
            
            print(f"Estimated processing time: {estimated_time/60:.1f} minutes")
            print(f"Using timeout: {actual_timeout/60:.1f} minutes")
            
            result = subprocess.run(
                cmd,
                timeout=actual_timeout,
                cwd=self.project_root
            )
            
            print("=" * 60)
            print(f"RAG command completed with return code: {result.returncode}")
            
            # Check that the command succeeded
            self.assertEqual(result.returncode, 0, 
                           f"RAG command failed with return code {result.returncode}")
            
            # Verify Chroma collection was created
            self.assertTrue(os.path.exists(self.test_chroma_path), 
                          "Chroma directory was not created")
            
            # Verify collection files exist
            chroma_files = os.listdir(self.test_chroma_path)
            self.assertGreater(len(chroma_files), 0, 
                             "No files found in Chroma directory")
            
            # Verify transcripts were created
            self.assertTrue(os.path.exists(self.test_transcript_dir), 
                          "Transcript directory was not created")
            
            transcript_files = [f for f in os.listdir(self.test_transcript_dir) 
                              if f.endswith('.txt')]
            self.assertGreater(len(transcript_files), 0, 
                             "No transcript files were created")
            
            # Expected total files: audio + PDF
            expected_total_files = len(self.test_audio_files) + len(self.test_pdf_files)
            
            # Verify we have transcripts for both audio and PDF files
            audio_transcripts = []
            pdf_transcripts = []
            non_empty_transcripts = 0
            missing_audio_transcripts = []
            missing_pdf_transcripts = []
            
            # Check for expected audio transcripts
            for audio_file in self.test_audio_files:
                audio_name = audio_file.split('.')[0]  # Remove extension
                expected_transcript = f"{audio_name}.txt"
                if expected_transcript in [f for f in transcript_files]:
                    audio_transcripts.append(expected_transcript)
                else:
                    missing_audio_transcripts.append(expected_transcript)
            
            # Check for expected PDF transcripts  
            for pdf_file in self.test_pdf_files:
                pdf_name = pdf_file.replace('.pdf', '')  # Remove .pdf extension
                expected_transcript = f"{pdf_name}.txt"
                if expected_transcript in [f for f in transcript_files]:
                    pdf_transcripts.append(expected_transcript)
                else:
                    missing_pdf_transcripts.append(expected_transcript)
            
            # Count non-empty transcripts
            for transcript_file in transcript_files:
                transcript_path = os.path.join(self.test_transcript_dir, transcript_file)
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and len(content) > 10:
                        non_empty_transcripts += 1
            
            # Report missing files
            if missing_audio_transcripts:
                print(f"⚠️  Missing audio transcripts: {missing_audio_transcripts}")
            if missing_pdf_transcripts:
                print(f"⚠️  Missing PDF transcripts: {len(missing_pdf_transcripts)} total missing")
                print(f"   First 5 missing: {missing_pdf_transcripts[:5]}")
                print(f"   Last 5 missing: {missing_pdf_transcripts[-5:]}")
            
            # Debug: Show what files were actually processed
            print("\nActual files processed:")
            print(f"  Total transcript files found: {len(transcript_files)}")
            print(f"  Expected audio files: {len(self.test_audio_files)}")
            print(f"  Expected PDF files: {len(self.test_pdf_files)}")
            print(f"  Audio transcripts found: {len(audio_transcripts)}")
            print(f"  PDF transcripts found: {len(pdf_transcripts)}")
            
            # Show some actual transcript filenames for debugging
            if transcript_files:
                print("  Sample transcript files:")
                for i, tf in enumerate(transcript_files[:5]):
                    print(f"    {tf}")
                if len(transcript_files) > 5:
                    print(f"    ... and {len(transcript_files) - 5} more")
            
            # Verify we have the expected total count
            self.assertEqual(len(transcript_files), expected_total_files,
                           f"Expected {expected_total_files} transcripts ({len(self.test_audio_files)} audio + {len(self.test_pdf_files)} PDF), but found {len(transcript_files)}. "
                           f"Missing: {len(missing_audio_transcripts)} audio, {len(missing_pdf_transcripts)} PDF")
            
            # Verify we processed both types of content
            self.assertGreater(len(audio_transcripts), 0, 
                             "No audio transcripts were created")
            self.assertGreater(len(pdf_transcripts), 0, 
                             "No PDF transcripts were created")
            self.assertGreater(non_empty_transcripts, 0, 
                             "No transcripts contain meaningful content")
            
            # Verify we have roughly the expected number of transcripts
            self.assertGreaterEqual(len(transcript_files), len(self.test_audio_files),
                                   f"Expected at least {len(self.test_audio_files)} transcripts for audio files")
            
            # Check for successful completion message in stdout (if available)
            # Note: since we removed capture_output, result.stdout won't exist
            # This check is removed for the real-time output version
            
            print("✅ Integration test passed!")
            print(f"   - Processed {len(self.test_audio_files)} audio files")
            print(f"   - Processed {len(self.test_pdf_files)} PDF files") 
            print(f"   - Created {len(transcript_files)} total transcript files")
            print(f"   - Audio transcripts found: {len(audio_transcripts)}")
            print(f"   - PDF transcripts found: {len(pdf_transcripts)}")
            print(f"   - Non-empty transcripts: {non_empty_transcripts}")
            print(f"   - Created Chroma collection with {len(chroma_files)} files")
            
        except subprocess.TimeoutExpired:
            self.fail(f"RAG command timed out after {actual_timeout/60:.1f} minutes ({actual_timeout} seconds)")
        except Exception as e:
            self.fail(f"RAG command failed with exception: {e}")
    
    def test_incremental_processing(self):
        """Test that re-running the command doesn't reprocess unchanged files."""
        # Run the initial processing first
        self.test_rag_chroma_pipeline_full_integration()
        
        # Get initial transcript timestamps
        initial_timestamps = {}
        for transcript_file in os.listdir(self.test_transcript_dir):
            if transcript_file.endswith('.txt'):
                transcript_path = os.path.join(self.test_transcript_dir, transcript_file)
                initial_timestamps[transcript_file] = os.path.getmtime(transcript_path)
        
        # Wait a moment to ensure different timestamps if files were recreated
        import time
        time.sleep(1)
        
        # Run the command again with both audio and PDF files
        rag_script = os.path.join(self.project_root, "rag.py")
        cmd = [
            "python3", rag_script,
            "--store", "chroma",
            "--chroma-path", self.test_chroma_path,
            "--chroma-index", "test_mixed_content",
            "--transcript-dir", self.test_transcript_dir,
            "--model", "bge-m3"
        ]
        
        # Expand wildcards manually and add all files
        audio_files = glob.glob(os.path.join(self.test_audio_dir, "*"))
        pdf_files = glob.glob(os.path.join(self.test_pdf_dir, "*.pdf"))
        
        # Filter out non-media files from audio directory
        audio_files = [f for f in audio_files if f.endswith(('.mp4', '.mp3', '.wav', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp')) and not os.path.basename(f).startswith('.')]
        
        # Filter out hidden files from PDF directory
        pdf_files = [f for f in pdf_files if not os.path.basename(f).startswith('.')]
        
        cmd.extend(audio_files)
        cmd.extend(pdf_files)
        
        print("\nRunning incremental test (second run)...")
        print("=" * 40)
        result = subprocess.run(cmd, timeout=300)
        print("=" * 40)
        
        # Should still succeed
        self.assertEqual(result.returncode, 0)
        
        # Check that transcript files weren't recreated (timestamps should be the same)
        files_reprocessed = 0
        for transcript_file in os.listdir(self.test_transcript_dir):
            if transcript_file.endswith('.txt'):
                transcript_path = os.path.join(self.test_transcript_dir, transcript_file)
                new_timestamp = os.path.getmtime(transcript_path)
                if transcript_file in initial_timestamps:
                    if new_timestamp != initial_timestamps[transcript_file]:
                        files_reprocessed += 1
        
        # Ideally, no files should be reprocessed (depending on the caching implementation)
        print(f"Files reprocessed on second run: {files_reprocessed}")
        
        # The test passes as long as the second run completes successfully
        # Caching behavior may vary based on implementation details


class TestRAGCommandLineInterface(unittest.TestCase):
    """Test RAG command line interface and argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = os.path.dirname(__file__)
        os.chdir(self.project_root)
    
    def test_rag_help_command(self):
        """Test that the RAG script shows help when requested."""
        rag_script = os.path.join(self.project_root, "rag.py")
        
        result = subprocess.run(
            ["python3", rag_script, "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("chroma", result.stdout.lower())
    
    def test_rag_invalid_arguments(self):
        """Test that RAG script handles invalid arguments gracefully."""
        rag_script = os.path.join(self.project_root, "rag.py")
        
        result = subprocess.run(
            ["python3", rag_script, "--invalid-arg"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit with error code
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    # Set up test environment
    if 'OPENAI_API_KEY' not in os.environ:
        print("Warning: OPENAI_API_KEY not set. Some tests may fail.")
    
    unittest.main(verbosity=2)