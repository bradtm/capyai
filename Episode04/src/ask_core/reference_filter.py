"""Reference filtering module to identify documents that actually contributed to the final answer."""

import re
from typing import List, Tuple
from langchain_core.documents import Document


class ReferenceFilter:
    """Filters reference documents to show only those that contributed to the final answer."""
    
    def __init__(self):
        self.min_overlap_threshold = 0.08  # Minimum overlap ratio to consider a document contributing  
        self.min_shared_phrases = 2  # Minimum number of shared key phrases
    
    def filter_contributing_references(self, answer: str, docs_with_scores: List[Tuple[Document, float]], 
                                     verbose: bool = False) -> List[Tuple[Document, float]]:
        """
        Filter documents to only those that contributed to the final answer.
        
        Args:
            answer: The final generated answer
            docs_with_scores: All retrieved documents with their scores
            verbose: Whether to show filtering details
            
        Returns:
            Filtered list of documents that contributed to the answer
        """
        if not answer or not docs_with_scores or answer.lower().strip() == "i don't know":
            # If no answer or "I don't know", no documents contributed
            return []
        
        contributing_docs = []
        answer_phrases = self._extract_key_phrases(answer)
        
        if verbose:
            print(f"*** Filtering references: analyzing {len(docs_with_scores)} documents ***")
            print(f"Answer key phrases: {answer_phrases[:5]}...")  # Show first few
        
        for doc, score in docs_with_scores:
            contribution_score = self._calculate_contribution_score(answer, answer_phrases, doc)
            
            if contribution_score >= self.min_overlap_threshold:
                contributing_docs.append((doc, score))
                if verbose:
                    source = doc.metadata.get('source', 'unknown')
                    print(f"  ✓ {source} (contribution: {contribution_score:.3f})")
            elif verbose:
                source = doc.metadata.get('source', 'unknown')
                print(f"  ✗ {source} (contribution: {contribution_score:.3f}) - filtered out")
        
        if verbose:
            print(f"*** Filtered {len(docs_with_scores)} → {len(contributing_docs)} contributing references ***")
        
        return contributing_docs
    
    def _calculate_contribution_score(self, answer: str, answer_phrases: List[str], doc: Document) -> float:
        """
        Calculate how much a document contributed to the final answer.
        
        Args:
            answer: The final answer text
            answer_phrases: Key phrases extracted from the answer
            doc: Document to analyze
            
        Returns:
            Contribution score between 0.0 and 1.0
        """
        doc_content = doc.page_content.lower()
        answer_lower = answer.lower()
        
        # Method 1: Exact phrase matching
        phrase_matches = 0
        for phrase in answer_phrases:
            if len(phrase) > 3 and phrase.lower() in doc_content:
                phrase_matches += 1
        
        phrase_contribution = phrase_matches / len(answer_phrases) if answer_phrases else 0
        
        # Method 2: Word overlap analysis
        answer_words = set(self._extract_significant_words(answer_lower))
        doc_words = set(self._extract_significant_words(doc_content))
        
        if not answer_words:
            word_contribution = 0
        else:
            shared_words = answer_words.intersection(doc_words)
            word_contribution = len(shared_words) / len(answer_words)
        
        # Method 3: Entity and number matching (high weight for specific facts)
        entity_contribution = self._calculate_entity_overlap(answer_lower, doc_content)
        
        # Combine contributions with weights
        total_contribution = (
            0.4 * phrase_contribution +
            0.3 * word_contribution +
            0.3 * entity_contribution
        )
        
        return min(1.0, total_contribution)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text that are likely important for matching."""
        phrases = []
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Extract noun phrases and important segments
            # Look for patterns like "X is Y", "X was Y", "X has Y"
            patterns = [
                r'\b[A-Z][a-z]+ (?:is|was|are|were|has|have) [^.!?]+',  # Factual statements
                r'\b\d{4}\b[^.!?]*',  # Years with context
                r'\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion|thousand|dollars?)[^.!?]*',  # Numbers with units
                r'\b(?:founded|established|created|built) (?:in|by|on) [^.!?]+',  # Founding information
                r'\b(?:located|based|situated) (?:in|at|on) [^.!?]+',  # Location information
                r'\bknown (?:for|as) [^.!?]+',  # Known for statements
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    phrase = match.group().strip()
                    if len(phrase) > 5:
                        phrases.append(phrase)
        
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase_lower)
        
        return unique_phrases[:15]  # Limit to avoid too many phrases
    
    def _extract_significant_words(self, text: str) -> List[str]:
        """Extract significant words (exclude common stop words)."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'us',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
        
        # Extract words, keeping numbers and proper nouns
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        significant_words = []
        
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                not (word.isalpha() and len(word) < 4)):  # Skip very short common words
                significant_words.append(word)
        
        return significant_words
    
    def _calculate_entity_overlap(self, answer: str, doc_content: str) -> float:
        """Calculate overlap of specific entities (numbers, dates, names, etc.)."""
        # Extract specific entities from both texts
        answer_entities = set()
        doc_entities = set()
        
        entity_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion|thousand|dollars?)\b',  # Numbers with units
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Potential names (Title Case)
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b',  # Dates
        ]
        
        for pattern in entity_patterns:
            answer_matches = re.findall(pattern, answer)
            doc_matches = re.findall(pattern, doc_content)
            
            answer_entities.update([m.lower().strip() for m in answer_matches])
            doc_entities.update([m.lower().strip() for m in doc_matches])
        
        if not answer_entities:
            return 0.0
        
        shared_entities = answer_entities.intersection(doc_entities)
        return len(shared_entities) / len(answer_entities)
