"""Answer validation module for detecting hallucinations and ensuring answer quality."""

import re
from typing import Tuple, Dict, Any, List, Optional
from langchain_core.documents import Document
from .llm_models import LLMManager


class AnswerValidator:
    """Validates generated answers against context to detect hallucinations."""
    
    def __init__(self, validator_llm: Optional[LLMManager] = None):
        """
        Initialize the answer validator.
        
        Args:
            validator_llm: Optional separate LLM for validation (uses fast/cheap model)
        """
        # Use a fast, cheap model for validation
        self.validator_llm = validator_llm or LLMManager("openai", "gpt-3.5-turbo")
        
        # Confidence threshold for accepting answers
        self.confidence_threshold = 0.6
        
        # Context coverage threshold
        self.coverage_threshold = 0.7
        
        # Uncertainty phrases that indicate low confidence answers
        self.uncertainty_phrases = [
            "i think", "maybe", "possibly", "i'm not sure", "it seems", 
            "appears to be", "might be", "could be", "probably", 
            "i believe", "it's likely", "presumably", "supposedly",
            "i don't know", "i cannot", "i can't", "unclear", "uncertain"
        ]
    
    def validate_answer(self, answer: str, context: str, query: str, 
                       verbose: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive answer validation using multiple methods.
        
        Args:
            answer: Generated answer to validate
            context: Source context used to generate the answer
            query: Original user query
            verbose: Whether to print validation details
            
        Returns:
            Tuple of (is_valid, final_answer, validation_details)
        """
        validation_details = {
            "confidence_check": {"passed": True, "score": 1.0, "reason": ""},
            "llm_validation": {"passed": True, "confidence": 1.0, "issues": []},
            "coverage_check": {"passed": True, "coverage": 1.0, "unsupported_claims": []},
            "final_decision": {"passed": True, "reason": "All checks passed"}
        }
        
        if verbose:
            print(f"*** Validating answer against context ***")
        
        # Step 1: Quick confidence check
        confidence_score = self._check_answer_confidence(answer)
        validation_details["confidence_check"] = {
            "passed": confidence_score >= self.confidence_threshold,
            "score": confidence_score,
            "reason": f"Confidence score: {confidence_score:.2f}"
        }
        
        if verbose:
            print(f"Confidence check: {confidence_score:.2f} (threshold: {self.confidence_threshold})")
        
        if confidence_score < self.confidence_threshold:
            if verbose:
                print("❌ Failed confidence check - answer shows uncertainty")
            validation_details["final_decision"] = {
                "passed": False, 
                "reason": f"Low confidence: {confidence_score:.2f} < {self.confidence_threshold}"
            }
            return False, "I don't know", validation_details
        
        # Step 2: LLM-based validation for hallucinations
        try:
            is_valid_llm, llm_confidence, issues = self._validate_with_llm(answer, context, query)
            validation_details["llm_validation"] = {
                "passed": is_valid_llm,
                "confidence": llm_confidence,
                "issues": issues
            }
            
            if verbose:
                print(f"LLM validation: {'✓' if is_valid_llm else '❌'} (confidence: {llm_confidence:.2f})")
                if issues:
                    print(f"Issues found: {', '.join(issues)}")
            
            if not is_valid_llm:
                validation_details["final_decision"] = {
                    "passed": False,
                    "reason": f"LLM validation failed: {', '.join(issues)}"
                }
                return False, "I don't know", validation_details
                
        except Exception as e:
            if verbose:
                print(f"Warning: LLM validation failed with error: {e}")
            # Continue with other validation methods if LLM validation fails
            validation_details["llm_validation"] = {
                "passed": True,
                "confidence": 0.5,
                "issues": [f"Validation error: {str(e)}"]
            }
        
        # Step 3: Context coverage analysis (lightweight fallback)
        coverage_result = self._analyze_context_coverage(answer, context)
        validation_details["coverage_check"] = coverage_result
        
        if verbose:
            print(f"Context coverage: {coverage_result['coverage']:.2f} (threshold: {self.coverage_threshold})")
            if coverage_result['unsupported_claims']:
                print(f"Unsupported claims: {coverage_result['unsupported_claims']}")
        
        if not coverage_result["passed"]:
            validation_details["final_decision"] = {
                "passed": False,
                "reason": f"Insufficient context coverage: {coverage_result['coverage']:.2f} < {self.coverage_threshold}"
            }
            return False, "I don't know", validation_details
        
        if verbose:
            print("Answer passed all validation checks")
        
        return True, answer, validation_details
    
    def _check_answer_confidence(self, answer: str) -> float:
        """
        Analyze answer text for uncertainty markers.
        
        Args:
            answer: Answer text to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        answer_lower = answer.lower()
        
        # Start with full confidence
        confidence = 1.0
        
        # Check for uncertainty phrases
        uncertainty_count = 0
        for phrase in self.uncertainty_phrases:
            if phrase in answer_lower:
                uncertainty_count += 1
                confidence -= 0.15  # Reduce confidence for each uncertainty phrase
        
        # Check for hedging patterns
        hedging_patterns = [
            r'\bmay be\b', r'\bcould be\b', r'\bmight be\b',
            r'\bseems? to\b', r'\bappears? to\b', r'\blooks? like\b',
            r'\bperhaps\b', r'\bmaybe\b'
        ]
        
        for pattern in hedging_patterns:
            if re.search(pattern, answer_lower):
                confidence -= 0.1
        
        # Bonus for definitive statements
        definitive_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bwill\b', r'\bdoes\b', r'\bdid\b'
        ]
        
        definitive_count = sum(1 for pattern in definitive_patterns 
                              if len(re.findall(pattern, answer_lower)) > 0)
        
        if definitive_count > 2:  # Multiple definitive statements
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    def _validate_with_llm(self, answer: str, context: str, query: str) -> Tuple[bool, float, List[str]]:
        """
        Use LLM to validate answer against context for hallucinations.
        
        Args:
            answer: Generated answer to validate
            context: Source context
            query: Original query
            
        Returns:
            Tuple of (is_valid, confidence, list_of_issues)
        """
        validation_prompt = f"""You are an expert fact-checker. Evaluate if the given answer contains hallucinations or unsupported claims.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER TO EVALUATE:
{answer}

Your task:
1. Check for CONTRADICTIONS: Claims that directly oppose the context
2. Check for UNSUPPORTED CLAIMS: Information not found in the context
3. Evaluate if the answer stays grounded in the provided context

Respond in this exact format:
Valid: [Yes/No]
Confidence: [0.0-1.0]
Issues: [List specific problems separated by semicolons, or "None"]

Example:
Valid: No
Confidence: 0.3
Issues: Claims the company was founded in 1995 but context says 1992; Mentions headquarters in Chicago but context says New York"""

        try:
            response = self.validator_llm.generate_response("", validation_prompt)
            
            # Parse the response
            lines = response.strip().split('\n')
            
            is_valid = True
            confidence = 0.8  # Default
            issues = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Valid:'):
                    is_valid = 'yes' in line.lower()
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        confidence = 0.5
                elif line.startswith('Issues:'):
                    issues_text = line.split(':', 1)[1].strip()
                    if issues_text.lower() != 'none':
                        issues = [issue.strip() for issue in issues_text.split(';') if issue.strip()]
            
            return is_valid, confidence, issues
            
        except Exception as e:
            # If LLM validation fails, be conservative but don't block
            return True, 0.5, [f"Validation error: {str(e)}"]
    
    def _analyze_context_coverage(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Analyze how much of the answer is supported by the context using keyword analysis.
        
        Args:
            answer: Generated answer
            context: Source context
            
        Returns:
            Dictionary with coverage analysis results
        """
        # Extract key phrases from answer (simple approach)
        answer_phrases = self._extract_key_phrases(answer)
        context_lower = context.lower()
        
        if not answer_phrases:
            return {
                "passed": True,
                "coverage": 1.0,
                "unsupported_claims": [],
                "method": "no_claims_to_verify"
            }
        
        # Check which phrases are supported by context
        supported_phrases = []
        unsupported_phrases = []
        
        for phrase in answer_phrases:
            phrase_lower = phrase.lower()
            # Simple containment check (could be enhanced with embeddings)
            if any(word in context_lower for word in phrase_lower.split() if len(word) > 3):
                supported_phrases.append(phrase)
            else:
                unsupported_phrases.append(phrase)
        
        coverage_ratio = len(supported_phrases) / len(answer_phrases) if answer_phrases else 1.0
        
        return {
            "passed": coverage_ratio >= self.coverage_threshold,
            "coverage": coverage_ratio,
            "unsupported_claims": unsupported_phrases,
            "total_claims": len(answer_phrases),
            "supported_claims": len(supported_phrases),
            "method": "keyword_analysis"
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text that should be verifiable against context.
        
        Args:
            text: Text to extract phrases from
            
        Returns:
            List of key phrases
        """
        # Simple approach: extract noun phrases and factual statements
        phrases = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Look for factual patterns
            factual_patterns = [
                r'\b\d{4}\b',  # Years
                r'\b\d+\s*(percent|%|dollars?|years?|months?|days?)\b',  # Numbers with units
                r'\bfounded in\b.*',  # Company founding
                r'\blocated in\b.*',  # Locations  
                r'\bknown (for|as)\b.*',  # Known for statements
                r'\b(is|are|was|were)\s+.*\b',  # Definitional statements
            ]
            
            for pattern in factual_patterns:
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
        
        return unique_phrases[:10]  # Limit to first 10 phrases to avoid too much analysis
