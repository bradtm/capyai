"""
Modular LLM implementation supporting OpenAI and HuggingFace models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

# OpenAI imports
try:
    from langchain_openai.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Ollama imports
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model_name: str
    usage: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM wrapper using LangChain.
    """
    
    DEFAULT_MODELS = {
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
        "gpt-4-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini", 
        "gpt-4.1-nano": "gpt-4.1-nano",
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        "o3": "o3",
        "o3-mini": "o3-mini",
        "o4-mini": "o4-mini"
    }
    
    def __init__(self, model_name: str = "gpt-3.5", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: Model name or preset
            api_key: OpenAI API key (if None, uses environment variable)
            **kwargs: Additional arguments for ChatOpenAI
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI dependencies not available. "
                "Install with: pip install langchain-openai"
            )
        
        # Handle preset model names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            **kwargs
        )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            # Filter out unsupported parameters for OpenAI
            openai_kwargs = {}
            
            # Only add temperature if it's not the default (1.0)
            if 'temperature' in kwargs and kwargs['temperature'] != 1.0:
                # Some models only support default temperature
                try:
                    openai_kwargs['temperature'] = kwargs['temperature']
                except Exception:
                    pass  # Skip temperature if not supported
                    
            if 'max_tokens' in kwargs:
                openai_kwargs['max_tokens'] = kwargs['max_tokens']
                
            response = self.llm.invoke(prompt, **openai_kwargs)
            return LLMResponse(
                content=response.content,
                model_name=self.model_name
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def get_model_name(self) -> str:
        return self.model_name


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace LLM wrapper with optimized models.
    """
    
    DEFAULT_MODELS = {
        "gemma-3-1b": "google/gemma-3-1b-it",
        "gemma-2-2b": "google/gemma-2-2b-it",
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "qwen2-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2-3b": "Qwen/Qwen2.5-3B-Instruct"
    }
    
    def __init__(
        self, 
        model_name: str = "gemma-3-1b", 
        device: Optional[str] = None,
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model_name: Model name or preset
            device: Device to run on (auto-detects if None)
            max_length: Maximum generation length
            **kwargs: Additional model arguments
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace dependencies not available. "
                "Install with: pip install transformers torch"
            )
        
        # Handle preset model names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Device selection - CPU only for M2 reliability
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                # MPS has persistent memory issues on M2 - use CPU only
                self.device = "cpu"
        else:
            # Still allow manual override but warn about MPS
            if device == "mps":
                print("Warning: MPS may crash on M2 with transformer models")
            self.device = device
        
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters."""
        model_name_lower = model_name.lower()
        
        # Extract parameter count from model name
        if "1b" in model_name_lower or "1.b" in model_name_lower:
            return 1.0
        elif "2b" in model_name_lower or "2.b" in model_name_lower:
            return 2.0
        elif "3b" in model_name_lower or "3.b" in model_name_lower:
            return 3.0
        elif "4b" in model_name_lower or "4.b" in model_name_lower:
            return 4.0
        elif "7b" in model_name_lower or "7.b" in model_name_lower:
            return 7.0
        elif "9b" in model_name_lower or "9.b" in model_name_lower:
            return 9.0
        elif "14b" in model_name_lower or "14.b" in model_name_lower:
            return 14.0
        else:
            # Default assumption for unknown models
            return 1.0
    
    def _load_model(self, verbose=False):
        """Lazy load the model on first use."""
        if not self._is_loaded:
            try:
                if verbose:
                    print(f"*** Loading HuggingFace LLM: {self.model_name} ***")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    token=True
                )
                
                # Add padding token if missing
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # Load model with optimized settings for Apple Silicon
                if self.device == "mps":
                    # MPS (Apple Silicon) settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        token=True,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                elif self.device == "cuda":
                    # CUDA GPU settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        token=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    # Highly optimized CPU settings for Apple Silicon M2
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,  # Float32 more stable on CPU
                        trust_remote_code=True,
                        token=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager"  # Use eager attention for CPU
                    ).to(self.device)
                    
                    # Enable CPU optimizations
                    if hasattr(torch.backends, 'opt_einsum'):
                        torch.backends.opt_einsum.enabled = True
                
                self._model.eval()
                self._is_loaded = True
                
                if verbose:
                    print(f"*** HuggingFace LLM loaded successfully on {self.device} ***")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model_name}: {e}")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using HuggingFace model."""
        verbose = kwargs.pop('verbose', False)  # Remove verbose from kwargs
        self._load_model(verbose)
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            # Generate response with better parameters for focused answers
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=min(kwargs.get('max_tokens', self.max_length), 150),  # Limit to 150 tokens
                    temperature=0.1,  # Lower temperature for more focused responses
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Higher penalty to reduce repetition
                    no_repeat_ngram_size=4,  # Prevent longer repeating patterns
                    top_p=0.9,  # Use nucleus sampling for better quality
                    top_k=50   # Limit vocabulary to top 50 tokens
                )
            
            # Decode response (skip input tokens)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response - remove excessive repetition and artifacts
            response_text = response_text.strip()
            
            import re
            
            # Stop at first occurrence of code blocks, excessive repetition, or off-topic content
            stop_patterns = [
                r'```',  # Code blocks
                r'def\s+\w+\(',  # Function definitions
                r'print\s*\(',  # Print statements
                r'Question:',  # New questions
                r'Let me know',  # Conversational endings
                r'Do you want',  # Questions back to user
                r'I\'m waiting',  # Conversational artifacts
                r'I apologize',  # Apologies
                r'Thank you',  # Thank yous
                r'You\'re welcome',  # Responses
                r'My intention',  # Meta-commentary
                r'My previous response',  # Self-references
                r'I was attempting',  # Self-explanation
                r'Therefore,',  # Concluding statements that lead to rambling
                r'In conclusion,',  # Conclusion markers
                r'---',  # Horizontal rules/separators
                r'\.{3,}',  # Multiple dots
            ]
            
            for pattern in stop_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    response_text = response_text[:match.start()].strip()
                    break
            
            # Clean up remaining artifacts
            response_text = re.sub(r'\.{3,}', '.', response_text)  # Replace 3+ dots with single dot
            response_text = re.sub(r'\.{2,}$', '.', response_text)  # Clean trailing multiple dots
            response_text = re.sub(r'\n\s*\.\s*\n', '\n', response_text)  # Remove standalone dots on lines
            response_text = re.sub(r'\s+', ' ', response_text)  # Normalize whitespace
            
            # Remove Unicode corruption and non-printable characters
            response_text = re.sub(r'[^\x00-\x7F]+', '', response_text)  # Remove non-ASCII characters
            response_text = re.sub(r'[^\w\s.,!?;:\-\'"()]+', '', response_text)  # Keep only common punctuation
            
            # Remove conversational artifacts more aggressively
            artifact_patterns = [
                r'\b(ame|---).*',  # Random fragments
                r'The provided text.*',  # Meta-commentary about the text
                r'Based solely on.*given text.*',  # Instruction following artifacts
                r'I apologize.*',  # Remove entire apology sections
                r'Thank you.*',  # Remove thank you sections
                r'My intention.*',  # Remove self-explanation
            ]
            
            for pattern in artifact_patterns:
                response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL)
                
            response_text = response_text.strip()
            
            # Ensure response ends properly
            if response_text and not response_text.endswith(('.', '!', '?')):
                # Find last complete sentence
                sentences = re.split(r'[.!?]+', response_text)
                if len(sentences) > 1:
                    response_text = '. '.join(sentences[:-1]) + '.'
            
            return LLMResponse(
                content=response_text.strip(),
                model_name=self.model_name
            )
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {e}")
    
    def get_model_name(self) -> str:
        return self.model_name


class OllamaLLM(BaseLLM):
    """
    Ollama LLM wrapper optimized for Apple Silicon with Metal GPU acceleration.
    """
    
    DEFAULT_MODELS = {
        "gemma3-4b": "gemma3:4b",
        "mistral": "mistral:latest",
        "llama3.2": "llama3.2:latest"
    }
    
    def __init__(
        self,
        model_name: str = "gemma3-4b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Model name or preset
            base_url: Ollama server URL
            **kwargs: Additional model arguments
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama dependencies not available. "
                "Install with: pip install requests"
            )
        
        # Handle preset model names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.kwargs = kwargs
        
        # Check if Ollama server is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama server not responding")
        except requests.exceptions.RequestException:
            raise RuntimeError(
                "Ollama server not running. Start with: ollama serve"
            )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.1),
                    "num_predict": kwargs.get('max_tokens', 512),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 50)
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            content = result.get('response', '').strip()
            
            # Basic cleanup for Ollama responses
            import re
            
            # Remove common artifacts
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Normalize line breaks
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.strip()
            
            return LLMResponse(
                content=content,
                model_name=self.model_name,
                usage=result.get('usage')
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def get_model_name(self) -> str:
        return self.model_name


def create_llm(
    llm_type: str = "openai",
    model_name: str = "gpt-3.5", 
    **kwargs
) -> BaseLLM:
    """
    Factory function to create appropriate LLM based on type.
    
    Args:
        llm_type: Type of LLM ("openai", "huggingface", or "ollama")
        model_name: Model name or preset
        **kwargs: Additional model arguments
        
    Returns:
        BaseLLM instance
        
    Raises:
        ValueError: If llm_type is not supported
    """
    if llm_type.lower() == "openai":
        return OpenAILLM(model_name, **kwargs)
    elif llm_type.lower() == "huggingface":
        return HuggingFaceLLM(model_name, **kwargs)
    elif llm_type.lower() == "ollama":
        return OllamaLLM(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'openai', 'huggingface', or 'ollama'")


def get_available_llm_models() -> Dict[str, Dict[str, str]]:
    """Get available preset models for all LLM types."""
    models = {
        "openai": OpenAILLM.DEFAULT_MODELS.copy(),
        "huggingface": HuggingFaceLLM.DEFAULT_MODELS.copy(),
        "ollama": OllamaLLM.DEFAULT_MODELS.copy()
    }
    return models
