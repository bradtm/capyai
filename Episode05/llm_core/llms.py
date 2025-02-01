"""
Modular LLM implementation supporting OpenAI and HuggingFace models.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
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
        "gpt-4o": "gpt-4o"
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
                except:
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
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct", 
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-4b": "Qwen/Qwen3-4B-Base",
        "t5gemma": "google/t5gemma-b-b-prefixlm",
        "gemma-3-1b": "google/gemma-3-1b-it"
    }
    
    def __init__(
        self, 
        model_name: str = "llama-3.1-8b", 
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
        
        # Device selection with Apple Silicon optimization
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
    
    def _load_model(self, verbose=False):
        """Lazy load the model on first use."""
        if not self._is_loaded:
            try:
                if verbose:
                    print(f"*** Loading HuggingFace LLM: {self.model_name} ***")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # Load model with device-specific settings
                if self.device == "mps":
                    # MPS (Apple Silicon) settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    ).to(self.device)
                elif self.device == "cuda":
                    # CUDA GPU settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    # CPU settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    ).to(self.device)
                
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
            
            # Generate response
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', self.max_length),
                    temperature=kwargs.get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode response (skip input tokens)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return LLMResponse(
                content=response_text.strip(),
                model_name=self.model_name
            )
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {e}")
    
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
        llm_type: Type of LLM ("openai" or "huggingface")
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
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'openai' or 'huggingface'")


def get_available_llm_models() -> Dict[str, Dict[str, str]]:
    """Get available preset models for all LLM types."""
    models = {
        "openai": OpenAILLM.DEFAULT_MODELS.copy(),
        "huggingface": HuggingFaceLLM.DEFAULT_MODELS.copy()
    }
    return models