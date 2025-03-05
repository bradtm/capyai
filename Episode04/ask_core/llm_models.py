"""LLM model providers for the ask_core system."""

import os
from typing import Optional, Any
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Optional imports for external LLM providers
try:
    from llm_core import create_llm
    LLM_CORE_AVAILABLE = True
except ImportError:
    LLM_CORE_AVAILABLE = False


class LLMManager:
    """Manager for different LLM providers."""
    
    def __init__(self, llm_type: str = "openai", model_name: str = "gpt-3.5-turbo", 
                 api_key: Optional[str] = None, device: str = "auto"):
        self.llm_type = llm_type
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.device = device
        self.llm = None
        self.model = None  # For backward compatibility
        
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup the appropriate LLM based on type."""
        if self.llm_type == "openai":
            self._setup_openai_llm()
        elif self.llm_type == "huggingface" and LLM_CORE_AVAILABLE:
            self._setup_huggingface_llm()
        elif self.llm_type == "ollama" and LLM_CORE_AVAILABLE:
            self._setup_ollama_llm()
        else:
            # Fallback to OpenAI
            self._setup_openai_llm()
    
    def _setup_openai_llm(self):
        """Setup OpenAI LLM."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI models")
        
        if LLM_CORE_AVAILABLE:
            self.llm = create_llm(
                llm_type="openai",
                model_name=self.model_name,
                api_key=self.api_key
            )
        else:
            # Legacy ChatOpenAI for backward compatibility
            self.model = ChatOpenAI(
                openai_api_key=self.api_key, 
                model=self.model_name
            )
    
    def _setup_huggingface_llm(self):
        """Setup HuggingFace LLM."""
        try:
            self.llm = create_llm(
                llm_type="huggingface", 
                model_name=self.model_name,
                max_length=512,
                device=None if self.device == "auto" else self.device
            )
        except Exception as e:
            print(f"Error initializing HuggingFace LLM: {e}")
            print("Falling back to OpenAI...")
            self._setup_openai_llm()
    
    def _setup_ollama_llm(self):
        """Setup Ollama LLM."""
        try:
            self.llm = create_llm(
                llm_type="ollama",
                model_name=self.model_name
            )
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")
            print("Falling back to OpenAI...")
            self._setup_openai_llm()
    
    def generate_response(self, context: str, question: str) -> str:
        """Generate response using the configured LLM."""
        template = """
You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
- You must ONLY use information from the context provided below
- If the context does not contain information to answer the question, you MUST respond with "I don't know"
- Do NOT use your general knowledge or training data
- Do NOT make assumptions or provide information not explicitly in the context

Context: {context}

Question: {question}

Answer based ONLY on the context above:"""

        if self.llm is not None:
            # Use modular LLM system - call generate method directly
            if self.llm_type in ["huggingface", "ollama"]:
                # Use more direct template for local models
                direct_template = """Based on the following context, answer the question. If the answer is not in the context, say "I don't know".

Context: {context}

Question: {question}

Answer:"""
                formatted_prompt = direct_template.format(context=context, question=question)
                response = self.llm.generate(formatted_prompt)
                # Extract content if it's an LLMResponse object
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # For OpenAI with modular LLM, use direct generation
                formatted_prompt = template.format(context=context, question=question)
                response = self.llm.generate(formatted_prompt)
                # Extract content if it's an LLMResponse object
                return response.content if hasattr(response, 'content') else str(response)
        else:
            # Legacy ChatOpenAI
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
    
    def get_model_info(self) -> dict:
        """Get information about the current LLM."""
        info = {
            "llm_type": self.llm_type,
            "model_name": self.model_name,
            "has_modular_llm": self.llm is not None,
            "has_legacy_model": self.model is not None
        }
        
        if self.llm and hasattr(self.llm, 'get_model_name'):
            info["actual_model_name"] = self.llm.get_model_name()
        
        return info