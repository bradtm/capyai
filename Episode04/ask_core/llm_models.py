"""LLM model providers for the ask_core system."""

import os
from typing import Optional
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
            # Provide user-friendly Ollama error messages
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                available_models = self._get_ollama_models()
                if available_models:
                    models_list = ", ".join(available_models[:5])
                    more_text = f" (and {len(available_models) - 5} more)" if len(available_models) > 5 else ""
                    suggestions = self._suggest_similar_models(available_models)
                    print(f"Error: Ollama model '{self.model_name}' not found locally.")
                    print(f"Available models: {models_list}{more_text}{suggestions}")
                    print(f"To download the model: ollama pull {self.model_name}")
                    print("To list all models: ollama list")
                else:
                    print(f"Error: Ollama model '{self.model_name}' not found locally.")
                    print(f"To download the model: ollama pull {self.model_name}")
                    print("Make sure Ollama is running: ollama serve")
            elif "connection" in error_str or "refused" in error_str:
                print("Error: Cannot connect to Ollama server.")
                print("Make sure Ollama is running: ollama serve")
            else:
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
            try:
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
            except Exception as e:
                # Handle Ollama-specific errors with user-friendly messages
                if self.llm_type == "ollama":
                    error_str = str(e).lower()
                    if "404" in error_str or "not found" in error_str:
                        available_models = self._get_ollama_models()
                        if available_models:
                            models_list = ", ".join(available_models[:5])  # Show first 5 models
                            more_text = f" (and {len(available_models) - 5} more)" if len(available_models) > 5 else ""
                            suggestions = self._suggest_similar_models(available_models)
                            raise RuntimeError(
                                f"Ollama model '{self.model_name}' not found locally.\n"
                                f"Available models: {models_list}{more_text}{suggestions}\n"
                                f"To download a model: ollama pull {self.model_name}\n"
                                f"To list all models: ollama list"
                            )
                        else:
                            raise RuntimeError(
                                f"Ollama model '{self.model_name}' not found locally.\n"
                                f"To download the model: ollama pull {self.model_name}\n"
                                f"To list available models: ollama list\n"
                                f"Make sure Ollama is running: ollama serve"
                            )
                    elif "connection" in error_str or "refused" in error_str:
                        raise RuntimeError(
                            "Cannot connect to Ollama server.\n"
                            "Make sure Ollama is running: ollama serve\n"
                            "Or check if Ollama is running on a different port."
                        )
                # Re-raise other errors as-is
                raise
        else:
            # Legacy ChatOpenAI
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
    
    def _get_ollama_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            # If we can't get models list, return empty list
            pass
        return []
    
    def _suggest_similar_models(self, available_models: list) -> str:
        """Suggest similar model names based on the requested model."""
        if not available_models:
            return ""
        
        model_lower = self.model_name.lower()
        suggestions = []
        
        # Look for models that contain part of the requested name
        for model in available_models:
            model_name_lower = model.lower()
            if any(part in model_name_lower for part in model_lower.split() if len(part) > 2):
                suggestions.append(model)
        
        # If no similar matches, suggest common models
        if not suggestions:
            common_patterns = ["llama", "gemma", "mistral", "phi", "qwen"]
            for pattern in common_patterns:
                matches = [m for m in available_models if pattern in m.lower()]
                suggestions.extend(matches[:2])  # Add up to 2 matches per pattern
        
        if suggestions:
            # Remove duplicates and limit suggestions
            unique_suggestions = list(dict.fromkeys(suggestions))[:3]
            return f"\nSimilar available models: {', '.join(unique_suggestions)}"
        
        return ""
    
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
