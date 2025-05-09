"""
Ollama LLM integration for local LLM inference
"""
import os
import sys
import json
import requests
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple

# Add project root to Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings
from src.prompts.prompt_templates import get_rag_prompt, get_reflection_prompt, get_structured_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """Client for Ollama local LLM"""
    
    def __init__(self, 
                 model: str = "mistral:7b-instruct-v0.3-q4_1",
                 base_url: str = "http://localhost:11434",
                 system_prompt: Optional[str] = None,
                 use_reflection: bool = True):
        """Initialize Ollama client
        
        Args:
            model: Model name to use
            base_url: Ollama API base URL
            system_prompt: Optional system prompt to set context
            use_reflection: Whether to use self-reflection for complex questions
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        self.system_prompt = system_prompt or settings.SYSTEM_PROMPT
        self.use_reflection = use_reflection
        
        # Check if Ollama is available
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if not model_names:
                    logger.warning("No models found in Ollama.")
                elif self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available models: {', '.join(model_names)}")
                    logger.info(f"You can pull it using: ollama pull {self.model}")
                else:
                    logger.info(f"Connected to Ollama. Using model: {self.model}")
            else:
                logger.warning(f"Ollama returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {str(e)}")
    
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate text from the LLM
        
        Args:
            prompt: User prompt/question
            context: Optional list of context documents
            
        Returns:
            Generated text response
        """
        try:
            if not context:
                # Simple query without RAG context
                return self._simple_generate(prompt)
            
            # Analyze question to determine best prompting strategy
            question_type = self._analyze_question_type(prompt)
            
            # Select appropriate prompting strategy
            if question_type == "factual":
                return self._generate_with_rag(prompt, context)
            elif question_type == "comparative":
                return self._generate_structured(prompt, context, "comparison")
            elif question_type == "procedural":
                return self._generate_structured(prompt, context, "steps")
            elif question_type == "complex":
                return self._generate_with_reflection(prompt, context)
            else:
                # Default to standard RAG approach
                return self._generate_with_rag(prompt, context)
                
        except requests.exceptions.ConnectionError:
            logger.error("Connection to Ollama failed. Is Ollama running?")
            return (
                "# Connection Error\n\n"
                "Cannot connect to Ollama. Please make sure the Ollama server is running.\n\n"
                "To start Ollama:\n"
                "1. Open a new terminal\n"
                "2. Run the Ollama application\n"
                "3. Try your question again"
            )
        except Exception as e:
            logger.error(f"Error generating from Ollama: {str(e)}")
            return f"# Error\n\n{str(e)}\n\nPlease try again or check the logs for more information."
    
    def _simple_generate(self, prompt: str) -> str:
        """Generate response for a simple prompt without RAG context
        
        Args:
            prompt: User prompt/question
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "system": self.system_prompt,
            "keep_alive": 0
        }
        
        response = requests.post(self.api_url, json=payload)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error: Failed to generate response from Ollama (Status {response.status_code})"
    
    def _generate_with_rag(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using standard RAG prompt
        
        Args:
            question: User question
            context: Retrieved context documents
            
        Returns:
            Generated text response
        """
        rag_prompt = get_rag_prompt(question, context)
        
        payload = {
            "model": self.model,
            "prompt": rag_prompt,
            "stream": False,
            "system": self.system_prompt
        }
        
        response = requests.post(self.api_url, json=payload)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error: Failed to generate response from Ollama (Status {response.status_code})"
    
    def _generate_with_reflection(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Generate response with self-reflection for complex questions
        
        Args:
            question: User question
            context: Retrieved context documents
            
        Returns:
            Generated text response with reflection-based improvements
        """
        # First generate an initial answer
        initial_answer = self._generate_with_rag(question, context)
        
        # For very basic models or short outputs, skip reflection
        if len(initial_answer) < 200 or "mistral:7b" not in self.model:
            return initial_answer
        
        # Generate reflection prompt
        reflection_prompt = get_reflection_prompt(question, initial_answer, context)
        
        payload = {
            "model": self.model,
            "prompt": reflection_prompt,
            "stream": False,
            "system": self.system_prompt,
            "keep_alive": 0
        }
        
        response = requests.post(self.api_url, json=payload)
        
        if response.status_code == 200:
            reflection_output = response.json()["response"]
            
            # Extract just the improved answer part
            if "## Improved Answer" in reflection_output:
                improved_answer = reflection_output.split("## Improved Answer")[-1].strip()
                return improved_answer
            
            # If format isn't followed, return the whole reflection output
            return reflection_output
        else:
            # Fallback to the initial answer if reflection fails
            logger.error(f"Reflection generation failed: {response.status_code}")
            return initial_answer
    
    def _generate_structured(self, question: str, context: List[Dict[str, Any]], format_type: str) -> str:
        """Generate structured response in a specific format
        
        Args:
            question: User question
            context: Retrieved context documents
            format_type: Type of structure to use (table, steps, comparison)
            
        Returns:
            Generated text response in structured format
        """
        structured_prompt = get_structured_prompt(question, context, format_type)
        
        payload = {
            "model": self.model,
            "prompt": structured_prompt,
            "stream": False,
            "system": self.system_prompt,
            "keep_alive": 0
        }
        
        response = requests.post(self.api_url, json=payload)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            # Fallback to standard RAG if structured generation fails
            logger.error(f"Structured generation failed: {response.status_code}")
            return self._generate_with_rag(question, context)
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the question to determine the best prompting strategy
        
        Args:
            question: User question
            
        Returns:
            Question type classification
        """
        question_lower = question.lower()
        
        # Check for comparative questions
        if any(phrase in question_lower for phrase in [
            "compare", "difference between", "versus", "vs", "pros and cons",
            "advantages and disadvantages", "similarities", "differences"
        ]):
            return "comparative"
        
        # Check for procedural questions
        if any(phrase in question_lower for phrase in [
            "how to", "steps", "process", "procedure", "implement", "develop",
            "create", "build", "set up", "configure", "method", "approach"
        ]):
            return "procedural"
        
        # Check for complex questions
        if any(phrase in question_lower for phrase in [
            "why", "explain", "analyze", "evaluate", "assess", "implications", 
            "impact", "effect", "relationship", "critically"
        ]) or len(question) > 100 or question.count(" ") > 15:
            return "complex"
        
        # Default to factual for simpler questions
        return "factual"
    
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [model.get("name") for model in response.json().get("models", [])]
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
