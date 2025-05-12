"""
RagChitChat - Main entry point
Owner: @nxdun
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Now we can import from the project root
from config import settings

from src.document_processor.processor import DocumentProcessorFactory
from src.vector_store.chroma_store import ChromaVectorStore
from src.retriever.haystack_retriever import HaystackRetriever
from src.llm.ollama_client import OllamaLLM
from src.interface.terminal_ui import TerminalUI
from src.prompts.prompt_templates import NO_CONTEXT_MESSAGE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RagChitChat:
    """Main application class"""
    
    def __init__(self,
                 data_dir: str = "data",
                 processed_dir: str = "processed",
                 db_dir: str = "chroma_db",
                 model_name: str = settings.DEFAULT_MODEL) -> None:
        """Initialize the RAG chatbot
        
        Args:
            data_dir: Directory containing lecture notes
            processed_dir: Directory to save processed documents
            db_dir: Directory for ChromaDB storage
            model_name: Ollama model to use
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.db_dir = db_dir
        self.model_name = model_name
        
        # Initialize components
        self.vector_store = ChromaVectorStore(persist_directory=db_dir)
        self.retriever = HaystackRetriever(vector_store=self.vector_store)
        self.llm = OllamaLLM(model=model_name)
        
        # Check if data needs processing
        self._ensure_data_processed()
    
    def _ensure_data_processed(self) -> None:
        """Process lecture notes if needed"""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Check if we need to process files
        data_files = list(Path(self.data_dir).glob("**/*.*"))
        processed_files = list(Path(self.processed_dir).glob("*.txt"))
        
        if not processed_files or len(data_files) > len(processed_files):
            logger.info("Processing lecture notes...")
            self._process_lecture_notes()
        else:
            logger.info("Using existing processed lecture notes")
    
    def _process_lecture_notes(self) -> None:
        """Process lecture notes from data directory"""
        data_files = list(Path(self.data_dir).glob("**/*.*"))
        all_documents = []
        
        # Note: Each file gets processed, chunked, and stored in both the DB and as text
        for file_path in data_files:
            # Skip non-document files
            if file_path.suffix.lower() not in ['.pdf', '.pptx']:
                continue
            
            # Get appropriate processor
            processor = DocumentProcessorFactory.get_processor(str(file_path))
            if processor:
                # Process the document
                chunks = processor.process(str(file_path))
                
                # Save processed chunks
                output_filename = f"{file_path.stem}.txt"
                processor.save_chunks(chunks, self.processed_dir, output_filename)
                
                # Add to combined documents
                all_documents.extend(chunks)
        
        # Add to vector store and retriever
        if all_documents:
            logger.info(f"Adding {len(all_documents)} document chunks to vector store")
            self.vector_store.add_documents(all_documents)
            self.retriever.add_documents(all_documents)
    
    def generate_response(self, question: str, progress_callback: Optional[Callable] = None) -> str:
        """Generate response for a question
        
        Args:
            question: User question
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Generated answer
        """
        logger.info(f"Received question: {question}")
        # Retrieve relevant documents
        context_docs = self.retriever.hybrid_retrieve(question, top_k=settings.TOP_K_RESULTS)
        
        if not context_docs:
            logger.info("No relevant documents found by retriever for the question.")
            if progress_callback:
                progress_callback("retrieval_complete", {"num_docs": 0})
            # Return the specific message indicating no context was found.
            # The UI will display this.
            return NO_CONTEXT_MESSAGE
            
        logger.info(f"Retrieved {len(context_docs)} documents for context.")

        # Notify about retrieval completion if callback provided
        if progress_callback:
            progress_callback("retrieval_complete", {"num_docs": len(context_docs)})
        
        # Generate response with context
        response = self.llm.generate(question, context_docs)
        
        return response
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a new LLM instance with the new model
            new_llm = OllamaLLM(model=model_name)
            
            #unload running models
            
            
            # Check if model is available
            available_models = new_llm.list_models()
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not available in Ollama")
                return False
            
            # If successful, update the class property
            self.llm = new_llm
            self.model_name = model_name
            logger.info(f"Successfully switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system for the UI"""
        # Count documents in the document store
        vector_store_stats = self.vector_store.get_collection_stats()
        
        # Get list of available models from Ollama
        available_models = self.llm.list_models()
        
        # List processed files
        processed_files = [f.name for f in Path(self.processed_dir).glob("*.txt")]
        
        return {
            "current_model": self.model_name,
            "available_models": available_models,
            "document_count": vector_store_stats.get("document_count", 0),
            "processed_files": processed_files[:5] + (["..."] if len(processed_files) > 5 else []),
            "vector_db": "ChromaDB"
        }
    
    def run(self) -> None:
        """Run the chatbot interface"""
        system_info = self.get_system_info()
        ui = TerminalUI(
            generate_fn=self.generate_response,
            system_info=system_info,
            model_switch_fn=self.switch_model
        )
        ui.run()


if __name__ == "__main__":
    try:
        chatbot = RagChitChat()
        chatbot.run()
    except KeyboardInterrupt:
        print("\nExiting RagChitChat...")
    except Exception as e:
        logger.error(f"Error running RagChitChat: {str(e)}", exc_info=True)
