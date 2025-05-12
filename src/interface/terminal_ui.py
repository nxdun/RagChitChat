"""
Rich-powered terminal UI for the chatbot
"""
import time
import os
from typing import List, Dict, Any, Optional, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.rule import Rule
from rich.syntax import Syntax
from rich import box, print


class TerminalUI:
    """Terminal UI using Rich for the chatbot interface"""
    
    def __init__(self, 
                 generate_fn: Callable[[str], str],
                 history_capacity: int = 10,
                 system_info: Dict[str, Any] = None,
                 model_switch_fn: Optional[Callable[[str], bool]] = None):
        """Initialize the terminal UI
        
        Args:
            generate_fn: Function that takes a question and returns an answer
            history_capacity: Maximum number of conversations to keep in history
            system_info: Dictionary containing system information to display
            model_switch_fn: Function to switch between different models
        """
        self.console = Console()
        self.generate_fn = generate_fn
        self.history = []
        self.history_capacity = history_capacity
        self.system_info = system_info or {}
        self.model_switch_fn = model_switch_fn
        
        # Theme colors
        self.theme = {
            "accent": "blue",
            "secondary": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "muted": "dim white"
        }
        
        # Clear screen on start
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_splash_screen(self):
        """Show an animated splash screen"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Initializing RagChitChat...[/]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Loading components...", total=100)
            
            for i in range(101):
                time.sleep(0.015)  # Adjust for faster/slower animation
                progress.update(task, completed=i, 
                               description=f"[cyan]{self._get_loading_message(i)}[/]")
        
        self._display_logo()
    
    def _get_loading_message(self, progress):
        """Return different messages based on loading progress"""
        if progress < 20:
            return "Initializing components..."
        elif progress < 40:
            return "Loading document store..."
        elif progress < 60:
            return "Warming up embeddings..."
        elif progress < 80:
            return "Connecting to Ollama..."
        else:
            return "Almost ready..."
    
    def _display_logo(self):
        """Display the app logo"""
        logo = """
  _____              _____ _     _ _    _____ _           _   
 |  __ \\            / ____| |   (_) |  / ____| |         | |  
 | |__) |__ _  __ _| |    | |__  _| |_| |    | |__   __ _| |_ 
 |  _  // _` |/ _` | |    | '_ \\| | __| |    | '_ \\ / _` | __|
 | | \\ \\ (_| | (_| | |____| | | | | |_| |____| | | | (_| | |_ 
 |_|  \\_\\__,_|\\__, |\\_____|_| |_|_|\\__|\\_____|_| |_|\\__,_|\\__|
               __/ |                                          
              |___/                                           
        """
        
        self.console.print(Align(
            Panel(f"[bold {self.theme['accent']}]{logo}[/]", 
              border_style=self.theme["accent"],
              subtitle="[white]Your AI assistant for CTSE Lecture Notes[/]"),
            align="center"
        ))
    
    def show_welcome(self):
        """Display welcome message and instructions"""
        self.show_splash_screen()
        
        # Show system info
        if self.system_info:
            self.show_system_info()
        
        self.console.print()
        self.console.print(Align(
            Panel(
            "[bold]Ask me anything about Current Trends in Software Engineering![/]\n\n"
            "I can answer questions about CTSE lecture content, explain concepts,\n"
            "and help you understand course materials better.",
            title=f"[{self.theme['secondary']}]How to use RagChitChat[/]",
            border_style=self.theme["secondary"],
            expand=False
            ),
            align="center"
        ))
        
        self.console.print(Align(
            "\nType [bold cyan]/help[/bold cyan] for commands or [bold cyan]/exit[/bold cyan] to quit.\n", 
            align="center"
        ))
    
    def show_system_info(self):
        """Display system information"""
        table = Table(box=box.ROUNDED, title="System Information", border_style=self.theme["secondary"])
        
        table.add_column("Component", style=f"bold {self.theme['secondary']}")
        table.add_column("Details", style="white")
        
        # Add system info rows
        for key, value in self.system_info.items():
            # Format keys and values nicely
            key_display = key.replace('_', ' ').title()
            if isinstance(value, (list, tuple)):
                value_display = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                value_display = ', '.join(f"{k}: {v}" for k, v in value.items())
            else:
                value_display = str(value)
                
            table.add_row(key_display, value_display)
            
        self.console.print(Align(table, align="center"))
    
    def show_help(self):
        """Display help information"""
        table = Table(box=box.ROUNDED, title="Available Commands", border_style=self.theme["accent"])
        
        table.add_column("Command", style=f"bold {self.theme['secondary']}")
        table.add_column("Description")
        
        table.add_row("/help", "Show this help message")
        table.add_row("/exit", "Exit the chatbot")
        table.add_row("/clear", "Clear the conversation history")
        table.add_row("/history", "Show conversation history")
        table.add_row("/info", "Show system information")
        table.add_row("/about", "About this application")
        table.add_row("/models", "List available models")
        table.add_row("/model <name>", "Switch to a different model")
        
        self.console.print(Align(table, align="center"))
        
        # Show example questions
        examples = Table(
            box=box.SIMPLE,
            title="Example Questions",
            show_header=False,
            border_style=self.theme["muted"]
        )
        
        examples.add_column("", style=self.theme["secondary"])
        
        examples.add_row("- What is continuous integration?")
        examples.add_row("- Explain the difference between DevOps and DevSecOps")
        examples.add_row("- What are the benefits of microservices architecture?")
        examples.add_row("- How does containerization improve software deployment?")
        
        self.console.print(Panel(examples, border_style=self.theme["muted"], expand=False))
    
    def show_about(self):
        """Display information about the application"""
        about_text = """
RagChitChat is a Retrieval-Augmented Generation (RAG) chatbot designed to help students
learn about Current Trends in Software Engineering. It uses:

- **Local LLM** via [Ollama](https://ollama.ai) for AI inference
- **RAG Pipeline** with Haystack for intelligent document retrieval
- **Vector Database** using ChromaDB for semantic search
- **Rich UI** in the terminal for a pleasant user experience

This project demonstrates how generative AI can be applied to educational contexts
while maintaining privacy by keeping all operations local.

*Created by Nadun for the CTSE module assignment*
        """
        
        self.console.print(Panel(
            Markdown(about_text),
            title="About RagChitChat",
            border_style=self.theme["accent"],
            padding=(1, 2)
        ))
    
    def show_history(self, page: int = 1, items_per_page: int = 5):
        """Display conversation history with pagination
        
        Args:
            page: Page number to display
            items_per_page: Number of conversations per page
        """
        if not self.history:
            self.console.print(Panel(
                "[italic]You haven't asked any questions yet.[/]", 
                border_style=self.theme["warning"],
                title="History Empty"
            ))
            return
        
        # Calculate pagination
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_items = self.history[start_idx:end_idx]
        total_pages = (len(self.history) + items_per_page - 1) // items_per_page
        
        # Create paginated display
        self.console.print(Rule(f"[bold]Conversation History (Page {page}/{total_pages})[/]"))
        
        for i, (question, answer) in enumerate(current_items, start=start_idx+1):
            self.console.print()
            self.console.print(Panel(
                f"[bold {self.theme['secondary']}]Q: {question}[/]",
                border_style=self.theme["secondary"],
                padding=(0, 1)
            ))
            
            try:
                # Try to render as markdown
                self.console.print(Panel(
                    Markdown(answer),
                    border_style=self.theme["success"],
                    padding=(0, 1)
                ))
            except Exception:
                # Fallback to plain text
                self.console.print(Panel(
                    answer,
                    border_style=self.theme["success"],
                    padding=(0, 1)
                ))
        
        # Pagination controls
        self.console.print()
        if total_pages > 1:
            pagination = ""
            if page > 1:
                pagination += "[bold]/prev[/] "
            pagination += f"Page {page}/{total_pages}"
            if page < total_pages:
                pagination += " [bold]/next[/]"
                
            self.console.print(Align(pagination, align="center"))
    
    def handle_question(self, question: str) -> None:
        """Process a user question and display the response
        
        Args:
            question: User's question text
        """
        # Display the question banner
        self.console.print()
        self.console.print(Panel(
            f"[bold]{question}[/]", 
            border_style=self.theme["accent"],
            title="Question"
        ))
        
        # Show spinner while generating response
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Thinking...[/bold cyan]"),
            BarColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            expand=True
        ) as progress:
            retrieval_task = progress.add_task("Retrieving context...", total=None)
            
            # Use closure to update progress
            def progress_callback(stage, details=None):
                if stage == "retrieval_complete":
                    progress.update(retrieval_task, description="Generating answer...")
            
            # Record time for response metrics
            start_time = time.time()
            
            # Generate response
            answer = self.generate_fn(question, progress_callback)
            
            # Calculate response time
            elapsed = time.time() - start_time
        
        # Store in history
        self.history.append((question, answer))
        if len(self.history) > self.history_capacity:
            self.history.pop(0)
        
        # Display the response
        self.console.print()
        self.console.print(f"[{self.theme['muted']}]Generated in {elapsed:.2f}s[/]")
        
        try:
            # Try to render as markdown
            self.console.print(Panel(
                Markdown(answer),
                border_style=self.theme["success"],
                title="Answer",
                padding=(1, 2)
            ))
        except Exception:
            # Fallback to plain text
            self.console.print(Panel(
                answer,
                border_style=self.theme["success"],
                title="Answer",
                padding=(1, 2)
            ))
    
    def list_available_models(self):
        """Display available models that can be selected"""
        if not self.model_switch_fn or not self.system_info or 'available_models' not in self.system_info:
            self.console.print(Panel(
                "[italic]Model switching is not available.[/]", 
                border_style=self.theme["warning"],
                title="Models"
            ))
            return
            
        # Get current model and available models
        current_model = self.system_info.get('current_model', 'unknown')
        models = self.system_info.get('available_models', [])
        
        if not models:
            self.console.print(Panel(
                "[italic]No models found in Ollama. You can pull models using the Ollama CLI:[/]\n" +
                "ollama pull mistral:7b-instruct-v0.3-q4_1", 
                border_style=self.theme["warning"],
                title="No Models Available"
            ))
            return
            
        # Create a table of models
        table = Table(title="Available Models", box=box.ROUNDED, border_style=self.theme["secondary"])
        table.add_column("Model", style=f"bold {self.theme['secondary']}")
        table.add_column("Status")
        
        for model in models:
            if model == current_model:
                table.add_row(model, f"[{self.theme['success']}]ACTIVE[/]")
            else:
                table.add_row(model, "")
                
        self.console.print(Panel(
            table,
            border_style=self.theme["secondary"],
            title="Available Models"
        ))
        
        self.console.print("\nTo switch models, use: [bold]/model model_name[/]")
        
    def switch_model(self, model_name: str):
        """Switch to a different model
        
        Args:
            model_name: Name of the model to switch to
        """
        if not self.model_switch_fn:
            self.console.print(Panel(
                "[italic]Model switching is not available.[/]", 
                border_style=self.theme["warning"],
                title="Models"
            ))
            return
            
        # Show loading indicator while switching model
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Switching to model {model_name}...[/cyan]"),
            transient=True
        ) as progress:
            task = progress.add_task("", total=None)
            
            # Try to switch the model
            success = self.model_switch_fn(model_name)
        
        # Show the result
        if success:
            # Update the current model in system info
            if self.system_info and 'current_model' in self.system_info:
                self.system_info['current_model'] = model_name
                
            self.console.print(Panel(
                f"[bold]Successfully switched to model: [green]{model_name}[/green][/]",
                border_style=self.theme["success"],
                title="Model Switched"
            ))
        else:
            self.console.print(Panel(
                f"[bold]Failed to switch to model: [red]{model_name}[/red][/]\n\n"
                f"Make sure the model is available in Ollama.\n"
                f"You can pull it using: ollama pull {model_name}",
                border_style=self.theme["error"],
                title="Error"
            ))
    
    def run(self) -> None:
        """Run the main chat loop"""
        self.show_welcome()
        
        # Main interaction loop
        current_history_page = 1
        while True:
            self.console.print()
            question = Prompt.ask(f"[bold {self.theme['accent']}]Ask a question (or type /help)[/]")
            
            # Handle commands
            if question.lower() == "/exit":
                if Confirm.ask("[yellow]Are you sure you want to exit?[/]"):
                    self.console.print("[yellow]Thank you for using RagChitChat! Goodbye![/]")
                    break
            elif question.lower() == "/help":
                self.show_help()
            elif question.lower() == "/clear":
                self.history = []
                os.system('cls' if os.name == 'nt' else 'clear')
                self.show_welcome()
            elif question.lower() == "/history":
                current_history_page = 1
                self.show_history(page=current_history_page)
            elif question.lower() == "/next" and self.history:
                current_history_page += 1
                self.show_history(page=current_history_page)
            elif question.lower() == "/prev" and self.history:
                current_history_page = max(1, current_history_page - 1)
                self.show_history(page=current_history_page)
            elif question.lower() == "/info":
                self.show_system_info()
            elif question.lower() == "/about":
                self.show_about()
            elif question.lower() == "/models":
                self.list_available_models()
            elif question.lower().startswith("/model "):
                model_name = question[7:].strip()
                if model_name:
                    self.switch_model(model_name)
                else:
                    self.console.print("[yellow]Please specify a model name. Example: /model mistral-7b[/]")
            elif question.strip():
                # Process regular questions
                self.handle_question(question)
