"""
Advanced prompt engineering templates for RagChitChat
"""
import os
import sys
from typing import List, Dict, Any, Optional
import random

# Add project root to Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings

NO_CONTEXT_MESSAGE = "No relevant context information was found in the documents to answer this question."


def get_rag_prompt(question: str, context: List[Dict[str, Any]]) -> str:
    """Generate an advanced RAG prompt with chain-of-thought reasoning
    
    Args:
        question: The user's question
        context: List of context documents
    
    Returns:
        Formatted prompt for the LLM
    """
    # Format context documents into a unified text
    context_text = format_context_documents(context)
    
    # Replace placeholders in the template
    # Add instruction for out-of-context handling
    prompt = settings.RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    ) + (
        "\n\nImportant: Answer based *only* on the provided context. "
        "If the answer cannot be found in the context, state that "
        f"'{NO_CONTEXT_MESSAGE}' or that the information is not available in the provided documents.EXPLICITLY SEND THIS REPLY 'NO RELEVANT DATA FOUND TO REPLY'"
        "When using information from the context, cite the document and page/slide number (e.g., Document 1: filename.pdf)."
    )
    
    # Add few-shot examples if we have space
    # Only include for complex questions (determined by length and question words)
    if _is_complex_question(question) and len(context_text) + len(prompt) < 6000:
        prompt = _add_few_shot_examples(prompt)
    
    return prompt


def get_reflection_prompt(question: str, initial_answer: str, context: List[Dict[str, Any]]) -> str:
    """Generate a self-reflection prompt for answer improvement
    
    Args:
        question: The original question
        initial_answer: The initial response from the LLM
        context: The context documents used
    
    Returns:
        Reflection prompt for the LLM to improve its answer
    """
    context_text = format_context_documents(context)
    
    # Create evaluation criteria questions
    evaluation_questions = "\n".join([
        f"- {criterion}: {question}" 
        for criterion, question_text in settings.SELF_EVALUATION_CRITERIA.items()
    ])
    
    # Create the reflection prompt
    reflection_prompt = f"""
You are an educational assistant that's reviewing your previous response to ensure it meets high academic standards.

Original question: {question}

Your previous response: 
{initial_answer}

Please reflect on your response using these criteria:
{evaluation_questions}
- Citation: Does the response properly cite sources and page/slide numbers from the context when information is used (e.g., Document 1: filename.pdf, Page X)?

First, identify any issues with your previous response, including missing citations.
Then, provide an improved version that addresses these issues while maintaining academic accuracy.
Base your improved answer strictly on this context information:

{context_text}

Important: Your improved answer must be based *only* on the provided context. 
If the answer cannot be found in the context, state that the information is not available in the provided documents.
Ensure all information taken from the context is properly cited with document and page/slide number.

Begin your response with "## Self-Reflection" followed by your analysis, then "## Improved Answer" with your revised response.
    """
    
    return reflection_prompt


def get_structured_prompt(question: str, context: List[Dict[str, Any]], 
                         output_format: str = "default") -> str:
    """Generate a prompt requiring structured output in a specific format
    
    Args:
        question: The user's question
        context: List of context documents
        output_format: The desired output format (default, table, steps, comparison)
    
    Returns:
        Formatted prompt for the LLM
    """
    context_text = format_context_documents(context)
    
    # Define different output format instructions
    format_instructions = {
        "default": """
Format your response using appropriate Markdown with:
- Clear headings with ## and ### for sections
- Bullet points for lists
- **Bold** for important concepts
- `code blocks` for technical terms or code
- > blockquotes for definitions
        """,
        
        "table": """
Include a Markdown table in your response to summarize key points:
| Concept | Description | Example |
| ------- | ----------- | ------- |
| Concept 1 | Description 1 | Example 1 |
| ... | ... | ... |
        """,
        
        "steps": """
Format your response as a step-by-step guide using:
## Process Overview
Brief overview of the process

## Step 1: [Step Name]
Explanation of step 1

## Step 2: [Step Name]
Explanation of step 2

And so on, with clear numbered steps and explanations.
        """,
        
        "comparison": """
Format your response as a comparison between concepts:
## Concept A
- Key characteristics
- Advantages
- Disadvantages

## Concept B
- Key characteristics
- Advantages
- Disadvantages

## Comparison
| Aspect | Concept A | Concept B |
| ------ | --------- | --------- |
| Aspect 1 | Value for A | Value for B |
| ... | ... | ... |
        """
    }
    
    # Determine the most appropriate format if not specified
    if output_format == "default":
        output_format = _detect_appropriate_format(question)
    
    # Get the format instructions
    format_instruction = format_instructions.get(output_format, format_instructions["default"])
    
    # Create the structured prompt
    structured_prompt = f"""
You are a university-level educational assistant specialized in Current Trends in Software Engineering.

Based on the following context information from CTSE lecture notes:
{context_text}

Please answer this question:
{question}

{format_instruction}

Important: Answer based *only* on the provided context. 
If the answer cannot be found in the context, state that the information is not available in the provided documents.
When using information from the context, cite the specific document and page/slide number (e.g., "According to Document 1: lecture_notes.pdf, Page 5...").
Be concise but comprehensive, and ensure all information is accurate according to the provided context.
    """
    
    return structured_prompt


def format_context_documents(context: List[Dict[str, Any]]) -> str:
    """Format context documents with metadata for better reference
    
    Args:
        context: List of context documents
    
    Returns:
        Formatted context text
    """
    if not context:
        return NO_CONTEXT_MESSAGE
    
    context_sections = []
    
    for i, doc in enumerate(context):
        # Extract metadata
        source = doc.get('metadata', {}).get('source', 'Unknown Source')
        page_num = doc.get('metadata', {}).get('page_num', 'N/A')
        content = doc.get('content', '').strip()
        
        # Calculate relevance if available
        relevance = ""
        if 'distance' in doc and doc['distance'] is not None:
            similarity = 1.0 - float(doc['distance'])
            relevance = f" [Relevance: {similarity:.2f}]"
        
        # Format document section
        section = f"[DOCUMENT {i+1}]: {source} (Page/Slide: {page_num}){relevance}\n{content}"
        context_sections.append(section)
    
    return "\n\n" + "\n\n".join(context_sections)


def _detect_appropriate_format(question: str) -> str:
    """Detect the most appropriate output format based on the question
    
    Args:
        question: The user question
    
    Returns:
        Suggested output format
    """
    question_lower = question.lower()
    
    # Check for step-by-step requests
    if any(phrase in question_lower for phrase in ["steps", "process", "how to", "procedure", "workflow"]):
        return "steps"
    
    # Check for comparison requests
    if any(phrase in question_lower for phrase in ["compare", "difference between", "versus", "vs", "pros and cons"]):
        return "comparison"
    
    # Check for listing/table appropriate requests
    if any(phrase in question_lower for phrase in ["list", "summarize", "overview", "key aspects", "characteristics"]):
        return "table"
    
    # Default format for other questions
    return "default"


def _is_complex_question(question: str) -> bool:
    """Determine if a question is complex enough to warrant few-shot examples
    
    Args:
        question: The user question
    
    Returns:
        True if the question appears complex
    """
    # Long questions are often more complex
    if len(question) > 100:
        return True
    
    # Questions asking for comparison, analysis, or explanation tend to be complex
    complex_indicators = ["compare", "explain", "analyze", "evaluate", "why", 
                         "how does", "implications", "relationship between"]
    
    return any(indicator in question.lower() for indicator in complex_indicators)


def _add_few_shot_examples(prompt: str) -> str:
    """Add few-shot learning examples to the prompt
    
    Args:
        prompt: The original prompt
    
    Returns:
        Prompt enhanced with examples
    """
    # Select one random example to include
    example = random.choice(settings.FEW_SHOT_EXAMPLES)
    
    few_shot_text = f"""
Here's an example of how to answer a similar question:

EXAMPLE QUESTION: {example['question']}

EXAMPLE ANSWER: {example['answer']}

Now, please answer the original question following a similar format and approach:
"""
    
    return prompt + few_shot_text
