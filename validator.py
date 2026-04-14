from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
MODEL_NAME = "ministral-3:3b" 
llm = OllamaLLM(model=MODEL_NAME)

def validate_system(content):
    """
    Validates a system prompt with a 1-5 rating.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following SYSTEM PROMPT for clarity and consistency. "
        "Rules: very clear, no contradictions, no confusing rules.\n\n"
        "Scoring Criteria (1-5 total):\n"
        "- Clarity (1-2 points)\n"
        "- No Contradictions (1-2 points)\n"
        "- Simplicity (1 point)\n\n"
        "System Prompt: {content}\n\n"
        "Provide your analysis followed by a 'Rating: X/5' section."
    )
    chain = prompt | llm
    return chain.stream({"content": content})

def validate_user(content):
    """
    Validates a user prompt with a 1-5 rating.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following USER PROMPT.\n\n"
        "Scoring Criteria (1-5 total):\n"
        "- Understandable (1-2 points)\n"
        "- Logical (1-2 points)\n"
        "- Intent Clarity (1 point)\n\n"
        "User Prompt: {content}\n\n"
        "Provide your analysis followed by a 'Rating: X/5' section."
    )
    chain = prompt | llm
    return chain.stream({"content": content})

def validate_thinking(content):
    """
    Validates a thinking cell with a 1-5 rating.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following THINKING CELL.\n\n"
        "Scoring Criteria (1-5 total):\n"
        "- Detail levels (1-2 points)\n"
        "- Step-by-step transparency (1-2 points)\n"
        "- Calculation completeness (1 point)\n\n"
        "Thinking Content: {content}\n\n"
        "Provide your analysis followed by a 'Rating: X/5' section."
    )
    chain = prompt | llm
    return chain.stream({"content": content})

def validate_assistant(content, thinking_context):
    """
    Validates an assistant response with a 1-5 rating.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following ASSISTANT RESPONSE against its THINKING PROCESS.\n\n"
        "Scoring Criteria (1-5 total):\n"
        "- Safety (1-2 points)\n"
        "- No Hallucinated improvements/new info (1-3 points)\n\n"
        "Thinking Process: {thinking_context}\n"
        "Assistant Response: {content}\n\n"
        "Provide your analysis followed by a 'Rating: X/5' section."
    )
    chain = prompt | llm
    return chain.stream({"content": content, "thinking_context": thinking_context})

def validate_cell(label, content, thinking_context=None):
    """
    Route the validation based on the cell label and return a stream.
    """
    label = label.lower()
    if "system" in label:
        return validate_system(content)
    elif "user" in label:
        return validate_user(content)
    elif "thinking" in label:
        return validate_thinking(content)
    elif "assistant" in label:
        return validate_assistant(content, thinking_context or "No previous thinking context provided.")
    else:
        # For unknown labels, return a dummy generator
        def generator():
            yield f"Unknown cell label: {label}. No specific rules defined."
        return generator()
