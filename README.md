# 🧪 LLM Conversation Validator

An automated tool to audit and validate the quality of LLM conversation datasets. This validator parses Jupyter Notebooks (`.ipynb`) and uses **LangChain** with **Ollama (Ministral-3b)** to score conversation cells based on logic, clarity, and grounding.

## 🚀 Features

- **Multi-Cell Validation**: Automatically scores System Prompts, User Queries, Thinking Processes, and Assistant Responses.
- **Grounding Checks**: Validates asssistant responses against their thinking process to detect hallucinations or missing information.
- **Real-Time Auditing**: Powered by a **Gradio** web interface with streaming feedback.
- **Notebook Support**: Directly process `.ipynb` files used in LLM training and fine-tuning pipelines.

## 🛠️ Tech Stack

- **Logic**: Python, LangChain
- **LLM Engine**: Ollama (Ministral-3b)
- **UI**: Gradio
- **Dependency Management**: uv

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/RIPYO6/analyze.git
    cd analyze
    ```

2.  **Install dependencies** (recommended via `uv`):
    ```bash
    uv sync
    ```

3.  **Run the application**:
    ```bash
    python main.py
    ```

## 📝 Usage

1. Launch the app.
2. Upload a `.ipynb` file containing LLM conversation cells.
3. Click "Run Validation" to see real-time scores and analysis for every cell in the notebook.
