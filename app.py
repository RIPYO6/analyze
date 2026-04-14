import gradio as gr
import nbformat
import validator
import os

def process_notebook(file_obj):
    if file_obj is None:
        yield "### Please upload a notebook file."
        return

    try:
        # file_obj is a temp file path in Gradio
        with open(file_obj.name, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        yield f"### Error reading notebook: {e}"
        return

    last_thinking_context = None
    full_log = ""

    for i, cell in enumerate(nb.cells):
        if i == 0:
            continue
        
        cell_type = cell.cell_type
        source = cell.source or ""
        
        lines = source.split("\n")
        cell_label = lines[0] if lines else ""
        cell_content = "\n".join(lines[1:]) if len(lines) > 1 else ""
        
        # Add entry for the cell
        header = f"\n---\n### Cell {i} - {cell_label}\n"
        full_log += header
        yield full_log
        
        # Validate cell streaming
        full_log += "**Validation Analysis & Rating**:\n\n"
        try:
            stream = validator.validate_cell(cell_label, cell_content, last_thinking_context)
            for chunk in stream:
                full_log += chunk
                yield full_log
        except Exception as e:
            full_log += f"\n\n**Error during validation**: {e}"
            yield full_log
        
        full_log += "\n\n"
        yield full_log

        # Update thinking context if it's a thinking cell
        if "thinking" in cell_label.lower():
            last_thinking_context = cell_content

# Gradio UI
with gr.Blocks(title="LLM Conversation Validator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧪 LLM Conversation Validator")
    gr.Markdown("Upload an `.ipynb` file to validate the quality of system prompts, user queries, thinking processes, and assistant responses.")
    
    with gr.Row():
        file_input = gr.File(label="Select .ipynb Notebook", file_count="single", file_types=[".ipynb"])
        
    with gr.Row():
        start_btn = gr.Button("🚀 Run Validation", variant="primary")
        
    output_log = gr.Markdown(label="Validation Results", value="Results will appear here...")

    start_btn.click(fn=process_notebook, inputs=file_input, outputs=output_log)

if __name__ == "__main__":
    # Launch on localhost
    demo.launch(server_name="127.0.0.1", server_port=7860)
