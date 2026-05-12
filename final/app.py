"""
RAG QA System — Gradio Web Demo
Variant 5: Question-Answering on ML Research Papers

Usage: python app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from config import GRADIO_PORT, TOP_K
from src.rag_pipeline import RAGPipeline

# ─── Custom CSS for premium look ──────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.header-section {
    text-align: center;
    padding: 20px 0 10px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
}

.header-section h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    color: white !important;
}

.header-section p {
    font-size: 14px;
    opacity: 0.9;
    margin: 5px 0 0 0;
    color: white !important;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.2);
    color: white;
    margin-top: 8px;
}

.source-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
}

.dark .source-card {
    background: #1e1e2e;
    border-left-color: #89b4fa;
}

.metric-box {
    text-align: center;
    padding: 10px;
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border-radius: 12px;
    border: 1px solid #667eea33;
}

.metric-box .value {
    font-size: 20px;
    font-weight: 700;
    color: #667eea;
}

.metric-box .label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

footer { display: none !important; }
"""


def create_app():
    """Create and configure the Gradio application."""

    # Initialize RAG pipeline
    pipeline = RAGPipeline()

    def answer_question(question: str, top_k: int) -> tuple:
        """Process a question through the RAG pipeline."""
        if not question.strip():
            return (
                "⚠️ Please enter a question.",
                "No sources retrieved.",
                "", "", ""
            )

        try:
            result = pipeline.ask(question, top_k=int(top_k))

            # Format answer
            answer_md = f"### 💡 Answer\n\n{result['answer']}"

            # Format sources
            sources_parts = []
            for i, src in enumerate(result["sources"], 1):
                score_pct = src["score"] * 100
                sources_parts.append(
                    f"**[{i}] {src['doc_name']}** — relevance: {score_pct:.1f}%\n\n"
                    f"> {src['text_preview']}\n"
                )
            sources_md = "### 📚 Retrieved Sources\n\n" + "\n---\n".join(sources_parts)

            # Metrics
            ret_time = f"{result['retrieval_time']*1000:.0f} ms"
            gen_time = f"{result['generation_time']:.1f} s"
            chunks_used = str(result['num_chunks_used'])

            return answer_md, sources_md, ret_time, gen_time, chunks_used

        except Exception as e:
            return (
                f"❌ Error: {str(e)}",
                "No sources retrieved.",
                "—", "—", "—"
            )

    # Build UI
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="RAG QA System — ML Paper Q&A") as demo:

        # Header
        gr.HTML(f"""
        <div class="header-section">
            <h1>🔬 RAG QA System</h1>
            <p>Retrieval-Augmented Generation for ML Research Papers</p>
            <div class="status-badge">🟢 {pipeline.llm.backend_name} • {pipeline.retriever.index.ntotal} chunks indexed</div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Ask a question about ML research",
                    placeholder="e.g., What is the transformer architecture and how does self-attention work?",
                    lines=2,
                    max_lines=4,
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=TOP_K, step=1,
                    label="Number of sources (Top-K)",
                )
                submit_btn = gr.Button("🔍 Ask", variant="primary", size="lg")

        # Metrics row
        with gr.Row():
            retrieval_time = gr.Textbox(label="⏱️ Retrieval", interactive=False, scale=1)
            generation_time = gr.Textbox(label="⏱️ Generation", interactive=False, scale=1)
            chunks_used = gr.Textbox(label="📄 Chunks Used", interactive=False, scale=1)

        # Results
        with gr.Row():
            with gr.Column(scale=1):
                answer_output = gr.Markdown(label="Answer")
            with gr.Column(scale=1):
                sources_output = gr.Markdown(label="Sources")

        # Wire events
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, retrieval_time, generation_time, chunks_used],
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, retrieval_time, generation_time, chunks_used],
        )

        # Example questions
        gr.Examples(
            examples=[
                ["What is the transformer architecture and how does self-attention work?"],
                ["How do convolutional neural networks process images?"],
                ["What are generative adversarial networks?"],
                ["How does reinforcement learning work?"],
                ["What is transfer learning in NLP?"],
                ["What are diffusion models and how do they generate images?"],
                ["What is knowledge distillation?"],
                ["How does batch normalization improve training?"],
            ],
            inputs=[question_input],
            label="💡 Example Questions",
        )

        # Footer info
        gr.Markdown(
            "---\n"
            "*Variant 5: RAG QA System for ML Research Papers • "
            "Built with SBERT + FAISS + Llama 2 + Gradio*"
        )

    return demo, theme


if __name__ == "__main__":
    demo, theme = create_app()
    demo.launch(
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=theme,
    )
