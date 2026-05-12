"""
Full RAG pipeline: Retrieval → Prompt Assembly → LLM Generation.
Orchestrates the retriever and LLM into a single ask() interface.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K
from src.retriever import Retriever
from src.llm import LLM


# Llama 2 Chat prompt template
PROMPT_TEMPLATE = """[INST] <<SYS>>
You are a knowledgeable AI assistant specialized in machine learning research.
Answer the user's question ONLY using the provided context from research papers.
Be precise and cite the source documents when possible.
If the answer cannot be found in the context, say "Based on the provided papers, I cannot find information about this topic."
<</SYS>>

Context from research papers:
{context}

Question: {question} [/INST]"""


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(self, llm_backend: str = "auto"):
        """Initialize retriever and LLM."""
        print("=" * 50)
        print("Initializing RAG Pipeline")
        print("=" * 50)

        self.retriever = Retriever()
        self.llm = LLM(backend=llm_backend)

        print("=" * 50)
        print("RAG Pipeline ready!")
        print(f"  Retriever: {self.retriever.index.ntotal} chunks")
        print(f"  LLM: {self.llm.backend_name}")
        print("=" * 50)

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dict with keys: answer, sources, retrieval_time, generation_time
        """
        from config import LLM_CONTEXT_LENGTH, LLM_MAX_TOKENS

        # Step 1: Retrieve relevant chunks
        t0 = time.time()
        results = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - t0

        # Step 2: Build context, respecting token budget
        # Rough estimate: 1 token ≈ 4 chars
        # Reserve tokens for: system prompt (~200 tokens), question (~50), answer (max_tokens)
        max_context_chars = (LLM_CONTEXT_LENGTH - LLM_MAX_TOKENS - 250) * 4
        
        context_parts = []
        used_chars = 0
        included_results = []
        
        for i, r in enumerate(results, 1):
            chunk_text = r['text']
            entry = f"[{i}] (Source: {r['doc_name']})\n{chunk_text}"
            
            if used_chars + len(entry) > max_context_chars:
                # Truncate this chunk to fit remaining budget
                remaining = max_context_chars - used_chars
                if remaining > 200:  # Only include if we can fit at least 200 chars
                    entry = entry[:remaining] + "..."
                    context_parts.append(entry)
                    included_results.append(r)
                break
            
            context_parts.append(entry)
            included_results.append(r)
            used_chars += len(entry)

        context = "\n\n".join(context_parts)

        # Step 3: Assemble prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # Step 4: Generate answer
        t1 = time.time()
        answer = self.llm.generate(prompt)
        generation_time = time.time() - t1

        # Step 5: Format sources
        sources = []
        for r in included_results:
            sources.append({
                "doc_name": r["doc_name"],
                "score": r["score"],
                "text_preview": r["text"][:200] + "...",
                "chunk_id": r["chunk_id"],
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "num_chunks_used": len(included_results),
            "prompt_length": len(prompt),
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print()

    test_questions = [
        "What is the transformer architecture?",
        "How does batch normalization work?",
        "What are the advantages of deep learning?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer'][:300]}")
        print(f"   Sources: {[s['doc_name'] for s in result['sources']]}")
        print(f"   Retrieval: {result['retrieval_time']:.3f}s, Generation: {result['generation_time']:.3f}s")
