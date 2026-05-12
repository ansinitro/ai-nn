# Defense Speech — RAG QA System (Variant 5)
**Course:** AI & Neural Networks | **Group:** AAI-2501  
**Students:** Angsar Shaumen, Bekzat Sundetkhan

---

## Slide 1 — Title (30 sec)

> Good day! We are Angsar Shaumen and Bekzat Sundetkhan from group AAI-2501.
> Today we present our final project — a **Retrieval-Augmented Generation** system
> for question-answering on machine learning research papers.

---

## Slide 2 — Task Description (45 sec)

> Our task was **Variant 5**: to build a RAG system that can answer questions
> using a corpus of 50 to 100 PDF articles on machine learning.
>
> The requirements were: use SBERT or Instructor embeddings, FAISS or Chroma
> as the vector database, and an LLM — either via API or running locally.
>
> The deliverables include a **web demo**, quality metrics using ROUGE and BERTScore,
> and an analysis of both successful and unsuccessful cases.

---

## Slide 3 — Architecture (1 min)

> Here is our system architecture.
> The pipeline has two phases: **offline indexing** and **online querying**.
>
> In the offline phase, we take 85 PDF papers, extract text using PyMuPDF,
> chunk them into passages using LangChain, encode each chunk with SBERT
> into 768-dimensional vectors, and store them in a FAISS index.
>
> During the online phase, when a user asks a question, we encode the query
> with the same SBERT model, search FAISS for the top-K most relevant chunks,
> assemble a prompt with the retrieved context, and pass it to Llama 2 7B
> which generates the answer. The answer and source papers are displayed
> in our Gradio web interface.

---

## Slide 4 — Data Collection (45 sec)

> We collected **85 PDF papers** from arXiv covering 15 different ML topics:
> deep learning, transformers, CNNs, GANs, reinforcement learning, NLP,
> graph neural networks, diffusion models, meta-learning, and more.
>
> After extraction and cleaning, **74 papers** were successfully parsed,
> producing nearly **1 million tokens** of text.
>
> We chunked this text into **1700 passages** with an average of about 650 tokens each,
> using a 400-character overlap to preserve context across chunk boundaries.

---

## Slide 5 — Embedding Model (30 sec)

> For embeddings, we chose Sentence-BERT with the **all-mpnet-base-v2** model.
> It produces 768-dimensional vectors with strong semantic understanding.
>
> Encoding all 1700 chunks took just **35 seconds** at about 49 chunks per second.
> The embeddings are L2-normalized, which allows us to use inner product
> as a cosine similarity metric in FAISS.

---

## Slide 6 — Vector Database (30 sec)

> We use **FAISS with a FlatIP index** — this means exact search using
> inner product, which equals cosine similarity for our normalized vectors.
>
> For our corpus size of 1700 chunks, exact search is actually faster than
> approximate methods like HNSW, and there is zero accuracy loss.
> The entire index is only 5 megabytes and queries complete in under 15 milliseconds.

---

## Slide 7 — LLM (1 min)

> The language model is **Llama 2 7B Chat**, quantized to 4 bits using GGUF format.
> This reduces the model from 14 gigabytes to just **3.8 gigabytes**,
> making it possible to run on our RTX 2060 with only 6 GB of VRAM.
>
> We use **llama-cpp-python** with CUDA support as the inference engine,
> achieving about **12 tokens per second** with 20 layers offloaded to GPU.
>
> An important feature is our **auto-retry mechanism**: if a GPU configuration
> fails due to insufficient VRAM, the system automatically tries fewer GPU layers,
> eventually falling back to CPU-only mode. This makes the system robust
> across different hardware configurations.
>
> We also implemented an OpenAI API fallback as a second option.

---

## Slide 8 — RAG Pipeline (45 sec)

> The RAG pipeline orchestrates everything. When a question comes in,
> we retrieve the top 5 most relevant chunks from FAISS.
>
> Then we assemble a prompt using the **Llama 2 Chat format** with a system instruction
> that tells the model to answer only from the provided context
> and cite the source documents.
>
> We also implemented **token-aware context truncation** — the system calculates
> how many chunks can fit within the 2048-token context window and includes
> only as many as will fit, preventing overflow errors.
>
> Retrieval takes about 15 milliseconds, and answer generation takes about 5 seconds.

---

## Slide 9 — Web Demo (30 sec)

> Our web interface is built with **Gradio 6.14**.
> It features a question input field, a Top-K slider to control
> how many source chunks are retrieved, and displays both the answer
> and the retrieved sources with relevance scores.
>
> We also include 8 example questions for quick testing
> and real-time performance metrics showing retrieval and generation times.

---

## Slide 10 — Evaluation (45 sec)

> For evaluation, we created **15 test questions** with manually written
> reference answers, covering 10 different ML topics.
>
> Each question is run through the full RAG pipeline, and the predicted answer
> is compared to the reference using two types of metrics:
>
> **ROUGE** measures word-level overlap — both unigram (ROUGE-1)
> and longest common subsequence (ROUGE-L).
>
> **BERTScore** uses RoBERTa embeddings to measure **semantic similarity**,
> which is more meaningful for open-ended QA since there are many valid ways
> to phrase the same answer. We run BERTScore on CPU to avoid GPU memory conflicts.

---

## Slide 11 — Results (1 min)

> Our results show a **ROUGE-1 F1 of 0.75** and a **BERTScore F1 of 0.96**,
> which indicates high semantic similarity between our generated answers
> and the reference answers.
>
> In our **success analysis**, the system performs well on questions
> directly matching paper topics — like questions about transformers,
> CNNs, or GANs — where relevant chunks are retrieved with high confidence.
>
> In our **failure analysis**, we identified three main causes of weaker answers:
> first, niche topics with very few matching papers in our corpus;
> second, questions requiring synthesis across multiple papers;
> and third, overly broad questions that exceed the context window budget.
>
> All results are logged to CSV and JSON for reproducibility.

---

## Slide 12 — Challenges (30 sec)

> We encountered several technical challenges.
> The biggest was fitting a 7-billion parameter model into 6 GB of VRAM,
> which we solved with 4-bit quantization and our auto-retry GPU layer mechanism.
>
> We also had to handle context window overflow through smart truncation,
> resolve CUDA compatibility issues by building from source,
> and manage GPU memory sharing between the LLM and evaluation models.

---

## Slide 13 — Tech Stack (15 sec)

> Here is our complete technology stack and hardware configuration.
> The project consists of 16 source files with approximately 1500 lines of Python code.

---

## Slide 14 — Conclusion (30 sec)

> To summarize: we successfully built a complete RAG system that meets all
> the task requirements — 85 ML papers, SBERT embeddings, FAISS indexing,
> local Llama 2 inference, a Gradio web demo, evaluation with ROUGE and BERTScore,
> and thorough analysis of success and failure cases.
>
> Thank you for your attention.

---

## Slide 15 — Q&A

> We are happy to take your questions and demonstrate the system live.

---

## Possible Q&A Questions and Answers

**Q: Why did you choose SBERT over Instructor embeddings?**
> SBERT all-mpnet-base-v2 offers the best balance of quality and speed for our hardware. 
> It encodes 49 chunks/second vs Instructor which is 3-5x slower. For 1700 chunks, speed matters.

**Q: Why FAISS instead of Chroma?**
> For our corpus size (~1700 vectors), FAISS Flat provides exact search with zero approximation loss.
> It's faster and simpler. Chroma is better when you need built-in metadata filtering, which we didn't need.

**Q: Why 4-bit quantization? Does it hurt quality?**
> 4-bit reduces model size from 14GB to 3.8GB. Research shows Q4_K_M quantization preserves 
> 98-99% of the full model's quality. Without it, the model simply wouldn't fit on our GPU.

**Q: How do you handle hallucination?**
> Our prompt explicitly instructs the model to answer ONLY from the provided context.
> If no relevant information is found, it says "I cannot find information about this topic."
> Source attribution also helps verify factuality.

**Q: What would you improve with more time?**
> Three things: (1) Add a re-ranking step after retrieval using a cross-encoder model;
> (2) Implement hybrid search combining semantic + keyword matching;
> (3) Upgrade to a larger context window model like Mistral 7B (32K context).

**Q: How long does it take to answer a question?**
> Retrieval: ~15ms. Answer generation: ~5 seconds on GPU. Total: ~5 seconds end-to-end.

**Q: What happens if the answer isn't in the papers?**
> The system retrieves the most similar chunks anyway, but the LLM is instructed to say
> "Based on the provided papers, I cannot find information about this topic" if the context 
> doesn't contain the answer. This is a deliberate design choice to reduce hallucination.
