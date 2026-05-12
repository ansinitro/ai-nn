"""
Evaluate RAG system quality using ROUGE and BERTScore.
Runs test questions through the pipeline and logs metrics.
Usage: python -m src.evaluate
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR, PROCESSED_DIR
from src.rag_pipeline import RAGPipeline


def load_test_questions() -> list[dict]:
    """Load test Q/A pairs from JSON."""
    qa_path = EVAL_DIR / "test_questions.json"
    if not qa_path.exists():
        print(f"No test questions found at {qa_path}")
        print("Creating a template file...")
        create_template_questions()

    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_template_questions():
    """Create a template test questions file."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    template = [
        {
            "question": "What is the transformer architecture and how does self-attention work?",
            "reference_answer": "The transformer architecture uses self-attention mechanisms to process sequences in parallel, computing attention weights between all positions simultaneously.",
            "topic": "transformers"
        },
        {
            "question": "What is batch normalization and why is it useful in deep learning?",
            "reference_answer": "Batch normalization normalizes layer inputs by adjusting and scaling activations, which helps stabilize training, allows higher learning rates, and acts as a regularizer.",
            "topic": "optimization"
        },
        {
            "question": "How do convolutional neural networks process images?",
            "reference_answer": "CNNs use convolutional layers with learnable filters to extract spatial features from images, using pooling to reduce dimensions and building hierarchical feature representations.",
            "topic": "computer_vision"
        },
        {
            "question": "What is transfer learning and how is it applied in NLP?",
            "reference_answer": "Transfer learning involves pretraining a model on a large dataset then fine-tuning it on a smaller task-specific dataset. In NLP, models like BERT are pretrained on large text corpora then fine-tuned for downstream tasks.",
            "topic": "nlp"
        },
        {
            "question": "What are generative adversarial networks?",
            "reference_answer": "GANs consist of a generator and discriminator network trained adversarially. The generator creates synthetic data while the discriminator tries to distinguish real from fake data.",
            "topic": "generative_models"
        },
        {
            "question": "How does reinforcement learning differ from supervised learning?",
            "reference_answer": "Reinforcement learning learns through trial and error by maximizing cumulative reward from environment interactions, while supervised learning learns from labeled input-output pairs.",
            "topic": "reinforcement_learning"
        },
        {
            "question": "What is the purpose of dropout regularization?",
            "reference_answer": "Dropout randomly deactivates neurons during training to prevent co-adaptation and overfitting, effectively training an ensemble of sub-networks.",
            "topic": "regularization"
        },
        {
            "question": "What are graph neural networks used for?",
            "reference_answer": "Graph neural networks process graph-structured data by propagating information between connected nodes, used for social networks, molecular analysis, and recommendation systems.",
            "topic": "graph_networks"
        },
        {
            "question": "How do attention mechanisms improve sequence models?",
            "reference_answer": "Attention mechanisms allow models to focus on relevant parts of the input when generating each output, solving the information bottleneck problem in encoder-decoder architectures.",
            "topic": "attention"
        },
        {
            "question": "What is few-shot learning?",
            "reference_answer": "Few-shot learning aims to learn new concepts from very few labeled examples, often using meta-learning approaches that learn to learn across many tasks.",
            "topic": "meta_learning"
        },
        {
            "question": "What is the role of the learning rate in neural network training?",
            "reference_answer": "The learning rate controls the step size during gradient descent optimization. Too high causes divergence, too low causes slow convergence. Schedulers and adaptive methods like Adam help manage it.",
            "topic": "optimization"
        },
        {
            "question": "How do diffusion models generate images?",
            "reference_answer": "Diffusion models learn to gradually denoise data by reversing a forward diffusion process that adds Gaussian noise. They generate high-quality images by iteratively refining noise into structured outputs.",
            "topic": "generative_models"
        },
        {
            "question": "What is knowledge distillation?",
            "reference_answer": "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model by training the student to match the teacher's soft probability outputs.",
            "topic": "model_compression"
        },
        {
            "question": "What are the main challenges in federated learning?",
            "reference_answer": "Federated learning faces challenges including non-IID data across clients, communication efficiency, privacy guarantees, and handling heterogeneous devices.",
            "topic": "federated_learning"
        },
        {
            "question": "What is self-supervised learning?",
            "reference_answer": "Self-supervised learning creates supervised signals from unlabeled data using pretext tasks like masked language modeling or contrastive learning to learn useful representations.",
            "topic": "self_supervised"
        },
    ]

    qa_path = EVAL_DIR / "test_questions.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"Created template with {len(template)} questions at {qa_path}")


def evaluate():
    """Run full evaluation: pipeline + metrics."""
    # Import evaluation libraries
    try:
        from rouge_score import rouge_scorer
        import bert_score
    except ImportError:
        print("Missing evaluation libraries. Install with:")
        print("  pip install rouge-score bert-score")
        return

    # Load test questions
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} test questions")

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Setup ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    # Run evaluation
    results = []
    predictions = []
    references = []

    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    for i, qa in enumerate(test_questions, 1):
        question = qa["question"]
        reference = qa["reference_answer"]

        print(f"\n[{i}/{len(test_questions)}] {question[:60]}...")

        try:
            result = pipeline.ask(question)
            predicted = result["answer"]

            # ROUGE scores
            rouge_scores = scorer.score(reference, predicted)

            predictions.append(predicted)
            references.append(reference)

            entry = {
                "question": question,
                "reference_answer": reference,
                "predicted_answer": predicted,
                "topic": qa.get("topic", ""),
                "rouge1_f1": rouge_scores["rouge1"].fmeasure,
                "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                "sources": [s["doc_name"] for s in result["sources"]],
                "retrieval_time": result["retrieval_time"],
                "generation_time": result["generation_time"],
            }
            results.append(entry)

            print(f"  ROUGE-1: {entry['rouge1_f1']:.3f}, ROUGE-L: {entry['rougeL_f1']:.3f}")
            print(f"  Answer: {predicted[:100]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": question,
                "reference_answer": reference,
                "predicted_answer": f"ERROR: {e}",
                "rouge1_f1": 0.0,
                "rougeL_f1": 0.0,
                "sources": [],
            })

    # Compute BERTScore for all predictions at once (on CPU to avoid GPU OOM)
    print("\nComputing BERTScore (on CPU)...")
    if predictions:
        P, R, F1 = bert_score.score(
            predictions, references, lang="en", verbose=False, device="cpu"
        )
        for i, entry in enumerate(results):
            if "ERROR" not in entry.get("predicted_answer", ""):
                entry["bert_f1"] = F1[i].item()

    # Save results CSV
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "eval_results.csv"
    fieldnames = [
        "question", "reference_answer", "predicted_answer", "topic",
        "rouge1_f1", "rougeL_f1", "bert_f1", "sources",
        "retrieval_time", "generation_time"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    valid = [r for r in results if r.get("rouge1_f1", 0) > 0 or r.get("bert_f1", 0) > 0]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if valid:
        avg_r1 = sum(r.get("rouge1_f1", 0) for r in valid) / len(valid)
        avg_rl = sum(r.get("rougeL_f1", 0) for r in valid) / len(valid)
        avg_bert = sum(r.get("bert_f1", 0) for r in valid) / len(valid)
        print(f"  Questions evaluated: {len(valid)}/{len(test_questions)}")
        print(f"  Avg ROUGE-1 F1:     {avg_r1:.4f}")
        print(f"  Avg ROUGE-L F1:     {avg_rl:.4f}")
        print(f"  Avg BERTScore F1:   {avg_bert:.4f}")

        # Failure analysis
        low_score = [r for r in valid if r.get("bert_f1", 0) < 0.5]
        high_score = [r for r in valid if r.get("bert_f1", 0) >= 0.7]
        print(f"\n  Successful answers (BERTScore ≥ 0.7): {len(high_score)}")
        print(f"  Weak answers (BERTScore < 0.5):       {len(low_score)}")

        if low_score:
            print("\n  --- Failure Analysis ---")
            for r in low_score:
                print(f"  Q: {r['question'][:60]}")
                print(f"    BERTScore: {r.get('bert_f1', 0):.3f}")
                print(f"    Possible cause: Retrieval missed relevant context or LLM hallucinated")
    else:
        print("  No valid results to analyze.")

    print(f"\n  Results saved to: {csv_path}")

    # Save detailed JSON report
    report_path = EVAL_DIR / "eval_report.json"
    report = {
        "summary": {
            "total_questions": len(test_questions),
            "evaluated": len(valid),
            "avg_rouge1_f1": avg_r1 if valid else 0,
            "avg_rougeL_f1": avg_rl if valid else 0,
            "avg_bert_f1": avg_bert if valid else 0,
            "successful": len(high_score) if valid else 0,
            "weak": len(low_score) if valid else 0,
        },
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Full report saved to: {report_path}")


if __name__ == "__main__":
    evaluate()
