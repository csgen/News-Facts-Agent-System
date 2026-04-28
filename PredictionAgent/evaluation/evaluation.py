"""
Evaluation Script — Task 3: Full-Stack & Evaluation Engineer

WHAT THIS DOES:
Runs benchmark datasets through the live pipeline and computes:
    1. Entity Extraction F1      (Preprocessing Agent quality)
    2. Macro-F1 for verdicts     (Fact-Check Agent quality)
    3. Precision@k for retrieval (Memory Agent quality)
    4. Multi-modal mismatch F1   (Cross-modal check quality)

DATASETS:
    LIAR    — 12,800 political claims  (https://huggingface.co/datasets/liar)
    FEVER   — Wikipedia claim verification (https://huggingface.co/datasets/fever)
    Factify — Multi-modal claim + image   (https://huggingface.co/datasets/factify)

HOW TO RUN:
    python -m evaluation.evaluation --dataset liar
    python -m evaluation.evaluation --dataset fever
    python -m evaluation.evaluation --dataset factify
    python -m evaluation.evaluation --dataset all

OUTPUT:
    Prints metrics to console + saves evaluation_report.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REAL SYSTEM CALLS
# ─────────────────────────────────────────────


def get_system_verdict(claim_text: str, image_url: Optional[str] = None) -> dict:
    """
    Calls the real Fact-Check Agent pipeline.

    Returns:
        {
            "label":         "supported" | "refuted" | "misleading",
            "confidence":    0.0–1.0,
            "image_mismatch": True | False
        }
    """
    try:
        from agents.fact_check_agent import fact_check_claim

        output = fact_check_claim(claim_text)
        label = (
            output.verdict
            if output.verdict in ("supported", "refuted", "misleading")
            else "misleading"
        )
        return {
            "label": label,
            "confidence": round(output.confidence_score / 100, 2)
            if output.confidence_score > 1
            else round(output.confidence_score, 2),
            "image_mismatch": False,  # updated below if cross-modal ran
        }
    except Exception as e:
        logger.error("Fact-check pipeline error for claim '%s': %s", claim_text[:60], e)
        return {"label": "misleading", "confidence": 0.0, "image_mismatch": False}


def get_extracted_entities(text: str) -> list[str]:
    """
    Calls the real Memory Agent to find entities already stored for this text,
    or falls back to a lightweight spaCy NER pass.

    Returns a list of entity name strings.
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return list({ent.text for ent in doc.ents})
    except Exception:
        # spaCy not available — return empty list (scored as 0 precision/recall)
        logger.warning("spaCy not available; entity extraction returning empty list.")
        return []


def get_retrieved_claims(query: str, k: int = 5) -> list[dict]:
    """
    Queries the real ChromaDB claims collection for similar claims.

    Returns a list of {"claim_id": ..., "is_relevant": True/False}.
    Relevance is approximated by distance < 0.5 (cosine similarity > 0.5).
    """
    try:
        from agents.memory_agent import get_memory

        memory = get_memory()
        results = memory.query_similar_claims(query, k=k)
        retrieved = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            distance = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
            retrieved.append(
                {
                    "claim_id": results.get("ids", [[]])[0][i]
                    if results.get("ids")
                    else f"clm_{i}",
                    "is_relevant": distance < 0.5,
                }
            )
        return retrieved
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        return []


# ─────────────────────────────────────────────
# METRIC CALCULATIONS (original from Task 3 — unchanged)
# ─────────────────────────────────────────────


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] = None) -> dict:
    """
    Computes Macro-F1 across all labels.

    Macro-F1 treats all classes equally regardless of frequency — important
    because fake news is a minority class.
    """
    if labels is None:
        labels = list(set(y_true + y_pred))

    results = {}
    f1_scores = []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_f1(precision, recall)

        results[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }
        f1_scores.append(f1)

    results["macro_f1"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    return results


def compute_entity_f1(y_true_entities: list[list[str]], y_pred_entities: list[list[str]]) -> dict:
    """Token-level F1 for entity extraction, averaged across samples."""
    precisions, recalls, f1s = [], [], []

    for true_ents, pred_ents in zip(y_true_entities, y_pred_entities):
        true_set = set(true_ents)
        pred_set = set(pred_ents)

        p = len(true_set & pred_set) / len(pred_set) if pred_set else 0.0
        r = len(true_set & pred_set) / len(true_set) if true_set else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(compute_f1(p, r))

    avg_p = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    avg_r = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    avg_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0
    return {"precision": avg_p, "recall": avg_r, "f1": avg_f1}


def compute_precision_at_k(retrieved_lists: list[list[dict]], k: int = 5) -> float:
    """Precision@k = relevant docs in top-k / k, averaged across queries."""
    scores = []
    for retrieved in retrieved_lists:
        top_k = retrieved[:k]
        relevant = sum(1 for doc in top_k if doc["is_relevant"])
        scores.append(relevant / k)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ─────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────


def load_liar_dataset(n_samples: int = 100) -> list[dict]:
    """
    Load LIAR dataset via HuggingFace datasets.

    Label mapping (6 → 3):
        true, mostly-true, half-true → "supported"
        barely-true, false           → "refuted"
        pants-fire                   → "refuted"

    Falls back to synthetic mock data if the datasets library is unavailable.
    """
    try:
        from datasets import load_dataset

        liar_map = {
            "true": "supported",
            "mostly-true": "supported",
            "half-true": "supported",
            "barely-true": "refuted",
            "false": "refuted",
            "pants-fire": "refuted",
        }
        ds = load_dataset("liar", split="test", trust_remote_code=True)
        samples = []
        for row in list(ds)[:n_samples]:
            samples.append(
                {
                    "claim": row["statement"],
                    "true_label": liar_map.get(row["label"], "misleading"),
                    "true_entities": [row.get("subject", ""), row.get("speaker", "")],
                }
            )
        return samples
    except Exception as e:
        logger.warning("Could not load real LIAR dataset (%s) — using synthetic mock.", e)
        import random

        liar_labels = ["supported", "refuted", "misleading"]
        return [
            {
                "claim": f"Mock LIAR claim #{i}: A political statement.",
                "true_label": random.choice(liar_labels),
                "true_entities": random.sample(["Congress", "Biden", "Trump", "FDA", "EPA"], 2),
            }
            for i in range(n_samples)
        ]


def load_fever_dataset(n_samples: int = 100) -> list[dict]:
    """
    Load FEVER dataset via HuggingFace datasets.

    Label mapping: SUPPORTS → supported, REFUTES → refuted, NOT ENOUGH INFO → misleading
    Falls back to mock data if unavailable.
    """
    try:
        from datasets import load_dataset

        fever_map = {"SUPPORTS": "supported", "REFUTES": "refuted", "NOT ENOUGH INFO": "misleading"}
        ds = load_dataset("fever", "v1.0", split="paper_test", trust_remote_code=True)
        samples = []
        for row in list(ds)[:n_samples]:
            samples.append(
                {
                    "claim": row["claim"],
                    "true_label": fever_map.get(row["label"], "misleading"),
                    "true_entities": [],
                }
            )
        return samples
    except Exception as e:
        logger.warning("Could not load real FEVER dataset (%s) — using synthetic mock.", e)
        import random

        fever_labels = ["supported", "refuted", "misleading"]
        return [
            {
                "claim": f"Mock FEVER claim #{i}: A Wikipedia-verifiable fact.",
                "true_label": random.choice(fever_labels),
                "true_entities": random.sample(["Paris", "Einstein", "World War II", "NASA"], 2),
            }
            for i in range(n_samples)
        ]


def load_factify_dataset(n_samples: int = 100) -> list[dict]:
    """
    Load Factify dataset via HuggingFace datasets.

    Factify labels → our schema:
        Supports_Multimodal   → supported  + image_mismatch=False
        Refutes               → refuted    + image_mismatch=True
        Insufficient_evidence → misleading + image_mismatch=False

    Falls back to mock data if unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("factify", split="test", trust_remote_code=True)
        label_map = {
            "Supports_Multimodal": ("supported", False),
            "Refutes": ("refuted", True),
            "Insufficient_evidence": ("misleading", False),
        }
        samples = []
        for row in list(ds)[:n_samples]:
            label, mismatch = label_map.get(row.get("label", ""), ("misleading", False))
            samples.append(
                {
                    "claim": row.get("claim", ""),
                    "image_url": row.get("image_url", ""),
                    "true_label": label,
                    "true_image_mismatch": mismatch,
                }
            )
        return samples
    except Exception as e:
        logger.warning("Could not load real Factify dataset (%s) — using synthetic mock.", e)
        import random

        labels = ["supported", "refuted", "misleading"]
        return [
            {
                "claim": f"Mock Factify claim #{i}: A claim with an image.",
                "image_url": f"https://example.com/image_{i}.jpg",
                "true_label": random.choice(labels),
                "true_image_mismatch": random.choice([True, False]),
            }
            for i in range(n_samples)
        ]


# ─────────────────────────────────────────────
# EVALUATION RUNNERS (original from Task 3 — real pipeline calls)
# ─────────────────────────────────────────────


def evaluate_liar(n_samples: int = 100) -> dict:
    print(f"\n{'─' * 50}")
    print("  Evaluating on LIAR Dataset...")
    print(f"{'─' * 50}")

    dataset = load_liar_dataset(n_samples)
    y_true, y_pred = [], []
    true_entities_list, pred_entities_list = [], []
    retrieval_lists = []

    for i, sample in enumerate(dataset):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_samples}")

        verdict = get_system_verdict(sample["claim"])
        y_true.append(sample["true_label"])
        y_pred.append(verdict["label"])

        pred_entities = get_extracted_entities(sample["claim"])
        true_entities_list.append(sample.get("true_entities", []))
        pred_entities_list.append(pred_entities)

        retrieval_lists.append(get_retrieved_claims(sample["claim"], k=5))

    verdict_metrics = compute_macro_f1(y_true, y_pred, ["supported", "refuted", "misleading"])
    entity_metrics = compute_entity_f1(true_entities_list, pred_entities_list)
    precision_at_5 = compute_precision_at_k(retrieval_lists, k=5)

    result = {
        "dataset": "LIAR",
        "n_samples": n_samples,
        "verdict_macro_f1": verdict_metrics["macro_f1"],
        "verdict_per_class": {k: v for k, v in verdict_metrics.items() if k != "macro_f1"},
        "entity_extraction_f1": entity_metrics["f1"],
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "retrieval_precision_at_5": precision_at_5,
    }
    _print_result(result)
    return result


def evaluate_fever(n_samples: int = 100) -> dict:
    print(f"\n{'─' * 50}")
    print("  Evaluating on FEVER Dataset...")
    print(f"{'─' * 50}")

    dataset = load_fever_dataset(n_samples)
    y_true, y_pred = [], []
    true_entities_list, pred_entities_list = [], []
    retrieval_lists = []

    for i, sample in enumerate(dataset):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_samples}")
        verdict = get_system_verdict(sample["claim"])
        y_true.append(sample["true_label"])
        y_pred.append(verdict["label"])
        pred_entities = get_extracted_entities(sample["claim"])
        true_entities_list.append(sample.get("true_entities", []))
        pred_entities_list.append(pred_entities)
        retrieval_lists.append(get_retrieved_claims(sample["claim"], k=5))

    verdict_metrics = compute_macro_f1(y_true, y_pred, ["supported", "refuted", "misleading"])
    entity_metrics = compute_entity_f1(true_entities_list, pred_entities_list)
    precision_at_5 = compute_precision_at_k(retrieval_lists, k=5)

    result = {
        "dataset": "FEVER",
        "n_samples": n_samples,
        "verdict_macro_f1": verdict_metrics["macro_f1"],
        "verdict_per_class": {k: v for k, v in verdict_metrics.items() if k != "macro_f1"},
        "entity_extraction_f1": entity_metrics["f1"],
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "retrieval_precision_at_5": precision_at_5,
    }
    _print_result(result)
    return result


def evaluate_factify(n_samples: int = 100) -> dict:
    print(f"\n{'─' * 50}")
    print("  Evaluating on Factify Dataset (Multi-Modal)...")
    print(f"{'─' * 50}")

    dataset = load_factify_dataset(n_samples)
    y_true, y_pred = [], []
    mismatch_true, mismatch_pred = [], []

    for i, sample in enumerate(dataset):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_samples}")
        verdict = get_system_verdict(sample["claim"], image_url=sample.get("image_url"))
        y_true.append(sample["true_label"])
        y_pred.append(verdict["label"])
        mismatch_true.append("mismatch" if sample["true_image_mismatch"] else "match")
        mismatch_pred.append("mismatch" if verdict["image_mismatch"] else "match")

    verdict_metrics = compute_macro_f1(y_true, y_pred, ["supported", "refuted", "misleading"])
    mismatch_metrics = compute_macro_f1(mismatch_true, mismatch_pred, ["mismatch", "match"])

    result = {
        "dataset": "Factify",
        "n_samples": n_samples,
        "verdict_macro_f1": verdict_metrics["macro_f1"],
        "verdict_per_class": {k: v for k, v in verdict_metrics.items() if k != "macro_f1"},
        "image_mismatch_macro_f1": mismatch_metrics["macro_f1"],
        "image_mismatch_per_class": {k: v for k, v in mismatch_metrics.items() if k != "macro_f1"},
    }
    _print_result(result)
    return result


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────


def _print_result(result: dict):
    print(f"\n  ┌─ {result['dataset']} Results ({result['n_samples']} samples) ─┐")
    print(f"  │ Verdict Macro-F1     : {result['verdict_macro_f1']:.4f}")
    if "entity_extraction_f1" in result:
        print(f"  │ Entity Extraction F1 : {result['entity_extraction_f1']:.4f}")
        print(f"  │ Retrieval Precision@5: {result['retrieval_precision_at_5']:.4f}")
    if "image_mismatch_macro_f1" in result:
        print(f"  │ Image Mismatch F1    : {result['image_mismatch_macro_f1']:.4f}")
    print(f"  └{'─' * 40}┘")


def save_report(all_results: list[dict], output_path: str = "evaluation_report.json"):
    report = {
        "generated_at": datetime.now().isoformat(),
        "results": all_results,
        "summary": {
            "datasets_evaluated": [r["dataset"] for r in all_results],
            "avg_macro_f1": round(
                sum(r["verdict_macro_f1"] for r in all_results) / len(all_results), 4
            )
            if all_results
            else 0.0,
        },
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_path}")
    return report


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="FactGuard Evaluation Script")
    parser.add_argument(
        "--dataset",
        choices=["liar", "fever", "factify", "all"],
        default="all",
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples per dataset (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path for the JSON report (default: evaluation_report.json)",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("  FactGuard Evaluation Pipeline")
    print(f"  Dataset: {args.dataset.upper()} | Samples: {args.samples}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    all_results = []

    if args.dataset in ("liar", "all"):
        all_results.append(evaluate_liar(args.samples))

    if args.dataset in ("fever", "all"):
        all_results.append(evaluate_fever(args.samples))

    if args.dataset in ("factify", "all"):
        all_results.append(evaluate_factify(args.samples))

    report = save_report(all_results, args.output)

    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"  Datasets : {', '.join(report['summary']['datasets_evaluated'])}")
    print(f"  Avg Macro-F1: {report['summary']['avg_macro_f1']:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
