"""
Query the PyTorch embedding index: retrieve top-k chunks for a question and return a grounded short answer.
"""

# PyTorch-only: avoid transformers loading TensorFlow/Keras (stub integration_utils)
import os
import sys
import types

os.environ["TRANSFORMERS_NO_TF"] = "1"

def _make_stub_module():
    stub = types.ModuleType("transformers.integrations.integration_utils")
    _noop_class = type("_Noop", (), {})
    _noop_fn = lambda *a, **k: None
    _noop_dict = {}
    def is_available(*a, **k):
        return False
    def get_integrations(*a, **k):
        return []
    def get_callbacks(*a, **k):
        return []
    def rewrite_logs(d):
        return d if d is not None else {}
    stub.CodeCarbonCallback = _noop_class
    stub.WandbCallback = _noop_class
    stub.AzureMLCallback = _noop_class
    stub.ClearMLCallback = _noop_class
    stub.CometCallback = _noop_class
    stub.DagsHubCallback = _noop_class
    stub.DVCLiveCallback = _noop_class
    stub.FlyteCallback = _noop_class
    stub.MLflowCallback = _noop_class
    stub.NeptuneCallback = _noop_class
    stub.NeptuneMissingConfiguration = _noop_class
    stub.SwanLabCallback = _noop_class
    stub.TensorBoardCallback = _noop_class
    stub.INTEGRATION_TO_CALLBACK = _noop_dict
    stub.hp_params = _noop_dict
    stub.rewrite_logs = rewrite_logs
    stub.get_available_reporting_integrations = get_integrations
    stub.get_reporting_integration_callbacks = get_callbacks
    for attr in ("is_azureml_available", "is_clearml_available", "is_codecarbon_available", "is_comet_available",
                 "is_dagshub_available", "is_dvclive_available", "is_flyte_deck_standard_available", "is_flytekit_available",
                 "is_mlflow_available", "is_neptune_available", "is_optuna_available", "is_ray_available",
                 "is_ray_tune_available", "is_sigopt_available", "is_swanlab_available", "is_tensorboard_available", "is_wandb_available"):
        setattr(stub, attr, is_available)
    for attr in ("run_hp_search_optuna", "run_hp_search_ray", "run_hp_search_sigopt", "run_hp_search_wandb"):
        setattr(stub, attr, _noop_fn)
    return stub

sys.modules["transformers.integrations.integration_utils"] = _make_stub_module()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
MAX_ANSWER_CONTEXT_CHARS = 1500
SNIPPET_LEN = 200


def load_index_and_meta(data_dir: str) -> tuple[torch.Tensor, dict, SentenceTransformer]:
    """Load embeddings.npy, meta.json, and embedding model. Returns (embeddings, meta, model)."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    embeddings_path = data_dir / "embeddings.npy"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run ingest.py first.")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}. Run ingest.py first.")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Load from .npy to avoid pickle/torch.load class-instantiation issues
    arr = np.load(embeddings_path, allow_pickle=False)
    embeddings = torch.from_numpy(arr).float()

    model_name = meta.get("model_name", DEFAULT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return embeddings, meta, model


def retrieve(
    question: str,
    embeddings: torch.Tensor,
    meta: dict,
    model: SentenceTransformer,
    k: int = 3,
) -> list[dict]:
    """Return top-k chunks with source and text (L2 distance, same as FAISS IndexFlatL2)."""
    chunks_meta = meta["chunks"]
    device = next(model.parameters()).device
    emb_device = embeddings.device
    if emb_device != device:
        embeddings = embeddings.to(device)

    q_emb = model.encode(
        [question],
        convert_to_tensor=True,
        device=device,
    ).float()

    k = min(k, len(chunks_meta))
    # L2 distances: (1, n_chunks)
    distances = torch.cdist(q_emb, embeddings, p=2).squeeze(0)
    scores, indices = torch.topk(distances, k, largest=False)

    results = []
    for i in range(k):
        idx = indices[i].item()
        c = chunks_meta[idx]
        results.append({
            "rank": i + 1,
            "source": c["source"],
            "text": c["text"],
            "distance": float(scores[i].item()),
        })
    return results


def _sentences(text: str) -> list[str]:
    """Simple sentence split by period, question mark, exclamation."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Build a short answer from retrieved chunks only. No hallucination.
    Uses first sentences that relate to question keywords, then caps length.
    """
    combined = " ".join(c["text"] for c in chunks)
    combined = combined[:MAX_ANSWER_CONTEXT_CHARS]
    sentences = _sentences(combined)
    if not sentences:
        return "Based on the documents, no matching content was found for this question."

    q_lower = question.lower()
    q_words = set(re.findall(r"\w+", q_lower)) - {"what", "how", "when", "where", "why", "which", "who", "is", "are", "the", "a", "an"}

    selected = []
    for s in sentences:
        s_lower = s.lower()
        if q_words and any(w in s_lower for w in q_words):
            selected.append(s)
        if len(selected) >= 4:
            break
    if not selected:
        selected = sentences[:3]
    else:
        selected = selected[:4]

    answer_text = " ".join(selected)
    if not answer_text.endswith("."):
        answer_text = answer_text.rstrip() + "."
    return "Based on the documents, " + answer_text


def main():
    parser = argparse.ArgumentParser(description="Query the RAG index and get a grounded answer")
    parser.add_argument("--question", "-q", default=None, help="Question to ask (or omit for interactive)")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--data_dir", default="data", help="Directory containing embeddings.npy and meta.json")
    args = parser.parse_args()

    question = args.question
    if question is None or question.strip() == "":
        question = input("Question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    data_dir = Path(args.data_dir)
    print("Loading index and model...")
    embeddings, meta, model = load_index_and_meta(data_dir)
    print("Retrieving...")
    chunks = retrieve(question, embeddings, meta, model, k=args.k)

    print("\n--- Top retrieved chunks ---\n")
    for c in chunks:
        snippet = c["text"]
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
        print(f"[{c['rank']}] Source: {c['source']}")
        print(f"    {snippet}\n")

    answer = generate_answer(question, chunks)
    print("--- Answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
