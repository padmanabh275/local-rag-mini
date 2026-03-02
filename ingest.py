"""
Build a PyTorch vector index from .txt documents in docs/.
Chunks text, computes embeddings with sentence-transformers (PyTorch), and saves embeddings + metadata.
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
    # is_*_available: return False; get_*: return empty list/dict; rewrite_logs: return first arg
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
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    """Load all .txt files from docs_dir. Returns list of (filename, full_text)."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents = []
    for path in sorted(docs_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            print(f"Warning: skipped {path}: {e}")
            continue
        if not text:
            continue
        rel = path.relative_to(docs_dir)
        documents.append((str(rel), text))
    return documents


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Sliding-window chunks. Returns list of (chunk_text, start_char, end_char)."""
    if not text or chunk_size <= 0:
        return []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end >= len(text):
            break
        start += step
    return chunks


def build_chunks(
    documents: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Build list of chunk dicts with id, source, text, start_char, end_char."""
    chunks_meta = []
    chunk_id = 0
    for source, text in documents:
        for chunk_text_val, start_char, end_char in chunk_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ):
            chunks_meta.append({
                "id": chunk_id,
                "source": source,
                "text": chunk_text_val,
                "start_char": start_char,
                "end_char": end_char,
            })
            chunk_id += 1
    return chunks_meta


def main():
    parser = argparse.ArgumentParser(description="Build PyTorch embedding index from .txt docs")
    parser.add_argument("--docs_dir", default="docs", help="Directory containing .txt files")
    parser.add_argument("--output_dir", default="data", help="Directory for embeddings and metadata")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap between chunks")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda); default: auto")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)

    print("Loading documents...")
    documents = load_documents(docs_dir)
    if not documents:
        print("No .txt documents found. Exiting.")
        return

    print(f"Building chunks (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = build_chunks(documents, args.chunk_size, args.chunk_overlap)
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model: {DEFAULT_MODEL}")
    model = SentenceTransformer(DEFAULT_MODEL, device=device)

    print("Computing embeddings (PyTorch)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )
    # Ensure float32 and contiguous for storage
    embeddings = embeddings.float().contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "meta.json"

    # Save as NumPy to avoid pickle/torch.load class-instantiation issues
    np.save(embeddings_path, embeddings.cpu().numpy(), allow_pickle=False)

    meta = {
        "model_name": DEFAULT_MODEL,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "index_type": "pytorch",
        "embedding_dim": int(embeddings.shape[1]),
        "chunks": [{"id": c["id"], "source": c["source"], "text": c["text"]} for c in chunks],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Documents: {len(documents)}, Chunks: {len(chunks)}")
    print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    main()
