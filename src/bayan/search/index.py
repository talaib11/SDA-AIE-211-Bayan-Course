"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/search/bayan_cases.csv"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

PREPROC_VERSION = "lab5-v1"


def normalize_text(text: str) -> str:
    """Apply the search preprocessing contract."""
    return " ".join(str(text).strip().split())


def build_index(
    prefix: str,
    limit: int | None = None,
    data_path: str = DATA_PATH,
    model_name: str = MODEL_NAME,
):
    """Build and persist the bilingual FAISS case index."""

    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load case corpus
    # ---------------------------------------------------------

    df = pd.read_csv(data_path)

    if limit is not None:
        df = df.head(limit).copy()
    else:
        df = df.copy()

    if df.empty:
        raise ValueError(
            "Cannot build a search index from an empty corpus."
        )

    if "case_text" not in df.columns:
        raise ValueError(
            "Case data must contain a 'case_text' column."
        )

    # Keep the display/original text while creating a separate
    # normalized search representation.
    df["search_text"] = (
        df["case_text"]
        .fillna("")
        .map(normalize_text)
    )

    # ---------------------------------------------------------
    # Encode corpus
    # ---------------------------------------------------------

    print(
        f"Loading bi-encoder: {model_name}"
    )

    encoder = SentenceTransformer(
        model_name
    )

    print(
        f"Encoding {len(df):,} cases..."
    )

    vectors = encoder.encode(
        df["search_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectors = vectors.astype("float32")

    # ---------------------------------------------------------
    # L2 normalization
    #
    # IndexFlatIP + normalized vectors = cosine similarity.
    # ---------------------------------------------------------

    faiss.normalize_L2(vectors)

    n_vectors, dim = vectors.shape

    # ---------------------------------------------------------
    # Build FAISS index
    # ---------------------------------------------------------

    index = faiss.IndexFlatIP(
        int(dim)
    )

    index.add(vectors)

    if index.ntotal != n_vectors:
        raise RuntimeError(
            "FAISS vector count does not match corpus size."
        )

    # ---------------------------------------------------------
    # Persist index
    # ---------------------------------------------------------

    index_path = Path(
        f"{prefix}_index.faiss"
    )

    metadata_path = Path(
        f"{prefix}_metadata.jsonl"
    )

    manifest_path = Path(
        f"{prefix}_manifest.json"
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    # ---------------------------------------------------------
    # Persist metadata
    # ---------------------------------------------------------

    metadata_columns = [
        column
        for column in [
            "case_id",
            "lang",
            "topic",
            "case_text",
            "resolution",
            "search_text",
        ]
        if column in df.columns
    ]

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in df[
            metadata_columns
        ].to_dict(orient="records"):

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Persist versioned manifest
    # ---------------------------------------------------------

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(n_vectors),
        "dim": int(dim),
        "metric": "cosine",
        "normalised": True,
        "index_type": "IndexFlatIP",
        "data_path": str(data_path),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("FAISS index built successfully")
    print("Vectors:", n_vectors)
    print("Dimension:", dim)
    print("Index:", index_path)
    print("Metadata:", metadata_path)
    print("Manifest:", manifest_path)

    return manifest