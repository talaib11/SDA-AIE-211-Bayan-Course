"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.search.index import PREPROC_VERSION, normalize_text


RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        """Load and validate the persisted search artefacts."""

        self.prefix = prefix

        manifest_path = Path(
            f"{prefix}_manifest.json"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Search manifest not found: {manifest_path}"
            )

        self.manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        # -----------------------------------------------------
        # Validate manifest contract
        # -----------------------------------------------------

        required = {
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        }

        missing = required - set(
            self.manifest
        )

        if missing:
            raise ValueError(
                "Search manifest is missing required keys: "
                + ", ".join(sorted(missing))
            )

        if (
            self.manifest["preproc_version"]
            != PREPROC_VERSION
        ):
            raise ValueError(
                "Search preprocessing version mismatch: "
                f'index={self.manifest["preproc_version"]}, '
                f"runtime={PREPROC_VERSION}"
            )

        index_path = Path(
            self.manifest.get(
                "index_path",
                f"{prefix}_index.faiss",
            )
        )

        metadata_path = Path(
            self.manifest.get(
                "metadata_path",
                f"{prefix}_metadata.jsonl",
            )
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Search metadata not found: {metadata_path}"
            )

        # -----------------------------------------------------
        # Load FAISS + metadata
        # -----------------------------------------------------

        self.index = faiss.read_index(
            str(index_path)
        )

        self.metadata = []

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if line:
                    self.metadata.append(
                        json.loads(line)
                    )

        expected_vectors = int(
            self.manifest["n_vectors"]
        )

        expected_dim = int(
            self.manifest["dim"]
        )

        if self.index.ntotal != expected_vectors:
            raise ValueError(
                "FAISS index vector count does not "
                "match manifest."
            )

        if len(self.metadata) != expected_vectors:
            raise ValueError(
                "Metadata count does not match manifest."
            )

        if self.index.d != expected_dim:
            raise ValueError(
                "FAISS dimension does not match manifest."
            )

        # -----------------------------------------------------
        # Load models pinned by the search artefact
        # -----------------------------------------------------

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            RERANKER_NAME
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        """Retrieve with bi-encoder then rerank with cross-encoder."""

        if k <= 0:
            return []

        if candidates <= 0:
            return []

        if not str(query).strip():
            return []

        # -----------------------------------------------------
        # Consistent query preprocessing
        # -----------------------------------------------------

        normalized_query = normalize_text(
            query
        )

        # -----------------------------------------------------
        # Bi-encoder query embedding
        # -----------------------------------------------------

        query_vector = self.encoder.encode(
            [normalized_query],
            convert_to_numpy=True,
        ).astype("float32")

        # Same normalization contract as corpus vectors.
        faiss.normalize_L2(
            query_vector
        )

        candidate_count = min(
            max(candidates, k),
            int(self.index.ntotal),
        )

        bi_scores, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        candidate_rows = []

        for bi_score, index_id in zip(
            bi_scores[0],
            indices[0],
        ):
            if index_id < 0:
                continue

            metadata = dict(
                self.metadata[int(index_id)]
            )

            candidate_rows.append(
                {
                    "index_id": int(index_id),
                    "bi_score": float(bi_score),
                    "metadata": metadata,
                }
            )

        if not candidate_rows:
            return []

        # -----------------------------------------------------
        # Cross-encoder reranking
        # -----------------------------------------------------

        pairs = [
            [
                normalized_query,
                row["metadata"].get(
                    "search_text",
                    row["metadata"].get(
                        "case_text",
                        "",
                    ),
                ),
            ]
            for row in candidate_rows
        ]

        ce_scores = self.reranker.predict(
            pairs
        )

        ce_scores = np.asarray(
            ce_scores
        ).reshape(-1)

        reranked = []

        for row, ce_score in zip(
            candidate_rows,
            ce_scores,
        ):
            item = dict(
                row["metadata"]
            )

            item["bi_score"] = row[
                "bi_score"
            ]

            item["score"] = float(
                ce_score
            )

            reranked.append(
                item
            )

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # Honest no-result threshold
        # -----------------------------------------------------

        filtered = [
            item
            for item in reranked
            if item["score"] >= min_score
        ]

        return filtered[:k]