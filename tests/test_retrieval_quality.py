"""
Retrieval-quality benchmark for the knowledge base.

Unlike test_store.py (which checks correctness — visibility filters, idempotency,
privacy), this file measures *how good* retrieval actually is: given a realistic
query, does the right document come back near the top? It builds a small fixture
corpus of paper summaries and markdown notes, runs a fixed set of golden queries
through the real search() pipeline (embeddings + re-ranking), and reports
hit-rate@5 and MRR@5.

The point is to make retrieval-accuracy changes measurable. Record the metrics
before a change, re-run after, and ratchet the assertion thresholds up as the
pipeline improves. It uses only local cached models (same policy as test_store.py),
so it runs as a normal unit test with no integration marker.

The corpus deliberately includes:
  - confusable near-neighbours (image classification vs image generation;
    "gradient descent" note vs "policy gradient" paper) to catch coarse matching
  - acronym / proper-noun targets (LoRA, Dr. Tanaka) that pure dense retrieval
    tends to miss — this is what hybrid BM25+RRF retrieval (see
    jarvis/kb/store.py::_hybrid_search, gated by [rag] hybrid) targets
  - author-name queries (e.g. "papers by Vaswani"), which only the
    embed_header prepended to every paper chunk can satisfy — the dense
    summary text itself never mentions an author's name

Runs with the default config (hybrid=True), the same as production.
"""

import uuid
from pathlib import Path

import pytest
from langchain_chroma import Chroma

from jarvis.kb.store import add_paper, add_texts, search

# Same gitignored store directory the other KB tests use (see conftest.py).
TEST_CHROMA_DIR = Path(__file__).parent / ".chroma"


# ── Fixture corpus ───────────────────────────────────────────────────────────

# Paper summaries: (source, title, dense_summary, authors). authors defaults to ""
# except where a golden query below specifically targets an author's name — the
# summary text alone never mentions authors, so the header embedded by
# add_paper() is the only thing that can make an author-name query hit.
# Stored via add_paper in summary mode (title + link + authors + summary → one or two chunks).
_PAPERS = [
    ("paper-transformers", "Attention Is All You Need",
     "We introduce the Transformer, a sequence model based entirely on self-attention "
     "that dispenses with recurrence and convolutions. Multi-head attention lets the "
     "model capture long-range dependencies between tokens in parallel.",
     "Ashish Vaswani, Noam Shazeer, Niki Parmar"),
    ("paper-cnn", "Deep Convolutional Networks for Image Classification",
     "A deep convolutional neural network classifies natural images into categories. "
     "Stacked convolutional filters and pooling layers learn a hierarchy of visual "
     "features from pixels to objects.", ""),
    ("paper-vit", "An Image is Worth 16x16 Words: Vision Transformers",
     "A vision transformer splits an image into fixed-size patches, embeds each patch, "
     "and applies a standard transformer encoder with self-attention over the patch "
     "sequence to classify the image, using no convolutions at all.", ""),
    ("paper-segmentation", "Semantic Segmentation with Encoder-Decoder Networks",
     "A fully convolutional encoder-decoder network assigns a class label to every pixel "
     "of an image, producing a dense segmentation mask instead of a single image-level "
     "category.", ""),
    ("paper-rl", "Policy Gradient Methods for Reinforcement Learning",
     "We study policy gradient algorithms that optimise an agent's expected reward by "
     "following the gradient of the policy. The method learns control policies through "
     "trial-and-error interaction with the environment.", ""),
    ("paper-lora", "LoRA: Low-Rank Adaptation of Large Language Models",
     "LoRA freezes the pretrained weights of a large language model and injects trainable "
     "low-rank matrices into each layer, drastically cutting the number of parameters "
     "needed to fine-tune the model for a downstream task.", "Edward Hu, Yelong Shen"),
    ("paper-diffusion", "Denoising Diffusion Probabilistic Models",
     "Diffusion models generate images by learning to reverse a gradual noising process. "
     "Starting from pure noise, the model iteratively denoises the sample until a "
     "realistic image emerges.", ""),
    ("paper-gnn", "Graph Neural Networks for Relational Data",
     "Graph neural networks propagate information along the edges of a graph, learning "
     "node representations that account for the structure of relationships between "
     "entities.", ""),
    ("paper-bert", "BERT: Pretraining Deep Bidirectional Transformers",
     "BERT pretrains a bidirectional transformer using a masked language modelling "
     "objective, predicting randomly masked tokens from their surrounding context to "
     "learn general-purpose language representations.", ""),
    ("paper-federated", "Privacy-Preserving Federated Learning",
     "Federated learning trains a shared model across many devices without moving raw "
     "data off each device, aggregating only model updates to preserve user privacy.", ""),
    ("paper-protein", "Predicting Three-Dimensional Protein Structure from Sequence",
     "A deep network predicts the folded three-dimensional structure of a protein "
     "directly from its amino-acid sequence, reaching near-experimental accuracy on "
     "structure prediction benchmarks.", ""),
    ("paper-scrna", "Clustering Cells from Single-Cell RNA Sequencing",
     "We cluster cells profiled by single-cell RNA sequencing into cell types based on "
     "their gene-expression signatures, revealing heterogeneity within a tissue sample.", ""),
]

# Markdown notes: (source, markdown_text). Stored via add_texts as full-text notes
# with real headers so section-aware chunking has something to work with.
_NOTES = [
    ("note-optimisers",
     "# Optimisation algorithms\n\n"
     "## Momentum\nMomentum accumulates a velocity vector across steps to smooth updates.\n\n"
     "## Adam\nAdam combines momentum with per-parameter adaptive learning rates and is "
     "the default optimiser for training most neural networks."),
    ("note-crispr",
     "# CRISPR knockout screen\n\n"
     "## Methods\nWe designed a guide-RNA library targeting 500 genes and transduced the "
     "cell line at low multiplicity.\n\n"
     "## Results\nThe screen identified twelve genes whose knockout reduced proliferation, "
     "with two hits validated by individual knockout."),
    ("note-bayesian",
     "# Bayesian inference notes\n\n"
     "## Priors\nThe choice of prior encodes belief before seeing data.\n\n"
     "## MCMC\nMarkov chain Monte Carlo draws samples from the posterior distribution when "
     "it cannot be computed in closed form."),
    ("note-docker",
     "# Docker deployment workflow\n\n"
     "Build the image, push it to the registry, and roll out the new container to the "
     "cluster with a health check before switching traffic."),
    ("note-tanaka",
     "# Meeting with Dr. Tanaka\n\n"
     "Dr. Tanaka proposed a collaboration on spatial transcriptomics and offered to share "
     "the imaging dataset from the lab."),
    ("note-vectordb",
     "# Vector databases\n\n"
     "A vector database stores embeddings and retrieves the nearest neighbours of a query "
     "vector, which is the backbone of semantic search and retrieval-augmented generation."),
    ("note-hparam",
     "# Hyperparameter tuning\n\n"
     "Grid search is exhaustive but expensive; random search and Bayesian optimisation "
     "find good configurations with far fewer trials."),
    ("note-pandas",
     "# Pandas data cleaning tricks\n\n"
     "Drop duplicate rows, fill missing values, and convert column dtypes before merging "
     "two data frames on a shared key."),
    ("note-causal",
     "# Causal inference\n\n"
     "A confounder influences both treatment and outcome; failing to adjust for it biases "
     "the estimated causal effect."),
    ("note-timeseries",
     "# Time-series forecasting\n\n"
     "Forecasting predicts future values of a series from its past, accounting for trend "
     "and seasonality in the historical data."),
]

# Golden queries: (query, expected_source). Mix of paraphrase, keyword/acronym,
# proper-noun, and section-specific queries.
_GOLDEN = [
    ("how does self-attention capture long range dependencies between tokens", "paper-transformers"),
    ("low-rank adaptation for fine-tuning large language models", "paper-lora"),
    ("LoRA", "paper-lora"),
    ("generating realistic images by reversing a noising process", "paper-diffusion"),
    ("policy gradient methods in reinforcement learning", "paper-rl"),
    ("predicting the folded 3D structure of a protein from its sequence", "paper-protein"),
    ("clustering cells from single-cell RNA sequencing data", "paper-scrna"),
    ("Adam optimiser and momentum for training neural networks", "note-optimisers"),
    ("results of the CRISPR knockout screen experiment", "note-crispr"),
    ("Markov chain Monte Carlo sampling from the posterior", "note-bayesian"),
    ("meeting notes with Dr. Tanaka about collaboration", "note-tanaka"),
    ("storing embeddings in a vector database for semantic search", "note-vectordb"),
    ("forecasting future values of a time series with trend and seasonality", "note-timeseries"),
    ("adjusting for confounders to estimate a causal effect", "note-causal"),
    # Confusable near-neighbours — several documents share vocabulary here.
    ("transformer applied to image patches with no convolutions", "paper-vit"),
    ("assigning a class label to every pixel of an image", "paper-segmentation"),
    ("classifying whole images with convolutional filters", "paper-cnn"),
    ("masked language model pretraining of a bidirectional encoder", "paper-bert"),
    # Terse keyword / acronym queries — the hardest for pure dense retrieval.
    ("BERT", "paper-bert"),
    ("aggregating model updates without sharing raw data across devices", "paper-federated"),
    ("adaptive per-parameter learning rate", "note-optimisers"),
    # Author-name queries — the summary text never mentions an author, so
    # only the embed_header (title + authors, prepended to every chunk) can
    # make these hit. This is the sentinel for the embed-header migration.
    ("papers by Vaswani", "paper-transformers"),
    ("paper written by Edward Hu on low-rank adaptation", "paper-lora"),
]

K = 5


# ── Metrics ──────────────────────────────────────────────────────────────────

def _hit_rate_at_k(ranked_by_query: list[tuple[str, list[str]]], k: int) -> float:
    """Fraction of queries whose expected source appears in the top k results."""
    hits = sum(1 for expected, sources in ranked_by_query if expected in sources[:k])
    return hits / len(ranked_by_query)


def _mrr_at_k(ranked_by_query: list[tuple[str, list[str]]], k: int) -> float:
    """Mean reciprocal rank of the expected source within the top k results."""
    total = 0.0
    for expected, sources in ranked_by_query:
        for rank, source in enumerate(sources[:k], start=1):
            if source == expected:
                total += 1.0 / rank
                break
    return total / len(ranked_by_query)


# ── Seeded corpus fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def quality_store(embeddings):
    """
    An isolated Chroma collection seeded once with the whole benchmark corpus.

    Module-scoped so the corpus is embedded a single time for all quality tests,
    mirroring the session-scoped real-model fixture in conftest.py.
    """
    TEST_CHROMA_DIR.mkdir(exist_ok=True)
    store = Chroma(
        collection_name=f"quality_{uuid.uuid4().hex[:8]}",
        embedding_function=embeddings,
        persist_directory=str(TEST_CHROMA_DIR),
    )
    for source, title, summary, authors in _PAPERS:
        add_paper({"link": source, "title": title, "authors": authors},
                  dense_summary=summary, store=store)
    for source, text in _NOTES:
        add_texts(content=text, doc_type="note", visibility="public",
                  source=source, store=store)
    yield store
    store.delete_collection()


def _run_golden(store) -> list[tuple[str, list[str]]]:
    """Run every golden query and return (expected_source, ranked_sources) pairs."""
    ranked_by_query = []
    for query, expected in _GOLDEN:
        results = search(query, n_results=K, store=store)
        sources = [doc.metadata["source"] for doc in results]
        ranked_by_query.append((expected, sources))
    return ranked_by_query


# ── Benchmarks ───────────────────────────────────────────────────────────────

def test_hit_rate_at_5(quality_store):
    """The expected document should land in the top 5 for the great majority of queries."""
    ranked = _run_golden(quality_store)
    hit_rate = _hit_rate_at_k(ranked, K)
    misses = [expected for expected, sources in ranked if expected not in sources[:K]]
    assert hit_rate >= 0.85, f"hit_rate@{K}={hit_rate:.3f}; missed: {misses}"


def test_mrr_at_5(quality_store):
    """The expected document should usually rank at or near the very top."""
    ranked = _run_golden(quality_store)
    mrr = _mrr_at_k(ranked, K)
    assert mrr >= 0.70, f"mrr@{K}={mrr:.3f}"
