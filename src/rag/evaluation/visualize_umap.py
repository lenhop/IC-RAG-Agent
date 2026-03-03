"""
UMAP Embedding Visualization for RAG Evaluation.

Per RAG_EVALUATION_IMPLEMENTATION_PLAN.md Phase 2.1.
Generates scatter plots of query and chunk embeddings in 2D UMAP space.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# Color coding per spec: Red=queries, Green=relevant, Gray=irrelevant
COLOR_QUERY = "rgb(220, 53, 69)"
COLOR_RELEVANT = "rgb(40, 167, 69)"
COLOR_IRRELEVANT = "rgb(108, 117, 125)"


def _get_relevant_texts(case: Dict[str, Any]) -> List[str]:
    """Extract relevant chunk texts from test case (contexts or ground_truth)."""
    if "contexts" in case and case["contexts"]:
        ctx = case["contexts"]
        return ctx if isinstance(ctx, list) else [str(ctx)]
    if case.get("ground_truth"):
        return [str(case["ground_truth"])[:500]]
    return []


def _sample_irrelevant_from_collection(
    collection: Any,
    relevant_texts: List[str],
    max_irrelevant: int = 50,
) -> tuple[List[List[float]], List[str]]:
    """Sample documents from Chroma collection, excluding relevant chunks.

    Args:
        collection: Chroma Collection instance.
        relevant_texts: Texts to exclude (relevant chunks).
        max_irrelevant: Max number of irrelevant chunks to sample.

    Returns:
        (embeddings, documents) for irrelevant chunks.
    """
    relevant_set = {t.strip().lower()[:200] for t in relevant_texts if t}
    try:
        result = collection.get(
            limit=max_irrelevant * 2,
            include=["documents", "embeddings"],
        )
    except Exception:
        return [], []

    docs = result.get("documents") or []
    embs = result.get("embeddings") or []

    if not docs or not embs:
        return [], []

    # Chroma returns ids, documents, embeddings as parallel lists
    ids = result.get("ids") or []
    embeddings_out = []
    docs_out = []

    for i, (doc, emb) in enumerate(zip(docs, embs)):
        if emb is None:
            continue
        doc_str = doc if isinstance(doc, str) else str(doc)
        if doc_str.strip().lower()[:200] in relevant_set:
            continue
        embeddings_out.append(emb)
        docs_out.append(doc_str[:100] + "..." if len(doc_str) > 100 else doc_str)
        if len(embeddings_out) >= max_irrelevant:
            break

    return embeddings_out, docs_out


def generate_umap_plot(
    test_cases: List[Dict[str, Any]],
    pipeline: Any,
    output_path: Optional[Path] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_irrelevant: int = 50,
    outlier_threshold: float = 2.0,
) -> str:
    """Generate UMAP 2D scatter plot of query and chunk embeddings.

    Color coding:
        Red: Query embeddings
        Green: Relevant chunks (from contexts or ground_truth)
        Gray: Irrelevant chunks (sampled from collection)

    Args:
        test_cases: List of test case dicts with question, ground_truth, optional contexts.
        pipeline: RAGPipeline with embedder and vector_store.
        output_path: Directory or file path for output. Default: tests/evaluation/
        n_neighbors: UMAP n_neighbors (default 15).
        min_dist: UMAP min_dist (default 0.1).
        max_irrelevant: Max irrelevant chunks to sample (default 50).
        outlier_threshold: Distance threshold for outlier annotation (default 2.0).

    Returns:
        Path to generated HTML file.
    """
    import numpy as np

    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly required for UMAP visualization. Run: pip install plotly"
        ) from e

    # Prefer UMAP; fallback to PCA when umap-learn not available (e.g. LLVM build issues)
    reducer = None
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    from ai_toolkit.chroma import get_chroma_collection

    embedder = pipeline.embedder
    collection = None
    try:
        collection = get_chroma_collection(pipeline.vector_store)
    except Exception:
        pass

    # Collect embeddings and labels
    all_vectors = []
    all_labels = []
    all_types = []
    all_ids = []
    query_2d_indices = []
    query_case_map = []

    for idx, case in enumerate(test_cases):
        question = case.get("question", "")
        case_id = case.get("id", f"case_{idx}")

        if not question:
            continue

        # Embed query
        try:
            q_vec = embedder.embed_query(question)
        except Exception:
            continue

        start_idx = len(all_vectors)
        all_vectors.append(q_vec)
        all_labels.append(question[:80] + "..." if len(question) > 80 else question)
        all_types.append("query")
        all_ids.append(case_id)
        query_2d_indices.append(start_idx)
        query_case_map.append((case_id, question, case))

        # Embed relevant chunks
        relevant_texts = _get_relevant_texts(case)
        for rtext in relevant_texts:
            if not rtext.strip():
                continue
            try:
                r_vec = embedder.embed_query(rtext)
            except Exception:
                r_vec = embedder.embed_documents([rtext])[0]
            all_vectors.append(r_vec)
            all_labels.append(rtext[:80] + "..." if len(rtext) > 80 else rtext)
            all_types.append("relevant")
            all_ids.append(case_id)

    # Sample irrelevant chunks from collection
    all_relevant = []
    for c in test_cases:
        all_relevant.extend(_get_relevant_texts(c))

    if collection:
        irr_embs, irr_docs = _sample_irrelevant_from_collection(
            collection, all_relevant, max_irrelevant
        )
        for emb, doc in zip(irr_embs, irr_docs):
            all_vectors.append(emb)
            all_labels.append(doc)
            all_types.append("irrelevant")
            all_ids.append("")

    if not all_vectors:
        raise ValueError("No embeddings to visualize")

    # Reduce to 2D (UMAP or PCA)
    embedding_2d = reducer.fit_transform(all_vectors)

    # Build Plotly figure
    fig = go.Figure()

    for point_type, color, name in [
        ("query", COLOR_QUERY, "Query"),
        ("relevant", COLOR_RELEVANT, "Relevant chunk"),
        ("irrelevant", COLOR_IRRELEVANT, "Irrelevant chunk"),
    ]:
        mask = [t == point_type for t in all_types]
        if not any(mask):
            continue
        x = [embedding_2d[i][0] for i in range(len(mask)) if mask[i]]
        y = [embedding_2d[i][1] for i in range(len(mask)) if mask[i]]
        labels = [all_labels[i] for i in range(len(mask)) if mask[i]]
        ids = [all_ids[i] for i in range(len(mask)) if mask[i]]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(size=10 if point_type == "query" else 6, color=color),
                text=[f"{i}: {l}" for i, l in zip(ids, labels)],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Annotate outliers: queries far from their relevant chunks
    _annotate_outliers(
        fig,
        embedding_2d,
        all_types,
        all_ids,
        all_labels,
        query_2d_indices,
        query_case_map,
        outlier_threshold,
    )

    reducer_name = "UMAP" if "umap" in str(type(reducer)).lower() else "PCA"
    fig.update_layout(
        title=f"{reducer_name} Embedding Space: Queries vs Chunks",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
        margin=dict(r=200),
    )

    # Resolve output path
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "tests" / "evaluation"
    output_path = Path(output_path)
    if output_path.suffix != ".html":
        output_path = output_path / "umap_visualization.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = str(output_path)
    fig.write_html(html_path)

    # Also write PNG if kaleido available
    png_path = output_path.with_suffix(".png")
    try:
        fig.write_image(str(png_path))
    except Exception:
        pass

    return html_path


def annotate_outliers(
    fig: Any,
    embedding_2d: List[List[float]],
    all_types: List[str],
    all_ids: List[str],
    all_labels: List[str],
    query_indices: List[int],
    query_case_map: List[tuple],
    threshold: float = 2.0,
) -> None:
    """Highlight queries far from relevant chunks (public API per spec).

    Call this with data from generate_umap_plot if you need to re-annotate
    with a different threshold. Normally annotation is done inside generate_umap_plot.
    """
    _annotate_outliers(
        fig, embedding_2d, all_types, all_ids, all_labels,
        query_indices, query_case_map, threshold,
    )


def _annotate_outliers(
    fig: Any,
    embedding_2d: List[List[float]],
    all_types: List[str],
    all_ids: List[str],
    all_labels: List[str],
    query_indices: List[int],
    query_case_map: List[tuple],
    threshold: float,
) -> None:
    """Highlight queries far from relevant chunks and add hover text.

    Args:
        fig: Plotly figure to annotate.
        embedding_2d: 2D coordinates for all points.
        all_types: Type of each point (query, relevant, irrelevant).
        all_ids: ID for each point.
        all_labels: Label for each point.
        query_indices: Indices of query points.
        query_case_map: (case_id, question, case) for each query.
        threshold: Distance threshold for outlier (in 2D UMAP space).
    """
    import numpy as np

    outliers = []
    for q_idx in query_indices:
        if q_idx >= len(embedding_2d):
            continue
        q_pt = np.array(embedding_2d[q_idx])
        case_id = all_ids[q_idx] if q_idx < len(all_ids) else ""

        # Find min distance to any relevant chunk for this case
        min_dist = float("inf")
        for i, t in enumerate(all_types):
            if t != "relevant":
                continue
            if all_ids[i] != case_id and case_id:
                continue
            r_pt = np.array(embedding_2d[i])
            d = np.linalg.norm(q_pt - r_pt)
            min_dist = min(min_dist, d)

        if min_dist != float("inf") and min_dist > threshold:
            outliers.append((q_idx, min_dist, case_id, all_labels[q_idx]))

    if not outliers:
        return

    # Add outlier annotations
    annotations = list(fig.layout.annotations or [])
    for q_idx, dist, case_id, label in outliers:
        x, y = embedding_2d[q_idx][0], embedding_2d[q_idx][1]
        annotations.append(
            dict(
                x=x,
                y=y,
                text=f"OUTLIER: {case_id} (d={dist:.2f})",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(color="red", size=10),
            )
        )
    fig.update_layout(annotations=annotations)
