"""
Evaluation Report Generator - Phase 2.2.

Generates HTML report with retrieval metrics, generation metrics,
UMAP visualization, and actionable issue list.
"""

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Targets from RAG_EVALUATION_IMPLEMENTATION_PLAN.md
TARGET_FAITHFULNESS = 0.85
TARGET_RELEVANCE = 4.0
TARGET_RECALL_MIN = 0.4  # Reasonable minimum for FAQ evaluation


def _build_issue_list(
    retrieval_results: Optional[Dict[str, Any]],
    generation_results: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build categorized issue list from evaluation results.

    Returns:
        List of dicts with keys: type, id, message, severity.
    """
    issues: List[Dict[str, str]] = []

    if retrieval_results and "per_case" in retrieval_results:
        for case in retrieval_results["per_case"]:
            case_id = case.get("id", "?")
            recall = case.get("recall", 0)
            if recall == 0.0:
                issues.append({
                    "type": "retrieval",
                    "id": case_id,
                    "message": f"Recall@5 = 0.0 - no relevant chunks retrieved",
                    "severity": "high",
                })
            elif recall < TARGET_RECALL_MIN:
                issues.append({
                    "type": "retrieval",
                    "id": case_id,
                    "message": f"Recall@5 = {recall:.2f} (below {TARGET_RECALL_MIN})",
                    "severity": "medium",
                })

    if generation_results and "per_case" in generation_results:
        for case in generation_results["per_case"]:
            case_id = case.get("id", "?")
            if not case.get("is_faithful", True):
                reasoning = case.get("faithfulness_reasoning", "")[:100]
                issues.append({
                    "type": "faithfulness",
                    "id": case_id,
                    "message": f"Unfaithful: {reasoning}...",
                    "severity": "high",
                })
            score = case.get("relevance_score", 5)
            if score < TARGET_RELEVANCE:
                reasoning = case.get("relevance_reasoning", "")[:80]
                issues.append({
                    "type": "relevance",
                    "id": case_id,
                    "message": f"Score {score}/5 - {reasoning}...",
                    "severity": "medium",
                })

    return issues


def _html_escape(s: str) -> str:
    """Escape string for HTML."""
    return html.escape(str(s)) if s else ""


def generate_html_report(
    retrieval_results: Optional[Dict[str, Any]] = None,
    generation_results: Optional[Dict[str, Any]] = None,
    umap_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate professional HTML evaluation report.

    Args:
        retrieval_results: Output from RetrievalEvaluator.evaluate_batch.
        generation_results: Output from GenerationEvaluator.evaluate_batch.
        umap_path: Path to umap_visualization.html (embedded via iframe).
        output_path: Output file path (default: evaluation_report.html in cwd).
        config: Optional metadata (dataset, limit, mode, etc.).

    Returns:
        Path to generated HTML file.
    """
    retrieval_results = retrieval_results or {}
    generation_results = generation_results or {}
    config = config or {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Executive summary
    retrieval_pass = True
    if retrieval_results:
        avg_r = retrieval_results.get("avg_recall", 0)
        retrieval_pass = avg_r >= TARGET_RECALL_MIN

    gen_pass = True
    if generation_results:
        faith = generation_results.get("faithfulness_rate", 0)
        rel = generation_results.get("avg_relevance_score", 0)
        gen_pass = faith >= TARGET_FAITHFULNESS and rel >= TARGET_RELEVANCE

    overall_pass = retrieval_pass and gen_pass
    issues = _build_issue_list(retrieval_results, generation_results)

    # Resolve output path early for UMAP relative path
    if output_path is None:
        output_path = Path.cwd() / "evaluation_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build HTML
    html_parts: List[str] = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Evaluation Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
  <h1>RAG Evaluation Report</h1>
  <p class="text-muted">Generated: """ + _html_escape(timestamp) + """</p>
""")

    # Config metadata
    if config:
        html_parts.append('  <div class="card mb-3"><div class="card-body"><h5>Configuration</h5><pre class="mb-0">')
        for k, v in config.items():
            html_parts.append(f"  {_html_escape(str(k))}: {_html_escape(str(v))}\n")
        html_parts.append("</pre></div></div>\n")

    # 1. Executive Summary
    status_class = "success" if overall_pass else "danger"
    html_parts.append(f"""
  <div class="card mb-4">
    <div class="card-header bg-{status_class} text-white">
      <h5 class="mb-0">Executive Summary</h5>
    </div>
    <div class="card-body">
      <p><strong>Overall:</strong> {"PASS" if overall_pass else "FAIL"}</p>
      <ul>
        <li>Retrieval: {"PASS" if retrieval_pass else "FAIL"} (Recall@5 target &gt;= {TARGET_RECALL_MIN})</li>
        <li>Generation: {"PASS" if gen_pass else "FAIL"} (Faithfulness &gt;= {TARGET_FAITHFULNESS*100:.0f}%, Relevance &gt;= {TARGET_RELEVANCE}/5)</li>
      </ul>
      <p class="mb-0"><strong>Issues:</strong> {len(issues)}</p>
    </div>
  </div>
""")

    # 2. Retrieval Metrics Table
    if retrieval_results and retrieval_results.get("per_case"):
        html_parts.append("""
  <div class="card mb-4">
    <div class="card-header"><h5 class="mb-0">Retrieval Metrics</h5></div>
    <div class="card-body">
      <p><strong>Averages:</strong> Recall@5: """ + f"{retrieval_results.get('avg_recall', 0):.2f}" +
            """, Precision@5: """ + f"{retrieval_results.get('avg_precision', 0):.2f}" +
            """, MRR: """ + f"{retrieval_results.get('avg_mrr', 0):.2f}" + """</p>
      <div class="table-responsive">
        <table class="table table-sm table-striped">
          <thead><tr><th>ID</th><th>Question</th><th>Recall@5</th><th>Precision@5</th><th>MRR</th><th>Min Dist</th></tr></thead>
          <tbody>
""")
        for case in retrieval_results["per_case"]:
            r = case.get("recall", 0)
            p = case.get("precision", 0)
            m = case.get("mrr", 0)
            d = case.get("min_distance", 0)
            row_class = "table-danger" if r == 0 else ""
            html_parts.append(
                f'            <tr class="{row_class}"><td>{_html_escape(case.get("id", ""))}</td>'
                f'<td>{_html_escape(case.get("question", ""))}</td>'
                f'<td>{r:.2f}</td><td>{p:.2f}</td><td>{m:.2f}</td><td>{d:.4f}</td></tr>\n'
            )
        html_parts.append("          </tbody>\n        </table>\n      </div>\n    </div>\n  </div>\n")

    # 3. Generation Metrics Table
    if generation_results and generation_results.get("per_case"):
        html_parts.append("""
  <div class="card mb-4">
    <div class="card-header"><h5 class="mb-0">Generation Metrics</h5></div>
    <div class="card-body">
      <p><strong>Summary:</strong> Faithfulness: """ +
            f"{generation_results.get('faithful_count', 0)}/{generation_results.get('total_count', 0)} " +
            f"({generation_results.get('faithfulness_rate', 0):.1%})" +
            """, Avg Relevance: """ + f"{generation_results.get('avg_relevance_score', 0):.2f}" + """/5</p>
      <div class="table-responsive">
        <table class="table table-sm table-striped">
          <thead><tr><th>ID</th><th>Question</th><th>Faithful</th><th>Relevance</th><th>Mode</th></tr></thead>
          <tbody>
""")
        for case in generation_results["per_case"]:
            faithful = case.get("is_faithful", False)
            score = case.get("relevance_score", 0)
            f_class = "text-danger" if not faithful else "text-success"
            s_class = "text-danger" if score < TARGET_RELEVANCE else "text-success"
            html_parts.append(
                f'            <tr><td>{_html_escape(case.get("id", ""))}</td>'
                f'<td>{_html_escape(case.get("question", ""))}</td>'
                f'<td class="{f_class}">{"Yes" if faithful else "No"}</td>'
                f'<td class="{s_class}">{score}/5</td>'
                f'<td>{_html_escape(case.get("selected_mode", ""))}</td></tr>\n'
            )
        html_parts.append("          </tbody>\n        </table>\n      </div>\n    </div>\n  </div>\n")

    # 4. UMAP Visualization
    if umap_path and Path(umap_path).exists():
        # Use relative path from report to UMAP file for same-folder deployment
        out_dir = output_path.parent
        umap_file = Path(umap_path).resolve()
        try:
            rel_path = umap_file.relative_to(out_dir)
        except ValueError:
            rel_path = umap_file.name
        html_parts.append(f"""
  <div class="card mb-4">
    <div class="card-header"><h5 class="mb-0">UMAP Embedding Visualization</h5></div>
    <div class="card-body">
      <iframe src="{_html_escape(str(rel_path))}" width="100%" height="600" frameborder="0" title="UMAP plot"></iframe>
      <p class="text-muted mt-2"><small><a href="{_html_escape(str(rel_path))}" target="_blank">Open UMAP in new tab</a></small></p>
    </div>
  </div>
""")
    elif umap_path:
        html_parts.append(f"""
  <div class="card mb-4">
    <div class="card-header"><h5 class="mb-0">UMAP Visualization</h5></div>
    <div class="card-body">
      <p class="text-muted">UMAP file not found: {_html_escape(str(umap_path))}</p>
    </div>
  </div>
""")

    # 5. Issue List
    html_parts.append("""
  <div class="card mb-4">
    <div class="card-header bg-warning"><h5 class="mb-0">Issue List</h5></div>
    <div class="card-body">
""")
    if not issues:
        html_parts.append('      <p class="text-success">No issues identified.</p>\n')
    else:
        html_parts.append('      <ul class="list-group">\n')
        for issue in issues:
            sev = issue.get("severity", "medium")
            badge = "danger" if sev == "high" else "warning"
            html_parts.append(
                f'        <li class="list-group-item d-flex justify-content-between align-items-start">'
                f'<span><strong>{_html_escape(issue.get("id", ""))}</strong> [{_html_escape(issue.get("type", ""))}]: '
                f'{_html_escape(issue.get("message", ""))}</span>'
                f'<span class="badge bg-{badge}">{_html_escape(sev)}</span></li>\n'
            )
        html_parts.append("      </ul>\n")
    html_parts.append("    </div>\n  </div>\n")

    html_parts.append("</div>\n</body>\n</html>")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    return str(output_path)
