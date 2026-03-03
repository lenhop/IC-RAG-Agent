"""
Generation Layer Evaluation - LLM-as-Judge

Implements faithfulness and relevance scoring per RAG_EVALUATION_IMPLEMENTATION_PLAN.md.
Targets: faithfulness >= 85%, relevance >= 4.0/5.
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown code blocks and extra text.

    Args:
        text: Raw LLM response string.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # Try finding outermost {...} (handles nested braces)
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i, c in enumerate(text[brace_start:], brace_start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def evaluate_faithfulness(
    answer: str,
    contexts: List[str],
    judge_llm: BaseChatModel,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate if answer is faithful to retrieved contexts (no hallucination).

    Args:
        answer: Generated answer from RAG.
        contexts: Retrieved document contexts (page_content strings).
        judge_llm: LLM instance for judging.
        verbose: Print reasoning.

    Returns:
        Dict with is_faithful (bool) and reasoning (str).
    """
    context_text = "\n\n".join(contexts) if contexts else "(No context provided)"

    prompt = f"""You are an expert evaluator. Assess if the ANSWER is completely faithful to the CONTEXT.

CONTEXT:
{context_text}

ANSWER:
{answer}

EVALUATION CRITERIA:
- Is the answer ONLY based on information in the context?
- Does it contain any fabricated, exaggerated, or external information?
- If context is insufficient, does the answer acknowledge this?

OUTPUT FORMAT (JSON):
{{
  "is_faithful": true/false,
  "reasoning": "Brief explanation of your judgment"
}}

Respond ONLY with valid JSON:"""

    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        result = _extract_json(response_text)
        if result is None:
            return {
                "is_faithful": False,
                "reasoning": f"Failed to parse LLM response as JSON: {response_text[:200]}...",
            }

        is_faithful = result.get("is_faithful", False)
        if isinstance(is_faithful, str):
            is_faithful = is_faithful.lower() in ("true", "yes", "1")
        reasoning = result.get("reasoning", "No reasoning provided")

        if verbose:
            print(f"Faithfulness: {is_faithful}")
            print(f"Reasoning: {reasoning}")

        return {"is_faithful": bool(is_faithful), "reasoning": str(reasoning)}
    except Exception as e:
        if verbose:
            print(f"Error evaluating faithfulness: {e}")
        return {"is_faithful": False, "reasoning": f"Evaluation error: {e}"}


def evaluate_relevance(
    question: str,
    answer: str,
    judge_llm: BaseChatModel,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate if answer is relevant and complete for the question.

    Args:
        question: User question.
        answer: Generated answer from RAG.
        judge_llm: LLM instance for judging.
        verbose: Print reasoning.

    Returns:
        Dict with relevance_score (1-5) and reasoning (str).
    """
    prompt = f"""You are an expert evaluator. Rate how well the ANSWER addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

EVALUATION CRITERIA:
- Does the answer directly address the question?
- Is the answer complete and accurate?
- Is the answer clear and well-structured?

RATING SCALE:
5 - Excellent: Fully answers the question, accurate, complete
4 - Good: Answers the question well, minor gaps
3 - Acceptable: Partially answers, missing some details
2 - Poor: Barely addresses the question
1 - Irrelevant: Does not answer the question

OUTPUT FORMAT (JSON):
{{
  "relevance_score": 1-5,
  "reasoning": "Brief explanation of your rating"
}}

Respond ONLY with valid JSON:"""

    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        result = _extract_json(response_text)
        if result is None:
            return {
                "relevance_score": 0,
                "reasoning": f"Failed to parse LLM response as JSON: {response_text[:200]}...",
            }

        score = result.get("relevance_score", 0)
        if isinstance(score, (int, float)):
            score = max(1, min(5, int(round(score))))
        else:
            score = 0
        reasoning = result.get("reasoning", "No reasoning provided")

        if verbose:
            print(f"Relevance Score: {score}/5")
            print(f"Reasoning: {reasoning}")

        return {"relevance_score": score, "reasoning": str(reasoning)}
    except Exception as e:
        if verbose:
            print(f"Error evaluating relevance: {e}")
        return {"relevance_score": 0, "reasoning": f"Evaluation error: {e}"}


class GenerationEvaluator:
    """Evaluator for generation metrics (faithfulness, relevance) via LLM-as-Judge."""

    def evaluate_batch(
        self,
        pipeline: Any,
        test_cases: List[Dict[str, Any]],
        mode: str = "hybrid",
        judge_llm: Optional[BaseChatModel] = None,
        verbose: bool = True,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate generation quality on a batch of test cases.

        Args:
            pipeline: RAGPipeline instance (must have query, llm_general).
            test_cases: List of dicts with 'question', optional 'id'.
            mode: Answer mode (documents, general, hybrid, auto).
            judge_llm: LLM for judging (default: pipeline.llm_general).
            verbose: Print per-case and summary results.
            progress: Show tqdm progress bar.

        Returns:
            Dict with faithfulness_rate, avg_relevance_score, per_case list, summary.
        """
        judge = judge_llm or pipeline.llm_general
        if judge is None:
            raise ValueError("judge_llm or pipeline.llm_general required")

        results: Dict[str, Any] = {
            "faithfulness": [],
            "relevance_scores": [],
            "per_case": [],
        }

        iterator = test_cases
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(test_cases, desc="Generation eval", unit="case")
            except ImportError:
                pass

        for case in iterator:
            question = case.get("question", "")
            case_id = case.get("id", "unknown")

            if verbose:
                print(f"\n{'='*60}")
                print(f"{case_id}: {question[:60]}...")

            # Generate answer via pipeline
            answer, retrieved_docs, selected_mode = pipeline.query(
                question, mode=mode, verbose=False
            )

            # Extract contexts from retrieved docs
            contexts = (
                [doc.page_content for doc in retrieved_docs]
                if retrieved_docs
                else []
            )

            # Evaluate faithfulness
            if verbose:
                print("Evaluating faithfulness...")
            faithfulness_result = evaluate_faithfulness(
                answer, contexts, judge, verbose=verbose
            )

            # Evaluate relevance
            if verbose:
                print("Evaluating relevance...")
            relevance_result = evaluate_relevance(
                question, answer, judge, verbose=verbose
            )

            results["faithfulness"].append(faithfulness_result["is_faithful"])
            results["relevance_scores"].append(relevance_result["relevance_score"])
            results["per_case"].append({
                "id": case_id,
                "question": question[:80] + "..." if len(question) > 80 else question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "selected_mode": str(selected_mode),
                "num_contexts": len(contexts),
                "is_faithful": faithfulness_result["is_faithful"],
                "faithfulness_reasoning": faithfulness_result["reasoning"],
                "relevance_score": relevance_result["relevance_score"],
                "relevance_reasoning": relevance_result["reasoning"],
            })

        # Summary
        n = len(results["faithfulness"])
        faithful_count = sum(results["faithfulness"])
        avg_relevance = (
            sum(results["relevance_scores"]) / n if n else 0.0
        )

        results["faithfulness_rate"] = faithful_count / n if n else 0.0
        results["faithful_count"] = faithful_count
        results["total_count"] = n
        results["avg_relevance_score"] = avg_relevance
        results["summary"] = {
            "faithfulness_rate": results["faithfulness_rate"],
            "faithful_count": faithful_count,
            "total_count": n,
            "avg_relevance_score": avg_relevance,
            "target_faithfulness": 0.85,
            "target_relevance": 4.0,
        }

        if verbose:
            print("\n" + "=" * 60)
            print("GENERATION EVALUATION SUMMARY")
            print("=" * 60)
            print(
                f"Faithfulness: {faithful_count}/{n} "
                f"({results['faithfulness_rate']:.1%})"
            )
            print("  Target: >= 85% (spec requirement)")
            print(f"Average Relevance: {avg_relevance:.2f}/5")
            print("  Target: >= 4.0 (spec requirement)")
            print("=" * 60)

        return results
