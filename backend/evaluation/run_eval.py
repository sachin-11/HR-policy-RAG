"""Basic RAG/agent evaluation runner for HR Policy Assistant."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.graph import HRPolicyAgent
from app.agent.llm import ExtractiveLLMClient
from app.rag.retriever import RetrievalResponse, RetrievedChunk


@dataclass
class EvaluationCase:
    id: str
    user_message: str
    expected_answer_term: str | None = None
    expected_source: str | None = None
    should_refuse: bool = False
    expected_tool_action: str | None = None
    expected_approval_required: bool = False


@dataclass
class EvaluationResult:
    case: EvaluationCase
    passed: bool
    reason: str
    answer: str
    retrieval_hit: bool
    source_match: bool
    tool_action_match: bool
    approval_required: bool


class StaticRetriever:
    """Simple rule-based retriever for local evaluation."""

    def __init__(self) -> None:
        self.chunks = [
            RetrievedChunk(
                id="leave-1",
                content=(
                    "Full-time employees in India can take sick leave when they are unwell and unable to work. "
                    "Employees should inform their manager as early as possible and submit the leave request in the HR portal. "
                    "For sick leave longer than two consecutive working days, HR may ask for a medical certificate."
                ),
                score=0.95,
                metadata={
                    "title": "Sick Leave Policy India",
                    "source": "hr_leave_policy_india_2026.md",
                    "section_title": "Sick Leave",
                    "policy_type": "leave",
                    "country": "India",
                    "employee_type": "full_time",
                },
            ),
            RetrievedChunk(
                id="reimbursement-1",
                content=(
                    "Employees may request reimbursement for approved laptop accessories used for work. "
                    "The request must include a valid invoice, manager approval, and the business reason. "
                    "Finance reviews reimbursement requests according to the employee's country and role."
                ),
                score=0.92,
                metadata={
                    "title": "Laptop Reimbursement Policy",
                    "source": "reimbursement_policy.txt",
                    "section_title": "Reimbursement",
                    "policy_type": "reimbursement",
                    "country": "India",
                    "employee_type": "full_time",
                },
            ),
        ]

    def retrieve(self, query: str, *, filters: Any = None, top_k: int | None = None, score_threshold: float | None = None) -> RetrievalResponse:
        query_text = query.lower()
        if any(keyword in query_text for keyword in ("sick", "leave")):
            return RetrievalResponse(query=query, chunks=[self.chunks[0]], metadata_filter={})
        if any(keyword in query_text for keyword in ("reimbursement", "invoice", "laptop")):
            return RetrievalResponse(query=query, chunks=[self.chunks[1]], metadata_filter={})
        return RetrievalResponse(query=query, chunks=[], metadata_filter={})


def load_golden_dataset(path: Path) -> list[EvaluationCase]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cases: list[EvaluationCase] = []
        for row in reader:
            cases.append(
                EvaluationCase(
                    id=row["id"].strip(),
                    user_message=row["user_message"].strip(),
                    expected_answer_term=row.get("expected_answer_term", "").strip() or None,
                    expected_source=row.get("expected_source", "").strip() or None,
                    should_refuse=row.get("should_refuse", "False").strip().lower() == "true",
                    expected_tool_action=row.get("expected_tool_action", "").strip() or None,
                    expected_approval_required=row.get("expected_approval_required", "False").strip().lower() == "true",
                )
            )
        return cases


def evaluate_case(agent: HRPolicyAgent, case: EvaluationCase) -> EvaluationResult:
    state = agent.run(user_message=case.user_message, user_id="emp_123", filters={})
    answer = state["final_answer"].strip()
    retrieval_hit = bool(state.get("sources"))
    source_match = False
    if case.expected_source and state.get("sources"):
        source_match = any(
            case.expected_source.lower() in str(source.get("source", "")).lower()
            or case.expected_source.lower() in str(source.get("title", "")).lower()
            for source in state["sources"]
        )

    tool_action_match = False
    if case.expected_tool_action:
        tool_actions = [f"{result['tool_name']}/{result['action']}" for result in state.get("tool_results", [])]
        tool_action_match = case.expected_tool_action in tool_actions

    approval_required = bool(state.get("approval_required_actions"))

    errors: list[str] = []
    if case.should_refuse:
        if not state.get("needs_human_confirmation", False):
            errors.append("expected refusal or human confirmation")
        if "could not find" not in answer.lower() and "no context" not in answer.lower():
            errors.append("expected no-context refusal answer")
    else:
        if case.expected_answer_term and case.expected_answer_term.lower() not in answer.lower():
            errors.append("answer did not contain expected term")
        if case.expected_source and not source_match:
            errors.append("expected source was not cited")
        if case.expected_tool_action and not tool_action_match:
            errors.append("expected tool action was not present")
        if case.expected_approval_required and not approval_required:
            errors.append("expected approval requirement not present")

    passed = not errors
    reason = "PASS" if passed else "; ".join(errors)
    return EvaluationResult(
        case=case,
        passed=passed,
        reason=reason,
        answer=answer,
        retrieval_hit=retrieval_hit,
        source_match=source_match,
        tool_action_match=tool_action_match,
        approval_required=approval_required,
    )


def format_report(results: list[EvaluationResult]) -> str:
    lines = [
        "Evaluation Results:",
        "-------------------",
    ]
    for result in results:
        lines.append(
            f"{result.case.id}: {'PASS' if result.passed else 'FAIL'} - {result.reason}"
        )
        lines.append(f"  Query: {result.case.user_message}")
        lines.append(f"  Answer: {result.answer}")
        if result.case.expected_source:
            lines.append(f"  Expected source: {result.case.expected_source}")
            lines.append(f"  Source match: {result.source_match}")
        if result.case.expected_tool_action:
            lines.append(f"  Expected tool action: {result.case.expected_tool_action}")
            lines.append(f"  Tool match: {result.tool_action_match}")
        lines.append(f"  Retrieval hit: {result.retrieval_hit}")
        lines.append(f"  Approval required: {result.approval_required}")
        lines.append("")
    summary = [
        f"Total cases: {len(results)}",
        f"Passed: {sum(1 for result in results if result.passed)}",
        f"Failed: {sum(1 for result in results if not result.passed)}",
    ]
    return "\n".join(lines + summary)


def run_evaluation(dataset_path: Path) -> list[EvaluationResult]:
    cases = load_golden_dataset(dataset_path)
    agent = HRPolicyAgent(retriever=StaticRetriever(), llm_client=ExtractiveLLMClient())
    results = [evaluate_case(agent, case) for case in cases]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HR Policy Assistant evaluation cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_dataset.csv",
        help="Path to the golden dataset CSV file.",
    )
    args = parser.parse_args()

    results = run_evaluation(args.dataset)
    print(format_report(results))


if __name__ == "__main__":
    main()
