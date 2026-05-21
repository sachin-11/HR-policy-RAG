from pathlib import Path

from evaluation.run_eval import load_golden_dataset, run_evaluation


def test_load_golden_dataset_reads_cases() -> None:
    dataset_path = Path(__file__).resolve().parents[2] / "evaluation" / "golden_dataset.csv"
    cases = load_golden_dataset(dataset_path)

    assert len(cases) == 5
    assert cases[0].id == "policy_1"
    assert cases[0].expected_source == "hr_leave_policy_india_2026.md"
    assert cases[0].should_refuse is False


def test_run_evaluation_returns_results() -> None:
    dataset_path = Path(__file__).resolve().parents[2] / "evaluation" / "golden_dataset.csv"
    results = run_evaluation(dataset_path)

    assert len(results) == 5
    assert all(hasattr(result, "passed") for result in results)
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is True
    assert results[4].passed is True
