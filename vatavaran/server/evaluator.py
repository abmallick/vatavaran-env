"""Vatavaran-compatible evaluator for root cause answers."""

from __future__ import annotations

import itertools
import re
from datetime import datetime


def _time_within_one_minute(expected: str, predicted: str) -> bool:
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t_expected = datetime.strptime(expected, fmt)
        t_predicted = datetime.strptime(predicted, fmt)
    except ValueError:
        return False
    return abs((t_expected - t_predicted).total_seconds()) <= 60


def _extract_prediction_items(prediction: str) -> list[dict[str, str]]:
    pattern = (
        r"{\s*"
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    matches = re.findall(pattern, prediction or "")
    items = []
    for dt_str, component, reason in matches:
        items.append(
            {
                "root cause occurrence datetime": dt_str,
                "root cause component": component,
                "root cause reason": reason,
            }
        )
    return items


def _extract_scoring_points(scoring_points: str) -> tuple[list[str], list[str], list[str]]:
    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    time_pattern = (
        r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes "
        r"\(i.e., <=1min\) of ([^\n]+)"
    )
    components = re.findall(component_pattern, scoring_points or "")
    reasons = re.findall(reason_pattern, scoring_points or "")
    times = re.findall(time_pattern, scoring_points or "")
    return components, reasons, times


def evaluate_prediction(prediction: str, scoring_points: str) -> dict:
    """Return evaluation details with Vatavaran-style partial credit."""

    predicted = _extract_prediction_items(prediction)
    components, reasons, times = _extract_scoring_points(scoring_points)

    target_count = max(len(components), len(reasons), len(times))
    criteria_count = len(components) + len(reasons) + len(times)

    if target_count == 0 or criteria_count == 0:
        return {
            "passed_criteria": [],
            "failed_criteria": [],
            "score": 0.01,
        }

    if len(predicted) != target_count:
        all_criteria = list(dict.fromkeys(components + reasons + times))
        return {
            "passed_criteria": [],
            "failed_criteria": all_criteria,
            "score": 0.01,
        }

    best_score = -1
    best_passing: list[str] = []

    for perm in itertools.permutations(predicted):
        current = 0
        passing: list[str] = []
        for idx in range(target_count):
            if len(components) == target_count:
                expected = components[idx]
                if perm[idx]["root cause component"] == expected:
                    current += 1
                    passing.append(expected)
            if len(reasons) == target_count:
                expected = reasons[idx]
                if perm[idx]["root cause reason"] == expected:
                    current += 1
                    passing.append(expected)
            if len(times) == target_count:
                expected = times[idx]
                if _time_within_one_minute(
                    expected, perm[idx]["root cause occurrence datetime"]
                ):
                    current += 1
                    passing.append(expected)
        if current > best_score:
            best_score = current
            best_passing = passing

    score = round(max(best_score, 0) / criteria_count, 2)
    score = max(0.01, min(0.99, score))
    all_criteria = list(dict.fromkeys(components + reasons + times))
    failed = [criterion for criterion in all_criteria if criterion not in best_passing]
    return {
        "passed_criteria": best_passing,
        "failed_criteria": failed,
        "score": score,
    }
