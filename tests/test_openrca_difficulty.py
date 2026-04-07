import pytest

from vatavaran.openrca_difficulty import (
    difficulty_from_openrca_number,
    difficulty_from_task_index,
    parse_openrca_task_number,
)


@pytest.mark.parametrize(
    "task_index,expected",
    [
        ("task_1", "easy"),
        ("task_3", "easy"),
        ("task_4", "middle"),
        ("task_6", "middle"),
        ("task_7", "hard"),
        ("task_99", "hard"),
    ],
)
def test_difficulty_from_task_index(task_index, expected):
    assert difficulty_from_task_index(task_index) == expected


def test_parse_none():
    assert parse_openrca_task_number(None) is None
    assert parse_openrca_task_number("bad") is None


def test_difficulty_from_number_direct():
    assert difficulty_from_openrca_number(2) == "easy"
    assert difficulty_from_openrca_number(5) == "middle"
    assert difficulty_from_openrca_number(7) == "hard"
