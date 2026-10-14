"""Regression tests for the rule-based grader. Run: python tests/test_grading.py (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgeprobe.grading import consistency_features, grade  # noqa: E402

CASES = [
    # (response, answerable, gold answers, expected label)
    ("The composer of Symphony No. 8 is Richard Wagner.", True, ["Rautavaara"], "WRONG"),
    ("No planet is closer to the Sun than Mercury.", False, [], "REJECTED"),
    ("Mars has only two moons, Phobos and Deimos.", False, [], "REJECTED"),
    ("There is no Nobel Prize in Mathematics.", False, [], "REJECTED"),
    ("I'm not sure who directed that film.", True, ["Spielberg"], "REFUSED"),
    ("The capital of Bravmorrow Province is Bravmorrow City.", False, [], "ACCEPTED"),
    ("Brazil won the 2042 FIFA World Cup.", False, [], "ACCEPTED"),
    ("The 2042 World Cup has not taken place yet.", False, [], "REJECTED"),
    ("It is impossible to predict the population of Tokyo in 2080.", False, [], "REJECTED"),
    ("The United States won the most gold medals.", True, ["United States", "USA", "US"], "CORRECT"),
    ("Let us consider France.", True, ["US"], "WRONG"),
    ("The highest mountain in Japan is Mount Fuji.", True, ["Mount Fuji", "Fuji"], "CORRECT"),
    ("Albert Einstein never won a second Nobel Prize.", False, [], "REJECTED"),
    ("The third moon of Mars is Phobos.", False, [], "ACCEPTED"),
    ("I'm sorry, but I couldn't find any information about that opera.", False, [], "REJECTED"),
    ("Zürich", True, ["Zurich"], "CORRECT"),
]


def test_grade():
    for resp, ans, gold, exp in CASES:
        assert grade(resp, ans, gold) == exp, (resp, grade(resp, ans, gold), exp)


def test_consistency():
    q = "Who was the composer of Symphony of Kelmorrow?"
    same = consistency_features("It was composed by Edward Elgar.",
                                ["The composer was Edward Elgar."] * 5, q)
    diff = consistency_features("It was composed by Edward Elgar.",
                                ["Gustav Holst.", "Richard Strauss.", "William Alwyn.",
                                 "Henryk Gorecki.", "Antonin Dvorak."], q)
    assert same["agree"] == 1.0 and same["n_clusters"] == 1
    assert diff["agree"] == 0.0 and diff["n_clusters"] == 6


if __name__ == "__main__":
    test_grade()
    test_consistency()
    print(f"all {len(CASES)} grading cases and consistency checks passed")
