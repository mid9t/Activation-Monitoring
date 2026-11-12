"""Hand-check allocation, the discount, and the mechanical branch."""

from amon.decide import condition_a_pass, decide, kept_shift
from amon.handcheck import allocate, tie_break_flag


def test_allocate_respects_the_floor_and_the_total():
    counts = {"A": 100, "B": 50, "C": 10}
    alloc = allocate(counts, n=30, minimum=3)
    assert sum(alloc.values()) == 30
    assert min(alloc.values()) >= 3
    assert alloc["A"] > alloc["B"] > alloc["C"]


def test_kept_shift_breaks_ties_a_then_c_then_b():
    discounts = {
        "A": {"F_adj": 200.0, "R_adj": 400.0},
        "C": {"F_adj": 200.0, "R_adj": 300.0},
        "B": {"F_adj": 50.0, "R_adj": 400.0},
    }
    assert kept_shift(discounts, min_f=200, min_frac=0.25) == "A"


def test_condition_a_needs_two_sets():
    sets = {
        "tune": {"probe_error": 2.0, "llm_error": 1.0},
        "A": {"probe_error": 1.0, "llm_error": 2.0},
        "B": {"probe_error": 2.0, "llm_error": 1.0},
        "C": {"probe_error": 1.0, "llm_error": 2.0},
    }
    assert condition_a_pass(sets)


def test_tie_break_flags_when_the_wilson_upper_bound_flips_a_shift():
    result = {
        "discount": {"A": {"F_adj": 200.0, "R_adj": 400.0}},
        "discount_at_wilson_upper": {"A": {"F_adj": 100.0, "R_adj": 400.0}},
    }
    assert tie_break_flag(result, min_f=200, min_frac=0.25)


def test_decide_pass_and_fail_a():
    hand = {
        "r_F": {"k": 1, "n": 30, "rate": 1 / 30, "wilson": [0.0, 0.2]},
        "r_U": {"k": 1, "n": 20, "rate": 0.05, "wilson": [0.0, 0.2]},
        "tie_break": False,
    }
    discounts = {"A": {"F_adj": 250.0, "R_adj": 500.0}, "B": {"F_adj": 10.0, "R_adj": 100.0}, "C": {"F_adj": 10.0, "R_adj": 100.0}}
    good = {
        "tune": {"probe_error": 2.0, "llm_error": 1.0},
        "A": {"probe_error": 2.0, "llm_error": 1.0},
        "B": {"probe_error": 1.0, "llm_error": 2.0},
        "C": {"probe_error": 1.0, "llm_error": 2.0},
    }
    passed = decide(
        {"sets": good},
        discounts,
        hand,
        min_f=200,
        min_frac=0.25,
        kill_mislabeled=15,
        materiality=0.10,
        deferred={"A": 300, "B": 100, "C": 100},
        deff={"A": 1.0, "B": 1.0, "C": 1.0},
    )
    assert passed["branch"] == "pass"
    assert passed["kept_shift"] == "A"
    bad = dict(good)
    bad = {key: {"probe_error": 1.0, "llm_error": 2.0} for key in good}
    failed = decide(
        {"sets": bad},
        discounts,
        hand,
        min_f=200,
        min_frac=0.25,
        kill_mislabeled=15,
        materiality=0.10,
        deferred={"A": 300, "B": 100, "C": 100},
        deff={"A": 1.0},
    )
    assert failed["branch"] == "fail_a"
