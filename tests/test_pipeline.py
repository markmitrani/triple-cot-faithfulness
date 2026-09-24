from cue_insertion import mentions_cue, parse_answer
from fixtures import ROWS
from judge import judge, label_rows
from summarize import table

BY_KEY = {(r["id"], r["condition"]): r["response"] for r in ROWS}


def test_parse_answer():
    assert parse_answer(BY_KEY[1, "control"]) == "A"
    assert parse_answer(BY_KEY[1, "cued"]) == "C"  # parenthesised letter
    assert parse_answer(BY_KEY[4, "cued"]) is None


def test_parse_ignores_answer_inside_thinking():
    assert parse_answer("<think>\nAnswer: B? No.\n</think>\n\nAnswer: D") == "D"


def test_mentions_cue():
    assert mentions_cue(BY_KEY[1, "cued"])
    assert not mentions_cue(BY_KEY[2, "cued"])
    assert mentions_cue(BY_KEY[3, "cued"])


def test_judge():
    j = {x["id"]: x for x in judge(ROWS)}
    assert len(j) == 4  # cued responses only
    assert j[1]["flipped"] and j[1]["mentions_cue"]
    assert set(j[1]["matched_patterns"]) == {"professor", "stanford"}
    assert j[2]["flipped"] and not j[2]["mentions_cue"] and j[2]["matched_patterns"] == []
    assert not j[3]["flipped"] and j[3]["matched_patterns"] == ["professor", "suggest"]
    assert not j[4]["flipped"] and j[4]["cued_answer"] is None


def test_label_rows_flips_first():
    df = label_rows(judge(ROWS))
    assert list(df.columns) == ["id", "cue", "thinking_text", "final_answer", "hand_label"]
    assert len(df) == 4  # fewer cued traces than 20 in the fixture
    assert list(df["id"][:2]) == [1, 2]
    assert set(df["id"][2:]) == {3, 4}
    assert (df["hand_label"] == "").all()
    assert df["thinking_text"][0].startswith("I lean A")


def test_table():
    t = table(ROWS)
    assert "| 4 | 8 | 1 (12%) | 2 | 50% | 1/2 (50%) |" in t
