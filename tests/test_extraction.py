from drugsafety.data import split_of
from drugsafety.extract import build_messages, parse_answer
from drugsafety.metrics import grounded, match_counts, score
from drugsafety.retrieve import ExampleRetriever


def test_build_examples_groups_pairs_and_adds_negatives(examples):
    by_text = {e.text: e for e in examples}
    assert len(examples) == 5
    assert len(by_text["Lithium toxicity and hypothyroidism after starting lithium carbonate."].pairs) == 2
    assert by_text["The patient recovered after supportive care."].pairs == []
    # The positive sentence is not added a second time as a negative.
    assert sum(e.text.startswith("Methotrexate-induced") for e in examples) == 1


def test_split_is_stable():
    assert split_of("same sentence") == split_of("same sentence  ")


def test_parse_answer():
    assert parse_answer('[{"drug": "a", "effect": "b"}]') == ([{"drug": "a", "effect": "b"}], True)
    assert parse_answer('```json\n[]\n```') == ([], True)
    assert parse_answer("Answer: [{\"drug\": \"a\", \"effect\": \"b\"}, {\"drug\": \"a\", \"effect\": \"b\"}]")[0] == [
        {"drug": "a", "effect": "b"}]
    assert parse_answer("no events") == ([], False)
    assert parse_answer('[{"drug": "a"}]') == ([], False)


def test_messages_include_shots(examples):
    msgs = build_messages("x", examples[:2])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_match_counts_strict_and_lenient():
    gold = [{"drug": "Methotrexate", "effect": "pneumonitis"}]
    assert match_counts([{"drug": "methotrexate", "effect": "Pneumonitis."}], gold) == (1, 1, 1)
    assert match_counts([{"drug": "methotrexate", "effect": "interstitial pneumonitis"}], gold) == (0, 1, 1)
    assert match_counts([{"drug": "methotrexate", "effect": "interstitial pneumonitis"}], gold, lenient=True)[0] == 1


def test_score_and_grounding():
    text = "Methotrexate-induced pneumonitis."
    rows = [
        {"text": text, "gold": [{"drug": "Methotrexate", "effect": "pneumonitis"}],
         "pred": [{"drug": "Methotrexate", "effect": "pneumonitis"}], "valid_json": True, "seconds": 1.0},
        {"text": "No event here.", "gold": [], "pred": [{"drug": "aspirin", "effect": "rash"}], "valid_json": True,
         "seconds": 2.0},
    ]
    s = score(rows)
    assert s["strict"]["precision"] == 0.5 and s["strict"]["recall"] == 1.0
    assert s["grounded_rate"] == 0.5
    assert grounded({"drug": "methotrexate", "effect": "pneumonitis"}, text)


def test_retriever_keeps_a_negative(examples):
    r = ExampleRetriever(examples)
    shots = r.top_k("lithium toxicity", k=2)
    assert len(shots) == 2 and any(not s.has_ade for s in shots)
