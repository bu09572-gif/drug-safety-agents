import json

from drugsafety.agents.evaluate import score_drug, summarize
from drugsafety.agents.graph import Tracer, _parse_plan, build_graph, render_report
from drugsafety.agents.tools import FDALabels, SentenceCorpus, drug_matches, effect_on_label
from drugsafety.llm import Generation


class FakeLLM:
    """Returns a fixed answer per sentence; used in place of a real model."""

    def __init__(self, answers, name="fake"):
        self.answers = answers
        self.name = name
        self.calls = 0

    def chat(self, messages, max_new_tokens=None):
        self.calls += 1
        last = messages[-1]["content"]
        for key, ans in self.answers.items():
            if key in last:
                return Generation(ans, 10, 5, 0.0)
        return Generation("[]", 10, 1, 0.0)


class FakeLabels(FDALabels):
    def __init__(self, texts):
        self.texts = texts

    def get(self, drug):
        t = self.texts.get(drug)
        return {"drug": drug, "found": t is not None, "set_id": "x" if t else None, "text": t or ""}


def test_drug_matching_and_label_check():
    assert drug_matches("lithium carbonate", "lithium")
    assert not drug_matches("lithium", "methotrexate")
    assert effect_on_label("interstitial pneumonitis", "Methotrexate can cause pneumonitis, interstitial.")
    assert effect_on_label("hepatotoxicity", "Serious hepatotoxicity has been reported.")
    assert not effect_on_label("ototoxicity", "Nausea and rash were reported.")


def test_parse_plan_drops_drugs_not_in_question():
    plan, valid = _parse_plan('{"drugs": ["methotrexate", "aspirin"]}', "Side effects of methotrexate?",
                              {"methotrexate"})
    assert plan["drugs"] == ["methotrexate"] and not valid
    plan, valid = _parse_plan("I think the drug is...", "Side effects of methotrexate?", {"methotrexate"})
    assert plan["drugs"] == ["methotrexate"] and plan["fallback"]


def test_graph_end_to_end(examples, tmp_path):
    corpus = SentenceCorpus(examples)
    planner = FakeLLM({"methotrexate": '{"drugs": ["methotrexate"], "check_label": true}'})
    # First pass: one good pair, and one pair that is not copied from the sentence.
    extractor = FakeLLM({
        "Methotrexate-induced pneumonitis": '[{"drug": "Methotrexate", "effect": "pneumonitis"}]',
        "54-year-old woman": '[{"drug": "methotrexate", "effect": "liver failure"}]',
    })
    second = FakeLLM({})  # second opinion says: no event
    tracer = Tracer(tmp_path / "trace.jsonl")
    g = build_graph(planner, extractor, corpus, FakeLabels({"methotrexate": "Risk of liver damage."}), tracer,
                    second_opinion_llm=second)
    state = g.invoke({"question": "What adverse events are reported for methotrexate?"})

    assert state["planner_valid"] and state["retries"] == 1
    assert second.calls == 1  # only the rejected sentence went back
    effects = state["report"]["methotrexate"]["effects"]
    assert [e["effect"] for e in effects] == ["pneumonitis"]
    assert state["report"]["methotrexate"]["possible_new_signals"] == ["pneumonitis"]
    nodes = [json.loads(line)["node"] for line in (tmp_path / "trace.jsonl").open()]
    assert nodes[:3] == ["planner", "search", "worker"] and nodes[-1] == "report" and "reviewer" in nodes
    assert "methotrexate" in render_report("q", state["report"])

    by_id = {e.id: e for e in examples}
    s = summarize([score_drug("methotrexate", state, by_id)])
    assert s["final"]["precision"] == 1.0 and s["first_pass"]["precision"] == 0.5
