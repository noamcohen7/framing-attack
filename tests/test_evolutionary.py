"""Tests for confirmation of winning rephrasings in the evolutionary search (no API)."""

from src.attack.evolutionary import EvolutionarySearch
from src.data.loader import QAItem
from src.equivalence.validator import EquivalenceResult
from src.evaluation.scorer import ScoreResult

QA = QAItem(id="q", question="Q?", answer="A")


class FakeParaphraser:
    def __init__(self, initial, mutations=()):
        self.initial = list(initial)
        self.mutations = list(mutations)

    def generate_initial(self, question, n=10):
        return list(self.initial)

    def generate_strategy_batch(self, question):
        return []

    def mutate(self, question, ancestor, n=2):
        return [self.mutations.pop(0)] if self.mutations else []

    def crossover(self, question, a, b, n=1):
        return []


class AlwaysEquivalent:
    def validate(self, original, paraphrase):
        return EquivalenceResult(is_equivalent=True)


class FakeScorer:
    """search[p] = score during search; fresh[p] = score on confirmation (num_samples given)."""

    def __init__(self, search, fresh):
        self.search, self.fresh = search, fresh
        self.confirm_calls = []

    def score(self, qa_item, paraphrase, num_samples=None):
        if num_samples is not None:
            self.confirm_calls.append(paraphrase)
            s, n = self.fresh[paraphrase], num_samples
        else:
            s, n = self.search[paraphrase], 5
        return ScoreResult(paraphrase=paraphrase, wrongness_score=s, num_samples=n,
                           num_correct=round(n * (1 - s)), num_incorrect=round(n * s), num_abstained=0)


def _search(paraphraser, scorer, **kw):
    kw.setdefault("population_size", 4)
    kw.setdefault("num_generations", 3)
    return EvolutionarySearch(paraphraser, AlwaysEquivalent(), scorer, confirm_samples=10, **kw)


def test_lucky_gen0_hit_is_rejected_and_search_continues():
    scorer = FakeScorer(search={"lucky": 1.0, "meh": 0.2, "m1": 0.0, "m2": 0.0},
                        fresh={"lucky": 0.1})
    result = _search(FakeParaphraser(["lucky", "meh"], ["m1", "m2"]), scorer).search(QA)
    assert not result.success
    assert scorer.confirm_calls == ["lucky"]          # tested once, not again as an elite
    lucky = next(c for c in result.all_candidates if c.paraphrase == "lucky")
    assert lucky.search_score == 1.0 and lucky.score == 0.1
    assert result.generations_run == 3                # did not stop early


def test_real_flip_is_confirmed_and_stops_early():
    scorer = FakeScorer(search={"real": 1.0, "meh": 0.2}, fresh={"real": 0.9})
    result = _search(FakeParaphraser(["real", "meh"]), scorer).search(QA)
    assert result.success and result.generations_run == 1
    assert result.best_candidate.paraphrase == "real" and result.best_score == 0.9


def test_second_candidate_gets_a_chance_after_first_fails():
    scorer = FakeScorer(search={"a": 1.0, "b": 1.0}, fresh={"a": 0.0, "b": 0.8})
    result = _search(FakeParaphraser(["a", "b"]), scorer).search(QA)
    assert result.success and result.best_candidate.paraphrase == "b"
    assert scorer.confirm_calls == ["a", "b"]


def test_end_of_search_majority_wrong_needs_confirmation():
    # 0.6 never hits early stop (1.0); at the end it must survive a fresh re-test.
    scorer = FakeScorer(search={"x": 0.6}, fresh={"x": 0.4})
    result = _search(FakeParaphraser(["x"]), scorer, num_generations=2).search(QA)
    assert not result.success and result.best_score == 0.4


def test_confirmation_disabled():
    scorer = FakeScorer(search={"x": 1.0}, fresh={})
    s = EvolutionarySearch(FakeParaphraser(["x"]), AlwaysEquivalent(), scorer,
                           population_size=4, num_generations=2, confirm_samples=0)
    result = s.search(QA)
    assert result.success and scorer.confirm_calls == []
