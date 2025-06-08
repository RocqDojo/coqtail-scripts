from dataclasses import dataclass
from typing import final, override
from coqtop import Coqtop
from search_tree import BFSearch, Env, MoveGenerator
from qwen3_gen import LLM
from collections.abc import Iterable
import xmlInterface


@dataclass(frozen=True)
class ProofGoal:
    hypotheses: tuple[str, ...]
    conclusion: str


def from_coqtail_goal(g: xmlInterface.Goal) -> ProofGoal:
    return ProofGoal(hypotheses=tuple(g.context), conclusion=g.goal)


@dataclass(frozen=True)
class ProofState:
    fg: tuple[ProofGoal, ...]
    bg: tuple[tuple[ProofGoal, ...], ...]
    shelved: tuple[ProofGoal, ...]
    given_up: tuple[ProofGoal, ...]

    def qed(self) -> bool:
        return (
            len(self.fg) == 0
            and len(self.bg) == 0
            and len(self.shelved) == 0
            and len(self.given_up) == 0
        )


def from_coqtail_goals(state: xmlInterface.Goals | None) -> ProofState:
    if not state:
        return ProofState((), (), (), ())

    def transform_goals(gs: list[xmlInterface.Goal]) -> tuple[ProofGoal, ...]:
        return tuple(from_coqtail_goal(g) for g in gs)

    return ProofState(
        fg=transform_goals(state.fg),
        bg=tuple(transform_goals(gs) for gs in state.bg),
        shelved=transform_goals(state.shelved),
        given_up=transform_goals(state.given_up),
    )


def static_tactic_recommendations(state: ProofState) -> Iterable[str]:
    print("[query]", state.fg[0].conclusion)
    x = ["n", "x", "y", "z"]

    yield from (
        "Proof.",
        "intros.",
        "rewrite mul_add_distr_r.",
        "rewrite mul_add_distr_r.",
        "rewrite (mul_comm x n).",
        "rewrite (mul_comm y n).",
        "rewrite (mul_comm z n).",
        "rewrite <- add_assoc.",
        "rewrite (add_comm (n*z) (n*y)).",
        "reflexivity.",
    )


llm = LLM()
SAMPLES = 10


def llm_tactic_recommendations(state: ProofState) -> Iterable[str]:
    g = state.fg[0]
    hyp = "\n".join(g.hypotheses)
    ccl = g.conclusion
    prompt = f"## Context\n{hyp} ## Goal\n{ccl}\nPlease provide a Coq tactic to proceed with the proof. Output the tactic with no explanation."
    candidates = llm.generate(prompt, SAMPLES)
    print("[recommendations]", *candidates, sep="\n")
    return (step for (step, _prob) in candidates)


@final
class CoqProofEnv(Env[ProofState, str]):
    def __init__(self, top: Coqtop) -> None:
        super().__init__()
        self.top = top

    @override
    def current_state(self) -> ProofState:
        ok, _, goals, _ = self.top.goals()
        assert ok, "Cannot query goals"
        return from_coqtail_goals(goals)

    @override
    def is_win(self) -> bool:
        return self.current_state().qed()

    @override
    def apply(self, step: str) -> bool:
        ok, _, _, _ = self.top.advance(step, True)
        return ok

    @override
    def revert(self, step: str) -> bool:
        _ = step
        ok, _, _, _ = self.top.rewind()
        return ok


def test_search(move_gen: MoveGenerator[ProofState, str]):
    top = Coqtop()
    _ = top.find_coq(None, None)
    [err, _] = top.start(
        filename="/tmp/test-llm-search.v",
        coqproject_args=[],
        use_dune=False,
        dune_compile_deps=False,
        timeout=10,
        stderr_is_warning=True,
    )
    assert err is None, "Coq start up error."

    _ = top.advance("Require Import PeanoNat.", False)
    _ = top.advance("Require Import Nat.", False)
    _ = top.advance("Import PeanoNat.", False)
    _ = top.advance("Import Nat.", False)
    _ = top.advance(
        "Theorem add_mul3 : forall x y z n, (x + y + z) * n = n * x + z * n + n * y.",
        True,
    )

    env = CoqProofEnv(top)
    bfs = BFSearch(env, move_gen)
    proof = bfs.run()
    print("A proof was found:", proof)

    pass


if __name__ == "__main__":
    test_search(static_tactic_recommendations)
    test_search(llm_tactic_recommendations)
