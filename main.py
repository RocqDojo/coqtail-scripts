import sys
from dataclasses import dataclass, field

from subprocess import check_output
import coqtop
import sentences
from serialization import json_dump
from xmlInterface import Goals

TIMEOUT = 30

top = coqtop.Coqtop()

filename = sys.argv[1]
save = filename + ".json"
args = sys.argv[2:]
print(f"working on [{filename}] with arguments {args}")
lines = open(filename, "rb").readlines()
steps = sentences.split_sentences(lines)

print(steps)

version = top.find_coq(None, None)
print("using coq version:", version)

[err, msg] = top.start(
    filename=filename,
    coqproject_args=args,
    use_dune=False,
    dune_compile_deps=False,
    timeout=TIMEOUT,
    stderr_is_warning=True,
)
assert err is None
print("coqtop start message", msg)

"""
notations and naming conventions from Ruixiang

(context)
H1 : P
H2 : P -> Q

============

(goal)
Q
"""


@dataclass
class Step:
    goals: Goals
    # when seeing (the context and goal), one should proceed with the tactic
    tactic: str = ""


def get_premises(coqtop_resp: str) -> list[str]:
    return coqtop_resp.split("\n")[1:]


@dataclass
class Theorem:
    kind: str = ""
    name: str = ""
    definition: str = ""
    steps: list[Step] = field(default_factory=lambda: [])
    cmds: list[str] = field(default_factory=lambda: [])
    premises: list[str] = field(default_factory=lambda: [])


"""
Whenever Qed./Admitted./Defined. is encountered

Call coqtop query:
Print Opaque Dependencies [theorem-name]

Collect the results in premises
"""


thm: Theorem | None = None
theorems: list[Theorem] = []

for cmd in steps:
    _, _, before_state, _ = top.subgoals()
    ok, _, _, _ = top.advance(cmd, False)
    assert ok
    _, _, after_state, _ = top.subgoals()

    # state transition: start proving a theorem
    if before_state is None and after_state is not None:
        if not thm:
            thm = Theorem()
            try:
                thm.kind, thm.name, thm.definition = sentences.proof_meta(cmd)
                print("+ working on", thm.kind, thm.name, thm.definition)
            except Exception as _:
                thm = None
    else:
        # update theorem recording
        if thm:
            thm.cmds.append(cmd)  # record the raw commands

            assert before_state is not None
            thm.steps.append(Step(before_state, cmd))

            # state transition: theorem proved
            if after_state is None:
                # extract opaque dependencies
                _, premises, _, _ = top.query(
                    f"Print Opaque Dependencies {thm.name}.", in_script=False
                )
                thm.premises = get_premises(premises)

                theorems.append(thm)
                thm = None
                print("- done with last theorem")


json_dump(theorems, save)

jq_filter = "map(.steps[].goals |= {focused_state: .fg, background_states: .bg, shelved: .shelved, given_up: .given_up})"

renamed = ""
with open(save, "r") as f:
    renamed = check_output(["jq", jq_filter], stdin=f)
with open(save, "wb") as f:
    _ = f.write(renamed)
