'''
brief: extract proof steps of proof scripts in one coq source code file
proof steps from partial-maps.v will be stored in partial-maps.v.json

usage: python extract.py <source code file> [coqtop arg0] [coqtop arg1] ...
'''

import sys
from dataclasses import dataclass, field

import coqtop
import queries
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

print(*steps, sep="\n")

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
print("coqtop start message", msg)
assert err is None


@dataclass
class Step:
    goals: Goals
    # when seeing (the context and goal), one should proceed with the tactic
    tactic: str = ""
    premises: list[queries.AboutResult] = field(default_factory=lambda: [])

    # real location in the buffer before applying this step
    loc_line: int = 0
    loc_col: int = 0


# Some common tactic names.
# These names should not be treated as potential lemmas
# https://www.cs.cornell.edu/courses/cs3110/2018sp/a5/coq-tactics-cheatsheet.html
blacklist_names: set[str] = set(
    [
        "intro",
        "intros",
        "unfold",
        "simpl",
        "induction",
        "destruct",
        "discriminate",
        "contradiction",
        "split",
        "left",
        "right",
        "apply",
        "assumption",
        "eapply",
        "exact",
        "inversion",
        "injection",
        "f_equal",
        "rewrite",
        "change",
        "assert",
        "auto",
        "trivial",
        "intuition",
        "eauto",
        "reflexivity",
        "symmetry",
        "transitivity",
    ]
)

# for now, external constants and construtors are extracted
dep_kinds: set[str] = set(
    [
        "Constant",
        "Constructor",
    ]
)


def find_dependent_names(cmd: str) -> list[queries.AboutResult]:
    deps: list[queries.AboutResult] = []
    qualids = sentences.qualids(cmd)
    for name in qualids:
        if name in blacklist_names:
            continue
        about = queries.about(top, name)
        if about is not None and about.kind in dep_kinds:
            deps.append(about)

    return deps


@dataclass
class Theorem:
    kind: str = ""
    name: str = ""
    definition: str = ""
    steps: list[Step] = field(default_factory=lambda: [])
    cmds: list[str] = field(default_factory=lambda: [])


thm: Theorem | None = None
theorems: list[Theorem] = []

for sentence in steps:
    cmd = sentence.text
    line, col = sentence.line, sentence.col

    _, _, before_state, _ = top.goals()
    ok, _, _, _ = top.advance(cmd, False)
    assert ok
    _, _, after_state, _ = top.goals()

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
            thm.steps.append(
                Step(
                    goals=before_state,  # the state on which the tactic is applied
                    tactic=cmd,  # the tactic command
                    premises=find_dependent_names(cmd),  # dependencies of this command
                    loc_line=line,
                    loc_col=col,
                )
            )

            # state transition: theorem proved
            if after_state is None:
                if "Abort" not in cmd:
                    # we will not import aborted proofs
                    theorems.append(thm)
                thm = None
                print("- done with last theorem")


json_dump(theorems, save)
