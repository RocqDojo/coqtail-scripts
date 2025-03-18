import sys
import json

import coqtop
import sentences
from xmlInterface import Goals, Goal

top = coqtop.Coqtop()

filename = sys.argv[1]
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
    timeout=60,
    stderr_is_warning=True,
)
assert err is None
print("coqtop start message", msg)


class Theorem:
    def __init__(self) -> None:
        self.name: str = ""
        self.typ: str = ""
        self.cmds: list[str] = []
        self.states: list[Goals] = []


thm: Theorem | None = None
theorems: list[Theorem] = []

for cmd in steps:
    _, _, before_state, _ = top.subgoals()
    _ = top.advance(cmd, False)
    _, _, after_state, _ = top.subgoals()

    # state transition: start proving a theorem
    if before_state is None and after_state is not None:
        if not thm:
            thm = Theorem()
            thm.name = cmd
            thm.typ = cmd
            print("+ working on theorem", cmd)

    # update theorem recording
    if thm:
        thm.cmds.append(cmd)
        if after_state is not None:
            thm.states.append(after_state)
        # state transition: theorem proved
        else:
            theorems.append(thm)
            thm = None
            print("- done with last theorem")


json.dump([thm.__dict__ for thm in theorems], open("out", "w"))
