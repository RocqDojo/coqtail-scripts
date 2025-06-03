"""
brief: compute tactic distributions for a bunch proof steps json files
usage: python count-steps.py
input: provide file path from stdin

example: find compcert/ -name '*.v.json' -type f | python count-steps.py
"""

import sys
import json
from collections import defaultdict


K = 30

if __name__ == "__main__":
    proofs: int = 0
    steps: int = 0
    step_distribution: defaultdict[str, int] = defaultdict(int)

    # for each proof steps json file
    for f in sys.stdin.readlines():
        f = f.strip()
        if len(f) == 0:
            continue
        data = json.load(open(f))
        # number of proofs
        proofs += len(data)
        for record in data:
            cmds = record["cmds"]
            steps += len(cmds)
            for cmd in (cmd.split(" ", maxsplit=1)[0] for cmd in cmds):
                step_distribution[cmd] += 1

    print(f"{steps} steps in {proofs} proofs")

    top_cmds = sorted(
        step_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:K]
    for word, count in top_cmds:
        print(word, ":", count, f"{count / steps * 100: .2f}%")
