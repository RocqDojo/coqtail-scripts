from runner import get_coqlib, coqlib_in_parents
from pathlib import Path
from subprocess import DEVNULL, check_output
import sys

"""
example:

coqdep -Q src/ "" src/ChargeCore/Logics/Pure.v | grep '.vo' -o | wc -l
"""


if __name__ == "__main__":
    coq_src = Path(sys.argv[1])
    try:
        conf = coqlib_in_parents(coq_src)
        assert conf is not None
        coqlib = get_coqlib(conf)
        assert coqlib is not None

        deps = check_output(
            ["coqdep", coq_src.as_posix()] + coqlib, stderr=DEVNULL, encoding="utf-8"
        )
        deps = deps.replace("\r", " ").replace("\n", " ")
        cnt = deps.count(".vo")
        print(coq_src.as_posix(), ":", cnt)
    except Exception as _:
        print("failed", coq_src.as_posix())
