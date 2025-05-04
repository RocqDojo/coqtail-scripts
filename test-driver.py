import sys
import os
from pathlib import Path
from subprocess import DEVNULL, check_call

from runner import get_coqlib, coqlib_in_parents

DEVICE = os.environ.get('TORCH_DEVICE')

def run_one(coq_src: Path):
    print(f'testing {coq_src} on {DEVICE}')
    try:
        conf = coqlib_in_parents(coq_src)
        assert conf is not None
        coqlib = get_coqlib(conf)
        assert coqlib is not None

        _ = check_call(
            ["python", "./tester.py", coq_src.as_posix()] + coqlib,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
    except Exception as _:
        print("failed", coq_src.as_posix())


if __name__ == "__main__":
    for f in sys.argv[1:]:
        run_one(Path(f))
