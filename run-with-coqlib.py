"""
brief:
Try to locate _CoqProject/Makefile.conf.coq/Makefile.conf in parent directories
and find the right command line arguments to supply to coqtop.
Then run a python script with the coqtop flags

usage: python run-with-coqlib extract.py /opt/coqgym/coq_projects/compcert/common/AST.v
This will start a process running
python extract.py /opt/coqgym/coq_projects/compcert/common/AST.v -Q compcert /opt/coqgym/coq_projects/compcert
"""

from pathlib import Path
from subprocess import DEVNULL, check_output, check_call
import sys


def get_coqlib(path: Path) -> list[str] | None:
    p = path.as_posix()
    coqlibs_raw = ""
    if p.endswith("Makefile.coq.conf"):
        out = check_output(["grep", "-m1", "COQMF_COQLIBS", p]).decode().strip()
        coqlibs_raw = out.split("=", 1)[1]
    elif p.endswith("Makefile.conf"):
        out = check_output(["grep", "-m1", "COQMF_COQLIBS", p]).decode().strip()
        coqlibs_raw = out.split("=", 1)[1]
    elif p.endswith("_CoqProject"):
        coqlibs_raw = (
            check_output(["cat", p])
            .decode()
            .strip()
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", "")
        )

    # main cases to handle
    # -Q <real> <logic>
    # -R <real> <logic>
    # -I <real>
    #
    # real path need to be processed
    if coqlibs_raw:
        args: list[str] = []
        fix_required = False
        for arg in coqlibs_raw.split(" "):
            if arg.startswith("-"):  # a real path after this argument
                fix_required = True
            elif fix_required:  # processing a physics path
                arg = path.parent.as_posix() + "/" + arg
                fix_required = False

            if arg.endswith(".v"):  # source files list in _CoqProject
                continue

            if len(arg) > 0:
                # special case for empty strings ...
                if arg == "''" or arg == '""':
                    arg = ""
                args.append(arg)

        return args

    return None


def coqlib_in_parents(path: Path) -> Path | None:
    """
    Recursively checks if coqlib configure file exists in `path`'s directory or any parent directory.

    :param path: A Path object representing the starting file or directory.
    :return: Path object of the found file or None if not found.
    """
    directory = path.parent  # Start from the directory of the given path

    while directory != directory.parent:  # Stop at the root directory
        mf_coq_conf = directory / "Makefile.coq.conf"
        mf_conf = directory / "Makefile.conf"
        coq_proj = directory / "_CoqProject"

        if mf_coq_conf.exists():
            return mf_coq_conf
        elif mf_conf.exists():
            return mf_conf
        elif coq_proj.exists():
            return coq_proj

        directory = directory.parent  # Move up to the parent directory

    return None


if __name__ == "__main__":
    script = Path(sys.argv[1])
    coq_src = Path(sys.argv[2])
    try:
        conf = coqlib_in_parents(coq_src)
        assert conf is not None
        coqlib = get_coqlib(conf)
        assert coqlib is not None

        print("starting", script.as_posix(), "on", coq_src.as_posix())
        _ = check_call(
            ["python", script.as_posix(), coq_src.as_posix()] + coqlib,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
    except Exception as e:
        print("failed", coq_src.as_posix())
