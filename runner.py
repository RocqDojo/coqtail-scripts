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
    coq_src = Path(sys.argv[1])
    try:
        conf = coqlib_in_parents(coq_src)
        assert conf is not None
        coqlib = get_coqlib(conf)
        assert coqlib is not None

        _ = check_call(
            ["python", "./main.py", coq_src.as_posix()] + coqlib, stdout=DEVNULL
        )
    except Exception as _:
        print("failed", coq_src.as_posix())
