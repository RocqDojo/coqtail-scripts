from pathlib import Path
from subprocess import check_output
from re import findall
import sys


def get_coqlib(path: Path) -> list[str] | None:
    p = str(path)
    coqlibs_raw = ""
    if p.endswith("Makefile.coq.conf"):
        out = check_output(["grep", "-m1", "COQMF_COQLIBS", p]).decode().strip()
        coqlibs_raw = out.split("=", 1)[1]
    elif p.endswith("Makefile.conf"):
        out = check_output(["grep", "-m1", "COQMF_COQLIBS", p]).decode().strip()
        coqlibs_raw = out.split("=", 1)[1]
    elif p.endswith("_CoqProject"):
        coqlibs_raw = check_output(["cat", p]).decode().strip()

    if coqlibs_raw:
        pattern = r"(-\w+)\s*(\S+)\s*(\S+)"  # -Q <real-path> <logic-path>
        res: list[str] = []
        for tag, real_path, logical_path in findall(pattern, coqlibs_raw):
            res.append(tag)
            res.append(real_path)
            res.append(logical_path)
        return res
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
    conf = coqlib_in_parents(coq_src)
    assert conf is not None
    coqlib = get_coqlib(conf)
    assert coqlib is not None

    _ = check_output(["python", "./main.py", str(coq_src)] + coqlib)
