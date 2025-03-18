import coqtail
import re


next_sentence = coqtail._find_next_sentence
get_text = coqtail._between


def purify(s: bytes | str) -> str:
    """
    clear out comments, whitespace in a coq command
    """
    if isinstance(s, str):
        s = s.encode()

    s, _ = coqtail._strip_comments(s)
    s = s.replace(b"\r", b" ")
    s = s.replace(b"\n", b" ")
    s = s.replace(b"\t", b" ")
    return re.sub(r"\s+", " ", s.decode()).strip()


def split_sentences(lines: list[bytes]) -> list[str]:
    """
    split a coq source code into step sentences
    """
    sline, scol = 0, 0
    steps: list[str] = []
    try:
        while True:
            nline, ncol = next_sentence(lines, sline, scol)
            text = get_text(lines, (sline, scol), (nline, ncol))
            text = purify(text)

            steps.append(text)

            if ncol == len(lines[nline]):
                sline = nline + 1
                scol = 0
            else:
                sline = nline
                scol = ncol + 1
    except Exception:
        pass
    return steps


def proof_meta(cmd: str) -> tuple[str, str, str]:
    """
    <kind> <name> : <type>.

    Theorem add_0_r : forall n, n + 0 = n.
    <kind> : Theorem
    <name> : add_0_r
    <type> : forall n, n + 0 = n
    """
    m = re.match(r"^(\S+)\s+(\S+)\s*:\s*(.+)\.$", cmd)
    assert m is not None
    kind, name, definition = m.groups()
    return (purify(kind), purify(name), purify(definition))
