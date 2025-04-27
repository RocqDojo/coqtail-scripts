from dataclasses import dataclass
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


@dataclass
class Sentence:
    text: str
    line: int
    col: int


def split_sentences(lines: list[bytes]) -> list[Sentence]:
    """
    split a coq source code into step sentences
    """
    sline, scol = 0, 0
    steps: list[Sentence] = []
    try:
        while True:
            nline, ncol = next_sentence(lines, sline, scol)
            raw = get_text(lines, (sline, scol), (nline, ncol))
            pure = purify(raw)

            steps.append(Sentence(pure, sline, scol))

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


def qualids(cmd: str) -> list[str]:
    """
    Find substrings that are potentially qualified identifiers in a command


    See rocq reference manual for the lexical rules
    https://rocq-prover.org/doc/V9.0.0/refman/language/core/modules.html#qualified-names
    https://rocq-prover.org/doc/V9.0.0/refman/language/core/basic.html#grammar-token-ident
    """
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_']*(?:\.[a-zA-Z_][a-zA-Z0-9_']*)*\b", cmd)


def test_qualids():
    cmd = "Definition foo_bar := 42. Lemma _baz123 := True. Check Module1.SubModule2.func_name."
    expect = [
        "Definition",
        "foo_bar",
        "Lemma",
        "_baz123",
        "True",
        "Check",
        "Module1.SubModule2.func_name",
    ]
    result = qualids(cmd)
    assert expect == result


if __name__ == "__main__":
    test_qualids()
