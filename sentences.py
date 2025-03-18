import coqtail
import re


next_sentence = coqtail._find_next_sentence
get_text = coqtail._between


def purify(s: bytes) -> str:
    """
    clear out comments, whitespace in a coq command
    """
    s, _ = coqtail._strip_comments(s)
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
