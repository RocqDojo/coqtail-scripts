import sys
import json
import os

import coqtop
import sentences

from tactic_gen import RandomMock as HintGenerator
# from tactic_gen import SftLlmQuery as HintGenerator

TIMEOUT = 30
MAX_SAMPLE = 10

top = coqtop.Coqtop()

hint_generator = HintGenerator()
result_suffix = '.' + os.environ.get("RESULT_SUFFIX", default="result")

coq_src = sys.argv[1]
json_records = coq_src + ".json"
test_results = coq_src + result_suffix
args = sys.argv[2:]
print(f"working on [{coq_src}] with arguments {args}")

if os.path.exists(test_results):
    print(f'result file {test_results} already exist, skipped')

lines = open(coq_src, "rb").readlines()
proofs = json.load(open(json_records))
steps = sentences.split_sentences(lines)


version = top.find_coq(None, None)
print("using coq version:", version)

[err, msg] = top.start(
    filename=coq_src,
    coqproject_args=args,
    use_dune=False,
    dune_compile_deps=False,
    timeout=TIMEOUT,
    stderr_is_warning=True,
)
print("coqtop start message", msg)
assert err is None


step_index = 0


def forward():
    """
    feed the next command to coqtop
    stop on error
    """
    global step_index

    step = steps[step_index]
    print("Feeding", step.text)
    ok, _, _, _ = top.advance(step.text, True)
    assert ok
    step_index += 1
    return step.line, step.col


def parse_goal(msg: str) -> tuple[list[str], list[str]]:
    """
    parse the local context the conclusion line from [Show.] output
    """

    # skip the 'goal 1 is:' line
    msg_lines = msg.splitlines()[1:]
    hypothesis: list[str] = []
    conclusion: list[str] = []
    seen_separator = False

    for line in msg_lines:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        # detect the ===== line
        if line.startswith("==="):
            seen_separator = True
            continue
        if seen_separator:
            conclusion.append(line)
        else:
            hypothesis.append(line)
    return hypothesis, conclusion


results: list[dict[str, bool | int | list[str]]] = []

for proof in proofs:
    for pstep in proof["steps"]:
        # step forward until the step for testing
        line, col = int(pstep["loc_line"]), int(pstep["loc_col"])
        while True:
            pos = forward()
            if pos == (line, col):
                break

        # make sure that we are working on a focused open proof
        _, msg, _, _ = top.query("Show 1.", False)
        if not msg.startswith("goal 1 is:"):
            continue
        hypothesis, conclusion = parse_goal(msg)
        hyp_text, ccl_text = "\n".join(hypothesis), "\n".join(conclusion)

        # sample many tactic hints and examine them
        attempts: list[str] = []
        succ = False
        for _ in range(MAX_SAMPLE):
            tactic = hint_generator.query(hyp_text, ccl_text)
            attempts.append(tactic)
            ok, _, _, _ = top.advance(tactic, True)
            if ok:
                succ = True
                _ = top.rewind()
                break
        print(f"Made {len(attempts)} attempts for {hypothesis} {conclusion}")
        results.append(
            dict(
                hypothesis=hypothesis,
                conclusion=conclusion,
                attempts=attempts,
                loc_line=line,
                loc_col=col,
                succeeded=succ,
            )
        )


json.dump(results, open(test_results, "w"))
