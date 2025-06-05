# various python script for handling Coq proof script

## Locate Coqlib for a coq source file

The `run-with-coqlib.py` can help find logical path and load path configurations in `_CoqProject` or `Makefile.conf.coq`.

Example use

```bash
python run-with-coqlib extract.py compcert/common/AST.v
```

1. The script will recursively find `_CoqProject` in parent directories of `AST.v`.
2. Identify flags for coqtop in `_CoqProject` file: primarily logical path and load path.
3. Start `extract.py` with the source code path and logical path supplied as command line arguments.

For example, it may execute a command similar to

```bash
python extract.py compcert/common/AST.v -Q compcert CompCert -I compcert/ml
```

## Extract proofs into JSON files

- `extract.py` uses coqtop to check proofs and extract steps in one source code file. The proofs in `theory.v` will be stored in `theory.v.json`
- `extract-many.sh` run `extract.py` on all `.v` files in a directory in parallel.

**Note:** `extract-many.sh` will call `run-with-coqlib.py` to find coqtop flags.

Example use

```bash
eval $(opam env)
bash extract-many.sh /opt/coqgym/coq_projects
```

## Proof scripts statistics

After running `extract-many.sh`, use `count-steps.py` to count number of proofs, number of tactics, and top 30 tactics.

Example use: run proof script statistics on compcert

```bash
find compcert/ -name '*.v.json' -type f | python count-steps.py
```

## Single step tactic generation pass rate testing

- `tactic_gen.py`: single step tactic recommendations.
- `tester.py`: test tactic recommendation on one coq source code file.
- `test-many.sh`: run `tester.py` on all `.v` file in a directory.
- `count-pass.py`: pass rate statistics.

**Note:** you will need to change some configurations to run tests

1. `extract-many.sh` will call `run-with-coqlib.py` to find coqtop flags.
2. `tactic_gen.py` contains the LLM prompt template and sampling configurations.
3. `test-many.sh` should set path to the base LLM, the fine tuned LLM, the theorems/lemmas vector database (if any).

Example use:

```bash
eval $(opam env)
bash test-many.sh /opt/coq_gym/coq_projects

find /opt/coq_gym/coq_projects/compcert -type f | python count-pass.py
```

## Known issues

- On leagacy python versions, type hinting is not supported. One have to use the `typing` package in the standard library. See [typing — Support for type hints — Python 3.13.3 documentation](https://docs.python.org/3/library/typing.html)
- On legacy coq versions, the subgoals query is not included in the XML protocol. One have to switch to the goals query. So one should change `coqtop.subgoals()` to `coqtop.goals()`.


## Acknowledgements

[whonore/Coqtail: Interactive Coq Proofs in Vim](https://github.com/whonore/Coqtail)

The `coqidetop` XML protocol client is taken from the `Coqtail` vim plugin.
We thank Wolf Honore for making the nice vim plugin.

## TODO

The following features and functionalities are to be implemented

- Finding dependencies of a proof
- Locating [top-level definitions](https://rocq-prover.org/doc/V9.0.0/refman/language/core/definitions.html#top-level-definitions)
- Locating [record types definitions](https://rocq-prover.org/doc/V9.0.0/refman/language/core/records.html)
- Locating [syntax extensions and notations](https://rocq-prover.org/doc/V9.0.0/refman/user-extensions/syntax-extensions.html)
- Locating [syntax extensions and notations](https://rocq-prover.org/doc/V9.0.0/refman/user-extensions/syntax-extensions.html)
- Support for [inductive definitions](https://rocq-prover.org/doc/V9.0.0/refman/language/core/inductive.html) and [coinductive definitions](https://rocq-prover.org/doc/V9.0.0/refman/language/core/coinductive.html)
