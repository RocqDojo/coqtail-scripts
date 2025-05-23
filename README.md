# various python script for handling Coq proof script

A set of python scripts that can be used to

- Extract [assertion and proofs](https://rocq-prover.org/doc/V9.0.0/refman/language/core/definitions.html#assertions-and-proofs) the definition of `Theorem` `Lemma` `Fact` `Remark` `Collary` `Proposition` `Property`.
- Record [proof steps and proof states of a proof](https://rocq-prover.org/doc/V9.0.0/refman/proofs/writing-proofs/proof-mode.html)

form a coq development

## Usage

Make sure `coqtop` can be located in `$PATH`.

```bash
# extract proof states and proof steps from $coq_src, the result is stored $coq_src.json
python main.py $coq_src $coqtop_flags
# run main.py but find coqtop flags in makefile or _CoqProject
python runner.py $coq_src
# run proof step extraction on every coq source file in $DIR directory.
bash run-with-root.sh $DIR
```

```bash
# test proof step generation for every proof step in $coq_src
python tester.py $coq_src $coqtop_flags
# similar to tester.py but coqtop flags are automatically resolved in _CoqProject or makefile
python test-driver.py $coq_src
# test tactic generation on every coq soruce file in $DIR directory.
bash test-project.sh $DIR
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
