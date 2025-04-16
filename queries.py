import coqtop
import re
from dataclasses import dataclass
import sentences


@dataclass
class AboutResult:
    name: str
    signature: str
    kind: str
    full_path: str


def about(top: coqtop.Coqtop, name: str) -> AboutResult | None:
    """
    Parse the About command result
    https://rocq-prover.org/doc/v8.16/refman/proof-engine/vernacular-commands.html
    """
    _, resp, _, _ = top.query(f"About {name}.", in_script=False)
    # breakpoint()
    return extract_about(resp)


def extract_about(about_resp: str) -> AboutResult | None:
    """
    extract fields from coqtop about output
    name, type signature, reference term kind, and fully qualified name
    """
    name, signature, kind, full_path = None, None, None, None

    try:
        match_header = re.search(r"^\s*(\w+)\s*:\s*(.*?)\n", about_resp, re.DOTALL)
        match_expands = re.search(r"Expands to:\s*(\w+)\s+([\w\.]+)", about_resp)

        if match_header and match_expands:
            name = match_header.group(1).strip()
            signature = match_header.group(2).strip()
            kind = match_expands.group(1).strip()
            full_path = match_expands.group(2).strip()

            name = sentences.purify(name)
            signature = sentences.purify(signature)
            kind = sentences.purify(kind)
            full_path = sentences.purify(full_path)

    except Exception as _:
        return None

    if name and signature and kind and full_path:
        return AboutResult(name, signature, kind, full_path)
    return None


def test_extract_about():
    test_extract_about1()
    test_extract_about2()
    test_extract_about3()
    test_extract_about4()
    test_extract_about5()


def test_extract_about1():
    input_text = """reflexivity :
forall {A : Type} {R : A -> A -> Prop}, Reflexive R -> forall x : A, R x x

reflexivity is not universe polymorphic
Arguments reflexivity {A}%type_scope {R}%function_scope {Reflexive} x
reflexivity is transparent
Expands to: Constant Coq.ssr.ssrclasses.reflexivity
"""
    about = extract_about(input_text)
    assert about is not None
    assert about.name == "reflexivity"
    assert (
        about.signature
        == "forall {A : Type} {R : A -> A -> Prop}, Reflexive R -> forall x : A, R x x"
    )
    assert about.kind == "Constant"
    assert about.full_path == "Coq.ssr.ssrclasses.reflexivity"


def test_extract_about2():
    input_text = """add_comm : forall n m : nat, n + m = m + n

add_comm is not universe polymorphic
Arguments add_comm (n m)%nat_scope
add_comm is opaque
Expands to: Constant Coq.Arith.PeanoNat.Nat.add_comm
"""
    about = extract_about(input_text)
    assert about is not None
    assert about.name == "add_comm"
    assert about.signature == "forall n m : nat, n + m = m + n"
    assert about.kind == "Constant"
    assert about.full_path == "Coq.Arith.PeanoNat.Nat.add_comm"


def test_extract_about3():
    input_text = """atrans_sound : (aexp -> aexp) -> Prop

atrans_sound is not universe polymorphic
Arguments atrans_sound atrans%function_scope
atrans_sound is transparent
Expands to: Constant PLF.Equiv.atrans_sound
"""
    about = extract_about(input_text)
    assert about is not None
    assert about.name == "atrans_sound"
    assert about.signature == "(aexp -> aexp) -> Prop"
    assert about.kind == "Constant"
    assert about.full_path == "PLF.Equiv.atrans_sound"


def test_extract_about4():
    input_text = """nat : Set

nat is not universe polymorphic
Expands to: Inductive Coq.Init.Datatypes.nat
"""
    about = extract_about(input_text)
    assert about is not None
    assert about.name == "nat"
    assert about.signature == "Set"
    assert about.kind == "Inductive"
    assert about.full_path == "Coq.Init.Datatypes.nat"


def test_extract_about5():
    input_text = """Ltac Coq.Init.Ltac.auto"""

    about = extract_about(input_text)
    assert about is None


if __name__ == "__main__":
    test_extract_about()
