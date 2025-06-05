"""
coqxml.py – minimal yet robust client for the Coq XML protocol (8.6 – 8.19).

Author : chatGPT-o3 demo (2025-06-04)
Licence: MIT
"""

from __future__ import annotations

import copy
import os
import select
import shutil
import subprocess
import threading
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from queue import Empty, Queue
from typing import IO, Any, final, override

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


# _log = print


def _log(*args, **kwargs):
    _ = args
    _ = kwargs
    pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


CHARMAP = {b"&nbsp;": b" ", b"&apos;": b"'", b"&#40;": b"(", b"&#41;": b")"}


def _unescape(cmd: bytes) -> bytes:
    """Replace escaped characters with the unescaped version."""
    for escape, unescape in CHARMAP.items():
        cmd = cmd.replace(escape, unescape)
    return cmd


def _bool(b: bool) -> str:
    return f'<bool val="{"true" if b else "false"}"/>'


# --------------------------------------------------------------------------- #
# Exceptions & result wrappers
# --------------------------------------------------------------------------- #


@final
class CoqError(RuntimeError):
    """Raised when a <value val="fail"> is received."""

    def __init__(self, message: str, loc_xml: str | None = None) -> None:
        super().__init__(message)
        self.loc_xml = loc_xml or ""


@final
class CoqReturn:
    """Wrapper around <value …> replies."""

    def __init__(self, root: ET.Element):
        self.root = root
        self.ok = root.get("val") == "good"
        if not self.ok:
            # Extract rich error string if available
            msg = "".join(root.itertext())
            loc = root.find(".//option")
            raise CoqError(
                msg.strip(),
                ET.tostring(loc, encoding="unicode") if loc is not None else None,
            )

    # convenience
    def state_id(self) -> int | None:
        sid = self.root.find(".//state_id")
        if sid is not None:
            val = sid.get("val")
            if val is not None:
                return int(val)
        return None

    def xml(self) -> str:
        return ET.tostring(self.root, encoding="unicode")

    # A tiny parser for <goals> (optional)
    def goals(self) -> Goals | None:
        g = self.root.find(".//goals")
        return Goals.from_xml(g) if g is not None else None


@final
class CoqFeedback:
    """Wrapper around <feedback …> messages."""

    def __init__(self, root: ET.Element):
        self.root = root
        self.kind = root.find("feedback_content").get("val")
        self.state_id: int | None = None
        sid = root.find("state_id")
        if sid is not None:
            self.state_id = int(sid.get("val"))

    def xml(self) -> str:
        return ET.tostring(self.root, encoding="unicode")

    @override
    def __repr__(self) -> str:
        return f"<Feedback {self.kind} state={self.state_id}>"


# --------------------------------------------------------------------------- #
# A *very* small goal structure helper
# --------------------------------------------------------------------------- #


@final
class Goal:
    def __init__(self, ident: str, hyps: list[str], concl: str):
        self.ident = ident
        self.hyps = hyps
        self.concl = concl

    @classmethod
    def from_xml(cls, g: ET.Element) -> Goal:
        ident = g.findtext("string", default="")
        hyps = [ET.tostring(h, encoding="unicode") for h in g.find("list")]
        concl = ET.tostring(g.find("richpp"), encoding="unicode")
        return cls(ident, hyps, concl)

    @override
    def __repr__(self) -> str:
        return f"<Goal {self.ident}: hyps={self.hyps} concl={self.concl}>"


@final
class Goals:
    """Very partial mapping of Coq’s complex goal structure."""

    def __init__(
        self,
        fg: list[Goal],
        bg: list[tuple[list[Goal], list[Goal]]],
        shelved: list[Goal],
        given_up: list[Goal],
    ):
        self.fg, self.bg, self.shelved, self.given_up = fg, bg, shelved, given_up

    @classmethod
    def from_xml(cls, g: ET.Element) -> "Goals":
        fg = [Goal.from_xml(goal) for goal in g.find("./list")]
        bg_pairs = []
        for pair in g.findall("./list[2]/pair"):
            lists = pair.findall("list")
            before = [Goal.from_xml(goal) for goal in lists[0].findall("goal")]
            after = [Goal.from_xml(goal) for goal in lists[1].findall("goal")]
            bg_pairs.append((before, after))

        def _optlist(idx: int) -> list[Goal]:
            opt = g.find(f"./list[{idx}]")
            return (
                [Goal.from_xml(goal) for goal in opt.findall("goal")]
                if opt is not None
                else []
            )

        return cls(fg, bg_pairs, _optlist(3), _optlist(4))


# --------------------------------------------------------------------------- #
# XML streaming reader (incremental, safe)
# --------------------------------------------------------------------------- #


@final
class _XMLStreamThread(threading.Thread):
    """
    Reads bytes from the subprocess stdout, feeds them to an XMLPullParser,
    and emits COMPLETE <value>, <feedback> or <prompt> elements into a Queue.
    """

    ROOTS = {"value", "feedback", "ltac_debug"}

    def __init__(
        self,
        raw_stdout: IO[bytes],
        resp_q: Queue[ET.Element],
        fb_q: Queue[ET.Element],
        ltac_dbg_q: Queue[ET.Element],
    ):
        super().__init__(daemon=True)
        self.raw = raw_stdout  # binary buffered reader
        self.resp_q = resp_q
        self.fb_q = fb_q
        self.ltac_dbg_q = ltac_dbg_q
        self.stop_flag = False
        self.parser = ET.XMLPullParser(events=("end",))

    @override
    def run(self):
        # NOTE: dummy root for XML stream
        """
        XML Pull Parser handles only one XML document.
        That is, one top-level element.
        However, we will receive multiple XML documents from the stream.
        To combat this misalignment, we shall add a fake root tag.
        """

        self.parser.feed(b"<coq_xml_stream>")

        while not self.stop_flag:
            # OS-level select for non-blocking read
            r, _, _ = select.select([self.raw], [], [], 0.1)
            if not r:
                continue
            chunk = os.read(self.raw.fileno(), 4096)
            if not chunk:  # EOF – Coq exited
                self.stop_flag = True
                break
            _log("[reader] feed", _unescape(chunk).decode())
            self.parser.feed(_unescape(chunk))
            for ev, elem in self.parser.read_events():
                if ev == "end" and elem.tag in self.ROOTS:
                    # Detach full element, push XML string to queue
                    # deepcopy, otherwise elem.clear() will clear up the underlying object
                    q = self.resp_q
                    if elem.tag == "feedback":
                        q = self.fb_q
                    elif elem.tag == "ltac_debug":
                        q = self.ltac_dbg_q
                    q.put(copy.deepcopy(elem))
                    _log("[reader] put", ET.tostring(elem).decode())
                    # Important: clear to free memory
                    elem.clear()


# --------------------------------------------------------------------------- #
# Main client
# --------------------------------------------------------------------------- #


@final
class CoqIdeTopClient:
    def __init__(self, exe: str | None = None, extra_args: Sequence[str] | None = None):
        exe = exe or _guess_exe()
        extra_args = ["-main-channel", "stdfds"] + list(extra_args or [])
        self.proc = subprocess.Popen(
            [exe, *extra_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # unbuffered binary
        )
        self._resp_q: Queue[ET.Element] = Queue()
        self._fb_q: Queue[ET.Element] = Queue()
        self._ltac_dbg_q: Queue[ET.Element] = Queue()
        self._reader = _XMLStreamThread(
            self.proc.stdout,
            self._resp_q,
            self._fb_q,
            self._ltac_dbg_q,
        )
        self._reader.start()

    # ------------- low-level I/O -------------

    def _send(self, xml: str) -> None:
        """Low-level send (adds trailing newline, UTF-8)."""
        _log("[driver] send", xml)
        if self.proc.stdin is None:
            raise RuntimeError("coq process already closed")
        self.proc.stdin.write(xml.encode("utf-8"))
        if not xml.endswith("\n"):
            self.proc.stdin.write(b"\n")
        self.proc.stdin.flush()

    def _recv_resp(self, timeout: float | None = None) -> CoqReturn:
        """receive response message for requests"""
        try:
            elem = self._resp_q.get(timeout=timeout)
            _log("[driver] receive", ET.tostring(elem).decode())
        except Empty:
            raise TimeoutError("no response from coqidetop")
        assert elem.tag == "value"
        return CoqReturn(elem)

    def _call(self, name: str, body: str, timeout: float | None = None) -> CoqReturn:
        self._send(f'<call val="{name}">{body}</call>')
        resp = self._recv_resp(timeout)
        return resp

    # ------------- public protocol helpers -------------

    # about/init/quit -------------------------------------------------------
    def about(self) -> CoqReturn:
        return self._call("About", "<unit/>")

    def init(self, v_file: str | None = None) -> CoqReturn:
        if v_file:
            body = f'<option val="some"><string>{_esc(v_file)}</string></option>'
        else:
            body = '<option val="none"/>'
        return self._call("Init", body)

    def quit(self) -> None:
        try:
            self._send('<call val="Quit"><unit/></call>')
            # best effort – ignore response
        finally:
            self.close()

    # ---------------------------------------------------------------------
    # Add / Edit_at
    # ---------------------------------------------------------------------

    def add(
        self,
        command: str,
        state_id: int,
        verbose: bool = False,
        bp: int = 0,
        line_nb: int = 1,
        bol_pos: int = 0,
        # <edit_id> has been discarded. It has no use in the protocol
        edit_id: int = 0,
    ) -> CoqReturn:
        body = f"""
        <pair>
          <pair>
            <pair>
              <pair>
                <string>{command}</string>
                <int>{edit_id}</int>
              </pair>
              <pair>
                <state_id val="{state_id}"/>
                {_bool(verbose)}
              </pair>
            </pair>
            <int>{bp}</int>
          </pair>
          <pair>
            <int>{line_nb}</int>
            <int>{bol_pos}</int>
          </pair>
        </pair>"""
        return self._call("Add", body, timeout=10)

    def edit_at(self, state_id: int) -> CoqReturn:
        return self._call("Edit_at", f'<state_id val="{state_id}"/>')

    # ---------------------------------------------------------------------
    # Goals & status
    # ---------------------------------------------------------------------

    def goal(self) -> CoqReturn:
        return self._call("Goal", "<unit/>")

    def subgoals(
        self,
        mode: str = "full",
        fg: bool = True,
        bg: bool = True,
        shelved: bool = True,
        given_up: bool = True,
    ) -> CoqReturn:
        flags = (
            f"<goal_flags><string>{mode}</string>"
            f"{_bool(fg)}{_bool(bg)}{_bool(shelved)}{_bool(given_up)}</goal_flags>"
        )
        return self._call("Subgoals", flags)

    def status(self, force: bool = False) -> CoqReturn:
        return self._call("Status", _bool(force))

    # ---------------------------------------------------------------------
    # Misc simple unit commands
    # ---------------------------------------------------------------------

    def evars(self) -> CoqReturn:
        return self._call("Evars", "<unit/>")

    def hints(self) -> CoqReturn:
        return self._call("Hints", "<unit/>")

    def stop_worker(self, w: str) -> CoqReturn:
        return self._call("StopWorker", f"<string>{w}</string>")

    def db_continue(self, mode: int) -> CoqReturn:
        return self._call("Db_continue", f"<int>{mode}</int>")

    def db_stack(self) -> CoqReturn:
        return self._call("Db_stack", "<unit/>")

    def db_vars(self, frame: int = 0) -> CoqReturn:
        return self._call("Db_vars", f"<int>{frame}</int>")

    def print_ast(self, sid: int) -> CoqReturn:
        return self._call("PrintAst", f'<state_id val="{sid}"/>')

    # ---------------------------------------------------------------------
    # Query / Search
    # ---------------------------------------------------------------------

    def query(self, query: str, state_id: int, route_id: int = 0) -> CoqReturn:
        body = (
            f'<pair><route_id val="{route_id}"/>'
            f"<pair><string>{query}</string>"
            f'<state_id val="{state_id}"/></pair></pair>'
        )
        return self._call("Query", body)

    def search(self, constraints: Sequence[tuple[str, str, bool]]) -> CoqReturn:
        """
        constraints = [(ctype, value, positive)]
        NB: For "include_blacklist" the *value must be omitted*.
        """
        items = []
        for ctype, cvalue, pos in constraints:
            if ctype == "include_blacklist":
                items.append(f'<pair><search_cst val="{ctype}"/>{_bool(pos)}</pair>')
            else:
                items.append(
                    f'<pair><search_cst val="{ctype}"><string>{cvalue}</string>'
                    f"</search_cst>{_bool(pos)}</pair>"
                )
        return self._call("Search", f"<list>{''.join(items)}</list>")

    # ---------------------------------------------------------------------
    # Options
    # ---------------------------------------------------------------------

    _OPT_INNER_TAG = {
        "intvalue": "int",
        "boolvalue": "bool",
        "stringvalue": "string",
        "identvalue": "string",
    }

    def get_options(self) -> CoqReturn:
        return self._call("GetOptions", "<unit/>")

    def set_options(self, options: Sequence[tuple[list[str], str, Any]]) -> CoqReturn:
        """
        options: [ ( ["Printing","Width"], "intvalue", 80 ), ... ]
        """
        chunks = []
        for names, variant, value in options:
            inner_tag = self._OPT_INNER_TAG[variant]
            names_xml = "".join(f"<string>{n}</string>" for n in names)
            value_xml = (
                f'<option_value val="{variant}"><option val="some">'
                f"<{inner_tag}>{str(value)}</{inner_tag}></option>"
                f"</option_value>"
            )
            chunks.append(f"<pair><list>{names_xml}</list>{value_xml}</pair>")
        return self._call("SetOptions", f"<list>{''.join(chunks)}</list>")

    # ---------------------------------------------------------------------
    # Breakpoints, annotate, mkcases (omitted for brevity – straight-forward)
    # ---------------------------------------------------------------------

    # ------------- housekeeping -------------

    def close(self):
        try:
            self.proc.kill()
        finally:
            self.proc.wait()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.quit()


# --------------------------------------------------------------------------- #
# util
# --------------------------------------------------------------------------- #


def _guess_exe() -> str:
    """Return best candidate for coqidetop executable."""
    for candidate in ("coqidetop.opt", "coqidetop"):
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError("coqidetop(.opt) not found in $PATH")


# --------------------------------------------------------------------------- #
# tiny manual test when run directly
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # with CoqIdeTopClient() as coq:
    coq = CoqIdeTopClient()
    print("Protocol-version :", coq.about().xml())
    init = coq.init()
    sid0 = init.state_id()
    print("Init state =", sid0)

    r = coq.add("Theorem add_zero_r: forall n, n + 0 = n.", sid0)
    sid1 = r.state_id()
    print("sid1 =", sid1)

    goals = coq.goal().goals()
    print("Goals:", *goals.fg)
    coq.quit()
