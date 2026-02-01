import streamlit as st
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

# --- VOYNICH ENGINE LOGIC (VVM v2) ------------------------------------------

OPERATORS = {
    'q':  'Initialization/Header',
    'p':  'Natural/Raw',
    'f':  'Processed/Pharma',
    'ch': 'Biological/Balneo',
    't':  'Transition/Boundary',
    'k':  'Potentia/Intensifier',   # add now even if unused
}

CORES = {
    'aiin': 'Primary Attribute (payload)',
    'oke':  'Index/Key',
    'ol':   'Transmission/Flow',
    'che':  'Interface/Integration',
}

FINALIZERS = {
    'dy': 'Stable/NULL (record end)',
    'y':  'Open/Continue',
    's':  'Terminal/Halt',
    'm':  'Pointer/Link',
}

# Rule R: [P0] - [C] - [S0]  (hyphenated form preferred)
TOKEN_RE = re.compile(r'^(?:(q|p|f|ch|t|k)-)?(aiin|oke|ol|che)(?:-(dy|y|s|m))?$')

@dataclass
class VVMResult:
    token: str
    p0: Optional[str]
    c: Optional[str]
    s0: Optional[str]
    op: str
    payload: str
    state: str
    ok: bool
    error_code: Optional[str] = None
    error_msg: Optional[str] = None

def classify_error(token: str, p: Optional[str], c: Optional[str], s: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Semantic checks beyond regex parse.
    Return (error_code, message) or (None, None)
    """
    # E2: Type mismatch rules (starter set, expand over time)
    # - q initializes; should not terminate
    if p == 'q' and s in ('s', 'dy'):
        return ("E2", "Type mismatch: 'q-' header cannot terminate with '-s' or '-dy'")

    # - Missing suffix after q (optional: either allowed or warned)
    # If you want q- to always start a record block, you can enforce suffix=None only.
    # We'll just warn via E2-lite; keep as warning not error.
    # (Handled separately as warnings, not fatal)

    # - k usually implies pharma/intensifier; allow for now but could flag in non-pharma contexts later (E1).
    return (None, None)

def parse_voynich(text: str) -> List[VVMResult]:
    raw_tokens = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', text.lower())
    results: List[VVMResult] = []

    for token in raw_tokens:
        m = TOKEN_RE.match(token)
        if not m:
            # Parse failure (syntax)
            results.append(VVMResult(
                token=token, p0=None, c=None, s0=None,
                op="ERROR", payload="Syntax Fault", state="Invalid",
                ok=False, error_code="PARSE", error_msg="Does not match Rule R: (P0-)?C(-S0)?"
            ))
            continue

        p, c, s = m.groups()

        # Base interpretation
        op = OPERATORS.get(p, "Data Only")
        payload = CORES.get(c, "Unknown Core")
        state = FINALIZERS.get(s, "In Transit")

        # Semantic checks (type/system constraints)
        ecode, emsg = classify_error(token, p, c, s)
        ok = (ecode is None)

        if not ok:
            results.append(VVMResult(
                token=token, p0=p, c=c, s0=s,
                op=op, payload=payload, state=state,
                ok=False, error_code=ecode, error_msg=emsg
            ))
        else:
            results.append(VVMResult(
                token=token, p0=p, c=c, s0=s,
                op=op, payload=payload, state=state,
                ok=True
            ))

    return results

def derive_header_stats(results: List[VVMResult]) -> Dict[str, int]:
    """
    Simple metrics you can paste to Gemini: counts of q- at beginnings etc.
    """
    q_count = sum(1 for r in results if r.p0 == 'q')
    dy_count = sum(1 for r in results if r.s0 == 'dy')
    s_count  = sum(1 for r in results if r.s0 == 's')
    parse_fail = sum(1 for r in results if r.error_code == "PARSE")
    semantic_fail = sum(1 for r in results if r.error_code and r.error_code != "PARSE")
    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    return {
        "total_tokens": total,
        "ok": ok_count,
        "parse_fail": parse_fail,
        "semantic_fail": semantic_fail,
        "q_headers": q_count,
        "dy_stable": dy_count,
        "s_terminal": s_count,
    }

# --- STREAMLIT UI ------------------------------------------------------------

st.set_page_config(page_title="Voynich Virtual Machine v2.0", layout="wide")
st.title("📜 Voynich Virtual Machine (VVM) — Validator + Linter")
st.markdown("### Rule R: `[P0-] C [-S0]`  (hyphenated tokens validated; errors logged as E1/E2/E3)")

default = "p-aiin-dy f-aiin-dy ch-aiin-s q-oke f-ol-m q-aiin-s"
input_text = st.text_area("Input EVA tokens (space-separated):", default, height=120)

if st.button("Execute Procedure"):
    output = parse_voynich(input_text)

    # Summary stats
    stats = derive_header_stats(output)
    st.subheader("Run Summary")
    st.json(stats)

    st.subheader("Execution Trace")
    for r in output:
        if not r.ok:
            # Error classification display
            st.error(f"❌ {r.token} [{r.error_code}] — {r.error_msg}")
        else:
            st.success(f"✔️ **{r.token}** → {r.op} | {r.payload} | {r.state}")

    # Exception / audit log (Gemini-friendly)
    st.subheader("Exception Log (copy/paste)")
    exception_rows = [asdict(r) for r in output if not r.ok]
    st.code(json.dumps(exception_rows, indent=2), language="json")

    st.subheader("Full Parse (copy/paste)")
    all_rows = [asdict(r) for r in output]
    st.code(json.dumps(all_rows, indent=2), language="json")

st.sidebar.info("VVM v2 acts as a *validator*: it does not silently normalize tokens.")
