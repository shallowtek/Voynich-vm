import re
import streamlit as st
import pandas as pd

# ============================================================
# Voynich Virtual Machine (VVM) — Validator + Linter
# Accepts:
#   1) Simple EVA tokens (space-separated), e.g. "p-aiin-dy f-aiin-dy ch-aiin-s"
#   2) Full Stolfi blocks with metadata and multiple channels (H/T/F/G/U)
#      e.g. lines like: <f111r.P.1;H>      {*}kchol!chda!r....
# ============================================================

# --- VOYNICH ENGINE LOGIC (Rule R dictionaries) ---
OPERATORS = {
    "q": "Initialization/Header",
    "p": "Natural/Raw",
    "f": "Processed/Pharma",
    "ch": "Biological/Balneo",
    "t": "Transition/Boundary",
    "k": "Potentia/Intensity",
}

# Keep your original canonical cores (you can expand later)
CORES = {
    "aiin": "Primary Attribute",
    "oke": "Index/Key",
    "ol": "Transmission/Flow",
    "che": "Interface/Integration",
}

FINALIZERS = {
    "dy": "Stable/NULL (End-of-record)",
    "y": "Open/Continue",
    "s": "Terminal/Halt",
    "m": "Pointer/Link",
}

# --- Stolfi line matcher: pulls out channel and payload
LINE_RE = re.compile(r"<[^>]+;([HTFGU])>\s*(.*)$")

# --- Strict Rule R (v3): [P0-] + C + [-S0]
# This mode only "validates" cores that are explicitly in CORES.
RULE_R_STRICT = re.compile(r"^(q|p|f|ch|t|k)?(aiin|oke|ol|che)(dy|y|s|m)?$")

# --- Relaxed Rule R: [P0-] + (any EVA-ish core) + [-S0]
# This mode is a LINTER: it classifies known P0/S0, but allows unknown cores.
RULE_R_RELAXED = re.compile(r"^(q|p|f|ch|t|k)?([a-z]+)(dy|y|s|m)?$")


def extract_stolfi_payload(text: str, channels=("H",)) -> str:
    """
    Extracts only the EVA payload from Stolfi transcription blocks.
    Keeps only selected channels: H/T/F/G/U
    """
    out_lines = []
    for raw_line in text.splitlines():
        m = LINE_RE.search(raw_line)
        if not m:
            continue
        chan, payload = m.group(1), m.group(2)
        if chan in channels:
            out_lines.append(payload)
    return "\n".join(out_lines)


def tokenize_eva(payload: str) -> list[str]:
    """
    Tokenizes EVA-ish payload from Stolfi lines.
    - strips brace annotations like {*} {&I} etc.
    - turns punctuation separators . ! * = , ; : into spaces
    - keeps hyphens if user supplied them
    """
    # Remove brace groups: { ... }
    payload = re.sub(r"\{[^}]*\}", " ", payload)

    # Replace typical Stolfi punctuation separators with spaces
    payload = re.sub(r"[.!*=,:;()\[\]\"<>]+", " ", payload)

    # Collapse whitespace
    payload = re.sub(r"\s+", " ", payload).strip()

    # Split and keep only [a-z-]+ tokens
    toks = []
    for t in payload.split(" "):
        t = t.strip().lower()
        if not t:
            continue
        if re.fullmatch(r"[a-z-]+", t):
            toks.append(t)
    return toks


def detect_input_mode(text: str) -> str:
    """
    Heuristic:
    - If it contains '<...;H>' style tags, treat as Stolfi block
    - else treat as plain EVA
    """
    if "<" in text and ";H>" in text:
        return "stolfi"
    if re.search(r"<[^>]+;[HTFGU]>", text):
        return "stolfi"
    return "plain"


def classify_error(token: str, p0: str | None, core: str | None, s0: str | None, paragraph_idx: int) -> str | None:
    """
    Error Log categories (E1/E2/E3) as best-effort lints.
    NOTE: Some of these are heuristic without folio/diagram context.
    """
    # E2: Type mismatch: q- initialization should not be paired with terminal "s"
    if p0 == "q" and s0 == "s":
        return "E2 (Type Mismatch: q- with -s)"

    # E3: Off-by-One Drift heuristic:
    # token looks close to known core 'aiin' but is not exact (e.g. aiin -> aiim/aiir etc.)
    if core and re.fullmatch(r"aii[nmrls]?", core) and core != "aiin":
        return "E3 (Off-by-One Drift: aiin-variant core)"

    # E1 is contextual (needs “module switch” signal). Approximate:
    # If paragraph_idx > 0 and token uses a strong operator (p/f/ch/k/t),
    # we can’t truly know context. We'll log E1 only if token is "p-" and later we see many "f-" in same paragraph,
    # which suggests a carryover. This is handled in paragraph-level pass, not here.
    return None


def parse_tokens(tokens: list[str], strict_cores: bool = True) -> list[dict]:
    """
    Parse a list of EVA-ish tokens according to Rule R.
    Returns records with parsed fields and lint flags.
    """
    results = []

    rule = RULE_R_STRICT if strict_cores else RULE_R_RELAXED

    for tok in tokens:
        clean = tok.replace("-", "")
        m = rule.match(clean)
        if m:
            p0, core, s0 = m.groups()
            p0 = p0 or ""
            s0 = s0 or ""

            op_label = OPERATORS.get(p0, "Data Only") if p0 else "Data Only"

            # payload label only if core in dictionary
            payload_label = CORES.get(core, "Unknown Core") if strict_cores else CORES.get(core, "Unregistered Core")

            state_label = FINALIZERS.get(s0, "In Transit") if s0 else "In Transit"

            results.append(
                {
                    "token": tok,
                    "P0": p0,
                    "C": core,
                    "S0": s0,
                    "Operator": op_label,
                    "CoreMeaning": payload_label,
                    "Finalizer": state_label,
                    "ValidRuleR": True,
                    "ErrorClass": "",
                }
            )
        else:
            results.append(
                {
                    "token": tok,
                    "P0": "",
                    "C": "",
                    "S0": "",
                    "Operator": "ERROR",
                    "CoreMeaning": "Syntax Fault",
                    "Finalizer": "Invalid",
                    "ValidRuleR": False,
                    "ErrorClass": "Rule R mismatch",
                }
            )
    return results


def paragraphize_payload(payload: str) -> list[str]:
    """
    Approximate paragraphs:
    - If payload contains lines, treat each line as a 'paragraph segment'
    - Otherwise split on '  ' or keep as single block.
    For Stolfi extraction we keep newlines from source lines.
    """
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
    if lines:
        return lines
    return [payload.strip()] if payload.strip() else []


def compute_paragraph_stats(records: list[dict], paragraph_boundaries: list[tuple[int, int]]) -> dict:
    """
    paragraph_boundaries: list of (start_idx, end_idx) inclusive indices in records
    Returns dy frequency stats.
    """
    dy_mid = 0
    dy_last = 0
    mid_total = 0
    last_total = 0

    for (a, b) in paragraph_boundaries:
        if a > b:
            continue
        # last token
        last_total += 1
        if records[b].get("S0") == "dy":
            dy_last += 1

        # middle tokens
        if b - a >= 1:
            for i in range(a, b):
                mid_total += 1
                if records[i].get("S0") == "dy":
                    dy_mid += 1

    return {
        "dy_last": dy_last,
        "last_total": last_total,
        "dy_mid": dy_mid,
        "mid_total": mid_total,
        "dy_last_rate": (dy_last / last_total) if last_total else 0.0,
        "dy_mid_rate": (dy_mid / mid_total) if mid_total else 0.0,
    }


def add_error_log(records: list[dict], paragraph_boundaries: list[tuple[int, int]]) -> list[dict]:
    """
    Adds ErrorClass for E1/E2/E3 where applicable.
    E1 is inferred: if within a paragraph the dominant operator is f/ch/k/t,
    and a rare p- appears, flag it as potential cache/carryover.
    """
    # First pass: E2/E3 per-token
    for idx, r in enumerate(records):
        if not r.get("ValidRuleR"):
            continue
        err = classify_error(r["token"], r["P0"] or None, r["C"] or None, r["S0"] or None, 0)
        if err:
            r["ErrorClass"] = err

    # Second pass: E1 paragraph-level heuristic
    for (a, b) in paragraph_boundaries:
        ops = [records[i].get("P0", "") for i in range(a, b + 1) if records[i].get("ValidRuleR")]
        if not ops:
            continue
        # Dominant op among recognized operator prefixes
        filtered = [o for o in ops if o in ("q", "p", "f", "ch", "t", "k")]
        if not filtered:
            continue
        # dominant non-empty operator
        counts = {}
        for o in filtered:
            counts[o] = counts.get(o, 0) + 1
        dominant = max(counts, key=counts.get)

        # if dominant is pharma-ish and we see p-, flag p- as E1
        pharmaish = dominant in ("f", "k", "ch", "t")
        if pharmaish:
            for i in range(a, b + 1):
                if records[i].get("P0") == "p":
                    # don't overwrite stronger errors
                    if not records[i].get("ErrorClass"):
                        records[i]["ErrorClass"] = "E1 (Contextual Cache: p- inside pharma-ish paragraph)"

    return records


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Voynich Virtual Machine (VVM)", layout="wide")

st.title("📜 Voynich Virtual Machine (VVM) — Validator + Linter")
st.markdown(
    """
**Rule R:**  \\( [P_0\\!-]\\; C \\;[-S_0] \\)  
Supports **plain EVA tokens** *and* **full Stolfi blocks** (with `<...;H>` etc).  
"""
)

with st.sidebar:
    st.header("Parsing Options")
    mode = st.radio("Input Mode", ["Auto-detect", "Plain EVA tokens", "Full Stolfi block"], index=0)

    st.subheader("Rule Strictness")
    strict = st.toggle("Strict cores (aiin/oke/ol/che only)", value=False)
    st.caption("Turn ON if you want to validate only the canonical cores. OFF = linter mode.")

    st.subheader("Stolfi Channels")
    channels = st.multiselect("Use channels", ["H", "T", "F", "G", "U"], default=["H"])
    st.caption("Only used for Stolfi input.")

    st.subheader("Outputs")
    show_table = st.toggle("Show table output", value=True)
    show_exec_trace = st.toggle("Show execution trace", value=True)
    show_stats = st.toggle("Show -dy paragraph statistics", value=True)
    show_errors = st.toggle("Show error log (E1/E2/E3)", value=True)

default_text = "p-aiin-dy f-aiin-dy ch-aiin-s q-oke f-ol-m q-aiin-s"

input_text = st.text_area("Paste EVA tokens OR full Stolfi block here:", value=default_text, height=220)

run = st.button("Execute / Lint")

if run:
    # Determine mode
    if mode == "Auto-detect":
        input_mode = detect_input_mode(input_text)
    elif mode == "Plain EVA tokens":
        input_mode = "plain"
    else:
        input_mode = "stolfi"

    if input_mode == "stolfi":
        payload = extract_stolfi_payload(input_text, channels=tuple(channels))
        if not payload.strip():
            st.error("No Stolfi payload found. Check you pasted lines with <...;H>/<...;T> etc and selected channels.")
            st.stop()

        # Paragraph segmentation by extracted lines
        paras = paragraphize_payload(payload)

        tokens_all = []
        paragraph_boundaries = []
        idx0 = 0
        for para in paras:
            toks = tokenize_eva(para)
            if not toks:
                continue
            tokens_all.extend(toks)
            idx1 = idx0 + len(toks) - 1
            paragraph_boundaries.append((idx0, idx1))
            idx0 = idx1 + 1

    else:
        # Plain EVA
        # Treat blank lines as paragraph separators:
        paras = [p.strip() for p in re.split(r"\n\s*\n", input_text.strip()) if p.strip()]
        if not paras:
            st.error("No tokens found.")
            st.stop()

        tokens_all = []
        paragraph_boundaries = []
        idx0 = 0
        for para in paras:
            # tokenization: keep hyphen tokens, split on whitespace
            toks = [t.lower() for t in re.findall(r"[a-z-]+", para.lower()) if t]
            if not toks:
                continue
            tokens_all.extend(toks)
            idx1 = idx0 + len(toks) - 1
            paragraph_boundaries.append((idx0, idx1))
            idx0 = idx1 + 1

    # Parse
    records = parse_tokens(tokens_all, strict_cores=strict)

    # Error log enrichment
    records = add_error_log(records, paragraph_boundaries)

    df = pd.DataFrame(records)

    # Summary ribbon
    valid_count = int(df["ValidRuleR"].sum())
    total = len(df)
    err_count = total - valid_count
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens", total)
    c2.metric("Valid Rule R", valid_count)
    c3.metric("Errors", err_count)

    # Execution trace (compact)
    if show_exec_trace:
        st.subheader("Execution Trace")
        for r in records[:500]:  # safety cap
            if not r["ValidRuleR"]:
                st.error(f"❌ {r['token']}  —  {r['ErrorClass'] or 'Rule R mismatch'}")
            else:
                msg = f"✔️ **{r['token']}** → {r['Operator']} | C={r['C']} ({r['CoreMeaning']}) | {r['Finalizer']}"
                if r.get("ErrorClass"):
                    st.warning(msg + f"  ⚠️  {r['ErrorClass']}")
                else:
                    st.success(msg)

        if len(records) > 500:
            st.info("Trace truncated at 500 tokens for performance. Use the table view for full results.")

    # Table view
    if show_table:
        st.subheader("Parsed Tokens Table")
        st.dataframe(df, use_container_width=True)

    # -dy paragraph statistics
    if show_stats:
        st.subheader("State-Transition Audit: -dy placement (EOF/NULL test)")
        stats = compute_paragraph_stats(records, paragraph_boundaries)
        st.write(
            {
                "dy in last token of paragraph": f"{stats['dy_last']} / {stats['last_total']}  ({stats['dy_last_rate']:.3f})",
                "dy in middle tokens": f"{stats['dy_mid']} / {stats['mid_total']}  ({stats['dy_mid_rate']:.3f})",
                "interpretation": "If dy_last_rate >> dy_mid_rate, dy behaves like an end-of-record/terminator marker.",
            }
        )

    # Error log view
    if show_errors:
        st.subheader("Error Log (E1/E2/E3 + Rule-R mismatches)")
        err_df = df[df["ErrorClass"].astype(str).str.len() > 0].copy()
        if err_df.empty:
            st.success("No classified errors found under current heuristics.")
        else:
            st.dataframe(err_df, use_container_width=True)

    # CSV download
    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vvm_results.csv",
        mime="text/csv",
    )

st.sidebar.info("Based on the 'Voynich as Procedural Almanac' theory. This app is a validator/linter for Rule R.")
```0            ))

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
