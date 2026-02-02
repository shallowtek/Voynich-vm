import re
import streamlit as st

# ============================================================
# Voynich Virtual Machine (VVM) — Robust Validator/Linter
# Works on Streamlit Cloud without pandas, without Python 3.10+ syntax.
# ============================================================

# ---- Rule R dictionaries (your dossier definitions) ----
OPERATORS = {
    "q": "Initialization/Header",
    "p": "Natural/Raw",
    "f": "Processed/Pharma",
    "ch": "Biological/Balneo",
    "t": "Transition/Boundary",
    "k": "Potentia/Intensity",
}

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

# ---- Stolfi line matcher: <f111r.P.1;H>  payload ----
LINE_RE = re.compile(r"<[^>]+;([HTFGU])>\s*(.*)$")

# Strict: only known cores
RULE_R_STRICT = re.compile(r"^(q|p|f|ch|t|k)?(aiin|oke|ol|che)(dy|y|s|m)?$")

# Linter: allow unknown cores but still enforce P0/S0
RULE_R_LINTER = re.compile(r"^(q|p|f|ch|t|k)?([a-z]+)(dy|y|s|m)?$")


def extract_stolfi_payload(text, channels):
    """Pull payload lines from selected Stolfi channels."""
    out = []
    for raw in text.splitlines():
        m = LINE_RE.search(raw)
        if not m:
            continue
        ch, payload = m.group(1), m.group(2)
        if ch in channels:
            out.append(payload)
    return "\n".join(out)


def strip_annotations(s):
    """Remove brace groups like {*}, {&I} and other markup."""
    s = re.sub(r"\{[^}]*\}", " ", s)
    return s


def tokenize_eva(s):
    """
    Convert payload into EVA-ish tokens.
    - Treat . ! = * , : ; etc as separators
    - Keep hyphens if present, but parse by removing hyphens later
    """
    s = strip_annotations(s)
    s = s.lower()

    # Replace common punctuation separators with spaces
    s = re.sub(r"[.!*=,:;()\[\]\"<>]+", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    toks = []
    for t in s.split(" "):
        if re.fullmatch(r"[a-z-]+", t):
            toks.append(t)
    return toks


def paragraphize(text):
    """
    For Stolfi: treat each extracted line as a paragraph segment.
    For plain: split on blank lines.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return blocks


def parse_token(tok, strict):
    """Parse a single token into (P0,C,S0) using Rule R."""
    clean = tok.replace("-", "")
    rule = RULE_R_STRICT if strict else RULE_R_LINTER
    m = rule.match(clean)
    if not m:
        return {
            "token": tok,
            "P0": "",
            "C": "",
            "S0": "",
            "Operator": "ERROR",
            "CoreMeaning": "Syntax Fault / Rule-R mismatch",
            "Finalizer": "Invalid",
            "ValidRuleR": False,
            "ErrorClass": "Rule R mismatch",
        }

    p0, core, s0 = m.groups()
    p0 = p0 or ""
    s0 = s0 or ""

    op_label = OPERATORS.get(p0, "Data Only") if p0 else "Data Only"
    core_label = CORES.get(core, "Unknown/Unregistered Core") if not strict else CORES.get(core, "Unknown Core")
    fin_label = FINALIZERS.get(s0, "In Transit") if s0 else "In Transit"

    return {
        "token": tok,
        "P0": p0,
        "C": core,
        "S0": s0,
        "Operator": op_label,
        "CoreMeaning": core_label,
        "Finalizer": fin_label,
        "ValidRuleR": True,
        "ErrorClass": "",
    }


def classify_errors(records, paragraph_bounds):
    """
    Adds E1/E2/E3 heuristics.

    E2: q- with -s
    E3: aiin drift (aiim/aiir/aiil etc) when in linter mode
    E1: p- inside a paragraph dominated by pharma-ish prefixes (f/k/ch/t)
    """
    # E2 + E3 token-level
    for r in records:
        if not r["ValidRuleR"]:
            continue
        if r["P0"] == "q" and r["S0"] == "s":
            r["ErrorClass"] = "E2 (Type Mismatch: q- with -s)"
        # E3 drift: only meaningful if we allow unknown cores (linter mode)
        if r["C"] and r["C"].startswith("aii") and r["C"] != "aiin":
            # very conservative drift rule
            if re.fullmatch(r"aii[a-z]{1,3}", r["C"]):
                if not r["ErrorClass"]:
                    r["ErrorClass"] = "E3 (Off-by-One Drift: aiin-variant core)"

    # E1 paragraph-level
    for (a, b) in paragraph_bounds:
        ops = [records[i]["P0"] for i in range(a, b + 1) if records[i]["ValidRuleR"] and records[i]["P0"]]
        if not ops:
            continue
        counts = {}
        for o in ops:
            counts[o] = counts.get(o, 0) + 1
        dominant = max(counts, key=counts.get)

        pharmaish = dominant in ("f", "k", "ch", "t")
        if pharmaish:
            for i in range(a, b + 1):
                if records[i]["ValidRuleR"] and records[i]["P0"] == "p" and not records[i]["ErrorClass"]:
                    records[i]["ErrorClass"] = "E1 (Contextual Cache: p- inside pharma-ish paragraph)"

    return records


def dy_paragraph_audit(records, paragraph_bounds):
    """Compare -dy rate at paragraph ends vs mid-paragraph."""
    dy_last = 0
    last_total = 0
    dy_mid = 0
    mid_total = 0

    for (a, b) in paragraph_bounds:
        if a > b:
            continue
        # last token
        last_total += 1
        if records[b].get("S0") == "dy":
            dy_last += 1
        # middle tokens: a..b-1
        for i in range(a, b):
            mid_total += 1
            if records[i].get("S0") == "dy":
                dy_mid += 1

    dy_last_rate = (float(dy_last) / float(last_total)) if last_total else 0.0
    dy_mid_rate = (float(dy_mid) / float(mid_total)) if mid_total else 0.0

    return {
        "dy_last": dy_last,
        "last_total": last_total,
        "dy_mid": dy_mid,
        "mid_total": mid_total,
        "dy_last_rate": dy_last_rate,
        "dy_mid_rate": dy_mid_rate,
    }


def to_csv(rows):
    """Minimal CSV serializer (no pandas)."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = [",".join(cols)]
    for r in rows:
        line = []
        for c in cols:
            v = str(r.get(c, ""))
            v = v.replace('"', '""')
            if "," in v or "\n" in v:
                v = f'"{v}"'
            line.append(v)
        out.append(",".join(line))
    return "\n".join(out)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Voynich VVM", layout="wide")
st.title("📜 Voynich Virtual Machine (VVM) — Validator + Linter")

st.markdown(
    "Paste **plain EVA tokens** or a **full Stolfi block**. "
    "This app extracts tokens, applies **Rule R**, and produces a lint/error log."
)

with st.sidebar:
    st.header("Options")

    input_mode = st.radio("Input type", ["Auto-detect", "Plain EVA", "Stolfi block"], index=0)
    strict = st.toggle("Strict cores only (aiin/oke/ol/che)", value=False)

    st.subheader("Stolfi channels")
    channels = st.multiselect("Use channels", ["H", "T", "F", "G", "U"], default=["H"])

    st.subheader("Display")
    max_trace = st.slider("Max trace rows (UI safety)", 0, 300, 80, 10)
    show_table = st.toggle("Show table", True)
    show_errors = st.toggle("Show error log", True)
    show_dy_audit = st.toggle("Show -dy audit", True)

default = "p-aiin-dy f-aiin-dy ch-aiin-s\n\nq-oke f-ol-m q-aiin-s"

text = st.text_area("Input text", value=default, height=260)

if st.button("Execute / Lint"):
    # detect
    detected = "plain"
    if input_mode == "Stolfi block":
        detected = "stolfi"
    elif input_mode == "Auto-detect":
        if re.search(r"<[^>]+;[HTFGU]>", text):
            detected = "stolfi"

    if detected == "stolfi":
        payload = extract_stolfi_payload(text, channels=set(channels))
        if not payload.strip():
            st.error("No Stolfi payload extracted. Check channels and that your paste includes '<...;H>' style lines.")
            st.stop()
        paras = paragraphize(payload)
        tokens = []
        bounds = []
        idx = 0
        for para in paras:
            tks = tokenize_eva(para)
            if not tks:
                continue
            tokens.extend(tks)
            bounds.append((idx, idx + len(tks) - 1))
            idx += len(tks)
    else:
        # plain: split on blank lines as paragraphs
        paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        tokens = []
        bounds = []
        idx = 0
        for para in paras:
            tks = [t.lower() for t in re.findall(r"[a-z-]+", para.lower())]
            if not tks:
                continue
            tokens.extend(tks)
            bounds.append((idx, idx + len(tks) - 1))
            idx += len(tks)

    # parse
    records = [parse_token(t, strict=strict) for t in tokens]
    records = classify_errors(records, bounds)

    total = len(records)
    valid = sum(1 for r in records if r["ValidRuleR"])
    errors = total - valid

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens", total)
    c2.metric("Valid Rule R", valid)
    c3.metric("Errors", errors)

    # trace
    if max_trace > 0:
        st.subheader("Execution Trace (preview)")
        shown = 0
        for r in records:
            if shown >= max_trace:
                break
            shown += 1
            if not r["ValidRuleR"]:
                st.error("❌ {0} — {1}".format(r["token"], r["ErrorClass"]))
            else:
                base = "✔️ {0} → {1} | C={2} ({3}) | {4}".format(
                    r["token"], r["Operator"], r["C"], r["CoreMeaning"], r["Finalizer"]
                )
                if r["ErrorClass"]:
                    st.warning(base + "  ⚠️ " + r["ErrorClass"])
                else:
                    st.success(base)

        if total > max_trace:
            st.info("Trace truncated for performance. Use the table for full output.")

    # table + error log
    if show_table:
        st.subheader("Parsed Table")
        st.dataframe(records, use_container_width=True)

    if show_errors:
        st.subheader("Error Log")
        errs = [r for r in records if r.get("ErrorClass")]
        if not errs:
            st.success("No classified errors under current heuristics.")
        else:
            st.dataframe(errs, use_container_width=True)

    # dy audit
    if show_dy_audit:
        st.subheader("State-Transition Audit: -dy placement")
        stats = dy_paragraph_audit(records, bounds)
        st.write(
            {
                "dy at paragraph end": "{0}/{1} ({2:.3f})".format(
                    stats["dy_last"], stats["last_total"], stats["dy_last_rate"]
                ),
                "dy mid-paragraph": "{0}/{1} ({2:.3f})".format(
                    stats["dy_mid"], stats["mid_total"], stats["dy_mid_rate"]
                ),
                "interpretation": "If paragraph-end rate >> mid rate, -dy behaves like an end-of-record terminator.",
            }
        )

    # download
    csv_data = to_csv(records).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="vvm_results.csv", mime="text/csv")
