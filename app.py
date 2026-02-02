import re
import streamlit as st

# ============================================================
# Voynich Virtual Machine (VVM) — Robust Validator/Linter
# Streamlit Cloud friendly: no pandas, no 3.10+ syntax.
# ============================================================

# ---- Rule R dictionaries (dossier definitions) ----
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

# Regex fallback for non-hyphen tokens
RULE_R_STRICT = re.compile(r"^(q|p|f|ch|t|k)?(aiin|oke|ol|che)(dy|y|s|m)?$")
RULE_R_LINTER = re.compile(r"^(q|p|f|ch|t|k)?([a-z]+)(dy|y|s|m)?$")

# Map of common unicode hyphen/dash chars -> ASCII hyphen
DASH_CHARS = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
    "\u00ad": "-",  # soft hyphen
}

def normalize_dashes(s):
    """Convert various unicode dashes/hyphens to ASCII '-'."""
    for k, v in DASH_CHARS.items():
        s = s.replace(k, v)
    return s

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

def clean_payload_for_tokenizing(s):
    """
    Clean Stolfi-ish payload into something tokenizable.
    Key behavior:
      - remove {...} annotations
      - normalize dashes
      - REMOVE '!' without splitting words (qa!al -> qaal)
      - treat '.' and '=' as separators
      - drop obvious garbage like '%' runs
    """
    s = strip_annotations(s)
    s = normalize_dashes(s)
    s = s.lower()

    # Remove uncertainty markers without splitting
    s = s.replace("!", "")

    # Drop long %%%%% blocks entirely
    s = re.sub(r"%{3,}", " ", s)

    # Make separators into spaces
    # '.' is the big one in Stolfi; '=' also appears
    s = re.sub(r"[.=]+", " ", s)

    # Other punctuation -> space
    s = re.sub(r"[,;:()\[\]\"<>]+", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_eva(s):
    """
    Convert cleaned payload into EVA-ish tokens.
    We keep hyphens if present (for user-entered tokens like p-aiin-dy).
    For Stolfi payload, tokens are usually bare a-z.
    """
    s = clean_payload_for_tokenizing(s)
    if not s:
        return []
    toks = []
    for t in s.split(" "):
        t = t.strip()
        if not t:
            continue
        # Keep only a-z and hyphen
        if re.fullmatch(r"[a-z-]+", t):
            toks.append(t)
    return toks

def paragraphize(text):
    """
    For extracted Stolfi payload: each non-empty line is its own paragraph unit.
    For plain EVA: paragraphs split by blank lines.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return blocks

def parse_token_hyphen_safe(tok, strict=True):
    """
    Parse EVA token using Rule R with correct hyphen semantics.

    Supports:
      p-aiin-dy
      ch-aiin-s
      q-oke
      f-ol-m
    Also supports bare tokens (aiin, qokeedy etc) via regex fallback.
    """
    original = tok
    token = normalize_dashes(tok.strip().lower())

    # Only EVA-ish
    if not re.fullmatch(r"[a-z-]+", token):
        return {
            "token": original, "P0": "", "C": "", "S0": "",
            "Operator": "ERROR",
            "CoreMeaning": "Non-EVA token",
            "Finalizer": "Invalid",
            "ValidRuleR": False,
            "ParseError": "Non-EVA token",
            "Exception": "",
        }

    # Hyphen-aware parse first
    if "-" in token:
        parts = [p for p in token.split("-") if p]  # protect against double hyphens
        p0 = ""
        s0 = ""
        core = ""

        # prefix (p, f, q, ch, t, k)
        if parts and parts[0] in OPERATORS:
            p0 = parts[0]
            parts = parts[1:]

        # suffix (dy, y, s, m)
        if parts and parts[-1] in FINALIZERS:
            s0 = parts[-1]
            parts = parts[:-1]

        core = "".join(parts)  # remaining bits, usually a single core

        # If strict, core must be known
        if strict and core and core not in CORES:
            return {
                "token": original, "P0": p0, "C": core, "S0": s0,
                "Operator": OPERATORS.get(p0, "Data Only") if p0 else "Data Only",
                "CoreMeaning": "Unknown Core (strict mode)",
                "Finalizer": FINALIZERS.get(s0, "In Transit") if s0 else "In Transit",
                "ValidRuleR": False,
                "ParseError": "Unknown core (strict)",
                "Exception": "",
            }

        # Accept (even if core unknown in linter mode)
        return {
            "token": original,
            "P0": p0,
            "C": core,
            "S0": s0,
            "Operator": OPERATORS.get(p0, "Data Only") if p0 else "Data Only",
            "CoreMeaning": CORES.get(core, "Unknown/Unregistered Core") if core else "Missing Core",
            "Finalizer": FINALIZERS.get(s0, "In Transit") if s0 else "In Transit",
            "ValidRuleR": True if core else False,
            "ParseError": "" if core else "Missing core",
            "Exception": "",
        }

    # Regex fallback for non-hyphen tokens
    clean = token
    rule = RULE_R_STRICT if strict else RULE_R_LINTER
    m = rule.match(clean)
    if not m:
        return {
            "token": original, "P0": "", "C": "", "S0": "",
            "Operator": "ERROR",
            "CoreMeaning": "Syntax Fault / Rule-R mismatch",
            "Finalizer": "Invalid",
            "ValidRuleR": False,
            "ParseError": "Rule R mismatch",
            "Exception": "",
        }

    p0, core, s0 = m.groups()
    p0 = p0 or ""
    s0 = s0 or ""

    if strict and core not in CORES:
        return {
            "token": original, "P0": p0, "C": core, "S0": s0,
            "Operator": OPERATORS.get(p0, "Data Only") if p0 else "Data Only",
            "CoreMeaning": "Unknown Core (strict mode)",
            "Finalizer": FINALIZERS.get(s0, "In Transit") if s0 else "In Transit",
            "ValidRuleR": False,
            "ParseError": "Unknown core (strict)",
            "Exception": "",
        }

    return {
        "token": original,
        "P0": p0,
        "C": core,
        "S0": s0,
        "Operator": OPERATORS.get(p0, "Data Only") if p0 else "Data Only",
        "CoreMeaning": CORES.get(core, "Unknown/Unregistered Core"),
        "Finalizer": FINALIZERS.get(s0, "In Transit") if s0 else "In Transit",
        "ValidRuleR": True,
        "ParseError": "",
        "Exception": "",
    }

def classify_exceptions(records, paragraph_bounds, strict):
    """
    Adds E1/E2/E3 heuristics as *exceptions*, not parse errors.

    E2: q- with -s
    E3: aiin drift (aiim/aiir/aiil etc) when NOT strict (linter mode)
    E1: p- inside a paragraph dominated by pharma-ish prefixes (f/k/ch/t)
    """
    # E2 + E3 token-level
    for r in records:
        if not r["ValidRuleR"]:
            continue

        # E2
        if r["P0"] == "q" and r["S0"] == "s":
            r["Exception"] = "E2 (Type Mismatch: q- with -s)"

        # E3 only in linter mode (strict=False)
        if (not strict) and r["C"]:
            # Detect aiin-like variants conservatively
            if r["C"].startswith("aii") and r["C"] != "aiin":
                if re.fullmatch(r"aii[a-z]{1,3}", r["C"]):
                    if not r["Exception"]:
                        r["Exception"] = "E3 (Off-by-One Drift: aiin-variant core)"

    # E1 paragraph-level (only if we have paragraph segmentation)
    for (a, b) in paragraph_bounds:
        ops = [records[i]["P0"] for i in range(a, b + 1)
               if records[i]["ValidRuleR"] and records[i]["P0"]]
        if not ops:
            continue

        counts = {}
        for o in ops:
            counts[o] = counts.get(o, 0) + 1
        dominant = max(counts, key=counts.get)

        pharmaish = dominant in ("f", "k", "ch", "t")
        if pharmaish:
            for i in range(a, b + 1):
                if records[i]["ValidRuleR"] and records[i]["P0"] == "p" and not records[i]["Exception"]:
                    records[i]["Exception"] = "E1 (Contextual Cache: p- inside pharma-ish paragraph)"

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
                v = '"{0}"'.format(v)
            line.append(v)
        out.append(",".join(line))
    return "\n".join(out)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Voynich VVM", layout="wide")
st.title("📜 Voynich Virtual Machine (VVM) — Validator + Linter")

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown("""
    [## About the Voynich Virtual Machine (VVM)

The **Voynich Virtual Machine (VVM)** is an experimental execution and validation environment for a structural theory of the Voynich Manuscript that treats the script as a **deterministic procedural system**, rather than a natural language.

Under this model, Voynich tokens are interpreted as **data packets** composed of three logical fields:

> **[ Header (P₀) | Payload (Core) | Footer (S₀) ]**

Where:

- **Headers (Operators)** define the *execution context*  
  (e.g. initialization, natural state, processed state)
- **Cores** encode *stable identity variables* shared across manuscript domains  
  (botanical, pharmaceutical, balneological, astronomical)
- **Footers (Finalizers)** encode *state termination or continuation* semantics

---

### Rule R: Formal Token Architecture

All tokens evaluated by the VVM are tested against **Rule R**, a strict three-field structural constraint:

[P₀] + [C] + [S₀]

- **P₀ (Operator / Header)**  
  `q` (Initialization), `p` (Natural), `f` (Processed),  
  `t` (Transition), `k` (Potentia), `ch` (Biological)

- **C (Core / Payload)**  
  `aiin`, `oke`, `ol`, `che` (extensible under linter mode)

- **S₀ (Finalizer / Footer)**  
  `dy` (Stable / End-of-record),  
  `y` (Open / Continue),  
  `s` (Terminal),  
  `m` (Pointer / Link)

Tokens that violate this structure are flagged as **logic errors**, not spelling mistakes.

---

### Execution Model

The VVM does **not translate** Voynichese.

Instead, it executes a **structural trace** of how identity variables move through different procedural contexts.

Example execution trace:

p-aiin-dy   → Natural state (Herbal) f-aiin-dy   → Processed state (Pharma) ch-aiin-s   → Biological execution (Balneological)

This demonstrates a **state transition** of a stable Core-ID (`aiin`) across manuscript sections.

---

### Error Detection & Human Execution Faults

The VVM includes a rule-based **linter** that detects execution anomalies consistent with *human procedural copying*, including:

- **E1 — Contextual Cache Errors**  
  (e.g. Natural prefixes inside Pharma-dominant paragraphs)

- **E2 — Type Mismatches**  
  (e.g. Initialization headers combined with terminal suffixes)

- **E3 — Off-by-One Drift**  
  (core mutations in dense repetitive sequences)

These are treated as evidence of **human rule execution**, not noise or cipher corruption.

---

### Input Formats

The VVM accepts:

- **Plain EVA-style tokens**
- **Full Stolfi transcription blocks**  
  (with channel selection: H / T / F / G / U)

Paragraph boundaries are respected when auditing termination behavior (e.g. `-dy` placement).

---

### What This Tool Is (and Is Not)

**This tool is:**
- A procedural validator
- A structural hypothesis tester
- A falsifiable execution model

**This tool is not:**
- A translation engine
- A linguistic decoder
- A claim of semantic meaning

---

### Status

This project is in **experimental / research mode**.

All outputs are traceable, reproducible, and intended for collaborative analysis.

Source code and documentation:
- https://github.com/shallowtek/Voynich-vm
- https://voynich-vm.streamlit.app/]

    """)
    
st.markdown(
    "Paste **plain EVA tokens** or a **full Stolfi block**. "
    "This app extracts tokens, applies **Rule R**, and produces a **Parse Error** + **Exception (E1/E2/E3)** log."
)

with st.sidebar:
    st.header("Options")
    input_mode = st.radio("Input type", ["Auto-detect", "Plain EVA", "Stolfi block"], index=0)
    strict = st.toggle("Strict cores only (aiin/oke/ol/che)", value=False)

    st.subheader("Stolfi channels")
    channels = st.multiselect("Use channels", ["H", "T", "F", "G", "U"], default=["H"])

    st.subheader("Display")
    max_trace = st.slider("Max trace rows (UI safety)", 0, 500, 120, 10)
    show_table = st.toggle("Show parsed table", True)
    show_parse_errors = st.toggle("Show parse errors", True)
    show_exceptions = st.toggle("Show exception log (E1/E2/E3)", True)
    show_dy_audit = st.toggle("Show -dy audit", True)

default = (
    "p-aiin-dy f-aiin-dy ch-aiin-s\n\n"
    "q-oke f-ol-m q-aiin-s\n\n"
    "# You can paste a Stolfi block too; choose 'Stolfi block' or auto-detect.\n"
)
st.sidebar.caption("These settings change validation logic and error detection")
text = st.text_area("Input text (see sidebar for parser & validation options)", value=default, height=280)

if st.button("Execute / Lint"):
    # Detect input kind
    detected = "plain"
    if input_mode == "Stolfi block":
        detected = "stolfi"
    elif input_mode == "Auto-detect":
        if re.search(r"<[^>]+;[HTFGU]>", text):
            detected = "stolfi"

    tokens = []
    bounds = []

    if detected == "stolfi":
        payload = extract_stolfi_payload(text, channels=set(channels))
        if not payload.strip():
            st.error("No Stolfi payload extracted. Check channels and ensure your paste contains '<...;H>' style lines.")
            st.stop()

        paras = paragraphize(payload)
        idx = 0
        for para in paras:
            tks = tokenize_eva(para)
            if not tks:
                continue
            tokens.extend(tks)
            bounds.append((idx, idx + len(tks) - 1))
            idx += len(tks)

    else:
        # Plain EVA: split on blank lines as paragraphs
        paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        idx = 0
        for para in paras:
            para = normalize_dashes(para)
            # extract eva-ish tokens including hyphenated
            tks = [t.lower() for t in re.findall(r"[a-zA-Z-]+", para) if re.fullmatch(r"[a-zA-Z-]+", t)]
            tks = [t.lower() for t in tks if re.search(r"[a-z]", t.lower())]
            if not tks:
                continue
            tokens.extend(tks)
            bounds.append((idx, idx + len(tks) - 1))
            idx += len(tks)

    # Parse tokens
    records = [parse_token_hyphen_safe(t, strict=strict) for t in tokens]
    records = classify_exceptions(records, bounds, strict=strict)

    total = len(records)
    valid = sum(1 for r in records if r["ValidRuleR"])
    parse_errors = sum(1 for r in records if not r["ValidRuleR"])
    exceptions = sum(1 for r in records if r.get("Exception"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens", total)
    c2.metric("Valid Rule R", valid)
    c3.metric("Parse Errors", parse_errors)
    c4.metric("Exceptions (E1/E2/E3)", exceptions)

    # Trace
    if max_trace > 0:
        st.subheader("Execution Trace (preview)")
        shown = 0
        for r in records:
            if shown >= max_trace:
                break
            shown += 1

            if not r["ValidRuleR"]:
                st.error("❌ {0} — {1}".format(r["token"], r.get("ParseError") or "Parse error"))
                continue

            base = "✔️ {0} → {1} | C={2} ({3}) | {4}".format(
                r["token"], r["Operator"], r["C"], r["CoreMeaning"], r["Finalizer"]
            )

            if r.get("Exception"):
                st.warning(base + "  ⚠️ " + r["Exception"])
            else:
                st.success(base)

        if total > max_trace:
            st.info("Trace truncated for performance. Use the table below for full output.")

    # Tables
    if show_table:
        st.subheader("Parsed Table")
        st.dataframe(records, use_container_width=True)

    if show_parse_errors:
        st.subheader("Parse Errors (Rule R mismatch)")
        pe = [r for r in records if not r["ValidRuleR"]]
        if not pe:
            st.success("No parse errors.")
        else:
            st.dataframe(pe, use_container_width=True)

    if show_exceptions:
        st.subheader("Exception Log (E1/E2/E3)")
        ex = [r for r in records if r.get("Exception")]
        if not ex:
            st.success("No exceptions under current heuristics.")
        else:
            st.dataframe(ex, use_container_width=True)

    # -dy audit
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

    # Download
    csv_data = to_csv(records).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="vvm_results.csv",
        mime="text/csv",
    )
