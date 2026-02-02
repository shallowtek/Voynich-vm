import re
import streamlit as st

# ============================================================
# Voynich Virtual Machine (VVM) — Validator + Linter (Enhanced UX)
# Streamlit Cloud friendly: no pandas, no 3.10+ syntax.
# Goals of this enhancement:
#  1) Mobile discoverability: make sidebar controls obvious.
#  2) Reduce confusion: explain "Data Only" / "Unknown Core" clearly.
#  3) Improve Stolfi ingestion: handle punctuation/marks more robustly.
#  4) Provide quick-start + About text that's readable and copy/paste safe.
#  5) Add optional "Summary" + "Core frequency" views without extra deps.
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
      - treat '.', '=' as separators
      - drop obvious garbage like '%' runs
      - treat stray '*' as separators (Stolfi sometimes has * / **)
      - treat runs of whitespace as separators
    """
    s = strip_annotations(s)
    s = normalize_dashes(s)
    s = s.lower()

    # Remove uncertainty markers without splitting
    s = s.replace("!", "")

    # Drop long %%%%% blocks entirely
    s = re.sub(r"%{3,}", " ", s)

    # Replace common separators with spaces
    s = re.sub(r"[.=]+", " ", s)

    # Make asterisks separators (often editorial / uncertain glyph markers)
    s = re.sub(r"\*+", " ", s)

    # Other punctuation -> space (keep hyphen)
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
    Also supports bare tokens via regex fallback.
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

        # prefix
        if parts and parts[0] in OPERATORS:
            p0 = parts[0]
            parts = parts[1:]

        # suffix
        if parts and parts[-1] in FINALIZERS:
            s0 = parts[-1]
            parts = parts[:-1]

        core = "".join(parts)

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

        if r["P0"] == "q" and r["S0"] == "s":
            r["Exception"] = "E2 (Type Mismatch: q- with -s)"

        if (not strict) and r["C"]:
            if r["C"].startswith("aii") and r["C"] != "aiin":
                if re.fullmatch(r"aii[a-z]{1,3}", r["C"]):
                    if not r["Exception"]:
                        r["Exception"] = "E3 (Off-by-One Drift: aiin-variant core)"

    # E1 paragraph-level
    for (a, b) in paragraph_bounds:
        ops = [
            records[i]["P0"] for i in range(a, b + 1)
            if records[i]["ValidRuleR"] and records[i]["P0"]
        ]
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
        last_total += 1
        if records[b].get("S0") == "dy":
            dy_last += 1
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


def counts_by_key(records, key, only_valid=True):
    """Simple frequency counter (no pandas)."""
    counts = {}
    for r in records:
        if only_valid and (not r.get("ValidRuleR")):
            continue
        v = r.get(key, "")
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
    # return sorted list of tuples
    items = list(counts.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Voynich VVM", layout="wide")
st.title("📜 Voynich Virtual Machine (VVM) — Validator + Linter")

# --- Mobile discoverability banner ---
st.info(
    "💡 **Mobile tip:** open the sidebar using the **☰ menu** to access filters (Strict mode, Stolfi channels, and display controls). "
    "Also: **‘Unknown/Unregistered Core’ is not a parse failure**—it means the core is not yet in the registered lexicon."
)

# --- Quick Start (reduces first-run confusion) ---
with st.expander("🚀 Quick Start (2 minutes)", expanded=True):
    st.markdown(
        """
**1) Plain EVA tokens**  
Paste space-separated tokens like: `p-aiin-dy f-aiin-dy ch-aiin-s`

**2) Stolfi blocks**  
Paste full `<...;H>` lines. Leave input mode on **Auto-detect** (or choose **Stolfi block**) and select channels (H/T/F/G/U) in the sidebar.

**3) Strict vs Linter mode**  
- **Strict ON**: only cores `aiin / oke / ol / che` are considered valid.  
- **Strict OFF (Linter)**: any core can be structurally valid; unknown cores are surfaced as *unregistered* (for lexicon expansion).

**4) Read the output**  
- **Parse Errors** = Rule R mismatch (token doesn’t fit the structure).  
- **Exceptions** = E1/E2/E3 “glitches” (token fits, but violates a heuristic consistency rule).
        """
    )

# --- About section (copy/paste safe, no bracket noise) ---
ABOUT_MD = """
## About the Voynich Virtual Machine (VVM)

The **Voynich Virtual Machine (VVM)** is an experimental validation environment for a structural hypothesis: that Voynich tokens behave like **deterministic procedural packets**, not natural language words.

Under this model, each token is parsed as:

> **[ Header (P₀) | Payload (Core) | Footer (S₀) ]**

### Rule R (Formal Token Architecture)

All tokens are evaluated against a strict 3-field constraint:

**P₀ + C + S₀**

- **P₀ (Operator / Header)**: `q` (Init), `p` (Natural), `f` (Processed), `t` (Transition), `k` (Potentia), `ch` (Biological)
- **C (Core / Payload)**: `aiin`, `oke`, `ol`, `che` *(extensible in Linter mode)*
- **S₀ (Finalizer / Footer)**: `dy` (Stable/NULL), `y` (Open), `s` (Terminal), `m` (Pointer)

### Execution Model

The VVM does **not translate** Voynichese.  
Instead, it produces a **structural trace** of how Core-IDs move through contexts.

Example trace:

- `p-aiin-dy` → Natural/Raw + `aiin` + Stable
- `f-aiin-dy` → Processed/Pharma + `aiin` + Stable
- `ch-aiin-s` → Biological/Balneo + `aiin` + Terminal

### Error Detection (Human Execution Faults)

The linter flags consistency anomalies treated as **execution glitches**, not spelling mistakes:

- **E1 — Contextual Cache**: `p-` occurs inside a paragraph dominated by pharma-ish operators (`f/k/ch/t`)
- **E2 — Type Mismatch**: `q-` occurs with terminal `-s`
- **E3 — Off-by-One Drift**: `aiin`-like core mutations (only in Linter mode)

### Inputs

- **Plain EVA tokens**
- **Stolfi transcription blocks** with selectable channels: H / T / F / G / U

### Links

- Repo: https://github.com/shallowtek/Voynich-vm
- App: https://voynich-vm.streamlit.app/
"""

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(ABOUT_MD)

# --- Main instruction line ---
st.markdown(
    "Paste **plain EVA tokens** or a **full Stolfi block**. "
    "This app extracts tokens, applies **Rule R**, and produces **Parse Errors** plus an **Exception (E1/E2/E3)** log."
)

with st.sidebar:
    st.markdown("## ⚙️ Controls & Filters")
    st.caption("On mobile, open this sidebar via the **☰ menu** at top-left.")

    input_mode = st.radio("Input type", ["Auto-detect", "Plain EVA", "Stolfi block"], index=0)
    strict = st.toggle("Strict cores only (aiin/oke/ol/che)", value=False)
    st.caption("Strict ON = unknown cores become **Parse Errors**. Strict OFF = unknown cores are **Unregistered** (useful for discovery).")

    st.subheader("Stolfi channels")
    channels = st.multiselect("Use channels", ["H", "T", "F", "G", "U"], default=["H"])
    st.caption("If you paste a Stolfi block, channel selection controls which witness stream is parsed.")

    st.subheader("Display")
    max_trace = st.slider("Max trace rows (UI safety)", 0, 500, 120, 10)

    # Mobile-friendly defaults: keep big tables optional
    show_table = st.toggle("Show parsed table", False)
    show_parse_errors = st.toggle("Show parse errors", True)
    show_exceptions = st.toggle("Show exception log (E1/E2/E3)", True)
    show_dy_audit = st.toggle("Show -dy audit", True)
    show_summary = st.toggle("Show summary (frequencies)", True)

default = (
    "p-aiin-dy f-aiin-dy ch-aiin-s\n\n"
    "q-oke f-ol-m q-aiin-s\n\n"
    "# Tip: paste Stolfi blocks too (auto-detect works)."
)
text = st.text_area("Input text", value=default, height=280)

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
            st.error("No Stolfi payload extracted. Check channels and ensure your paste includes '<...;H>' style lines.")
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
            # Extract eva-ish tokens including hyphenated tokens
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

    # Interpretive note right above outputs (reduces "it looks broken" reports)
    st.caption(
        "ℹ️ **Interpretation note**: `Data Only` means no operator (P₀) was detected. "
        "`Unknown/Unregistered Core` means the token fits Rule R structurally but the core is not currently defined in the lexicon "
        "(expected in **Linter mode**)."
    )

    # Summary (frequencies) helps on mobile: quick sanity check without scrolling tables
    if show_summary:
        st.subheader("Summary")
        op_counts = counts_by_key(records, "P0", only_valid=True)
        core_counts = counts_by_key(records, "C", only_valid=True)
        s0_counts = counts_by_key(records, "S0", only_valid=True)

        colA, colB, colC = st.columns(3)
        colA.markdown("**Operators (P₀)**")
        colA.write(op_counts[:12] if op_counts else [])

        colB.markdown("**Cores (C)**")
        colB.write(core_counts[:12] if core_counts else [])

        colC.markdown("**Finalizers (S₀)**")
        colC.write(s0_counts[:12] if s0_counts else [])

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
            st.info("Trace truncated for performance. Increase max trace in the sidebar or use the CSV download.")

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
