import re
import streamlit as st

# ============================================================
# Voynich Virtual Machine (VVM) — Robust Validator/Linter
# Streamlit Cloud friendly: no pandas, no Python 3.10+ syntax.
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
# Strict: core must be one of CORES (so greediness isn't an issue)
RULE_R_STRICT = re.compile(r"^(q|p|f|ch|t|k)?(aiin|oke|ol|che)(dy|y|s|m)?$")

# Linter: allow unknown cores, BUT make core non-greedy so suffix can match
RULE_R_LINTER = re.compile(r"^(q|p|f|ch|t|k)?([a-z]+?)(dy|y|s|m)?$")

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

FINALIZER_KEYS = sorted(FINALIZERS.keys(), key=len, reverse=True)  # ["dy","y","s","m"]


def normalize_dashes(s):
    """Convert various unicode dashes/hyphens to ASCII '-'."""
    for k, v in DASH_CHARS.items():
        s = s.replace(k, v)
    return s


def split_suffix(clean):
    """
    Return (stem, s0) where s0 is one of FINALIZERS if present at end.
    Conservative: only splits if the remaining stem is non-empty.
    """
    for suf in FINALIZER_KEYS:
        if clean.endswith(suf) and len(clean) > len(suf):
            return clean[:-len(suf)], suf
    return clean, ""


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
    Also supports bare tokens (chedy, qokeedy etc).
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

    # --------------------------
    # Hyphen-aware parse first
    # --------------------------
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

        # Strict validation
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

    # --------------------------
    # Non-hyphen token parsing
    # --------------------------
    clean = token

    # In linter mode, do deterministic split: (optional operator) + (stem) + (optional suffix)
    # This fixes the classic issue where "chedy" would never populate "dy".
    if not strict:
        p0 = ""
        stem = clean

        # operator at start: handle 'ch' before single-letter ops
        if stem.startswith("ch") and len(stem) > 2:
            p0, stem = "ch", stem[2:]
        elif stem and stem[0] in OPERATORS and len(stem) > 1:
            p0, stem = stem[0], stem[1:]

        stem, s0 = split_suffix(stem)
        core = stem

        if not core:
            return {
                "token": original, "P0": p0, "C": "", "S0": s0,
                "Operator": OPERATORS.get(p0, "Data Only") if p0 else "Data Only",
                "CoreMeaning": "Missing Core",
                "Finalizer": FINALIZERS.get(s0, "In Transit") if s0 else "In Transit",
                "ValidRuleR": False,
                "ParseError": "Missing core",
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

    # Strict mode: use enumerated-core regex
    m = RULE_R_STRICT.match(clean)
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
            if r["C"].startswith("aii") and r["C"] != "aiin":
                if re.fullmatch(r"aii[a-z]{1,3}", r["C"]):
                    if not r["Exception"]:
                        r["Exception"] = "E3 (Off-by-One Drift: aiin-variant core)"

    # E1 paragraph-level
    for (a, b) in paragraph_bounds:
        ops = []
        i = a
        while i <= b:
            if records[i]["ValidRuleR"] and records[i]["P0"]:
                ops.append(records[i]["P0"])
            i += 1

        if not ops:
            continue

        counts = {}
        for o in ops:
            counts[o] = counts.get(o, 0) + 1
        dominant = max(counts, key=counts.get)

        pharmaish = dominant in ("f", "k", "ch", "t")
        if pharmaish:
            i = a
            while i <= b:
                if records[i]["ValidRuleR"] and records[i]["P0"] == "p" and not records[i]["Exception"]:
                    records[i]["Exception"] = "E1 (Contextual Cache: p- inside pharma-ish paragraph)"
                i += 1

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

        i = a
        while i < b:
            mid_total += 1
            if records[i].get("S0") == "dy":
                dy_mid += 1
            i += 1

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

# Quick “hint panel” (because the sidebar is easy to miss on mobile)
with st.expander("👈 Controls are in the left sidebar (tap the arrow).", expanded=True):
    st.markdown(
        "- **Input type**: Auto / Plain EVA / Stolfi block\n"
        "- **Strict cores**: when ON, only aiin/oke/ol/che count as valid cores\n"
        "- **Channels**: choose H/T/F/G/U for Stolfi pastes\n"
        "- **Max trace rows**: limits UI output for performance\n"
        "- **Tables**: toggle parsed table / errors / exceptions / dy-audit\n"
    )

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
### About the Voynich Virtual Machine (VVM)

The **Voynich Virtual Machine (VVM)** is an experimental validation environment for a structural theory of the Voynich Manuscript that treats the script as a **deterministic procedural system**, rather than a natural language.

Under this model, Voynich tokens are interpreted as **data packets** composed of three logical fields:

> **[ Header (P₀) | Payload (Core) | Footer (S₀) ]**

- **Headers (Operators)** define the execution context (initialization, natural, processed, transition, etc.)
- **Cores** encode stable identity variables (shared across domains)
- **Footers (Finalizers)** encode termination/continuation semantics

**Rule R:**  
`[P₀] + [C] + [S₀]`

The app accepts:
- Plain EVA tokens (including hyphenated forms like `p-aiin-dy`)
- Full Stolfi transcription blocks (channel-selectable)

Outputs:
- Parse errors (Rule-R mismatches)
- Exception log (E1/E2/E3)
- `-dy` paragraph-end vs mid-paragraph audit
        """
    )

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

    st.caption("Tip: If finalizers look empty, turn OFF 'Strict cores' and run in linter mode.")

default = (
    "p-aiin-dy f-aiin-dy ch-aiin-s\n\n"
    "chedy qokeedy otedy qoky qokam\n\n"
    "# Paste a Stolfi block too; choose 'Stolfi block' or auto-detect.\n"
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
        paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        idx = 0
        for para in paras:
            para = normalize_dashes(para)
            # only accept a-z and hyphen tokens
            tks = [t.lower() for t in re.findall(r"[a-zA-Z-]+", para) if re.fullmatch(r"[a-zA-Z-]+", t)]
            tks = [t.lower() for t in tks if re.search(r"[a-z]", t)]
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
