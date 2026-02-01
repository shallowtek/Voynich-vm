import streamlit as st
import re

# --- VOYNICH ENGINE LOGIC ---
OPERATORS = {'q': 'Initialization', 'p': 'Natural/Raw', 'f': 'Processed/Pharma', 'ch': 'Biological/Balneo', 't': 'Transition'}
CORES = {'aiin': 'Vital Heat', 'oke': 'Structural Base', 'ol': 'Fluid Flow', 'che': 'Interface'}
FINALIZERS = {'dy': 'Stable/NULL', 's': 'Terminal/Halt', 'm': 'Pointer/Link'}

def parse_voynich(text):
    tokens = re.findall(r'\b[a-z-]+\b', text.lower())
    results = []
    for token in tokens:
        clean = token.replace('-', '')
        # Pattern: [Prefix] + [Core] + [Suffix]
        match = re.match(r'^(q|p|f|ch|t)?(aiin|oke|ol|che)(dy|s|m)?$', clean)
        if match:
            p, c, s = match.groups()
            results.append({
                "token": token,
                "op": OPERATORS.get(p, "Data Only"),
                "payload": CORES.get(c, "Unknown Core"),
                "state": FINALIZERS.get(s, "In Transit")
            })
        else:
            results.append({"token": token, "op": "ERROR", "payload": "Syntax Fault", "state": "Invalid"})
    return results

# --- STREAMLIT UI ---
st.set_page_config(page_title="Voynich Virtual Machine v1.0", layout="wide")
st.title("📜 Voynich Virtual Machine (VVM) PoC")
st.markdown("### A Deterministic Procedural Almanac Execution Engine")

input_text = st.text_area("Input EVA Transcription Here:", "p-aiin-dy f-aiin-dy ch-aiin-s")

if st.button("Execute Procedure"):
    output = parse_voynich(input_text)
    
    # Create Visual Execution Trace
    for res in output:
        if res['op'] == "ERROR":
            st.error(f"❌ {res['token']} : Logic Error (Type Mismatch)")
        else:
            st.success(f"✔️ **{res['token']}** → {res['op']} | {res['payload']} | {res['state']}")

st.sidebar.info("Based on the 'Voynich as Procedural Almanac' Theory.")
