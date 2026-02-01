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
