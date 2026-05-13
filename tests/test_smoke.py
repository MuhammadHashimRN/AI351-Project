"""Smoke tests — verify layout, parse the core scripts, and check for leaked secrets."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_layout():
    for fname in ("main.py", "streamlit_pages.py", "models.py", "explainers.py", "utils.py", "requirements.txt", ".env.example"):
        assert (ROOT / fname).is_file(), f"missing {fname}"


def test_core_scripts_parse():
    for fname in ("main.py", "streamlit_pages.py", "models.py", "explainers.py", "utils.py", "project.py"):
        src = (ROOT / fname).read_text(encoding="utf-8")
        ast.parse(src)


def test_no_hardcoded_alpha_vantage_key():
    """Regression guard for the previously-leaked Alpha Vantage key."""
    suspicious = re.compile(r'API_KEY\s*=\s*"[A-Z0-9]{12,}"')
    for py in ROOT.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not suspicious.search(text), f"hardcoded API key found in {py.name}"


def test_requirements_lists_core_deps():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for dep in ("tensorflow", "streamlit", "shap", "scikit-learn", "yfinance", "python-dotenv"):
        assert dep in reqs, f"requirements.txt missing {dep}"
