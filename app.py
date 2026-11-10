"""TaxMate Streamlit UI. All logic lives in the taxmate package; this file only renders."""
from __future__ import annotations

import uuid

import streamlit as st

from taxmate import copy, tracing
from taxmate.config import load_settings, missing_keys
from taxmate.profile import TaxProfile
from taxmate.types import Turn

st.set_page_config(page_title="TaxMate", page_icon="🧾", layout="centered")


def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.profile = TaxProfile()
    st.session_state.session_id = str(uuid.uuid4())


if "session_id" not in st.session_state:
    reset_session()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Your Tax Profile")
        for label, value in st.session_state.profile.display():
            st.markdown(f"**{label}:** {value}")
        if st.button("Start over", use_container_width=True):
            reset_session()
            st.rerun()
        st.divider()
        st.caption(copy.DISCLAIMER)


settings = load_settings()
missing = missing_keys(settings)
if missing:
    st.title("TaxMate")
    st.error(copy.MISSING_CONFIG.format(missing=", ".join(missing)))
    with st.sidebar:
        st.caption(copy.DISCLAIMER)
    st.stop()


@st.cache_resource(show_spinner="Loading BIR sources…")
def get_deps():
    from taxmate.live import build_live_deps
    tracing.init(settings)
    deps, _index = build_live_deps(settings)
    return deps


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['title']} — {s['section']}** (issued {s['issued']})")
            st.caption(s["snippet"])


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        kind = msg.get("kind")
        if kind in ("blocked", "unsafe"):
            st.warning(msg["content"])
        elif kind == "fallback":
            st.info(msg["content"])
        elif kind == "error":
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
        render_sources(msg.get("sources", []))


def handle(user_text: str) -> None:
    from taxmate.pipeline import run_turn
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.spinner("Checking BIR sources…"):
        result = run_turn(user_text, st.session_state.history, st.session_state.profile, get_deps(),
                          st.session_state.session_id)
    tracing.flush()
    masked = result.record.get("user_message_masked") or user_text
    if result.kind != "invalid":
        st.session_state.history += [Turn("user", masked), Turn("assistant", result.text)]
    st.session_state.profile = result.profile
    st.session_state.messages.append({
        "role": "assistant", "content": result.text, "kind": result.kind,
        "sources": [{"title": s.chunk.title, "section": s.chunk.section, "issued": s.chunk.issued,
                     "snippet": s.chunk.text[:400] + ("…" if len(s.chunk.text) > 400 else "")}
                    for s in result.sources],
    })


st.title("TaxMate")
st.caption("Philippine tax answers for freelancers and small businesses, grounded in BIR issuances.")

pending = st.chat_input("Ask about 8% vs. graduated rates, deductions, Form 2307, filing…")

render_message({"role": "assistant", "content": copy.GREETING})
if not st.session_state.messages and not pending:
    cols = st.columns(len(copy.SUGGESTED_QUESTIONS))
    for col, q in zip(cols, copy.SUGGESTED_QUESTIONS):
        if col.button(q, use_container_width=True):
            pending = q

if pending:
    handle(pending)

for msg in st.session_state.messages:
    render_message(msg)

render_sidebar()
