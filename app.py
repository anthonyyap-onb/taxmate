"""TaxMate Streamlit UI. All logic lives in the taxmate package; this file only renders."""
from __future__ import annotations

import uuid

import streamlit as st

from taxmate import copy, tracing
from taxmate.config import load_settings, missing_keys
from taxmate.logging_utils import log_turn
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
        if settings.admin:
            render_admin()


def render_admin() -> None:
    """Only with TAXMATE_ADMIN=1. Normal users never see this."""
    st.divider()
    st.subheader("Admin")
    try:
        cache = get_deps().cache
    except Exception:
        st.caption("Cache unavailable (the index failed to load).")
        return
    if cache is None:
        st.caption("Response cache is off.")
        return
    s = cache.stats()
    st.caption(f"Response cache: {s['entries']} entries · {s['hits_exact']} exact hits · "
               f"{s['hits_semantic']} semantic hits · {s['misses']} misses")
    if st.button("Clear cache", use_container_width=True):
        cache.clear()
        st.toast("Response cache cleared.")


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
    deps, _index = build_live_deps(settings, response_cache=settings.response_cache)
    return deps


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['title']} — {s['section']}** (issued {s['issued']})")
            st.caption(s["snippet"])


OPTION_LABELS = {"8_percent": "8% rate", "graduated_osd": "Graduated + OSD", "graduated_itemized": "Graduated + itemized"}


def peso(x: float) -> str:
    return f"₱{x:,.2f}"


def render_breakdown(result: dict) -> None:
    with st.expander("How this was computed"):
        st.table([{"Item": line["label"], "Amount": peso(line["amount"])} for line in result["lines"]])
        for note in result["notes"]:
            st.caption(note)


def render_tools(tools: list[dict]) -> None:
    """Calculator results as cards. Failed calls show nothing; the reply text explains them."""
    for t in tools:
        if not t["ok"]:
            continue
        r = t["result"]
        with st.container(border=True):
            if t["name"] == "compare_regimes":
                cols = st.columns(len(r["options"]))
                for col, o in zip(cols, r["options"]):
                    label = OPTION_LABELS[o["option"]] + (" ✓ lowest" if o["option"] == r["lowest"] else "")
                    col.metric(label, peso(o["total_tax"]))
                if not r["eight_percent_available"]:
                    st.caption("The 8% option isn't available above ₱3,000,000 gross sales/receipts.")
                lowest = next(o for o in r["options"] if o["option"] == r["lowest"])
                render_breakdown(lowest)
            else:
                label = "8% rate" if r["regime"] == "8_percent" else OPTION_LABELS["graduated_" + r["deduction_method"]]
                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated total tax", peso(r["total_tax"]))
                c2.metric("Income tax", peso(r["income_tax"]))
                c3.metric("Percentage tax", peso(r["percentage_tax"]))
                st.caption(f"Regime: {label}")
                render_breakdown(r)


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
        render_tools(msg.get("tools", []))
        if kind == "answer":
            st.caption(copy.VERIFY_NOTE)
        render_sources(msg.get("sources", []))


def handle(user_text: str) -> None:
    from taxmate.pipeline import run_turn
    st.session_state.messages.append({"role": "user", "content": user_text})
    try:
        deps = get_deps()  # builds the index on first use; st.cache_resource does not cache a failure
    except Exception as exc:
        log_turn({"request_id": str(uuid.uuid4()), "session_id": st.session_state.session_id,
                  "kind": "error", "stage": "startup", "error": type(exc).__name__})
        st.session_state.messages.append({"role": "assistant", "content": copy.ERROR_TEXT, "kind": "error"})
        return
    with st.spinner("Checking BIR sources…"):
        result = run_turn(user_text, st.session_state.history, st.session_state.profile, deps,
                          st.session_state.session_id)
    tracing.flush()
    masked = result.record.get("user_message_masked") or user_text
    if result.kind != "invalid":
        st.session_state.history += [Turn("user", masked), Turn("assistant", result.text)]
    st.session_state.profile = result.profile
    st.session_state.messages.append({
        "role": "assistant", "content": result.text, "kind": result.kind,
        "tools": [{"name": t.name, "ok": t.ok, "result": t.result} for t in result.tool_calls],
        "sources": [{"title": s.chunk.title, "section": s.chunk.section, "issued": s.chunk.issued,
                     "snippet": s.chunk.text[:400] + ("…" if len(s.chunk.text) > 400 else "")}
                    for s in result.sources],
    })


st.title("TaxMate")
st.caption("Philippine tax answers for freelancers and small businesses, grounded in BIR issuances.")

def pick_chip(question: str) -> None:
    # Callbacks run before the rerun, so the chips below are skipped on the same rerun as the click.
    st.session_state.chip_question = question


pending = (st.chat_input("Ask about 8% vs. graduated rates, deductions, Form 2307, filing…")
           or st.session_state.pop("chip_question", None))

render_message({"role": "assistant", "content": copy.GREETING})
if not st.session_state.messages and not pending:
    cols = st.columns(len(copy.SUGGESTED_QUESTIONS))
    for col, q in zip(cols, copy.SUGGESTED_QUESTIONS):
        col.button(q, use_container_width=True, on_click=pick_chip, args=(q,))

if pending:
    handle(pending)

for msg in st.session_state.messages:
    render_message(msg)

render_sidebar()
