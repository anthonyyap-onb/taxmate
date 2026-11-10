import json

from taxmate import tracing
from taxmate.config import Settings
from taxmate.logging_utils import log_turn


def settings(**kw):
    base = dict(openai_api_key="x", langfuse_public_key=None, langfuse_secret_key=None,
                langfuse_host="https://cloud.langfuse.com", chat_model="m", embed_model="e", log_level="INFO")
    base.update(kw)
    return Settings(**base)


def test_tracing_disabled_is_noop():
    tracing.init(settings())
    assert not tracing.tracing_enabled(settings())

    @tracing.observe("x")
    def f(a):
        return a + 1

    assert f(1) == 2 and tracing.current_trace_id() is None
    with tracing.trace_context(session_id="s", tags=["t"]):
        tracing.update_span(metadata={"k": "v"})
    tracing.flush()  # no raise


def test_disabled_client_is_plain_openai():
    import openai
    client = tracing.make_openai_client(settings())
    assert type(client) is openai.OpenAI


def test_log_turn_emits_json(caplog):
    with caplog.at_level("INFO", logger="taxmate"):
        log_turn({"request_id": "r1", "user_message_masked": "TIN [TIN]"})
    rec = json.loads(caplog.records[-1].getMessage())
    assert rec["request_id"] == "r1" and "ts" in rec
