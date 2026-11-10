from types import SimpleNamespace as NS

from taxmate import usage
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile
from taxmate.types import GenerationResult
from tests.test_pipeline import fake_deps


def chat_resp(model, p, c):
    return NS(model=model, usage=NS(prompt_tokens=p, completion_tokens=c, total_tokens=p + c))


def embed_resp(model, p):
    return NS(model=model, usage=NS(prompt_tokens=p, total_tokens=p))


def test_record_outside_a_turn_is_noop():
    usage.record(chat_resp("gpt-4o-mini", 10, 5))  # must not raise


def test_accumulates_and_prices_known_models():
    with usage.track() as u:
        usage.record(chat_resp("gpt-4o-mini-2024-07-18", 1000, 200))
        usage.record(chat_resp("gpt-4o-mini", 1000, 0))
        usage.record(embed_resp("text-embedding-3-small", 500))
    s = u.summary()
    assert s["tokens"] == {"prompt": 2000, "completion": 200, "embedding": 500}
    # 2000*0.15/1M + 200*0.60/1M + 500*0.02/1M
    assert abs(s["est_cost_usd"] - (0.0003 + 0.00012 + 0.00001)) < 1e-9


def test_unknown_model_makes_cost_none():
    with usage.track() as u:
        usage.record(chat_resp("some-new-model", 10, 10))
    assert u.summary()["est_cost_usd"] is None


def test_record_ignores_responses_without_usage():
    with usage.track() as u:
        usage.record(NS(model="gpt-4o-mini", usage=None))
        usage.record(object())
    assert u.summary()["tokens"] == {"prompt": 0, "completion": 0, "embedding": 0}


def test_pipeline_logs_tokens_and_cost():
    def gen(q, h, p, s, v, strict):
        usage.record(chat_resp("gpt-4o-mini", 100, 50))
        return GenerationResult("answer", "Yes [S1]", [1])

    r = run_turn("What is 2307?", [], TaxProfile(), fake_deps([], generate=gen), "s")
    assert r.record["tokens"] == {"prompt": 100, "completion": 50, "embedding": 0}
    assert r.record["est_cost_usd"] > 0


def test_every_openai_call_site_records_usage():
    """Drive each real make_* function with a fake client, so a missing import or record() call is caught."""
    from taxmate.generation import make_generator
    from taxmate.guardrails import make_topic_classifier
    from taxmate.profile import make_profile_extractor
    from taxmate.retrieval import make_embedder, make_query_rewriter
    from taxmate.types import Turn

    def chat(parsed=None, content="q"):
        r = chat_resp("gpt-4o-mini", 10, 1)
        r.choices = [NS(message=NS(parsed=parsed, content=content, tool_calls=None))]
        return r

    class FakeClient:
        def __init__(self):
            self.chat = NS(completions=NS(parse=lambda **k: chat(), create=lambda **k: chat()))
            self.embeddings = NS(create=lambda **k: NS(model="text-embedding-3-small", usage=NS(prompt_tokens=3),
                                                       data=[NS(embedding=[0.0, 1.0])]))

    c = FakeClient()
    with usage.track() as u:
        make_topic_classifier(c, "m")("hi")
        make_profile_extractor(c, "m")("hi", TaxProfile())
        make_generator(c, "m")("hi", [], TaxProfile(), [], "deliberate", False)
        make_query_rewriter(c, "m")([Turn("user", "a")], "b")
        make_embedder(c, "e")(["x"])
    assert u.summary()["tokens"] == {"prompt": 40, "completion": 4, "embedding": 3}
