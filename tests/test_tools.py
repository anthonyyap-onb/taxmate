import json

from taxmate.tools import TOOL_SCHEMAS, run_tool


def test_schemas_are_strict_function_tools():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["calculate_ph_tax", "compare_regimes"]
    assert all(t["function"]["strict"] for t in TOOL_SCHEMAS)


def test_run_calculate_ok():
    call = run_tool("calculate_ph_tax", json.dumps({
        "gross_receipts": 1_200_000, "regime": "8_percent", "deduction_method": None, "itemized_expenses": None,
        "non_operating_income": None, "compensation_income": None}))
    assert call.ok and call.result["total_tax"] == 76_000 and call.args["gross_receipts"] == 1_200_000


def test_run_compare_ok():
    call = run_tool("compare_regimes", json.dumps({"gross_receipts": 900_000, "itemized_expenses": 200_000,
                                                   "non_operating_income": None, "compensation_income": None}))
    assert call.ok and call.result["lowest"] == "8_percent"


def test_calc_error_becomes_structured_result():
    call = run_tool("calculate_ph_tax", json.dumps({
        "gross_receipts": -5, "regime": "8_percent", "deduction_method": None, "itemized_expenses": None,
        "non_operating_income": None, "compensation_income": None}))
    assert not call.ok and call.result["error"] == "invalid_amount" and "message" in call.result


def test_bad_json_and_unknown_tool_and_schema_violation_do_not_raise():
    assert run_tool("calculate_ph_tax", "{not json").result["error"] == "invalid_arguments"
    assert run_tool("delete_everything", "{}").result["error"] == "unknown_tool"
    assert run_tool("calculate_ph_tax", json.dumps({"gross_receipts": "one hundred thousand",
                                                    "regime": "8_percent"})).result["error"] == "invalid_arguments"


# --- preconditions: the model may not fill in what the user never said ---------------------------------------
import pytest  # noqa: E402

from taxmate.profile import TaxProfile  # noqa: E402
from taxmate.tools import check_call, stated_amounts  # noqa: E402


def calc(**kw):
    base = {"gross_receipts": 800000, "regime": "8_percent", "deduction_method": None, "itemized_expenses": None,
            "non_operating_income": None, "compensation_income": None}
    return {**base, **kw}


@pytest.mark.parametrize("text,expected", [
    ("I earned ₱800,000", {800000}), ("about 1.2M gross", {1200000}), ("900k a year", {900000}),
    ("P1,500,000 and 200,000 expenses", {1500000, 200000}), ("-₱500,000", {-500000}),
    ("2.5 million", {2500000}), ("PHP 60000", {60000}),
])
def test_stated_amounts(text, expected):
    assert expected <= stated_amounts([text])


def test_regime_must_be_known_or_stated():
    err = check_call("calculate_ph_tax", calc(), TaxProfile(), ["I earned ₱800,000. How much tax?"])
    assert err["error"] == "regime_not_stated"
    assert check_call("calculate_ph_tax", calc(), TaxProfile(tax_regime="8_percent"), ["₱800,000"]) is None
    assert check_call("calculate_ph_tax", calc(), TaxProfile(), ["what if I'm on 8% with ₱800,000?"]) is None


def test_deduction_method_must_be_known_or_stated():
    args = calc(regime="graduated", deduction_method="osd")
    p = TaxProfile(tax_regime="graduated")
    assert check_call("calculate_ph_tax", args, p, ["₱800,000"])["error"] == "deduction_method_not_stated"
    assert check_call("calculate_ph_tax", args, p, ["₱800,000 using the OSD"]) is None
    assert check_call("calculate_ph_tax", args, p.model_copy(update={"deduction_method": "osd"}), ["₱800,000"]) is None


def test_amounts_must_come_from_the_user():
    p = TaxProfile(tax_regime="8_percent")
    assert check_call("calculate_ph_tax", calc(gross_receipts=900000), p, ["₱75,000 a month"])["error"] == "amount_not_stated"
    assert check_call("calculate_ph_tax", calc(), p, ["first turn", "₱800,000 this year"]) is None  # any user turn


def test_negative_amount_from_user_is_rejected_even_if_model_drops_the_sign():
    err = check_call("calculate_ph_tax", calc(gross_receipts=500000), TaxProfile(tax_regime="8_percent"),
                     ["My gross receipts are -₱500,000."])
    assert err["error"] == "negative_amount"


def test_compare_needs_no_regime_but_amounts_are_checked():
    args = {"gross_receipts": 900000, "itemized_expenses": 200000, "non_operating_income": None, "compensation_income": None}
    assert check_call("compare_regimes", args, TaxProfile(), ["900k gross and ₱200,000 expenses"]) is None
    assert check_call("compare_regimes", args, TaxProfile(), ["900k gross"])["error"] == "amount_not_stated"


def test_run_tool_applies_preconditions_when_context_given():
    import json
    call = run_tool("calculate_ph_tax", json.dumps(calc()), TaxProfile(), ["₱800,000, how much?"])
    assert call.result["error"] == "regime_not_stated" and "ask" in call.result["message"].lower()


def test_zero_gross_receipts_must_also_be_stated():
    p = TaxProfile(tax_regime="8_percent")
    assert check_call("calculate_ph_tax", calc(gross_receipts=0), p, ["How much tax?"])["error"] == "amount_not_stated"
    assert check_call("calculate_ph_tax", calc(gross_receipts=0), p, ["I earned ₱0 this year"]) is None
    assert check_call("calculate_ph_tax", calc(compensation_income=0), p, ["₱800,000"]) is None  # optional zero is fine


def test_regime_error_points_model_back_to_an_existing_comparison():
    err = check_call("calculate_ph_tax", calc(), TaxProfile(), ["₱800,000"])
    assert "compare_regimes" in err["message"]


def test_itemize_verb_counts_as_stating_itemized():
    args = calc(regime="graduated", deduction_method="itemized", itemized_expenses=200000)
    assert check_call("calculate_ph_tax", args, TaxProfile(tax_regime="graduated"),
                      ["₱800,000 gross, I itemize ₱200,000 of expenses"]) is None
