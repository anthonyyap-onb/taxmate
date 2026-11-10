"""Expected values are hand-computed from the BIR rules (see comments)."""
import math

import pytest

from taxmate.calculator import CalcError, calculate_ph_tax, compare_regimes, graduated_tax


@pytest.mark.parametrize("taxable,expected", [
    (0, 0), (250_000, 0),
    (400_000, 22_500),                      # 15% x 150k
    (600_000, 62_500),                      # 22,500 + 20% x 200k
    (800_000, 102_500),
    (2_000_000, 402_500),                   # 102,500 + 25% x 1.2M
    (8_000_000, 2_202_500),                 # 402,500 + 30% x 6M
    (10_000_000, 2_902_500),                # 2,202,500 + 35% x 2M
])
def test_graduated_table_2023(taxable, expected):
    assert graduated_tax(taxable) == pytest.approx(expected)


def test_eight_percent_purely_self_employed_gets_250k_reduction():
    r = calculate_ph_tax(1_200_000, "8_percent")
    assert r.income_tax == pytest.approx(76_000)  # 8% x (1.2M - 250k)
    assert r.percentage_tax == 0 and r.total_tax == pytest.approx(76_000)


def test_eight_percent_includes_non_operating_income():
    assert calculate_ph_tax(1_000_000, "8_percent", non_operating_income=50_000).income_tax == pytest.approx(64_000)


def test_eight_percent_below_250k_is_zero():
    assert calculate_ph_tax(200_000, "8_percent").total_tax == 0


def test_eight_percent_mixed_income_no_reduction_and_compensation_graduated():
    r = calculate_ph_tax(600_000, "8_percent", compensation_income=500_000)
    # 8% x 600k = 48,000 (no 250k reduction) + graduated(500k) = 22,500 + 20% x 100k = 42,500
    assert r.income_tax == pytest.approx(90_500)


def test_eight_percent_above_3m_is_ineligible():
    with pytest.raises(CalcError) as e:
        calculate_ph_tax(2_900_000, "8_percent", non_operating_income=200_000)
    assert e.value.code == "eight_percent_not_available"


def test_graduated_osd_adds_percentage_tax():
    r = calculate_ph_tax(1_000_000, "graduated", deduction_method="osd")
    assert r.income_tax == pytest.approx(62_500)       # taxable 600k
    assert r.percentage_tax == pytest.approx(30_000)   # 3% x 1M
    assert r.total_tax == pytest.approx(92_500)


def test_graduated_itemized():
    r = calculate_ph_tax(1_500_000, "graduated", deduction_method="itemized", itemized_expenses=700_000)
    assert r.income_tax == pytest.approx(102_500) and r.percentage_tax == pytest.approx(45_000)


def test_graduated_itemized_loss_floors_business_income_at_zero():
    r = calculate_ph_tax(500_000, "graduated", deduction_method="itemized", itemized_expenses=900_000,
                         compensation_income=400_000)
    assert r.income_tax == pytest.approx(22_500)  # only compensation is taxed


def test_graduated_mixed_income_sums_compensation_and_business():
    r = calculate_ph_tax(500_000, "graduated", deduction_method="osd", compensation_income=300_000)
    assert r.income_tax == pytest.approx(graduated_tax(600_000))  # 300k + 60% x 500k


def test_graduated_above_3m_has_no_percentage_tax_and_notes_vat():
    r = calculate_ph_tax(4_000_000, "graduated", deduction_method="osd")
    assert r.percentage_tax == 0 and any("VAT" in n for n in r.notes)


@pytest.mark.parametrize("kwargs,code", [
    (dict(gross_receipts=-1, regime="8_percent"), "invalid_amount"),
    (dict(gross_receipts=math.nan, regime="8_percent"), "invalid_amount"),
    (dict(gross_receipts=1e6, regime="8_percent", compensation_income=-5), "invalid_amount"),
    (dict(gross_receipts=1e6, regime="graduated"), "missing_deduction_method"),
    (dict(gross_receipts=1e6, regime="graduated", deduction_method="itemized"), "missing_itemized_expenses"),
    (dict(gross_receipts=1e6, regime="flat"), "invalid_regime"),
])
def test_input_errors_are_structured(kwargs, code):
    with pytest.raises(CalcError) as e:
        calculate_ph_tax(**kwargs)
    assert e.value.code == code and e.value.message


def test_compare_picks_eight_percent_for_low_expenses():
    c = compare_regimes(900_000, itemized_expenses=200_000)
    totals = {o["option"]: o["total_tax"] for o in c["options"]}
    assert totals == pytest.approx({"8_percent": 52_000, "graduated_osd": 77_500, "graduated_itemized": 109_500})
    assert c["lowest"] == "8_percent"


def test_compare_picks_itemized_for_high_expenses():
    c = compare_regimes(1_000_000, itemized_expenses=800_000)
    assert c["lowest"] == "graduated_itemized"  # 0 income tax + 30,000 PT beats 8% (60,000)


def test_compare_without_expenses_skips_itemized():
    assert [o["option"] for o in compare_regimes(1_000_000)["options"]] == ["8_percent", "graduated_osd"]


def test_compare_above_3m_marks_eight_percent_unavailable():
    c = compare_regimes(4_000_000)
    assert [o["option"] for o in c["options"]] == ["graduated_osd"] and c["eight_percent_available"] is False


def test_breakdown_serialises():
    d = calculate_ph_tax(1_200_000, "8_percent").to_dict()
    assert d["total_tax"] == 76_000 and d["lines"] and d["regime"] == "8_percent"
