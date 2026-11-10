"""Deterministic Philippine income-tax calculator for self-employed individuals (annual, taxable year 2023 onwards).

Every rule below is taken from the corpus:
- Graduated rates "effective January 1, 2023 and onwards": RR 8-2018 (Sec. 24(A)(2)(a) table).
- 8% option on gross sales/receipts + other non-operating income in excess of P250,000; the P250,000 reduction is
  only for PURELY self-employed individuals, not mixed-income earners: RMC 50-2018 Q22-Q23, RR 8-2018.
- 8% only if gross sales/receipts + other non-operating income do not exceed the P3,000,000 VAT threshold: RR 8-2018.
- 8% is in lieu of the graduated rates AND the 3% percentage tax (Sec. 116): RMC 50-2018 Q21.
- Mixed-income earners: compensation is always taxed at graduated rates: RMC 50-2018 Q25.
- OSD = 40% of gross sales/receipts for individuals: RR 8-2018, 1701Q guidelines.
- Graduated-rate non-VAT taxpayers keep paying 3% percentage tax (2551Q): RR 8-2018, 2551Q guidelines.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# Bump when any rule or output format below changes: it is part of every response-cache key.
CALCULATOR_VERSION = "1"

VAT_THRESHOLD = 3_000_000.0
EIGHT_PCT_RATE = 0.08
EIGHT_PCT_REDUCTION = 250_000.0
OSD_RATE = 0.40
PERCENTAGE_TAX_RATE = 0.03

# (over, not_over, base_amount, rate_on_excess) — RR 8-2018, effective January 1, 2023 and onwards.
GRADUATED_2023: list[tuple[float, float, float, float]] = [
    (0, 250_000, 0, 0.0),
    (250_000, 400_000, 0, 0.15),
    (400_000, 800_000, 22_500, 0.20),
    (800_000, 2_000_000, 102_500, 0.25),
    (2_000_000, 8_000_000, 402_500, 0.30),
    (8_000_000, math.inf, 2_202_500, 0.35),
]

REGIMES = ("8_percent", "graduated")
DEDUCTION_METHODS = ("osd", "itemized")


class CalcError(ValueError):
    """A problem the user can fix (bad or missing input) or a rule that blocks the request."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message}


@dataclass
class TaxBreakdown:
    regime: str
    deduction_method: str | None
    lines: list[tuple[str, float]]
    income_tax: float
    percentage_tax: float
    total_tax: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lines"] = [{"label": label, "amount": amount} for label, amount in self.lines]
        return d


def _money(x: float) -> float:
    return round(x + 0.0, 2)


def _amount(name: str, value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise CalcError("invalid_amount", f"{name} must be a non-negative number of pesos, got {value!r}.")
    return float(value)


def graduated_tax(taxable_income: float) -> float:
    for over, not_over, base, rate in GRADUATED_2023:
        if taxable_income <= not_over:
            return _money(base + max(taxable_income - over, 0) * rate)
    raise AssertionError("unreachable")


def calculate_ph_tax(gross_receipts: float, regime: str, deduction_method: str | None = None,
                     itemized_expenses: float | None = None, non_operating_income: float | None = None,
                     compensation_income: float | None = None) -> TaxBreakdown:
    """Annual income tax (and percentage tax, where it applies) for one taxable year."""
    gross = _amount("gross_receipts", gross_receipts)
    nonop = _amount("non_operating_income", non_operating_income)
    comp = _amount("compensation_income", compensation_income)
    if regime not in REGIMES:
        raise CalcError("invalid_regime", f"regime must be one of {REGIMES}, got {regime!r}.")
    mixed = comp > 0
    notes = ["Annual estimate for taxable year 2023 onwards. Excludes tax credits (e.g. Form 2307 withholding) "
             "and quarterly payments already made."]

    if regime == "8_percent":
        if gross + nonop > VAT_THRESHOLD:
            raise CalcError("eight_percent_not_available",
                            "The 8% option is only available when gross sales/receipts plus other non-operating "
                            f"income do not exceed the ₱3,000,000 VAT threshold (here: ₱{gross + nonop:,.2f}). "
                            "Graduated rates apply instead.")
        reduction = 0.0 if mixed else EIGHT_PCT_REDUCTION
        base = max(gross + nonop - reduction, 0.0)
        business_tax = _money(base * EIGHT_PCT_RATE)
        lines = [("Gross sales/receipts", gross), ("Other non-operating income", nonop),
                 ("Less: ₱250,000 reduction" + (" (not allowed for mixed-income earners)" if mixed else ""),
                  -reduction),
                 ("Tax base", base), ("8% income tax on business income", business_tax)]
        comp_tax = 0.0
        if mixed:
            comp_tax = graduated_tax(comp)
            lines += [("Compensation income", comp), ("Graduated tax on compensation", comp_tax)]
        income_tax = _money(business_tax + comp_tax)
        notes.append("The 8% rate is in lieu of the graduated income tax and the 3% percentage tax.")
        return TaxBreakdown("8_percent", None, lines, income_tax, 0.0, income_tax, notes)

    if deduction_method not in DEDUCTION_METHODS:
        raise CalcError("missing_deduction_method",
                        "Graduated rates need a deduction method: 'osd' (40% of gross) or 'itemized'.")
    if deduction_method == "osd":
        deduction, label = _money(gross * OSD_RATE), "Less: Optional Standard Deduction (40% of gross)"
    else:
        if itemized_expenses is None:
            raise CalcError("missing_itemized_expenses", "Itemized deductions need the total allowable expenses.")
        deduction, label = _amount("itemized_expenses", itemized_expenses), "Less: itemized deductions"
    business_net = max(gross - deduction, 0.0)
    taxable = business_net + nonop + comp
    income_tax = graduated_tax(taxable)
    lines = [("Gross sales/receipts", gross), (label, -deduction), ("Net business income", business_net)]
    if nonop:
        lines.append(("Other non-operating income", nonop))
    if comp:
        lines.append(("Compensation income", comp))
    lines += [("Taxable income", taxable), ("Graduated income tax", income_tax)]

    if gross + nonop <= VAT_THRESHOLD:
        percentage_tax = _money(gross * PERCENTAGE_TAX_RATE)
        lines.append(("3% percentage tax (Form 2551Q)", percentage_tax))
    else:
        percentage_tax = 0.0
        notes.append("Above the ₱3,000,000 VAT threshold: VAT applies instead of the 3% percentage tax "
                     "(not computed here).")
    total = _money(income_tax + percentage_tax)
    return TaxBreakdown("graduated", deduction_method, lines, income_tax, percentage_tax, total, notes)


def compare_regimes(gross_receipts: float, itemized_expenses: float | None = None,
                    non_operating_income: float | None = None, compensation_income: float | None = None) -> dict:
    """Total tax under each option the taxpayer qualifies for, and which is lowest."""
    common = dict(non_operating_income=non_operating_income, compensation_income=compensation_income)
    candidates = [("8_percent", dict(regime="8_percent")),
                  ("graduated_osd", dict(regime="graduated", deduction_method="osd"))]
    if itemized_expenses is not None:
        candidates.append(("graduated_itemized", dict(regime="graduated", deduction_method="itemized",
                                                      itemized_expenses=itemized_expenses)))
    options, eight_ok = [], True
    for name, kw in candidates:
        try:
            options.append({"option": name, **calculate_ph_tax(gross_receipts, **kw, **common).to_dict()})
        except CalcError as e:
            if e.code != "eight_percent_not_available":
                raise
            eight_ok = False
    lowest = min(options, key=lambda o: o["total_tax"])["option"]
    return {"options": options, "lowest": lowest, "eight_percent_available": eight_ok}
