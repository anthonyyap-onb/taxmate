"""OpenAI function-calling wrappers around the deterministic calculator. A tool call never raises: bad input,
schema violations and rule blocks all come back as {"error": code, "message": ...} for the model to explain or
turn into a clarifying question."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

import openai
from pydantic import BaseModel, Field, ValidationError

from taxmate import tracing
from taxmate.calculator import CalcError, calculate_ph_tax, compare_regimes
from taxmate.profile import TaxProfile

_PESOS = "in Philippine pesos for the whole taxable year, as a plain number (e.g. 1200000)"


class CalculateArgs(BaseModel):
    gross_receipts: float = Field(description=f"Annual gross sales/receipts from business or profession, {_PESOS}.")
    regime: Literal["8_percent", "graduated"] = Field(description="The income tax regime the taxpayer is on.")
    deduction_method: Optional[Literal["osd", "itemized"]] = Field(
        description="Required for graduated: 'osd' (40% of gross) or 'itemized'. null for 8_percent.")
    itemized_expenses: Optional[float] = Field(description=f"Total allowable business expenses, {_PESOS}. "
                                                           "Required for itemized; otherwise null.")
    non_operating_income: Optional[float] = Field(description=f"Other non-operating business income, {_PESOS}. "
                                                              "null if none.")
    compensation_income: Optional[float] = Field(
        description=f"Salary/compensation income for mixed-income earners, {_PESOS}. null or 0 if purely "
                    "self-employed.")


class CompareArgs(BaseModel):
    gross_receipts: float = Field(description=f"Annual gross sales/receipts from business or profession, {_PESOS}.")
    itemized_expenses: Optional[float] = Field(description=f"Total allowable business expenses, {_PESOS}, if the "
                                                           "user gave them; otherwise null.")
    non_operating_income: Optional[float] = Field(description=f"Other non-operating business income, {_PESOS}. "
                                                              "null if none.")
    compensation_income: Optional[float] = Field(
        description=f"Salary/compensation income for mixed-income earners, {_PESOS}. null or 0 if purely "
                    "self-employed.")


_TOOLS = {
    "calculate_ph_tax": (CalculateArgs, lambda a: calculate_ph_tax(**a.model_dump()).to_dict(),
                         "Compute the annual Philippine income tax (plus 3% percentage tax for graduated filers) "
                         "for a self-employed individual or mixed-income earner on a KNOWN regime. Call this for "
                         "any tax amount; never compute tax yourself."),
    "compare_regimes": (CompareArgs, lambda a: compare_regimes(**a.model_dump()),
                        "Compare total annual tax under the 8% option, graduated with OSD, and graduated with "
                        "itemized deductions (if expenses are given), and say which is lowest. Use when the user "
                        "asks which regime is better or cheaper."),
}

TOOL_SCHEMAS: list[dict] = [openai.pydantic_function_tool(model, name=name, description=desc)
                            for name, (model, _fn, desc) in _TOOLS.items()]


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict

    @property
    def ok(self) -> bool:
        return "error" not in self.result

    def log_view(self) -> dict:
        """Numbers and codes only, safe for the JSON log."""
        return {"name": self.name, "args": self.args, "ok": self.ok, "error": self.result.get("error")}


# --- Preconditions -------------------------------------------------------------------------------------------
# The prompt tells the model to ask instead of guessing; these checks enforce it in code. A failed check returns an
# error the model must turn into a clarifying question, so the calculator never runs on an input the user didn't give.

_AMOUNT = re.compile(r"(?P<neg>-\s*)?(?:₱|\bphp\s*|\bp(?=\s?\d))?\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
                     r"\s*(?P<unit>k|m|mil|million|thousand)?\b", re.I)
_UNITS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6}
_REGIME_WORDS = {"8_percent": re.compile(r"\b8\s*%|\b8\s*percent|eight percent", re.I),
                 "graduated": re.compile(r"graduated", re.I)}
_DEDUCTION_WORDS = {"osd": re.compile(r"\bosd\b|optional standard", re.I),
                    "itemized": re.compile(r"\bitemi[sz]", re.I)}
PRECONDITION_ERRORS = frozenset({"regime_not_stated", "deduction_method_not_stated", "amount_not_stated",
                                 "negative_amount"})
_AMOUNT_FIELDS = ("gross_receipts", "itemized_expenses", "non_operating_income", "compensation_income")


def stated_amounts(user_texts: list[str]) -> set[float]:
    """Every peso amount the user wrote, with its sign ("1.2M" -> 1200000, "-₱500,000" -> -500000)."""
    found = set()
    for text in user_texts:
        for m in _AMOUNT.finditer(text):
            value = float(m["num"].replace(",", "") + (m["dec"] or ""))
            value *= _UNITS.get((m["unit"] or "").lower(), 1)
            found.add(-value if m["neg"] else value)
    return found


def _err(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def check_call(name: str, args: dict, profile: TaxProfile, user_texts: list[str]) -> dict | None:
    """None if the call may run; otherwise an error telling the model what to ask. user_texts: the user's
    messages in this conversation, the current one last."""
    current = user_texts[-1] if user_texts else ""
    if name == "calculate_ph_tax":
        regime = args.get("regime")
        if profile.tax_regime == "unknown" and not (regime in _REGIME_WORDS and _REGIME_WORDS[regime].search(current)):
            return _err("regime_not_stated", "The user has not said which regime they are on. Do not assume one: "
                        "ask whether they are on the 8% rate or graduated rates, and offer to compare both. If compare_regimes "
                        "already returned a result in this turn, answer from that comparison instead of asking.")
        method = args.get("deduction_method")
        if regime == "graduated" and profile.deduction_method == "unknown" and not (
                method in _DEDUCTION_WORDS and _DEDUCTION_WORDS[method].search(current)):
            return _err("deduction_method_not_stated", "The user has not said how they deduct expenses. Do not "
                        "assume: ask whether they use the Optional Standard Deduction (OSD, 40%) or itemized deductions.")
    stated = stated_amounts(user_texts)
    for field in _AMOUNT_FIELDS:
        value = args.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if value == 0 and field != "gross_receipts":  # 0 for an optional amount means "none"
            continue
        if any(s < 0 and abs(-value - s) <= 1 for s in stated):
            return _err("negative_amount", f"The user gave a negative amount for {field}. Ask them to confirm the "
                        "correct (positive) peso amount.")
        if not any(abs(value - s) <= 1 for s in stated):
            return _err("amount_not_stated", f"{field}={value:,.0f} is not an amount the user stated. Do not derive "
                        "or convert amounts yourself: ask the user for the annual figure.")
    return None


@tracing.observe("tool")
def _execute(name: str, args: dict) -> dict:
    model, fn, _desc = _TOOLS[name]
    try:
        return fn(model.model_validate(args))
    except ValidationError as e:
        return {"error": "invalid_arguments", "message": "; ".join(err["msg"] for err in e.errors())}
    except CalcError as e:
        return e.to_dict()


def run_tool(name: str, arguments_json: str, profile: TaxProfile | None = None,
             user_texts: list[str] | None = None) -> ToolCall:
    """Without profile/user_texts (unit tests, direct use) the preconditions are skipped."""
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        return ToolCall(name, {}, {"error": "invalid_arguments", "message": "Arguments were not a JSON object."})
    if name not in _TOOLS:
        return ToolCall(name, args, {"error": "unknown_tool", "message": f"No tool named {name!r}."})
    if profile is not None and user_texts is not None:
        problem = check_call(name, args, profile, user_texts)
        if problem:
            return ToolCall(name, args, problem)
    return ToolCall(name, args, _execute(name, args))
