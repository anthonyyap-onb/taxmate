"""Structured tax profile kept in session memory and shown in the sidebar."""
from __future__ import annotations

import re
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ValidationError

TaxpayerType = Literal["freelancer_professional", "sole_proprietor", "mixed_income"]
Regime = Literal["8_percent", "graduated"]
IncomeSource = Literal["purely_self_employed", "mixed"]
YesNo = Literal["yes", "no"]
Deduction = Literal["osd", "itemized"]


class TaxProfile(BaseModel):
    taxpayer_type: Literal["freelancer_professional", "sole_proprietor", "mixed_income", "unknown"] = "unknown"
    tax_regime: Literal["8_percent", "graduated", "unknown"] = "unknown"
    income_source: Literal["purely_self_employed", "mixed", "unknown"] = "unknown"
    under_3m: Literal["yes", "no", "unknown"] = "unknown"
    deduction_method: Literal["osd", "itemized", "unknown"] = "unknown"

    def display(self) -> list[tuple[str, str]]:
        return [(label, _VALUE_LABELS.get(getattr(self, field), "Unknown")) for field, label in _FIELD_LABELS]


_FIELD_LABELS = [
    ("taxpayer_type", "Taxpayer type"),
    ("tax_regime", "Tax regime"),
    ("income_source", "Income source"),
    ("under_3m", "Gross receipts under ₱3M"),
    ("deduction_method", "Deduction method"),
]
_VALUE_LABELS = {
    "freelancer_professional": "Freelancer/Professional",
    "sole_proprietor": "Sole proprietor",
    "mixed_income": "Mixed-income earner",
    "8_percent": "8%",
    "graduated": "Graduated",
    "purely_self_employed": "Purely self-employed",
    "mixed": "Mixed (with compensation)",
    "yes": "Yes",
    "no": "No",
    "osd": "OSD (40%)",
    "itemized": "Itemized",
}


class ProfileUpdate(BaseModel):
    taxpayer_type: Optional[TaxpayerType] = None
    tax_regime: Optional[Regime] = None
    income_source: Optional[IncomeSource] = None
    under_3m: Optional[YesNo] = None
    deduction_method: Optional[Deduction] = None


class Fact(BaseModel):
    field: Literal["taxpayer_type", "tax_regime", "income_source", "under_3m", "deduction_method"]
    value: str
    quote: str  # the user's exact words that state this fact


class _Extraction(BaseModel):
    facts: list[Fact]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def facts_to_update(message: str, facts: list[Fact]) -> ProfileUpdate:
    """Keep only facts whose quote appears verbatim in the message and whose value is valid for the field."""
    msg, fields = _norm(message), {}
    for f in facts:
        quote = _norm(f.quote)
        if not quote or quote not in msg:
            continue
        try:
            ProfileUpdate(**{f.field: f.value})
        except ValidationError:
            continue
        fields[f.field] = f.value
    return ProfileUpdate(**fields)


def merge_profile(p: TaxProfile, u: ProfileUpdate) -> TaxProfile:
    changes = {k: v for k, v in u.model_dump().items() if v is not None}
    return p.model_copy(update=changes)


_EXTRACT_SYSTEM = """You update a Philippine taxpayer's profile from their latest chat message.
Fill a field ONLY if the user's latest message explicitly states or corrects a fact about THEMSELVES for that
exact field. Never infer, guess, or pick a "likely" value; leave everything else null. Most messages fill 0-2 fields.
Rules:
- Questions are not statements. "Can I use the 8%?", "Pwede ba ako mag 8%?", "Should I pick OSD?" set NOTHING
  for tax_regime / deduction_method; the user is asking, not telling you their choice.
- tax_regime is set only when the user says they chose/opted for/are on a regime ("I opted for 8%", "naka-graduated ako").
- deduction_method is set only when the user says they use OSD or itemized. Income level or occupation never implies it.
- Do not derive one field from another (e.g. income under 3M does not imply 8%; "freelancer" alone does not imply
  income_source unless they say they have no employer / are purely self-employed, or mention an employer).
Example: "I'm a freelance designer, self-employed, earning about 1.2M a year" ->
  taxpayer_type=freelancer_professional, income_source=purely_self_employed, under_3m=yes; everything else null.
Fields:
- taxpayer_type: freelancer_professional (freelancer, consultant, professional, self-employed service provider),
  sole_proprietor (owns a registered business/shop), mixed_income (employee who also freelances/has a business).
- tax_regime: 8_percent (opted for the 8% income tax rate) or graduated (graduated income tax rates).
- income_source: purely_self_employed, or mixed (also earns compensation from an employer).
- under_3m: yes if annual gross sales/receipts are at or below PHP 3,000,000, no if above.
- deduction_method: osd (Optional Standard Deduction, 40%) or itemized.
Messages may be in English, Filipino, or Taglish.
Return a list of facts. For each fact give the field, its value, and `quote`: the user's exact words (copied verbatim
from the latest message) that state it. No supporting quote means no fact. Return an empty list when nothing is stated."""


def make_profile_extractor(client, model: str) -> Callable[[str, TaxProfile], ProfileUpdate]:
    def extract(message: str, current: TaxProfile) -> ProfileUpdate:
        completion = client.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Current profile: {current.model_dump_json()}\n\nLatest message: {message}"},
            ],
            response_format=_Extraction,
        )
        parsed = completion.choices[0].message.parsed
        return facts_to_update(message, parsed.facts) if parsed else ProfileUpdate()

    return extract
