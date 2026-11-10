import pytest
from pydantic import ValidationError

from taxmate.profile import Fact, ProfileUpdate, TaxProfile, facts_to_update, merge_profile


def test_defaults_unknown():
    assert all(v == "Unknown" for _, v in TaxProfile().display())


def test_merge_applies_only_stated_fields():
    p = merge_profile(TaxProfile(tax_regime="8_percent"), ProfileUpdate(taxpayer_type="freelancer_professional"))
    assert p.tax_regime == "8_percent" and p.taxpayer_type == "freelancer_professional"


def test_correction_overwrites():
    assert merge_profile(TaxProfile(tax_regime="8_percent"), ProfileUpdate(tax_regime="graduated")).tax_regime == "graduated"


def test_update_rejects_unknown():
    with pytest.raises(ValidationError):
        ProfileUpdate(tax_regime="unknown")


def test_display_labels():
    assert TaxProfile(tax_regime="8_percent").display()[1] == ("Tax regime", "8%")


MSG = "Hi! I'm a  Freelance designer, earning about 1.2M a year. I opted for the 8% rate."


def test_fact_with_quote_from_message_is_kept_case_and_space_insensitive():
    u = facts_to_update(MSG, [Fact(field="taxpayer_type", value="freelancer_professional", quote="i'm a freelance designer"),
                              Fact(field="tax_regime", value="8_percent", quote="opted for the 8% rate")])
    assert u.taxpayer_type == "freelancer_professional" and u.tax_regime == "8_percent"


def test_fact_whose_quote_is_not_in_message_is_dropped():
    u = facts_to_update(MSG, [Fact(field="deduction_method", value="osd", quote="I use OSD")])
    assert u == ProfileUpdate()


def test_fact_with_empty_quote_or_invalid_value_is_dropped():
    u = facts_to_update(MSG, [Fact(field="under_3m", value="yes", quote="  "),
                              Fact(field="tax_regime", value="unknown", quote="opted for the 8% rate")])
    assert u == ProfileUpdate()
