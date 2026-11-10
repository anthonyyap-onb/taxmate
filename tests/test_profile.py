import pytest
from pydantic import ValidationError

from taxmate.profile import ProfileUpdate, TaxProfile, merge_profile


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
