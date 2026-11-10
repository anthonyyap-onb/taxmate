from taxmate.pii import mask_pii


def test_masks_hyphenated_tin_with_branch():
    assert mask_pii("TIN 123-456-789-000 po") == ("TIN [TIN] po", ["tin"])


def test_masks_plain_9_digit_tin():
    assert mask_pii("my tin is 123456789")[0] == "my tin is [TIN]"


def test_masks_email_and_phone():
    out, kinds = mask_pii("email juan@x.ph or 09171234567 / +639171234567")
    assert out == "email [EMAIL] or [PHONE] / [PHONE]" and kinds == ["email", "phone"]


def test_amounts_not_masked():
    s = "I earned ₱3,000,000 and 250000 last year, plus PHP 300,000.50 and P 123456789"
    assert mask_pii(s) == (s, [])


def test_no_pii_unchanged():
    assert mask_pii("Can I deduct rent on 8%?") == ("Can I deduct rent on 8%?", [])
