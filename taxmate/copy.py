"""Fixed user-facing copy. Change wording here only."""

GREETING = ("Hi, I'm TaxMate! I answer Philippine tax questions for freelancers and small businesses "
            "using official BIR issuances. To tailor my answers: are you a freelancer/professional or a "
            "sole proprietor, and do you know if you're on the 8% rate or graduated rates?")
SUGGESTED_QUESTIONS = ["8% or graduated — which is better for me?",
                       "Can I deduct my internet bill?",
                       "What is Form 2307?"]
BLOCKED_TEXT = ("I can only help with Philippine tax questions for freelancers and small businesses — like the "
                "8% vs. graduated rates, deductions, Form 2307, and quarterly filing. Could you ask me something along those lines?")
UNSAFE_TEXT = ("I can't help with that. I can explain legal ways to manage your taxes, such as choosing between "
               "the 8% rate and graduated rates with the Optional Standard Deduction.")
FALLBACK_TEXT = ("I couldn't find guidance on that in my BIR sources, so I'd rather not guess. "
                 "Please check with a CPA or your Revenue District Office (RDO).")
ERROR_TEXT = "TaxMate is having trouble reaching its AI service right now. Please try again in a moment."
EMPTY_TEXT = "Please type a question."
TOO_LONG_TEXT = "That message is too long — please keep it under 2,000 characters."
DISCLAIMER = ("TaxMate gives general information based on BIR issuances — not legal or CPA advice. Figures are "
              "estimates. File officially through eBIRForms/eFPS, and consult a CPA or your RDO for your specific case.")
MISSING_CONFIG = "TaxMate isn't configured yet. Missing: {missing}. Add them to `.env` (local) or the app's Secrets (Streamlit Cloud)."
PROFILE_ACK_TEXT = ("Got it — I've noted that in your tax profile (see the sidebar). "
                    "What would you like to know about your taxes?")
VERIFY_NOTE = "ℹ️ General information from BIR issuances — verify with a CPA or your RDO before filing."
