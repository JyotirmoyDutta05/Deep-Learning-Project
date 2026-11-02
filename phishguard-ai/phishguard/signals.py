"""Attack-vector check: a transparent rule layer that runs next to the recurrent model.

The neural model judges *intent from wording*. This layer answers a different, concrete
question: does the message actually ask the reader to do something dangerous? A phishing
message needs at least one attack vector to work (a link to "verify", a request for a
password / OTP / CVV, a payment demand, or a threat with a deadline). A legitimate message
that merely *talks about* fraud usually has none.

The result is shown in the UI and API next to the model score. It only changes the verdict
in one narrow, explicit case (see `combine`), so the model's own score is always visible.
"""
from __future__ import annotations

import re

_I = re.IGNORECASE
_URL = re.compile(r"https?://\S+|www\.\S+|\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|rb\.gy|is\.gd)/\S+", _I)
_IP_URL = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", _I)
_CLICK = re.compile(r"\b(click|tap|follow|open)\b[^.\n]{0,40}\b(link|here|below|button)\b|\[\s*click[^\]]*\]", _I)
_CRED = re.compile(
    r"\b(confirm|verify|update|provide|enter|share|send|reply\s+with|give|submit|validate|re-?enter)\b"
    r"[^.\n]{0,60}\b(password|passcode|otp|one[- ]time|pin|cvv|cvc|card\s*(number|details)|login|user\s*id|username"
    r"|account\s*(information|details)|identity|security\s*code|ssn|aadhaar|pan|date\s+of\s+birth|code\s+on\s+your\s+screen|details)\b",
    _I)
_PAY = re.compile(
    r"\b(gift\s*card|processing\s+fee|verification\s+fee|security\s+deposit|pay\s+(a|the)\s+\w*\s*fee|transfer\b[^.\n]{0,40}\baccount\s+below"
    r"|wire\s+transfer|bitcoin|crypto|western\s+union|release\s+the\s+funds)\b", _I)
_CALL = re.compile(r"\bcall\b[^.\n]{0,40}(\+?\d[\d\s-]{8,})", _I)
_THREAT = re.compile(
    r"\b(within|in)\s+(the\s+next\s+)?\d+\s*(hours?|hrs?|minutes?|days?)\b|\b(permanently\s+(closed|blocked|suspended|cancelled)"
    r"|legal\s+action|will\s+be\s+(closed|blocked|suspended|terminated|cancelled|lost)|final\s+(notice|warning)|expires?\s+today"
    r"|do\s+not\s+disconnect)\b", _I)
# Signals that a message is written BY the account holder or is a safe bank notice.
_REPORTER = re.compile(
    r"\b(i\s+am\s+writing\s+to\s+(formally\s+)?report|i\s+(would\s+like|want)\s+to\s+(report|dispute)|i\s+did\s+not\s+(authori[sz]e|make)"
    r"|i\s+noticed\s+(a|this|an)\b|my\s+(card|wallet|account)\b[^.\n]{0,30}\b(stolen|used|compromised)|please\s+(block|freeze|cancel)\s+(my|this)\s+card"
    r"|dispute\s+the\s+charge|credit\s+the\s+(stolen\s+)?(funds|amount)\s+back|copy\s+of\s+(a|my)\s+police\s+(report|complaint))\b", _I)
_SAFE_BANK = re.compile(
    r"\b(never\s+ask(s)?\s+(you\s+)?for|do\s+not\s+share\s+(your\s+)?(otp|pin|cvv)|call\s+the\s+number\s+(on|printed\s+on)\s+(the\s+back\s+of\s+)?your\s+card"
    r"|official\s+app|no\s+action\s+is\s+needed|you\s+do\s+not\s+need\s+to\s+take\s+any)\b", _I)


def check(text: str) -> dict:
    vectors = []
    if _URL.search(text):
        vectors.append("ip_link" if _IP_URL.search(text) else "link")
    if _CLICK.search(text) and "link" not in vectors and "ip_link" not in vectors:
        vectors.append("click_request")
    if _CRED.search(text):
        vectors.append("credential_request")
    if _PAY.search(text):
        vectors.append("payment_demand")
    threat = bool(_THREAT.search(text))
    if threat:
        vectors.append("deadline_threat")
    reporter = bool(_REPORTER.search(text))
    safe_bank = bool(_SAFE_BANK.search(text))
    # A phone number is only an attack vector when combined with pressure (refund-scam pattern).
    if _CALL.search(text) and threat and not safe_bank:
        vectors.append("callback_number")
    # "verify your identity" style requests inside a self-written report are not attacks.
    if reporter and vectors == ["credential_request"]:
        vectors = []
    return {"attack_vectors": vectors, "reporter_voice": reporter, "safe_bank_language": safe_bank}


LABELS = {"link": "Link to an external site", "ip_link": "Link to a raw IP address", "click_request": "Asks you to click a link/button",
          "credential_request": "Asks for password / OTP / card / identity details", "payment_demand": "Demands a payment or fee",
          "deadline_threat": "Threat with a deadline", "callback_number": "Pressure to call a phone number"}


def combine(p_model: float, threshold: float, sig: dict) -> tuple[float, str]:
    """Return (final probability, rule note).

    Only one adjustment: if the model flags fraud but the message contains NO attack vector
    and reads as written by the account holder or as a safe bank notice, the final risk is
    capped just below the decision line ("talks about fraud, asks for nothing dangerous").
    """
    if p_model >= threshold and not sig["attack_vectors"] and (sig["reporter_voice"] or sig["safe_bank_language"]):
        return min(p_model, threshold * 0.6), "capped: no attack vector, reads as a report / safe bank notice"
    return p_model, ""
