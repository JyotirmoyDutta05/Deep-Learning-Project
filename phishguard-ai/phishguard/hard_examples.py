"""Contrastive hard examples for fraud-vocabulary false positives.

Problem: in public phishing corpora, words such as "unauthorized", "fraudulent", "account
suspended" or "urgent" appear almost only in scams. A model trained on them learns
"talks about fraud => is fraud" and flags legitimate mail such as a customer reporting a
stolen card, a genuine bank alert, or a bank's reply to a dispute.

This module generates PAIRS that share that vocabulary but differ in what actually makes a
message malicious: an attack vector (link to "verify", request for password/OTP/CVV,
payment demand, threat with a deadline). Legitimate examples talk about fraud but ask for
nothing dangerous; phishing examples always contain at least one attack vector.

    python -m phishguard.hard_examples --out ../data/hard_examples.csv
"""
from __future__ import annotations

import argparse
import csv
import random

BANKS = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Chase", "Bank of America", "Wells Fargo",
         "Barclays", "HSBC", "Citibank", "Kotak Mahindra Bank", "PayPal", "Capital One", "Amex"]
MERCHANTS = ["Amazon", "Flipkart", "Walmart", "Uber", "Swiggy", "Zomato", "Netflix", "Apple.com",
             "Best Buy", "Target", "Steam Games", "Shell Petrol", "Myntra", "eBay", "Starbucks"]
NAMES = ["Rahul Sharma", "Priya Nair", "John Miller", "Emily Davis", "Arjun Mehta", "Sarah Johnson",
         "Ananya Gupta", "Michael Brown", "Neha Verma", "David Wilson", "Kavya Reddy", "Chris Lee"]
SHADY = ["http://secure-{b}-verify.com/login", "http://{b}-alerts.support-center.net/auth",
         "bit.ly/{r}", "http://{b}.account-restore.info/id={r}", "www.{b}-kyc-update.co/verify",
         "http://192.168.{n}.{m}/secure/{b}", "tinyurl.com/{r}", "http://{b}-refund-desk.top/claim"]
PLACEHOLDER = ["[Your Full Name]", "[Your Phone Number]", "[Your Email Address]", "[Insert date]",
               "[Insert amount]", "[Card ending XXXX]", "[Account Number]"]


def _r(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _amount(rng):
    return _r(rng, ["$", "Rs ", "₹", "£", "USD "]) + f"{rng.randint(5, 9999):,}.{rng.randint(0, 99):02d}"


def _date(rng):
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/202{rng.randint(3, 6)}"


def _link(rng, bank):
    slug = bank.lower().replace(" ", "").replace(".", "")[:10]
    return _r(rng, SHADY).format(b=slug, r="".join(rng.choices("abcdefXYZ0123456789", k=6)),
                                 n=rng.randint(0, 255), m=rng.randint(1, 254))


def _sig(rng, placeholder: bool):
    if placeholder:
        return "\n".join(rng.sample(PLACEHOLDER[:3], k=rng.randint(1, 3)))
    name = _r(rng, NAMES)
    extra = _r(rng, ["", f"\n+91 9{rng.randint(100000000, 999999999)}", f"\nCustomer ID: {rng.randint(10000, 99999)}"])
    return name + extra


def _ctx(rng):
    b = _r(rng, BANKS)
    return dict(bank=b, merchant=_r(rng, MERCHANTS), amount=_amount(rng), date=_date(rng),
                last4=f"{rng.randint(0, 9999):04d}", case=f"{_r(rng, ['CASE', 'REF', 'DSP', 'SR'])}-{rng.randint(100000, 999999)}",
                link=_link(rng, b), hours=_r(rng, ["24", "12", "48", "2", "6"]))


# ----------------------------------------------------------------------------- legitimate (0)
def _legit_customer_dispute(rng, c):
    ph = rng.random() < 0.5
    subj = _r(rng, ["Urgent: Unauthorized Fraudulent Transaction on My Account",
                    "Report of unauthorized charge on card ending {last4}",
                    "Dispute request - fraudulent transaction", "Fraudulent charge on my account - please help",
                    "Unauthorized transaction - request to block card", "URGENT: I did not make this payment"])
    opening = _r(rng, ["Dear Customer Support Team,", "Dear {bank} Fraud Department,", "Hello,", "To whom it may concern,",
                       "Dear Sir/Madam,", "Hi {bank} team,"])
    body = [
        _r(rng, ["I am writing to formally report an unauthorized and fraudulent transaction on my account.",
                 "I would like to report a fraudulent charge that I did not make.",
                 "I noticed a transaction on my statement that I did not authorize.",
                 "My card appears to have been used without my permission and I want to report it.",
                 "I am contacting you to dispute an unauthorized payment from my account."]),
        _r(rng, ["I noticed this charge during a regular check of my recent statements.",
                 "I only found out when I received the transaction SMS.",
                 "I still have my card with me, so I believe the details were stolen online.",
                 "I have never shopped with this merchant before."]),
        "Here are the details of the transaction:",
        f"Card / Account: {'[Insert last 4 digits]' if ph else 'ending ' + c['last4']}",
        f"Date: {'[Insert date]' if ph else c['date']}",
        f"Merchant: {'[Insert merchant name]' if ph else c['merchant']}",
        f"Amount: {'[Insert amount, e.g., $150.00]' if ph else c['amount']}",
        _r(rng, ["I have already locked my card temporarily through the mobile app.",
                 "I have frozen the card from the app to prevent further misuse.",
                 "I have not shared my PIN or OTP with anyone."]),
        _r(rng, ["Please officially block this card and issue a replacement as soon as possible.",
                 "Please cancel the card and send me a new one.",
                 "Kindly freeze my card and reverse the charge."]),
        _r(rng, ["I request that you open a formal investigation, dispute the charge on my behalf, and credit the stolen funds back to my account.",
                 "Please raise a chargeback and refund the amount to my account.",
                 "I would like this charge investigated and the money returned."]),
        _r(rng, ["Please let me know if you require any further documentation, such as a dispute form or a copy of a police report.",
                 "I can provide a copy of my police complaint if needed.",
                 "I can visit the branch with ID if required."]),
        _r(rng, ["I look forward to your prompt confirmation.", "Thank you for your help.", "Looking forward to hearing from you."]),
    ]
    if rng.random() < 0.3:
        rng.shuffle(body[1:3])
    return "Subject: " + subj.format(**c) + "\n" + opening.format(**c) + "\n" + "\n".join(body) + \
        "\n" + _r(rng, ["Sincerely,", "Regards,", "Thanks,", "Best regards,"]) + "\n" + _sig(rng, ph)


def _legit_bank_alert(rng, c):
    return _r(rng, [
        "{bank}: {amount} debited from A/c XX{last4} on {date} at {merchant}. If this was not you, call the number on the back of your card or visit your nearest branch. We will never ask for your PIN, OTP or password.",
        "Transaction alert: Your {bank} card ending {last4} was used for {amount} at {merchant} on {date}. Not you? Open the official {bank} app and choose 'Report fraud', or call customer care. {bank} never asks for OTP or CVV over phone, SMS or email.",
        "Dear customer, we have blocked a suspicious transaction of {amount} at {merchant} on your card ending {last4}. No action is needed if you did not attempt this payment; your card remains secure. For questions, call the number printed on your card.",
        "Security notice from {bank}: a new device signed in to your account on {date}. If this was you, you can ignore this message. If not, please change your password from the official app. We will never send you a link asking for your login details.",
        "Your {bank} statement for card ending {last4} is ready. Total due {amount}. Please review it in the app. If you see any charge you do not recognise, report it to us from the app or by calling customer care.",
        "{bank} Fraud Prevention: We noticed unusual activity and have temporarily paused card ending {last4} to protect you. Please call the number on the back of your card to confirm recent transactions. Do not share OTP, PIN or CVV with anyone, including bank staff.",
    ]).format(**c)


def _legit_bank_reply(rng, c):
    return _r(rng, [
        "Subject: Your dispute {case} has been received\nDear {name},\nThank you for reporting the unauthorized transaction of {amount} at {merchant} on your card ending {last4}. We have blocked the card and opened dispute {case}. A provisional credit will appear within 10 working days while our fraud team investigates. A replacement card will be posted to your registered address. You do not need to take any further action.\nRegards,\n{bank} Disputes Team",
        "Subject: Update on fraud case {case}\nHello {name},\nOur investigation into the fraudulent charge of {amount} is complete and the amount has been credited back to your account. Thank you for your patience. If you have more questions, reply to this email or call customer care.\nKind regards,\nCustomer Care, {bank}",
        "Subject: Card replacement dispatched\nDear {name},\nFollowing your report of unauthorized use, your old card ending {last4} has been permanently blocked and a new card has been dispatched. Please remember that we will never ask for your full card number, PIN or OTP.\n{bank}",
    ]).format(name=_r(rng, NAMES), **c)


def _legit_scam_discussion(rng, c):
    return _r(rng, [
        "Hey, just a heads up - I got a fake message claiming my {bank} account was suspended and asking me to click a link. Don't fall for it, the real bank never asks for OTP. I reported it.",
        "Hi team, IT security reminder: we are seeing phishing emails pretending to be from {bank} about unauthorized transactions. Do not click links or enter passwords. Forward suspicious emails to the security team.",
        "Mom, someone called pretending to be from {bank} fraud department asking for the OTP. I hung up and called the bank directly. Never share it with anyone.",
        "Subject: Police complaint copy\nHi,\nAttaching the FIR copy for the unauthorized transaction of {amount} on {date} as requested for my dispute {case}. Let me know if anything else is needed.\nThanks,\n{name}",
        "Dear {bank} team, I received an email saying my account will be closed in {hours} hours unless I verify my details. I did not click it. Can you confirm it is a scam? I am forwarding it for your records.\nThanks,\n{name}",
    ]).format(name=_r(rng, NAMES), **c)


# ----------------------------------------------------------------------------- phishing (1)
def _phish_suspension(rng, c):
    return _r(rng, [
        "Subject: URGENT: Your account has been suspended due to suspicious activity!\nDear Customer,\nWe detected an unauthorized login attempt to your online banking portal from a new device. For your security, we have temporarily suspended your account. To restore access you must immediately confirm your account information by clicking the secure link below:\n{link}\nIf you do not update your details within the next {hours} hours, your account will be permanently closed and legal action may follow.\nSecurity Operations Team\n{bank}",
        "Subject: Unauthorized transaction detected on your account\nDear valued customer,\nAn unauthorized transaction of {amount} at {merchant} was attempted on your card ending {last4}. To cancel this transaction and avoid being charged, verify your identity now at {link} within {hours} hours.\nFailure to verify will result in the charge being completed.\n{bank} Fraud Department",
        "{bank} ALERT: Fraudulent activity detected. Your account is locked. Verify your login now at {link} or it will be blocked permanently in {hours} hours.",
        "Dear customer, your {bank} KYC has expired and your account will be suspended today. Update your PAN and Aadhaar immediately at {link} to avoid blocking.",
        "Subject: Final notice - account restriction\nWe noticed unusual sign-in activity. Your access is limited until you confirm your password and card details here: {link}. Accounts not verified within {hours} hours will be closed.\n{bank} Security",
    ]).format(**c)


def _phish_credential_request(rng, c):
    return _r(rng, [
        "Subject: Unauthorized charge of {amount} - action required\nDear Customer,\nTo reverse the fraudulent charge on your card, please reply to this email with your full card number, expiry date, CVV and the OTP you will receive. Our fraud team will process your refund immediately.\n{bank} Fraud Department",
        "This is {bank} fraud prevention. A fraudulent transaction of {amount} is pending on your account. To stop it, share the OTP sent to your phone with our agent now. Do not disconnect or the money will be lost.",
        "Dear user, we detected unauthorized access. Kindly provide your username, password and date of birth by replying to this message so we can secure your account within {hours} hours.",
        "Subject: Refund of {amount} pending\nWe owe you a refund for an unauthorized transaction. To receive it, please confirm your net banking user ID, password and the OTP at {link}.\nRefund Desk, {bank}",
        "{bank}: Your debit card ending {last4} is blocked due to fraud. To unblock, reply with your card number, CVV and ATM PIN. Ignore this and your card will be cancelled permanently.",
    ]).format(**c)


def _phish_payment_demand(rng, c):
    return _r(rng, [
        "Subject: Invoice for {amount} - auto-renewal\nYour subscription has been renewed and {amount} will be debited today. If you did not authorize this, call our cancellation desk immediately at +1-8{n} to get a refund. Charges cannot be reversed after {hours} hours.",
        "Your account has been flagged for fraud. To avoid legal action you must pay a verification fee of {amount} via gift card within {hours} hours. Reply with the gift card codes.",
        "Dear customer, a chargeback for {amount} was raised in your name. To cancel it, transfer a refundable security deposit to the account below and send the receipt to this email today.",
        "Congratulations! Your fraud refund of {amount} is approved. Pay a small processing fee at {link} to release the funds into your account.",
    ]).format(n=rng.randint(100000000, 999999999), **c)


def _phish_disguised_as_reply(rng, c):
    """Phishing that copies the polite dispute-reply style but still contains an attack vector."""
    return _r(rng, [
        "Subject: Your dispute {case} requires verification\nDear Customer,\nThank you for reporting the unauthorized transaction of {amount}. Before we can credit your account, you must verify your identity by logging in at {link} within {hours} hours, otherwise the dispute will be closed.\nRegards,\n{bank} Disputes Team",
        "Subject: Refund ready for case {case}\nHello,\nYour fraud claim has been approved. To receive {amount}, confirm your card number and CVV at {link}. Unclaimed refunds expire in {hours} hours.\n{bank}",
    ]).format(**c)


LEGIT = [(_legit_customer_dispute, 3), (_legit_bank_alert, 3), (_legit_bank_reply, 2), (_legit_scam_discussion, 2)]
PHISH = [(_phish_suspension, 3), (_phish_credential_request, 3), (_phish_payment_demand, 2), (_phish_disguised_as_reply, 2)]


def _weighted(rng, pool):
    fns = [f for f, w in pool for _ in range(w)]
    return _r(rng, fns)


def generate(n_per_class: int = 1500, seed: int = 7):
    rng = random.Random(seed)
    rows, seen = [], set()
    for label, pool in ((0, LEGIT), (1, PHISH)):
        tries = 0
        count = 0
        while count < n_per_class and tries < n_per_class * 20:
            tries += 1
            t = _weighted(rng, pool)(rng, _ctx(rng))
            if t not in seen:
                seen.add(t); rows.append((t, label)); count += 1
    rng.shuffle(rows)
    return rows


# Hand-written messages (NOT produced by the templates above) used only as a regression check.
REGRESSION = [
    ("Subject: Urgent: Unauthorized Fraudulent Transaction on Account [Your Account/Card Number]\nDear Customer Support Team,\nI am writing to formally report an unauthorized and fraudulent transaction that occurred on my account. I noticed this charge during a regular check of my recent statements and did not authorize or make this purchase.\nHere are the details of the transaction:\n• Account / Card Number: [Insert last 4 digits of your card or account number]\n• Date of Transaction: [Insert date]\n• Time of Transaction: [Insert time, if known]\n• Merchant Name: [Insert name of the merchant as it appears on the statement]\n• Amount: [Insert amount, e.g., $150.00]\nUpon recognizing this activity, I immediately temporarily locked my card through the mobile application to prevent further issues. Please officially freeze or cancel this card and issue a replacement as soon as possible.\nI request that you open a formal investigation into this matter, dispute the charge on my behalf, and credit the stolen funds back to my account.\nPlease let me know if you require any further documentation, such as a formal dispute form or a copy of a police report. I look forward to your prompt confirmation that this issue is being investigated.\nSincerely,\n[Your Full Name]\n[Your Phone Number]\n[Your Email Address]", 0),
    ("Subject: URGENT: Your account has been suspended due to suspicious activity!\nDear Customer,\nWe detected an unauthorized login attempt to your online banking portal from a new device located in a different country. For your security, we have temporarily suspended your account restrictions.\nTo restore access and verify your identity, you must immediately confirm your account information by clicking the secure link below:\n[CLICK HERE TO VERIFY YOUR ACCOUNT]\nIf you do not update your details within the next 24 hours, your account will be permanently closed, and legal actions may follow.\nThank you for your immediate cooperation.\nSecurity Operations Team\n[Name of Fake Bank]", 1),
    ("Hi, my wallet was stolen yesterday and I think someone used my credit card at a petrol station. Can you please block it and tell me what I need to do to get the money back? I'm at home all day if you want to call. Thanks, Meera", 0),
    ("INR 2,340.00 spent on your SBI Credit Card ending 8812 at BIGBASKET on 14-Oct. Avl limit INR 45,210. If not done by you, call 1800 180 1290 to block.", 0),
    ("Dear user your SBI YONO account will be blocked today. Please update your PAN card immediately by clicking http://sbi-yono-update.in/kyc", 1),
    ("Your Amazon order was flagged as fraudulent. To cancel the $899 charge, call our refund team at +1-888-512-0091 and give them the code on your screen.", 1),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/hard_examples.csv")
    ap.add_argument("--n", type=int, default=1500, help="examples per class")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rows = generate(a.n, a.seed)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["text", "label"]); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
