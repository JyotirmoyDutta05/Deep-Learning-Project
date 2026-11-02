"""PhishGuard AI dashboard.   streamlit run dashboard/app.py   (run from the project root)"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from phishguard.extract import extract_text_from_file  # noqa: E402
from phishguard.service import Predictor  # noqa: E402
from phishguard.signals import LABELS as AV_LABELS  # noqa: E402
from phishguard.text import tokenize  # noqa: E402

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

EXAMPLES = {
    "Bank SMS": ("sms", "URGENT: Your HDFC Bank account has been suspended due to incomplete KYC. Verify now at http://hdfc-kyc-secure.example-login.com or your account will be blocked within 24 hours."),
    "Prize SMS": ("sms", "Congratulations! You have won a free gift voucher worth 500. Claim your reward now by calling 09061701461. Offer expires today."),
    "Phishing email": ("email", "Dear customer, we detected unusual sign-in activity on your PayPal account. For your security we have limited access. Please confirm your password and card details at www.paypal-resolution-center.net/verify immediately to avoid permanent suspension."),
    "Transaction": ("transaction", "UPI/DR/PAYMENT REVERSAL DESK/URGENT chargeback pending - share OTP to refund Rs 49999 to your account immediately"),
    "Friendly note": ("sms", "Hey, are you still coming for lunch tomorrow? I booked the table by the window for 1pm. Let me know if that works."),
}
CONTROL = EXAMPLES["Friendly note"][1]
FILLER = ("Anyway, the weather has been lovely this week and the garden is finally coming along. We walked to the "
          "market on Saturday and picked up some fresh bread, tomatoes and a few apples for dinner.").split()
LAYERS = {"Evidence for the verdict": "contribution", "Input gate (what it stored)": "write",
          "Memory survival (what it still held)": "survival"}
CUE_LABELS = {"urgency": "Urgency", "credentials": "Asks for credentials", "money": "Money or prize",
              "impersonation": "Impersonation wording", "link": "Contains a link"}


def filler(n: int) -> str:
    return " ".join(FILLER[i % len(FILLER)] for i in range(n))


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str, mtime: float) -> Predictor:
    return Predictor(path)


def heat_html(tokens: list[dict], mode: str) -> str:
    vals = [t[mode] for t in tokens]
    mx = max(abs(v) for v in vals) or 1.0
    spans = []
    for t, v in zip(tokens, vals):
        tip = html.escape(f"evidence {t['contribution']:+.2f} | input gate {t['write']:.2f} | "
                          f"keep gate {t['keep']:.2f} | survival to end {t['survival']:.2f}")
        
        style = "padding:2px 5px;margin:1px;border-radius:4px;display:inline-block;transition:all 0.15s ease;"
        if mode == "contribution":
            s = v / mx
            mag = abs(s)
            if mag < 0.05:
                style += "color:#94a3b8;background:transparent;"
            elif s > 0:
                # Toward Fraud (Red / Crimson)
                if mag > 0.6:
                    style += "background:rgba(220,38,38,0.88);color:#ffffff;font-weight:600;box-shadow:0 0 8px rgba(239,68,68,0.5);"
                elif mag > 0.25:
                    style += "background:rgba(239,68,68,0.35);color:#fecaca;border:1px solid rgba(239,68,68,0.5);font-weight:500;"
                else:
                    style += "background:rgba(239,68,68,0.18);color:#fca5a5;"
            else:
                # Toward Legitimate (Blue / Azure)
                if mag > 0.6:
                    style += "background:rgba(37,99,235,0.88);color:#ffffff;font-weight:600;box-shadow:0 0 8px rgba(59,130,246,0.5);"
                elif mag > 0.25:
                    style += "background:rgba(59,130,246,0.35);color:#bfdbfe;border:1px solid rgba(59,130,246,0.5);font-weight:500;"
                else:
                    style += "background:rgba(59,130,246,0.18);color:#93c5fd;"
        elif mode == "write":
            # Input Gate (Amber / Gold) - matches chart yellow line
            if v > 0.6:
                style += "background:rgba(217,119,6,0.88);color:#ffffff;font-weight:600;box-shadow:0 0 8px rgba(245,158,11,0.5);"
            elif v > 0.35:
                style += "background:rgba(245,158,11,0.35);color:#fef08a;border:1px solid rgba(245,158,11,0.5);font-weight:500;"
            elif v > 0.15:
                style += "background:rgba(245,158,11,0.18);color:#fde047;"
            else:
                style += "color:#94a3b8;background:transparent;"
        else:
            # Memory Survival (Teal / Emerald) - matches chart cyan line
            if v > 0.65:
                style += "background:rgba(13,148,136,0.9);color:#ffffff;font-weight:600;box-shadow:0 0 8px rgba(20,184,166,0.5);"
            elif v > 0.35:
                style += "background:rgba(20,184,166,0.35);color:#99f6e4;border:1px solid rgba(20,184,166,0.5);font-weight:500;"
            elif v > 0.15:
                style += "background:rgba(20,184,166,0.18);color:#5eead4;"
            else:
                style += "color:#64748b;background:transparent;"
                
        spans.append(f'<span title="{tip}" style="{style}">{html.escape(t["t"])}</span>')

    legend_html = ""
    if mode == "contribution":
        legend_html = '''<div style="display:flex;align-items:center;gap:12px;margin-top:10px;font-size:12px;color:#94a3b8;">
            <span>🔵 Legitimate Evidence</span>
            <div style="flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg, #2563eb, #334155, #dc2626);max-width:260px;"></div>
            <span>🔴 Fraud Evidence</span>
        </div>'''
    elif mode == "write":
        legend_html = '''<div style="display:flex;align-items:center;gap:12px;margin-top:10px;font-size:12px;color:#94a3b8;">
            <span>Low write (ignored)</span>
            <div style="flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg, #334155, #d97706);max-width:200px;"></div>
            <span>High write (stored into memory)</span>
        </div>'''
    else:
        legend_html = '''<div style="display:flex;align-items:center;gap:12px;margin-top:10px;font-size:12px;color:#94a3b8;">
            <span>Decayed (forgotten)</span>
            <div style="flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg, #334155, #0d9488);max-width:200px;"></div>
            <span>High Survival (held in memory)</span>
        </div>'''

    return (f'<div style="font-family:\'IBM Plex Mono\',Consolas,monospace;font-size:14px;line-height:2.2;padding:16px;'
            f'border:1px solid #334155;border-radius:8px;background:#0f172a;color:#f8fafc;max-height:380px;overflow-y:auto;box-shadow:inset 0 2px 6px rgba(0,0,0,0.4);">'
            + " ".join(spans) + f"</div>{legend_html}")


def memory_note(res: dict) -> str:
    m = res.get("memory")
    if res["label"] == 1 and m:
        w = res["tokens"][m["top_index"]]["t"]
        if m["distance"] > 0:
            return (f"Strongest evidence: “{w}” (token {m['top_index'] + 1}). The network read {m['distance']} more tokens "
                    f"after it and still carried about {round(m['survival'] * 100)}% of what it stored there into the decision.")
        return f"Strongest evidence: “{w}”, at the very end, so nothing had to be remembered."
    if res["label"] == 1:
        return "The verdict comes from many small signals spread across the message rather than one strong token."
    return "No token pushed strongly toward fraud. Red tokens, if any, are weak signals the network weighed and outvoted."


def score_rows(P: Predictor, texts: list[str]) -> list[dict | None]:
    out: list[dict | None] = []
    for i in range(0, len(texts), 64):
        chunk = texts[i : i + 64]
        valid = [(j, t) for j, t in enumerate(chunk) if tokenize(t)]
        res = P.score_many([t for _, t in valid]) if valid else []
        mapped = {j: r for (j, _), r in zip(valid, res)}
        out += [mapped.get(j) for j in range(len(chunk))]
    return out


# ---------------------------------------------------------------- sidebar
st.sidebar.title("PhishGuard AI")
found = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("artifacts/*/model.pt"))
env = os.environ.get("PHISHGUARD_MODEL")
if env and env not in found:
    found.insert(0, env)
if not found:
    st.error("No trained model found.")
    st.code("python -m phishguard.train --dataset sms_spam --cell lstm --out artifacts/lstm")
    st.stop()
choice = st.sidebar.selectbox("Model", found, help="Each folder in artifacts/ holds one trained model.")
model_path_obj = ROOT / choice if not Path(choice).is_absolute() else Path(choice)
P = load_model(str(model_path_obj), model_path_obj.stat().st_mtime)
st.sidebar.markdown(f"**Architecture:** {P.cfg.cell.upper()}, {P.cfg.hidden} hidden units")
st.sidebar.markdown(f"**Decision line:** {P.threshold * 100:.0f} / 100")
st.sidebar.markdown(f"**Reads up to:** {P.cfg.max_len} tokens")
channel = st.sidebar.radio("Message type", ["sms", "email", "transaction"], format_func=str.capitalize)
st.sidebar.caption("Messages are scored locally and are not stored.")

st.title("PhishGuard AI: Phishing & Fraud Prevention")
tab_a, tab_m, tab_b, tab_r = st.tabs(["Check a message", "Memory stress test", "Scan a batch", "Model report"])

# ---------------------------------------------------------------- tab: analyze
with tab_a:
    st.session_state.setdefault("text", "")

    def load_example(name):
        st.session_state["text"] = EXAMPLES[name][1]

    cols = st.columns(len(EXAMPLES))
    for c, name in zip(cols, EXAMPLES):
        c.button(name, key=f"ex_{name}", on_click=load_example, args=(name,), use_container_width=True)

    with st.expander("📎 Upload Attachment or Screenshot (PDF, DOCX, PNG, JPG, EML)", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload suspicious attachment, invoice PDF, or email screenshot",
            type=["pdf", "docx", "png", "jpg", "jpeg", "txt", "eml"],
            key="file_attach_uploader",
            help="Extracts text via OCR (Tesseract) or document parsing (PyMuPDF / docx) into the text box below for sequence analysis.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            extracted_txt, desc, meta = extract_text_from_file(file_bytes, uploaded_file.name)
            if extracted_txt and not extracted_txt.startswith("[Error"):
                st.success(f"Extracted {len(extracted_txt.split())} words from {desc}")
                c_use, c_clear = st.columns([1, 4])
                if c_use.button("Load into text box", key="btn_use_ocr", type="primary"):
                    st.session_state["text"] = extracted_txt
                    st.rerun()
            else:
                st.error(f"Could not extract text: {extracted_txt}")

    st.text_area("Message to check", key="text", height=170,
                 placeholder="Paste the email body, SMS or transaction narrative here")
    pad = st.toggle("Add 100 filler words after the message", help="Tests whether the opening signal is still remembered.")
    if st.button("Check message", type="primary"):
        txt = st.session_state["text"]
        if not txt.strip():
            st.warning("Enter a message first.")
        else:
            try:
                st.session_state["res"] = P.analyze(txt + (" " + filler(100) if pad else ""), explain=True)
                st.session_state["res"]["channel"] = channel
            except ValueError as e:
                st.session_state.pop("res", None)
                st.error(str(e))

    res = st.session_state.get("res")
    if res is None:
        st.info("Nothing checked yet. Paste a message or pick an example, then choose Check message.")
    else:
        v1, v2, v3 = st.columns([2, 1, 3])
        (v1.error if res["label"] else v1.success)("Fraud / Malicious [1]" if res["label"] else "Legitimate [0]")
        v2.metric("Risk score", f"{res['risk_score']:.0f}/100")
        v3.progress(min(res["risk_score"] / 100, 1.0), text=f"Risk level: {res['risk_band']} (decision line {res['threshold']:.0f})")
        st.caption(f"{res['n_tokens']} tokens read in {res['latency_ms']} ms" + (". Long input: only the first tokens were read." if res["truncated"] else ""))

        st.markdown("**Attack-vector check** (does the message ask you to do something dangerous?)")
        av = res.get("attack_vectors", [])
        a1, a2 = st.columns([3, 2])
        if av:
            a1.error("Found: " + " · ".join(AV_LABELS.get(v, v) for v in av))
        else:
            a1.success("None found: no suspicious link, no request for password / OTP / card details, no payment demand, no deadline threat.")
        tags = [t for t, on in (("Written by the account holder", res.get("reporter_voice")),
                                ("Safe bank wording (\"we never ask for OTP\")", res.get("safe_bank_language"))) if on]
        a2.markdown(f"Neural model score: **{res.get('model_score', res['risk_score']):.0f}/100**" +
                    ("".join(f"<br>• {t}" for t in tags)), unsafe_allow_html=True)
        if res.get("rule_note"):
            st.warning(f"Final risk lowered from {res['model_score']:.0f} to {res['risk_score']:.0f}: the message talks about fraud "
                       "but contains no attack vector and reads as a report or a genuine bank notice.")

        st.info(memory_note(res))
        cues = sorted({CUE_LABELS[c["category"]] for c in res["cues"]})
        if cues:
            st.markdown("**Cues seen:** " + ", ".join(cues))

        st.subheader("Gate heatmap")
        layer = st.radio("Layer", list(LAYERS), horizontal=True, label_visibility="collapsed")
        st.html(heat_html(res["tokens"], LAYERS[layer]))
        st.caption("Hover a token for its gate values. Red pushes toward fraud, blue toward legitimate; teal shows gate strength.")

        df = pd.DataFrame(res["tokens"]).rename(columns={"t": "token"})
        df["position"] = df.index + 1
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X("position:Q", title="Token position"), y=alt.Y("contribution:Q", title="Evidence (logit)"),
            color=alt.condition(alt.datum.contribution > 0, alt.value("#c8102e"), alt.value("#1f6fb2")),
            tooltip=["position", "token", alt.Tooltip("contribution:Q", format="+.2f")]).properties(height=200)
        lines = alt.Chart(df.melt(["position", "token"], ["write", "keep", "survival"], "gate", "value")).mark_line().encode(
            x=alt.X("position:Q", title="Token position"), y=alt.Y("value:Q", title="Gate value", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("gate:N", title=None, scale=alt.Scale(domain=["write", "keep", "survival"], range=["#e0a100", "#1f6fb2", "#0e8a80"])),
            tooltip=["position", "token", "gate", alt.Tooltip("value:Q", format=".2f")]).properties(height=200)
        c1, c2 = st.columns(2)
        c1.altair_chart(bars, use_container_width=True)
        c2.altair_chart(lines, use_container_width=True)
        st.caption("Left: which tokens drove the verdict. Right: average input gate (write), keep gate, and how much of each token's memory survived to the end.")

# ---------------------------------------------------------------- tab: stress test
with tab_m:
    st.write("Append more and more harmless text after the message and watch whether the risk score holds. "
             "A model with working long-range memory keeps flagging the fraud and keeps the benign control low.")
    src = st.text_area("Message to test", value=EXAMPLES["Bank SMS"][1], height=120, key="stress_text")
    top = st.slider("Maximum filler words", 50, 350, 300, 50)
    if st.button("Run stress test"):
        ks = sorted({0, 25, 50, 100, 150, 200, 250, 300, 350} & set(range(0, top + 1)) | {top})
        rows = []
        try:
            for k in ks:
                for name, base in (("Your message", src), ("Benign control", CONTROL)):
                    r = P.analyze(f"{base} {filler(k)}".strip(), explain=False)
                    rows.append({"filler_words": k, "series": name, "risk_score": r["risk_score"], "truncated": r["truncated"]})
            d = pd.DataFrame(rows)
            ch = alt.Chart(d).mark_line(point=True).encode(
                x=alt.X("filler_words:Q", title="Filler words after the message"),
                y=alt.Y("risk_score:Q", title="Risk score", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#c8102e", "#1f6fb2"])),
                tooltip=["filler_words", "series", "risk_score"])
            rule = alt.Chart(pd.DataFrame({"y": [P.threshold * 100]})).mark_rule(strokeDash=[5, 4], color="#18222d").encode(y="y:Q")
            st.altair_chart((ch + rule).properties(height=320), use_container_width=True)
            mine = d[d.series == "Your message"]
            held = (mine.risk_score >= P.threshold * 100).all()
            st.success("The flag held at every filler length.") if held else st.warning(
                f"The score dropped below the decision line at {int(mine[mine.risk_score < P.threshold * 100].filler_words.min())} filler words.")
            if d.truncated.any():
                st.caption("Some runs exceeded the model's reading limit, so the end of the filler was cut off.")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- tab: batch
with tab_b:
    mode = st.radio("Input", ["Upload CSV", "Paste messages"], horizontal=True)
    texts: list[str] = []
    if mode == "Upload CSV":
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            col = st.selectbox("Column with the message text", raw.columns)
            texts = raw[col].astype(str).tolist()[:5000]
    else:
        pasted = st.text_area("One message per line", height=150)
        texts = [ln for ln in pasted.splitlines() if ln.strip()]
    if texts and st.button("Scan messages", type="primary"):
        with st.spinner(f"Scanning {len(texts)} messages..."):
            scored = score_rows(P, texts)
        out = pd.DataFrame({"message": texts,
                            "risk_score": [r["risk_score"] if r else None for r in scored],
                            "flag": [("Fraud [1]" if r["label"] else "Legitimate [0]") if r else "Unreadable" for r in scored],
                            "risk_level": [r["risk_band"] if r else None for r in scored]})
        out = out.sort_values("risk_score", ascending=False, na_position="last")
        flagged = int((out.flag == "Fraud [1]").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Messages scanned", len(out)); m2.metric("Flagged as fraud", flagged)
        m3.metric("Flag rate", f"{flagged / len(out):.0%}")
        st.dataframe(out, use_container_width=True, hide_index=True, column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f")})
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode(), "phishguard_results.csv", "text/csv")
        st.caption("Batch scans give scores only. Check a single message for the gate heatmap.")

# ---------------------------------------------------------------- tab: report
with tab_r:
    M = P.metrics
    if not M:
        st.info("This model file carries no evaluation report.")
    else:
        st.write(f"Trained on **{M.get('dataset', '?')}** with a **{M.get('cell', '?').upper()}**. "
                 f"Temperature {M.get('temperature', 0):.2f}, decision line {M.get('threshold', 0):.2f}.")
        rows = [{"Test set": n, **{k: v for k, v in M[key].items()}} for n, key in
                (("Short messages", "test_short"), ("Long messages (with filler)", "test_long")) if key in M]
        if rows:
            st.dataframe(pd.DataFrame(rows).rename(columns={"n": "Messages", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
                         "precision": "Precision", "recall": "Recall", "f1": "F1"}), hide_index=True, use_container_width=True)
        cur = M.get("retention_recall_by_filler_words")
        if cur:
            d = pd.DataFrame({"filler_words": [int(k) for k in cur], "recall": list(cur.values())})
            st.subheader("Does memory last? Fraud recall by amount of filler")
            st.altair_chart(alt.Chart(d).mark_line(point=True, color="#0e8a80").encode(
                x=alt.X("filler_words:Q", title="Filler words after the fraud message"),
                y=alt.Y("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                tooltip=["filler_words", alt.Tooltip("recall:Q", format=".2%")]).properties(height=300), use_container_width=True)
        st.caption("These numbers come from the held-out test split of your own training run.")
