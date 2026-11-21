"""ChatGuard - real-time chat moderation demo (Gradio).

Each message is labelled live (clean / offensive) with a confidence score, offensive
messages are masked, and the side panel logs the response time of every message and lets
the user switch between models. Default model: the quantized INT8 BERT.
Run:  python app.py   ->  http://127.0.0.1:7860
"""
import html
import time

import gradio as gr
import pandas as pd

from src.normalize import normalize
from src.predictors import load_all

print("Loading models ...")
MODELS = load_all()
DEFAULT = "BERT quantized (INT8)"
BUDGET_MS = 50.0
LEX = MODELS["Rule: edit-distance lexicon"].m
for p in MODELS.values():                      # warm-up so the first message is not slow
    p.predict_proba(["hello there"])

LOG_COLS = ["#", "model", "label", "confidence", "latency_ms", "within_50ms"]
CMP_COLS = ["model", "label", "confidence", "latency_ms"]
EMPTY_CMP = pd.DataFrame(columns=CMP_COLS)


def mask(text: str):
    """Mask offensive words (found by the edit-distance lexicon); if the model flagged the
    message but no word matched, hide the whole message."""
    words, masked_any = [], False
    for w in text.split():
        if any(LEX.match(t) for t in normalize(w).split()):
            words.append(w[0] + "*" * (len(w) - 1)); masked_any = True
        else:
            words.append(w)
    return " ".join(words) if masked_any else "█" * min(len(text), 30) + "  (message hidden)"


def render(history):
    rows = []
    for h in history:
        off = h["label"] == "offensive"
        color, badge = ("#fde2e1", "🚫 offensive") if off else ("#e3f6e8", "✅ clean")
        body = html.escape(mask(h["text"]) if off else h["text"])
        rows.append(
            f"<div style='background:{color};border-radius:10px;padding:8px 12px;margin:6px 0;color:#111'>"
            f"<b>player{h['n'] % 3 + 1}:</b> {body}<br>"
            f"<small>{badge} · conf {h['conf']:.2f} · {h['ms']:.1f} ms · {html.escape(h['model'])}</small></div>")
    return ("<div style='height:420px;overflow-y:auto;border:1px solid #ccc;border-radius:10px;padding:6px'>"
            + ("".join(rows) or "<i>No messages yet - type one below.</i>") + "</div>")


def classify(name, msg):
    t0 = time.perf_counter()
    p = float(MODELS[name].predict_proba([msg])[0])
    ms = (time.perf_counter() - t0) * 1000
    label = "offensive" if p >= 0.5 else "clean"
    return label, (p if label == "offensive" else 1 - p), ms


def send(msg, model_name, compare, history, log):
    msg = (msg or "").strip()
    if not msg:
        return "", render(history), history, log, pd.DataFrame(log[::-1], columns=LOG_COLS), EMPTY_CMP
    label, conf, ms = classify(model_name, msg)
    n = len(history) + 1
    history = history + [{"n": n, "text": msg, "label": label, "conf": conf, "ms": ms, "model": model_name}]
    log = log + [[n, model_name, label, round(conf, 3), round(ms, 2), "yes" if ms <= BUDGET_MS else "NO"]]
    cmp_df = EMPTY_CMP
    if compare:
        cmp = []
        for name in MODELS:
            l, c, t = classify(name, msg)
            cmp.append([name, l, round(c, 3), round(t, 2)])
        cmp_df = pd.DataFrame(cmp, columns=CMP_COLS)
    return "", render(history), history, log, pd.DataFrame(log[::-1], columns=LOG_COLS), cmp_df


def clear():
    return render([]), [], [], pd.DataFrame(columns=LOG_COLS), EMPTY_CMP


with gr.Blocks(title="ChatGuard - real-time chat moderation") as demo:
    gr.Markdown("## 🛡️ ChatGuard — real-time chat moderation (CPU only)\n"
                "Type a game-chat message. It is normalized (regex), classified live, and masked "
                f"if offensive. Latency budget: **{BUDGET_MS:.0f} ms / message**.")
    history, log = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.HTML(render([]))
            with gr.Row():
                box = gr.Textbox(placeholder="Type a message and press Enter ...", show_label=False, scale=5)
                btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(["gg wp, that was a great match!", "you are such an 1d10t, uninstall the game",
                         "f.u.c.k this stuuuupid lag", "nice shot, you killed it 🔥",
                         "go back to the kitchen, nobody wants you here"], inputs=box)
        with gr.Column(scale=2):
            model = gr.Dropdown(list(MODELS), value=DEFAULT, label="Model")
            compare = gr.Checkbox(False, label="Also compare all 5 models on each message")
            gr.Markdown("**Response-time log** (newest first)")
            log_df = gr.Dataframe(pd.DataFrame(columns=LOG_COLS), interactive=False)
            cmp_df = gr.Dataframe(EMPTY_CMP, label="All models on the last message", interactive=False)
            gr.Button("Clear chat").click(clear, None, [chat, history, log, log_df, cmp_df])
    outs = [box, chat, history, log, log_df, cmp_df]
    box.submit(send, [box, model, compare, history, log], outs)
    btn.click(send, [box, model, compare, history, log], outs)

if __name__ == "__main__":
    demo.launch()
