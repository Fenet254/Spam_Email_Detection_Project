import os
import math
import joblib
from flask import Flask, request, jsonify, render_template_string
from train import clean_text  # reuse the exact same preprocessing used in training

app = Flask(__name__)

MODEL = joblib.load("models/spam_model.pkl")
VECTORIZER = joblib.load("models/vectorizer.pkl")

# ... (PAGE template and routes exactly as you had them — no changes needed)
PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spam Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --blush:#fff2f6;
    --rose:#ffd9e5;
    --rose-deep:#f5a8c1;
    --magenta:#d6437c;
    --magenta-deep:#a92c5c;
    --plum:#4a1f33;
    --ham:#5e9c76;
    --line:rgba(74,31,51,0.14);
  }
  *{box-sizing:border-box;}
  html,body{ height:100%; }
  body{
    margin:0; min-height:100vh;
    background:
      radial-gradient(circle at 12% 8%, #ffe1ec 0%, transparent 40%),
      radial-gradient(circle at 88% 92%, #ffd0e0 0%, transparent 45%),
      linear-gradient(160deg, var(--blush) 0%, var(--rose) 100%);
    font-family:'Poppins',sans-serif;
    color:var(--plum);
    display:flex; align-items:center; justify-content:center;
    padding:60px 16px;
    position:relative; overflow-x:hidden;
  }

  .bg-pattern{
    position:fixed; inset:0; width:100%; height:100%;
    z-index:0; pointer-events:none;
  }

  .petal{
    position:fixed; width:14px; height:14px; border-radius:60% 0 60% 0;
    background:var(--rose-deep); opacity:.5; z-index:1;
    animation:drift 14s linear infinite;
  }
  @keyframes drift{
    0%{ transform:translateY(-40px) rotate(0deg); opacity:0; }
    10%{ opacity:.5; }
    100%{ transform:translateY(110vh) rotate(220deg); opacity:0; }
  }

  .envelope-wrap{
    width:100%; max-width:480px; position:relative; z-index:2;
  }
  .flap{
    width:100%; height:86px;
    background:linear-gradient(135deg, var(--rose-deep), var(--magenta));
    clip-path:polygon(0 0, 100% 0, 50% 100%);
    box-shadow:0 -6px 24px -10px rgba(169,44,92,0.4);
  }
  .seal{
    position:absolute; left:50%; top:86px; transform:translate(-50%,-50%);
    width:52px; height:52px; border-radius:50%; z-index:3;
    background:radial-gradient(circle at 35% 30%, #e26a95, var(--magenta-deep));
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 8px 16px -6px rgba(0,0,0,0.35), inset 0 -4px 8px rgba(0,0,0,0.15), inset 0 3px 5px rgba(255,255,255,0.3);
  }
  .seal svg{ width:22px; height:22px; }

  .card{
    width:100%;
    background:rgba(255,255,255,0.68);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.9);
    border-top:none;
    border-radius:0 0 28px 28px;
    box-shadow:0 40px 80px -30px rgba(169,44,92,0.35), 0 0 0 1px rgba(255,255,255,0.4) inset;
    padding:34px 38px 34px;
    position:relative;
  }

  .eyebrow{
    text-align:center; font-size:11px; letter-spacing:.24em; text-transform:uppercase;
    color:var(--magenta-deep); font-weight:500; margin-bottom:10px;
  }
  h1{
    text-align:center; font-family:'Playfair Display',serif; font-weight:700;
    font-size:clamp(28px,5vw,34px); margin:0 0 8px; color:var(--plum);
  }
  .sub{
    text-align:center; font-size:13.5px; color:var(--plum); opacity:.65;
    margin:0 0 28px; font-weight:300;
  }

  .gauge-wrap{ display:flex; justify-content:center; margin-bottom:2px; }
  .gauge-labels{
    display:flex; justify-content:space-between; width:200px; margin:0 auto 24px;
    font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; font-weight:500;
  }
  .gauge-labels .ham{ color:var(--ham); }
  .gauge-labels .spam{ color:var(--magenta-deep); }
  .needle{ transform-origin:100px 98px; transition:transform 1.2s cubic-bezier(.22,1.4,.36,1); }

  textarea{
    width:100%; min-height:120px; resize:vertical;
    background:rgba(255,255,255,0.6);
    border:1.5px solid rgba(214,67,124,0.18);
    border-radius:16px; padding:16px 18px;
    font-family:'Poppins',sans-serif; font-size:14.5px; color:var(--plum);
    outline:none; transition:border-color .2s ease, box-shadow .2s ease;
    margin-bottom:8px;
  }
  textarea::placeholder{ color:var(--plum); opacity:.4; }
  textarea:focus{
    border-color:var(--magenta);
    box-shadow:0 0 0 4px rgba(214,67,124,0.12);
  }
  .hint{ font-size:11px; color:var(--plum); opacity:.5; margin:0 0 22px; text-align:right; }

  button{
    width:100%; padding:15px; border:none; cursor:pointer; border-radius:16px;
    background:linear-gradient(135deg, var(--magenta), var(--magenta-deep));
    color:#fff; font-family:'Poppins',sans-serif; font-weight:500; font-size:14px;
    letter-spacing:.03em;
    box-shadow:0 14px 28px -10px rgba(169,44,92,0.55);
    transition:transform .18s ease, box-shadow .18s ease;
  }
  button:hover{ transform:translateY(-2px); box-shadow:0 18px 34px -10px rgba(169,44,92,0.65); }
  button:active{ transform:translateY(0) scale(0.98); }

  .result-block{
    margin-top:28px; padding-top:26px; border-top:1px dashed var(--line);
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    animation:rise .5s ease both;
  }
  @keyframes rise{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }

  .result-seal{
    width:84px; height:84px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; flex-direction:column;
    font-family:'Playfair Display',serif; font-weight:700; font-size:13px;
    color:#fff; text-transform:uppercase; letter-spacing:.05em;
    box-shadow:0 10px 20px -6px rgba(0,0,0,0.35), inset 0 -6px 10px rgba(0,0,0,0.15), inset 0 4px 6px rgba(255,255,255,0.25);
    animation:press .5s cubic-bezier(.3,1.7,.4,1) .25s both;
    position:relative;
  }
  .result-seal::after{
    content:""; position:absolute; inset:6px; border-radius:50%;
    border:1px dashed rgba(255,255,255,0.5);
  }
  .result-seal.spam{ background:radial-gradient(circle at 35% 30%, #e26a95, var(--magenta-deep)); }
  .result-seal.ham{ background:radial-gradient(circle at 35% 30%, #7cb896, #3f7856); }
  @keyframes press{
    0%{ transform:scale(2.6); opacity:0; }
    55%{ opacity:1; }
    100%{ transform:scale(1); opacity:1; }
  }
  .score-readout{ text-align:right; }
  .score-readout .n{ font-family:'Playfair Display',serif; font-size:26px; font-weight:700; color:var(--plum); }
  .score-readout .l{ font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--plum); opacity:.5; }

  @media (prefers-reduced-motion: reduce){
    .needle, .result-block, .result-seal, .petal{ animation:none !important; transition:none !important; }
  }
</style>
</head>
<body>

  <svg class="bg-pattern" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern id="mailTile" width="220" height="220" patternUnits="userSpaceOnUse" patternTransform="rotate(-6)">
        <rect x="18" y="24" width="52" height="36" rx="4" fill="none" stroke="#f3aecb" stroke-width="1.4" opacity="0.55"/>
        <polyline points="18,24 44,46 70,24" fill="none" stroke="#f3aecb" stroke-width="1.4" opacity="0.55"/>

        <circle cx="168" cy="42" r="17" fill="none" stroke="#eab6cf" stroke-width="1.2" stroke-dasharray="2 3" opacity="0.5"/>
        <text x="168" y="47" font-family="Poppins,sans-serif" font-size="9" font-weight="600" fill="#e2a0bf" text-anchor="middle" opacity="0.6">HAM</text>

        <text x="120" y="110" font-family="Georgia,serif" font-size="26" fill="#f0b9d1" opacity="0.45">@</text>

        <path d="M40,140 c4,-8 12,-8 16,0 c4,-8 12,-8 16,0 c0,10 -16,20 -16,20 c0,0 -16,-10 -16,-20 z"
              fill="#f6c9dd" opacity="0.4"/>

        <rect x="120" y="150" width="48" height="32" rx="4" fill="none" stroke="#eda9c6" stroke-width="1.3" opacity="0.5" transform="rotate(8 144 166)"/>
        <polyline points="120,150 144,168 168,150" fill="none" stroke="#eda9c6" stroke-width="1.3" opacity="0.5" transform="rotate(8 144 166)"/>

        <circle cx="30" cy="185" r="12" fill="none" stroke="#e59bbf" stroke-width="1.1" stroke-dasharray="2 3" opacity="0.45"/>
        <text x="30" y="189" font-family="Poppins,sans-serif" font-size="7.5" font-weight="600" fill="#dd8bb2" text-anchor="middle" opacity="0.55">SPAM</text>

        <path d="M10,90 q30,-14 60,0" fill="none" stroke="#f3b9d3" stroke-width="1" stroke-dasharray="1 5" opacity="0.4"/>
        <path d="M150,100 q20,10 40,-4" fill="none" stroke="#f3b9d3" stroke-width="1" stroke-dasharray="1 5" opacity="0.4"/>
      </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#mailTile)"/>
  </svg>

  <div class="petal" style="left:8%; animation-delay:0s;"></div>
  <div class="petal" style="left:22%; animation-delay:3s;"></div>
  <div class="petal" style="left:78%; animation-delay:6s;"></div>
  <div class="petal" style="left:90%; animation-delay:1.5s;"></div>
  <div class="petal" style="left:55%; animation-delay:9s;"></div>

  <div class="envelope-wrap">
    <div class="flap"></div>
    <div class="seal">
      <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"/>
        <polyline points="22,6 12,13 2,6"/>
      </svg>
    </div>

    <div class="card">
      <div class="eyebrow">Message Inspection</div>
      <h1>Spam Detector</h1>
      <p class="sub">Paste a message and let the detector decide.</p>

      <div class="gauge-wrap">
        <svg width="200" height="120" viewBox="0 0 200 120">
          <defs>
            <linearGradient id="hamGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#5e9c76"/>
              <stop offset="100%" stop-color="#a9d4b6"/>
            </linearGradient>
            <linearGradient id="spamGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#f2a9c5"/>
              <stop offset="100%" stop-color="#a92c5c"/>
            </linearGradient>
          </defs>
          <path d="M8,98 A92,92 0 0,1 100,6" fill="none" stroke="url(#hamGrad)" stroke-width="9" stroke-linecap="round" opacity="0.55"/>
          <path d="M100,6 A92,92 0 0,1 192,98" fill="none" stroke="url(#spamGrad)" stroke-width="9" stroke-linecap="round" opacity="0.55"/>
          {% if result %}
            {% set is_spam = 'spam' in result|lower %}
            {% set pct = score|float %}
            {% set angle = (pct * 90) if is_spam else -(pct * 90) %}
            <line class="needle" x1="100" y1="98" x2="100" y2="16"
                  stroke="{{ '#a92c5c' if is_spam else '#3f7856' }}"
                  stroke-width="3.5" stroke-linecap="round"
                  style="transform:rotate({{ angle }}deg)"/>
          {% else %}
            <line class="needle" x1="100" y1="98" x2="100" y2="16" stroke="#d6437c" stroke-width="3.5" stroke-linecap="round" style="transform:rotate(0deg)"/>
          {% endif %}
          <circle cx="100" cy="98" r="5.5" fill="#4a1f33"/>
        </svg>
      </div>
      <div class="gauge-labels"><span class="ham">Ham</span><span class="spam">Spam</span></div>

      <form method="POST" action="/predict-form">
        <textarea name="message" placeholder="Paste a message to inspect…" required></textarea>
        <p class="hint">Plain text only</p>
        <button type="submit">Check message</button>
      </form>

      {% if result %}
        {% set is_spam = 'spam' in result|lower %}
        {% set pct = score|float %}
        <div class="result-block">
          <div class="result-seal {{ 'spam' if is_spam else 'ham' }}">{{ result }}</div>
          <div class="score-readout">
            <div class="n">{{ (pct * 100)|round(1) }}%</div>
            <div class="l">Confidence</div>
          </div>
        </div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


def get_prediction(message: str):
    cleaned = clean_text(message)
    vec = VECTORIZER.transform([cleaned])
    pred = MODEL.predict(vec)[0]
    # LinearSVC has decision_function instead of predict_proba;
    # squash the unbounded distance into a 0-1 confidence
    raw_score = float(MODEL.decision_function(vec)[0])
    confidence = 1 / (1 + math.exp(-abs(raw_score)))
    label = "Spam" if pred == 1 else "Ham"
    return label, confidence


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE, result=None, score=None)


@app.route("/predict-form", methods=["POST"])
def predict_form():
    message = request.form.get("message", "")
    label, confidence = get_prediction(message)
    return render_template_string(PAGE, result=label, score=round(confidence, 4))


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "message field is required"}), 400
    label, confidence = get_prediction(message)
    return jsonify({"message": message, "prediction": label, "score": confidence})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)