"""SolarGuard - solar panel defect inspection UI.

Serves the locked baseline checkpoint (validation macro-F1 0.7534) for single-image
classification into six panel conditions.

Every performance number shown in this interface is a VALIDATION number measured on 117
images. The test split has never been evaluated, so nothing here is a generalization
estimate. The UI is built to surface that rather than hide it: the confidence figure is
labelled uncalibrated wherever it appears, and the two weak classes carry an explicit
caution on the result card.

Inference itself lives in solarguard.inference.predictor - this module only renders it.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solarguard.inference.predictor import SolarGuardPredictor  # noqa: E402

CHECKPOINT = REPO_ROOT / "models" / "solarguard_baseline_v1.pt"
PREPROCESSING_CONFIG = REPO_ROOT / "configs" / "preprocessing.yaml"
MAX_UPLOAD_MB = 20
VAL_SAMPLES = 117

# ---------------------------------------------------------------------------
# Verified figures. Sources: docs/MODEL_CARD.md S7/S10, docs/EXPERIMENT_REPORT.md S5-S8.
# Nothing in this block is estimated - if a number is not measured, it is not here.
# ---------------------------------------------------------------------------

# Locked baseline, per-class F1 on the 117-image validation set.
CLASS_F1 = {
    "Electrical-damage": 0.889,
    "Snow-Covered": 0.884,
    "Clean": 0.863,
    "Dusty": 0.784,
    "Physical-Damage": 0.556,
    "Bird-drop": 0.545,
}
CLASS_VAL_N = {
    "Electrical-damage": 13, "Snow-Covered": 22, "Clean": 27,
    "Dusty": 26, "Physical-Damage": 10, "Bird-drop": 19,
}
# 3-seed baseline arm: recall and false positives received.
CLASS_RECALL = {
    "Snow-Covered": 0.909, "Electrical-damage": 0.846, "Dusty": 0.833,
    "Clean": 0.827, "Bird-drop": 0.667, "Physical-Damage": 0.467,
}
CLASS_FP = {
    "Bird-drop": 28, "Clean": 15, "Dusty": 11,
    "Snow-Covered": 8, "Electrical-damage": 6, "Physical-Damage": 6,
}
CLASS_TRAIN_N = {
    "Clean": 126, "Dusty": 123, "Snow-Covered": 100,
    "Bird-drop": 87, "Electrical-damage": 58, "Physical-Damage": 46,
}
CLASS_WEIGHT = {
    "Clean": 0.7143, "Dusty": 0.7317, "Snow-Covered": 0.9000,
    "Bird-drop": 1.0345, "Electrical-damage": 1.5517, "Physical-Damage": 1.9565,
}
CLASS_WHAT = {
    "Clean": "No visible contamination or damage.",
    "Dusty": "Surface dust or soiling across the panel.",
    "Bird-drop": "Bird droppings - small, localised, high contrast.",
    "Snow-Covered": "Partial or full snow cover.",
    "Electrical-damage": "Visible electrical fault, burn or discolouration.",
    "Physical-Damage": "Cracking or physical breakage of the panel surface.",
}
WEAK_CLASSES = {"Physical-Damage", "Bird-drop"}

# Per-class caution shown on the result card. Facts only, no advice beyond the numbers.
CLASS_NOTE = {
    "Physical-Damage": (
        "Recall for this class is 46.7% on validation - the model misses more than half "
        "of physically damaged panels. 33% of true Physical-Damage images are predicted "
        "as Bird-drop. A result of 'not damaged' from this model is not evidence of "
        "absence."
    ),
    "Bird-drop": (
        "This class attracts the most false positives of any class (28 across the 3-seed "
        "baseline) while reaching only 66.7% recall. The model over-predicts it relative "
        "to how often it actually occurs."
    ),
    "Snow-Covered": "Strongest class on validation - 90.9% recall, 0.884 F1.",
    "Electrical-damage": "Strongest fault class on validation - 84.6% recall, 0.889 F1.",
    "Dusty": "Reliable on validation - 83.3% recall, 0.784 F1.",
    "Clean": "Reliable on validation - 82.7% recall, 0.863 F1. Receives 15 false positives.",
}

EXPERIMENTS = [
    {
        "name": "Baseline", "sub": "Class-weighted cross-entropy",
        "f1": 0.7629, "std": 0.0381, "delta": None, "ci": None, "d": None,
        "seeds_won": None, "epoch": 46.0, "verdict": "REFERENCE", "tone": "neutral",
        "note": "The arm every comparison is measured against. Its seed-to-seed standard "
                "deviation of 0.0381 is the noise floor any effect has to clear.",
    },
    {
        "name": "Focal loss", "sub": "gamma = 2, loss function only",
        "f1": 0.7441, "std": 0.0216, "delta": -0.0189, "ci": (-0.0685, 0.0308), "d": -0.608,
        "seeds_won": "1 / 3", "epoch": 27.0, "verdict": "REJECTED", "tone": "bad",
        "note": "The hypothesis was that focal loss would help Bird-drop, which is confusable "
                "rather than rare. Bird-drop got worse (-0.041). The only meaningful gain landed "
                "on Physical-Damage, the rarest class - the opposite of the proposed mechanism. "
                "One confound is unresolved: focal stopped systematically earlier (46 -> 27 "
                "epochs), so those models may be undertrained rather than inferior.",
    },
    {
        "name": "No rotation", "sub": "Rotation augmentation removed",
        "f1": 0.7729, "std": 0.0244, "delta": 0.0100, "ci": (-0.0413, 0.0612), "d": 0.312,
        "seeds_won": "2 / 3", "epoch": 59.3, "verdict": "PROVISIONALLY RETAINED", "tone": "warn",
        "note": "Retained on grounds other than the mean. RandomRotation(15) defaults to fill=0, "
                "which was measured blacking out 5.53% of pixels on average in ~91% of training "
                "images - an artifact present in 0% of validation images. Seed-to-seed std fell "
                "0.0381 -> 0.0244 (~36%) and the worst seed improved from 0.7189 to 0.7447.",
    },
]

PER_SEED = [
    ("42", 0.7844, 0.7331, 0.7877),
    ("123", 0.7189, 0.7301, 0.7863),
    ("456", 0.7855, 0.7690, 0.7447),
]

LIMITATIONS = [
    ("Test set never evaluated", "high",
     "The 115-image test split has never been loaded or evaluated, by design. No unbiased "
     "generalization estimate exists for this model. Every figure in this interface is a "
     "validation figure, measured on data that model selection was performed against."),
    ("Not production-ready", "high",
     "No certification, no external validation, no real-world evaluation, no defined "
     "failure-handling behaviour. This is a research and portfolio system, not a certified "
     "industrial inspection system."),
    ("Physical-Damage is weak", "high",
     "46.7% recall on validation - the model misses more than half of physically damaged "
     "panels, and 33% of them are predicted as Bird-drop instead. This is the least "
     "desirable direction of error for defect detection: damage is more likely to be "
     "missed than falsely reported."),
    ("Confidence is not calibrated", "high",
     "The number shown next to each prediction is a raw softmax output. No calibration "
     "analysis has been performed, so 90% does not mean the model is right 90% of the "
     "time. It should not be read as a reliability estimate."),
    ("No out-of-distribution rejection", "medium",
     "The model assigns one of six panel labels to any image it is given, including images "
     "that contain no solar panel at all. There is no abstention path and no rejection "
     "mechanism."),
    ("Dataset size", "medium",
     "540 training images across six classes, with 46 examples of the rarest class. This is "
     "small for image classification and is likely the binding constraint on performance - "
     "more so than the loss function or the augmentation policy."),
    ("Validation size caps what is measurable", "medium",
     "With 117 validation images, one flipped prediction is worth 0.0085 macro-F1 while "
     "seed noise is 0.0381. Only effects larger than roughly 0.06 are detectable at three "
     "seeds. Physical-Damage has 10 validation images, so its F1 moves ~0.10 per sample."),
    ("Class imbalance", "medium",
     "2.73:1 between the largest and smallest class, handled with inverse-frequency weighted "
     "cross-entropy. Weighting cannot manufacture information that 46 images do not contain, "
     "and it does nothing for Bird-drop, whose weight is a near-neutral 1.0345."),
    ("Domain shift", "medium",
     "Validation images come from the same aggregated public dataset as training. Real "
     "photographs differ in camera, lighting, angle and framing - and are often tilted, "
     "which the retained configuration is now less equipped to handle since rotation "
     "augmentation was removed."),
    ("Label quality is not independently verified", "low",
     "18 images were excluded for sitting in near-duplicate clusters carrying conflicting "
     "labels. The true label-error rate across the 772 manifest paths is unknown - "
     "duplicate-based detection cannot catch a uniquely-photographed mislabelled image."),
    ("Nothing is statistically established", "low",
     "Both experiments produced effects smaller than the baseline's own seed-to-seed "
     "variance, and both confidence intervals contain zero. No improvement in this project "
     "has been demonstrated to be real."),
]

SEVERITY = {
    "high": ("Significant", "var(--bad)"),
    "medium": ("Moderate", "var(--warn)"),
    "low": ("Noted", "var(--muted)"),
}

CSS = """
<style>
:root{
  --bg:#0F141B; --surface:#161D26; --surface2:#1C2531; --border:#26313F;
  --text:#E8EDF3; --muted:#8896A6; --accent:#F5B62C;
  --ok:#3FB950; --warn:#E3873C; --bad:#E5534B;
}
.stApp{background:var(--bg);}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1180px;}
#MainMenu,footer,header{visibility:hidden;}

.sg-head{display:flex;align-items:flex-end;justify-content:space-between;
  flex-wrap:wrap;gap:1rem;border-bottom:1px solid var(--border);
  padding-bottom:1.1rem;margin-bottom:1.6rem;}
.sg-brand{font-size:1.6rem;font-weight:700;letter-spacing:.16em;color:var(--text);
  margin:0;line-height:1.1;}
.sg-brand span{color:var(--accent);}
.sg-tag{color:var(--muted);font-size:.9rem;margin:.35rem 0 0;}
.sg-status{display:flex;align-items:center;gap:.5rem;font-size:.82rem;color:var(--muted);
  background:var(--surface);border:1px solid var(--border);border-radius:999px;
  padding:.4rem .85rem;white-space:nowrap;}
.sg-dot{width:7px;height:7px;border-radius:50%;background:var(--ok);flex:none;}

.sg-strip{display:flex;flex-wrap:wrap;gap:.75rem;margin-bottom:1.6rem;}
.sg-stat{flex:1 1 190px;background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:.85rem 1rem;}
.sg-stat .k{color:var(--muted);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.09em;margin-bottom:.3rem;}
.sg-stat .v{color:var(--text);font-size:1.15rem;font-weight:650;
  font-variant-numeric:tabular-nums;}
.sg-stat .n{color:var(--muted);font-size:.72rem;margin-top:.15rem;}

.sg-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:1.35rem 1.5rem;margin-bottom:1rem;}
.sg-label{color:var(--muted);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:.9rem;}

.sg-empty{border:1px dashed var(--border);border-radius:12px;padding:3rem 1.5rem;
  text-align:center;color:var(--muted);background:var(--surface);}
.sg-empty .t{color:var(--text);font-weight:600;margin-bottom:.35rem;}
.sg-empty .s{font-size:.86rem;}

.sg-pred{font-size:2.1rem;font-weight:700;color:var(--text);line-height:1.15;
  margin:.1rem 0 .1rem;word-break:break-word;}
.sg-conf{display:flex;align-items:baseline;gap:.55rem;margin-top:.55rem;}
.sg-conf .n{font-size:1.75rem;font-weight:650;color:var(--accent);
  font-variant-numeric:tabular-nums;}
.sg-conf .l{font-size:.8rem;color:var(--muted);}
.sg-uncal{font-size:.75rem;color:var(--muted);margin-top:.3rem;
  border-left:2px solid var(--border);padding-left:.55rem;}

.sg-bar{margin-bottom:.7rem;}
.sg-bar .r{display:flex;justify-content:space-between;font-size:.85rem;
  margin-bottom:.3rem;gap:1rem;}
.sg-bar .nm{color:var(--text);}
.sg-bar .pc{color:var(--muted);font-variant-numeric:tabular-nums;flex:none;}
.sg-track{height:7px;background:var(--surface2);border-radius:4px;overflow:hidden;}
.sg-fill{height:100%;border-radius:4px;}

.sg-note{font-size:.84rem;line-height:1.55;color:var(--muted);
  background:var(--surface2);border-left:2px solid var(--border);
  border-radius:0 8px 8px 0;padding:.75rem .9rem;margin-top:.2rem;}
.sg-note.warn{border-left-color:var(--warn);}
.sg-note.bad{border-left-color:var(--bad);}
.sg-note b{color:var(--text);font-weight:600;}

.sg-badge{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.09em;
  padding:.24rem .6rem;border-radius:5px;text-transform:uppercase;white-space:nowrap;}
.b-bad{background:rgba(229,83,75,.14);color:var(--bad);border:1px solid rgba(229,83,75,.35);}
.b-warn{background:rgba(227,135,60,.14);color:var(--warn);border:1px solid rgba(227,135,60,.35);}
.b-neutral{background:var(--surface2);color:var(--muted);border:1px solid var(--border);}

.sg-exp-h{display:flex;justify-content:space-between;align-items:center;gap:1rem;
  flex-wrap:wrap;margin-bottom:.9rem;}
.sg-exp-h .nm{font-size:1.05rem;font-weight:650;color:var(--text);}
.sg-exp-h .sb{font-size:.78rem;color:var(--muted);margin-top:.15rem;}
.sg-nums{display:flex;flex-wrap:wrap;gap:1.5rem;margin-bottom:.9rem;}
.sg-nums .i .k{font-size:.68rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.08em;}
.sg-nums .i .v{font-size:1.05rem;color:var(--text);font-variant-numeric:tabular-nums;
  margin-top:.15rem;}

table.sg-t{width:100%;border-collapse:collapse;font-size:.86rem;}
table.sg-t th{color:var(--muted);font-weight:600;font-size:.7rem;text-transform:uppercase;
  letter-spacing:.08em;text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--border);}
table.sg-t td{padding:.55rem .6rem;border-bottom:1px solid rgba(38,49,63,.5);
  color:var(--text);font-variant-numeric:tabular-nums;}
table.sg-t td.num{text-align:right;}
table.sg-t tr:last-child td{border-bottom:none;}
.sg-scroll{overflow-x:auto;}

.sg-lim{border-left:3px solid var(--border);padding:.1rem 0 .1rem 1rem;margin-bottom:1.4rem;}
.sg-lim .h{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;margin-bottom:.35rem;}
.sg-lim .t{font-weight:650;color:var(--text);font-size:.97rem;}
.sg-lim .sev{font-size:.66rem;text-transform:uppercase;letter-spacing:.09em;font-weight:700;}
.sg-lim .d{color:var(--muted);font-size:.87rem;line-height:1.6;}

/* --- result card ------------------------------------------------------- */
.sg-result{border-left:3px solid var(--border);padding-left:1.15rem;}
.sg-verdict{display:flex;align-items:center;justify-content:space-between;
  gap:.75rem;flex-wrap:wrap;margin-bottom:.15rem;}
.sg-chip{display:inline-flex;align-items:center;gap:.4rem;font-size:.7rem;font-weight:700;
  letter-spacing:.07em;text-transform:uppercase;padding:.26rem .6rem;border-radius:5px;
  border:1px solid var(--border);background:var(--surface2);white-space:nowrap;}
.sg-chip .cd{width:6px;height:6px;border-radius:50%;flex:none;}

.sg-meter{height:9px;background:var(--surface2);border-radius:5px;overflow:hidden;
  margin-top:.65rem;position:relative;}
.sg-meter i{display:block;height:100%;border-radius:5px;}
.sg-scale{display:flex;justify-content:space-between;font-size:.66rem;color:var(--muted);
  margin-top:.28rem;font-variant-numeric:tabular-nums;}

.sg-rank{display:inline-block;width:1.15rem;color:var(--muted);font-size:.78rem;
  font-variant-numeric:tabular-nums;flex:none;}
.sg-bar .r .nm{display:flex;align-items:center;gap:.1rem;min-width:0;}
.sg-bar.lead .nm{font-weight:650;}

.sg-ready{border:1px solid var(--border);border-radius:12px;padding:2.4rem 1.5rem;
  text-align:center;background:var(--surface);}
.sg-ready .t{color:var(--text);font-weight:650;margin-bottom:.3rem;}
.sg-ready .s{color:var(--muted);font-size:.85rem;line-height:1.55;}
.sg-ready .px{display:inline-block;margin-top:.9rem;font-size:.72rem;color:var(--muted);
  background:var(--surface2);border:1px solid var(--border);border-radius:6px;
  padding:.3rem .6rem;font-variant-numeric:tabular-nums;}

div[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--border);}
.sg-meta{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;
  font-size:.74rem;color:var(--muted);margin-top:.45rem;font-variant-numeric:tabular-nums;}

.stTabs [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{height:44px;padding:0 1.15rem;color:var(--muted);
  font-size:.9rem;font-weight:500;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--accent) !important;
  border-bottom:2px solid var(--accent);}
.stButton>button{border-radius:8px;font-weight:600;border:1px solid var(--border);}
.stButton>button[kind="primary"]{background:var(--accent);color:#12171F;border:none;}
.stButton>button[kind="primary"]:hover{background:#FFCA4D;color:#12171F;}
/* Without this the primary rule above keeps the button at full accent brightness while
   it is disabled, so it reads as clickable when there is no image to analyze. */
.stButton>button[kind="primary"]:disabled,
.stButton>button[kind="primary"]:disabled:hover{background:var(--surface2);
  color:var(--muted);border:1px solid var(--border);cursor:not-allowed;}
div[data-testid="stFileUploaderDropzone"]{background:var(--surface);
  border:1px dashed var(--border);border-radius:10px;}
/* Streamlit stacks columns below ~640px on its own; these just stop the type and
   padding from staying desktop-sized once that happens. */
@media (max-width:640px){
  .block-container{padding-top:1.4rem;padding-left:1rem;padding-right:1rem;}
  .sg-brand{font-size:1.25rem;letter-spacing:.12em;}
  .sg-pred{font-size:1.65rem;}
  .sg-conf .n{font-size:1.45rem;}
  .sg-head{padding-bottom:.9rem;margin-bottom:1.2rem;}
  .sg-stat{flex:1 1 100%;padding:.7rem .85rem;}
  .sg-strip{gap:.5rem;margin-bottom:1.2rem;}
  .sg-card{padding:1.05rem 1.1rem;}
  .sg-result{padding-left:.9rem;}
  .sg-nums{gap:1rem;}
  .stTabs [data-baseweb="tab"]{padding:0 .8rem;font-size:.84rem;}
}
</style>
"""


def pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


def bar_colour(name: str, rank: int) -> str:
    if rank == 0:
        return "var(--bad)" if name in WEAK_CLASSES else "var(--accent)"
    return "var(--border)"


def prob_bar(name: str, prob: float, rank: int, numbered: bool = False) -> str:
    """One horizontal bar. `numbered` prefixes the rank, used for the alternatives list."""
    width = max(prob * 100, 1.2)
    rank_html = f'<span class="sg-rank">{rank + 1}</span>' if numbered else ""
    lead = " lead" if numbered and rank == 0 else ""
    return (
        f'<div class="sg-bar{lead}"><div class="r">'
        f'<span class="nm">{rank_html}{name}</span>'
        f'<span class="pc">{pct(prob)}</span></div>'
        f'<div class="sg-track"><div class="sg-fill" style="width:{width:.2f}%;'
        f'background:{bar_colour(name, rank)}"></div></div></div>'
    )


# Display convention only. The per-class F1 values above are measured; bucketing them into
# three bands is a presentation choice so the result card can say at a glance how much the
# model is worth trusting for the class it just predicted.
def reliability(name: str) -> tuple[str, str]:
    f1 = CLASS_F1[name]
    if f1 >= 0.85:
        return "High", "var(--ok)"
    if f1 >= 0.70:
        return "Moderate", "var(--warn)"
    return "Low", "var(--bad)"


def stat(k: str, v: str, n: str = "") -> str:
    note = f'<div class="n">{n}</div>' if n else ""
    return f'<div class="sg-stat"><div class="k">{k}</div><div class="v">{v}</div>{note}</div>'


@st.cache_resource(show_spinner=False)
def load_predictor() -> SolarGuardPredictor:
    return SolarGuardPredictor(CHECKPOINT, PREPROCESSING_CONFIG)


def header(meta: dict) -> None:
    st.markdown(
        '<div class="sg-head"><div>'
        '<p class="sg-brand">SOLAR<span>GUARD</span></p>'
        '<p class="sg-tag">AI-powered solar panel defect inspection</p></div>'
        f'<div class="sg-status"><span class="sg-dot"></span>'
        f'Model loaded &middot; {meta["architecture"]} &middot; {meta["device"].upper()}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def metric_strip(meta: dict) -> None:
    st.markdown(
        '<div class="sg-strip">'
        + stat("Classes", str(meta["num_classes"]), "panel conditions")
        + stat("Parameters", f"{meta['parameters']:,}", "trained from scratch")
        + stat("Validation macro-F1", f"{meta['val_metric_value']:.4f}", "not a test-set score")
        + stat("Validation samples", str(VAL_SAMPLES), "test split never evaluated")
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def render_result(result: dict) -> None:
    predicted = result["predicted_class"]
    confidence = result["confidence"]
    weak = predicted in WEAK_CLASSES

    tier, tier_colour = reliability(predicted)
    fill = bar_colour(predicted, 0)

    st.markdown('<div class="sg-label">Inspection result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sg-result" style="border-left-color:{fill}">'
        f'<div class="sg-verdict"><div class="sg-pred">{predicted}</div>'
        f'<span class="sg-chip" style="color:{tier_colour}">'
        f'<span class="cd" style="background:{tier_colour}"></span>'
        f"{tier} validation reliability</span></div>"
        f'<div class="sg-conf"><span class="n">{pct(confidence)}</span>'
        f'<span class="l">model confidence</span></div>'
        f'<div class="sg-meter"><i style="width:{max(confidence * 100, 1.2):.2f}%;'
        f'background:{fill}"></i></div>'
        f'<div class="sg-scale"><span>0%</span><span>50%</span><span>100%</span></div>'
        f'<div class="sg-uncal" style="margin-top:.6rem">Confidence is not calibrated. '
        f"It is a raw softmax output, not a probability that this reading is correct.</div>"
        f'<div class="sg-meta"><span>Validation F1 {CLASS_F1[predicted]:.3f}</span>'
        f"<span>Recall {pct(CLASS_RECALL[predicted])}</span>"
        f"<span>{CLASS_VAL_N[predicted]} validation images</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-label">Alternatives considered</div>', unsafe_allow_html=True
    )
    bars = "".join(
        prob_bar(e["class"], e["probability"], i, numbered=True)
        for i, e in enumerate(result["top_k"])
    )
    st.markdown(bars, unsafe_allow_html=True)

    tone = "bad" if weak else ""
    note = CLASS_NOTE.get(predicted, "")
    st.markdown(
        f'<div class="sg-note {tone}"><b>Interpretation.</b> {note}</div>',
        unsafe_allow_html=True,
    )

    if confidence < 0.50:
        st.markdown(
            '<div class="sg-note warn"><b>Low separation.</b> The top score is below 50%, '
            'so the model is not clearly separating this image from the other classes.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("All six classes"):
        ordered = sorted(result["all_probabilities"].items(), key=lambda kv: -kv[1])
        rows = "".join(
            f'<tr><td>{n}</td><td class="num">{pct(p, 2)}</td>'
            f'<td class="num">{CLASS_F1[n]:.3f}</td></tr>'
            for n, p in ordered
        )
        st.markdown(
            '<div class="sg-scroll"><table class="sg-t"><thead><tr><th>Class</th>'
            '<th style="text-align:right">Probability</th>'
            '<th style="text-align:right">Val F1</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )


def page_inspect(predictor: SolarGuardPredictor) -> None:
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.result_for = None

    left, right = st.columns([1, 1.05], gap="large")

    with left:
        st.markdown('<div class="sg-label">Panel image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a solar panel photograph",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key=f"up_{st.session_state.upload_key}",
            label_visibility="collapsed",
            help=f"JPG, PNG, BMP or WEBP. Up to {MAX_UPLOAD_MB} MB. "
                 "Resized to 224x224 before inference.",
        )

        image = None
        signature = None
        if uploaded is not None:
            signature = (uploaded.name, uploaded.size)
            if uploaded.size > MAX_UPLOAD_MB * 1_000_000:
                st.error(
                    f"That file is {uploaded.size / 1e6:.1f} MB. The limit is "
                    f"{MAX_UPLOAD_MB} MB - try a smaller image."
                )
            else:
                try:
                    image = Image.open(uploaded)
                    image.load()
                except (UnidentifiedImageError, OSError):
                    st.error(
                        "That file could not be read as an image. It may be corrupt or "
                        "in a format this app does not support."
                    )
                    image = None

        # A new file invalidates any result still on screen from the previous one.
        if signature != st.session_state.result_for:
            st.session_state.result = None

        if image is not None:
            st.image(image, use_container_width=True)
            st.markdown(
                f'<div class="sg-meta"><span>{uploaded.name}</span>'
                f"<span>{image.width} x {image.height} px &middot; {image.mode} "
                f"&middot; {uploaded.size / 1000:,.0f} KB</span></div>",
                unsafe_allow_html=True,
            )

        analysed = (
            st.session_state.result is not None
            and st.session_state.result_for == signature
        )

        st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            analyze = st.button(
                "Re-analyze panel" if analysed else "Analyze panel",
                type="primary",
                disabled=image is None,
                use_container_width=True,
            )
        with c2:
            if st.button("Clear", disabled=uploaded is None, use_container_width=True):
                st.session_state.upload_key += 1
                st.session_state.result = None
                st.session_state.result_for = None
                st.rerun()

        if image is None:
            st.caption("Analysis is disabled until an image is loaded.")
        elif analysed:
            st.caption("Inference is deterministic - re-analyzing gives the same result.")
        else:
            st.caption("Image resized to 224x224 with ImageNet normalization before inference.")

    with right:
        if analyze and image is not None:
            with st.spinner("Analyzing panel..."):
                try:
                    st.session_state.result = predictor.predict(image, top_k=3)
                    st.session_state.result_for = signature
                except Exception:
                    st.session_state.result = None
                    st.session_state.result_for = None
                    st.error(
                        "Inference failed on this image. The model could not process it. "
                        "Try a different file - if it keeps happening, the checkpoint or "
                        "preprocessing config may be missing."
                    )

        if st.session_state.result is not None:
            render_result(st.session_state.result)
        elif image is not None:
            st.markdown(
                '<div class="sg-ready"><div class="t">Image loaded</div>'
                '<div class="s">Run <b>Analyze panel</b> to classify it into one of the '
                "six conditions.</div>"
                f'<div class="px">{image.width} x {image.height} &rarr; 224 x 224</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sg-empty"><div class="t">No inspection yet</div>'
                '<div class="s">Upload a panel photograph on the left, then run '
                "<b>Analyze panel</b>. Results appear here.</div></div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def page_model(meta: dict) -> None:
    st.markdown('<div class="sg-label">Defect classes</div>', unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td>{n}</td><td>{CLASS_WHAT[n]}</td>'
        f'<td class="num">{CLASS_TRAIN_N[n]}</td>'
        f'<td class="num">{CLASS_VAL_N[n]}</td>'
        f'<td class="num">{CLASS_WEIGHT[n]:.4f}</td>'
        f'<td class="num">{CLASS_F1[n]:.3f}</td></tr>'
        for n in sorted(CLASS_F1, key=lambda k: -CLASS_F1[k])
    )
    st.markdown(
        '<div class="sg-card sg-scroll"><table class="sg-t"><thead><tr>'
        "<th>Class</th><th>What it means</th>"
        '<th style="text-align:right">Train n</th><th style="text-align:right">Val n</th>'
        '<th style="text-align:right">Loss weight</th><th style="text-align:right">Val F1</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="sg-label">Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sg-card">'
            '<div class="sg-scroll"><table class="sg-t"><thead><tr><th>Stage</th>'
            "<th>Layers</th><th>Output</th></tr></thead><tbody>"
            "<tr><td>Input</td><td>&mdash;</td><td>3 x 224 x 224</td></tr>"
            "<tr><td>Blocks 1-4</td><td>Conv3x3 &rarr; BatchNorm &rarr; ReLU &rarr; MaxPool2x2</td>"
            "<td>16 &rarr; 32 &rarr; 64 &rarr; 128</td></tr>"
            "<tr><td>Head</td><td>GlobalAvgPool &rarr; Dropout(0.5) &rarr; Linear(128, 6)</td>"
            "<td>6 logits</td></tr></tbody></table></div>"
            '<div class="sg-note" style="margin-top:1rem"><b>Why it is this small.</b> '
            f"{meta['parameters']:,} parameters, trained from scratch on 540 images. The point "
            "of a baseline is to be a floor that transfer learning has to clear, not to win on "
            "its own. Global average pooling is used instead of Flatten+FC because a flatten "
            "head on the 128x14x14 feature map would need a 25,088-wide weight matrix and would "
            "encourage the model to memorise <i>where</i> a defect appeared - but dust, cracks "
            "and droppings can appear anywhere in frame.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="sg-label">Locked baseline &middot; validation</div>',
                    unsafe_allow_html=True)
        metrics = [
            ("Accuracy", "0.7692"), ("Macro precision", "0.7639"),
            ("Macro recall", "0.7504"), ("Macro F1", "0.7534"),
            ("Weighted F1", "0.7744"), ("Validation loss", "0.8091"),
        ]
        rows = "".join(
            f'<tr><td>{k}</td><td class="num">{v}</td></tr>' for k, v in metrics
        )
        st.markdown(
            '<div class="sg-card"><div class="sg-scroll"><table class="sg-t">'
            f"<tbody>{rows}</tbody></table></div>"
            '<div class="sg-note" style="margin-top:1rem"><b>Macro-F1 is the primary metric.</b> '
            "At a 2.73:1 imbalance, accuracy and weighted-F1 both weight by class size, so a "
            "model can score well on Clean and Dusty while failing the rarest classes. Macro-F1 "
            "averages per-class F1 without that weighting. Accuracy is reported but was never "
            "used for model selection - <code>TrainingConfig</code> rejects it outright.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sg-label">Per-class F1 &middot; locked baseline</div>',
                unsafe_allow_html=True)
    bars = "".join(
        prob_bar(n, CLASS_F1[n], 0 if n in WEAK_CLASSES else 1)
        for n in sorted(CLASS_F1, key=lambda k: -CLASS_F1[k])
    )
    st.markdown(f'<div class="sg-card">{bars}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-note bad"><b>The two weak classes fail for different reasons.</b> '
        "Physical-Damage has only 10 validation images, so a single prediction moves its F1 by "
        "roughly 0.10 - its score is as much a measurement artifact as a model property. "
        "Bird-drop is the more interesting failure: with 87 training and 19 validation images "
        "it is not among the rarest classes, and its class weight of 1.0345 is essentially "
        "neutral, so frequency-based rebalancing gives it no help at all. Its problem is visual "
        "confusability, not scarcity.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sg-note" style="margin-top:1rem">Serving '
        f'<code>{meta["source_checkpoint"]}</code> &middot; epoch {meta["trained_epoch"]} '
        f'&middot; seed {meta["seed"]} &middot; {meta["val_metric_name"]} '
        f'{meta["val_metric_value"]:.4f}. Preprocessing at serving time is the same '
        "<code>build_eval_transform()</code> used to compute every validation metric above, "
        "asserted equal by test.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def page_experiments() -> None:
    st.markdown(
        '<div class="sg-note bad"><b>Nothing below is statistically established.</b> '
        "Both experiments produced effects smaller than the baseline's own seed-to-seed "
        "standard deviation of 0.0381, and both confidence intervals contain zero. The "
        "verdicts are decisions made under that uncertainty, not demonstrated wins.</div>"
        '<div style="height:1.4rem"></div>',
        unsafe_allow_html=True,
    )

    for e in EXPERIMENTS:
        badge = {"bad": "b-bad", "warn": "b-warn", "neutral": "b-neutral"}[e["tone"]]
        nums = [("Macro-F1", f'{e["f1"]:.4f}'), ("Std", f'{e["std"]:.4f}')]
        if e["delta"] is not None:
            nums.append(("Delta", f'{e["delta"]:+.4f}'))
            nums.append(("Cohen's d", f'{e["d"]:+.3f}'))
            nums.append(("95% CI", f'[{e["ci"][0]:+.4f}, {e["ci"][1]:+.4f}]'))
            nums.append(("Seeds won", e["seeds_won"]))
        nums.append(("Mean best epoch", f'{e["epoch"]:.1f}'))
        num_html = "".join(
            f'<div class="i"><div class="k">{k}</div><div class="v">{v}</div></div>'
            for k, v in nums
        )
        st.markdown(
            f'<div class="sg-card">'
            f'<div class="sg-exp-h"><div><div class="nm">{e["name"]}</div>'
            f'<div class="sb">{e["sub"]}</div></div>'
            f'<span class="sg-badge {badge}">{e["verdict"]}</span></div>'
            f'<div class="sg-nums">{num_html}</div>'
            f'<div class="sg-note {e["tone"] if e["tone"] != "neutral" else ""}">{e["note"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sg-label">Per-seed macro-F1</div>', unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td>{s}</td><td class="num">{b:.4f}</td><td class="num">{f:.4f}</td>'
        f'<td class="num">{n:.4f}</td></tr>'
        for s, b, f, n in PER_SEED
    )
    st.markdown(
        '<div class="sg-card sg-scroll"><table class="sg-t"><thead><tr><th>Seed</th>'
        '<th style="text-align:right">Baseline</th>'
        '<th style="text-align:right">Focal</th>'
        '<th style="text-align:right">No rotation</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sg-note"><b>Why three seeds, and why that matters.</b> A single run '
        "conflates the effect being tested with the effect of initialization. Baseline runs "
        "here ranged from 0.7189 to 0.7855 purely from seed choice - a 0.0666 spread. An "
        "experiment reporting one run at 0.7855 against one baseline run at 0.7189 could have "
        "claimed a +0.067 improvement from a change that did nothing at all. With 117 "
        "validation images and three seeds per arm, only effects of roughly 0.06 macro-F1 or "
        "larger are separable from noise. Both experiments landed well below that.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

def page_limitations() -> None:
    st.markdown(
        '<div class="sg-note bad" style="font-size:.92rem">'
        "<b>This is a research and portfolio system, not a certified industrial inspection "
        "system.</b> It has no certification, meets no standard, and has not been externally "
        "validated. The limitations below are not disclaimers bolted on at the end - they are "
        "the measured boundaries of what this model was shown to do.</div>"
        '<div style="height:1.6rem"></div>',
        unsafe_allow_html=True,
    )

    counts = {"high": 0, "medium": 0, "low": 0}
    for _, sev, _ in LIMITATIONS:
        counts[sev] += 1
    st.markdown(
        '<div class="sg-strip">'
        + stat("Significant", str(counts["high"]), "blocks any deployment")
        + stat("Moderate", str(counts["medium"]), "constrains interpretation")
        + stat("Noted", str(counts["low"]), "known and documented")
        + stat("Test-set evaluations", "0", "held out by design")
        + "</div>",
        unsafe_allow_html=True,
    )

    order = {"high": 0, "medium": 1, "low": 2}
    for title, sev, desc in sorted(LIMITATIONS, key=lambda x: order[x[1]]):
        label, colour = SEVERITY[sev]
        st.markdown(
            f'<div class="sg-lim" style="border-left-color:{colour}">'
            f'<div class="h"><span class="t">{title}</span>'
            f'<span class="sev" style="color:{colour}">{label}</span></div>'
            f'<div class="d">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sg-note"><b>What would have to happen before any real use.</b> '
        "A single test-set evaluation, once all modeling decisions are frozen. Evaluation on "
        "photographs actually representative of the deployment setting rather than a held-out "
        "slice of the same public dataset. Confidence calibration. An abstention or "
        "escalate-to-human path for the Physical-Damage failure mode. Out-of-distribution "
        "rejection. None of these exist today.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SolarGuard - Panel Defect Inspection",
        page_icon="☀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if not CHECKPOINT.exists():
        st.markdown(
            '<p class="sg-brand">SOLAR<span>GUARD</span></p>', unsafe_allow_html=True
        )
        st.error(
            f"Deployment checkpoint not found at {CHECKPOINT.relative_to(REPO_ROOT)}. "
            "Generate it with:  PYTHONPATH=src python scripts/export_deployment_checkpoint.py"
        )
        st.stop()

    try:
        predictor = load_predictor()
    except Exception:
        st.markdown(
            '<p class="sg-brand">SOLAR<span>GUARD</span></p>', unsafe_allow_html=True
        )
        st.error(
            "The model could not be loaded. The checkpoint or the preprocessing config "
            "appears to be missing or unreadable."
        )
        st.stop()

    meta = predictor.metadata
    header(meta)
    metric_strip(meta)

    inspect_tab, model_tab, exp_tab, lim_tab = st.tabs(
        ["Inspect", "Model", "Experiments", "Limitations"]
    )
    with inspect_tab:
        page_inspect(predictor)
    with model_tab:
        page_model(meta)
    with exp_tab:
        page_experiments()
    with lim_tab:
        page_limitations()

    st.markdown(
        '<div style="margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid var(--border);'
        'color:var(--muted);font-size:.78rem;line-height:1.6">'
        "SolarGuard &middot; research and portfolio project. All figures are validation "
        "figures on 117 images; the 115-image test split has never been evaluated. "
        "Dataset: PV Panel Defect Dataset (CC BY-NC-SA 4.0, non-commercial), not "
        "redistributed here. Full detail in README.md, docs/MODEL_CARD.md and "
        "docs/EXPERIMENT_REPORT.md.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
