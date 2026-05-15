"""
streamlit_app.py  —  MindBridge AI  |  Face + Voice Dual Recognition
=====================================================================

FOLDER LAYOUT:
    FACE/
    ├── face_recogniton/
    │   ├── fsl_model.py
    │   ├── incremental_fsl_model.py
    │   ├── new_person_handler.py
    │   └── voice_output.py
    ├── voice_recognition/
    │   ├── realtime_listener.py   ← uses listen_once() + RealtimeListener
    │   └── voice_model.py
    ├── telegram_notifier.py
    ├── notify_config.json
    └── streamlit_app.py

DUAL RECOGNITION FLOW:
──────────────────────
  FACE (FSL)  +  VOICE (Resemblyzer)  run simultaneously.

  KNOWN face + KNOWN voice  → ✅ CONFIRMED  (highest trust, ~0.9+)
  KNOWN face + unknown voice → ⚠ FACE ONLY  (moderate trust)
  KNOWN voice + unknown face → ⚠ VOICE ONLY (moderate trust)
  Both unknown               → ❓ UNKNOWN    → ask name → Telegram → FSL train

  Combined trust = 0.6 * face_conf + 0.4 * voice_conf
  (face weighted higher because it is more reliable for dementia care)

NAME CAPTURE:
  listen_once() from realtime_listener.py — records mic for N seconds,
  runs Whisper, returns transcript string.

Run:  streamlit run streamlit_app.py
"""

import os, sys, time, json, pickle, queue, threading, datetime, shutil
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

try:
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ── sys.path ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
FACE_PKG  = os.path.join(ROOT, "face_recogniton")   # ← your folder (typo kept)
VOICE_PKG = os.path.join(ROOT, "voice_recognition")
for _d in [ROOT, FACE_PKG, VOICE_PKG]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBridge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
    --bg:#04040d; --bg1:#080818; --bg2:rgba(255,255,255,.038);
    --bg3:rgba(255,255,255,.06);
    --bo:rgba(100,120,255,.18); --bo-h:rgba(100,120,255,.42);
    --pu:#7c5cfc; --pu2:#a78bfa; --bl:#3b82f6;
    --cy:#06b6d4; --gr:#10b981; --am:#f59e0b; --re:#ef4444;
    --tg:#229ED9; --wh:#e2e8f0; --sl:#94a3b8;
    --grad:linear-gradient(135deg,#7c5cfc,#3b82f6,#06b6d4);
    --r:16px; --rsm:10px;
    --gp:0 0 24px rgba(124,92,252,.32);
    --gc2:0 0 24px rgba(6,182,212,.32);
    --gg:0 0 24px rgba(16,185,129,.32);
    --ga:0 0 24px rgba(245,158,11,.32);
    --gr2:0 0 24px rgba(239,68,68,.32);
    --gtg:0 0 24px rgba(34,158,217,.32);
}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:var(--bg)!important;color:var(--wh)!important;}
.block-container{padding:0!important;max-width:100%!important;background:var(--bg)!important;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--bo)!important;}
[data-testid="stSidebar"] *{color:var(--wh)!important;}
[data-testid="stSidebar"] hr{border-color:var(--bo)!important;}

/* Top bar */
.topbar{background:rgba(4,4,13,.96);border-bottom:1px solid var(--bo);backdrop-filter:blur(20px);
        padding:.75rem 2rem;display:flex;align-items:center;justify-content:space-between;
        position:sticky;top:0;z-index:200;}
.brand{font-family:'Orbitron',sans-serif;font-size:1.35rem;font-weight:700;
       background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
       background-clip:text;letter-spacing:.08em;}
.srow{display:flex;align-items:center;gap:16px;}
.sdot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.slbl{font-size:11px;color:var(--sl);letter-spacing:.06em;font-family:'JetBrains Mono';}

.wrap{padding:1.5rem 2rem 2.5rem;}

/* Page titles */
.ptitle{font-family:'Orbitron',sans-serif;font-size:1.7rem;font-weight:700;
        background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:.2rem;letter-spacing:.04em;}
.psub{font-size:.88rem;color:var(--sl);font-weight:300;margin-bottom:1.2rem;}

/* Glass card */
.gc{background:var(--bg2);border:1px solid var(--bo);border-radius:var(--r);
    backdrop-filter:blur(12px);padding:1.4rem 1.6rem;margin-bottom:.9rem;}
.gc-p{border-color:rgba(124,92,252,.4);box-shadow:var(--gp);}
.gc-b{border-color:rgba(59,130,246,.4);box-shadow:0 0 24px rgba(59,130,246,.3);}
.gc-c{border-color:rgba(6,182,212,.4);box-shadow:var(--gc2);}
.gc-g{border-color:rgba(16,185,129,.4);box-shadow:var(--gg);}
.gc-a{border-color:rgba(245,158,11,.4);box-shadow:var(--ga);}
.gc-r{border-color:rgba(239,68,68,.4);box-shadow:var(--gr2);}
.gc-tg{border-color:rgba(34,158,217,.45);box-shadow:var(--gtg);}

/* Metric card */
.mcard{background:var(--bg2);border:1px solid var(--bo);border-radius:var(--r);
       padding:1.1rem 1.3rem;text-align:center;position:relative;overflow:hidden;}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--grad);}
.mv{font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:700;}
.ml{font-size:10.5px;color:var(--sl);letter-spacing:.08em;text-transform:uppercase;
    margin-top:3px;font-family:'JetBrains Mono';}
.mi{font-size:1.6rem;margin-bottom:.3rem;display:block;}

/* Hero */
.hero{background:var(--bg2);border:1px solid var(--bo);border-radius:20px;
      padding:2rem 1.6rem;text-align:center;backdrop-filter:blur(14px);}
.hn-k{font-family:'Orbitron',sans-serif;font-size:2.6rem;font-weight:700;
      background:linear-gradient(135deg,#06b6d4,#a78bfa);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;margin:.3rem 0;line-height:1.1;}
.hn-u{font-family:'Orbitron',sans-serif;font-size:2rem;color:#ef4444;margin:.3rem 0;}
.hn-l{font-family:'Orbitron',sans-serif;font-size:1.8rem;color:#f59e0b;margin:.3rem 0;}
.hn-tg{font-family:'Orbitron',sans-serif;font-size:2rem;color:#229ED9;margin:.3rem 0;}
.hrel{font-size:.95rem;color:var(--cy);font-weight:500;margin-bottom:.4rem;}
.hhint{background:rgba(255,255,255,.04);border-left:3px solid var(--am);
       border-radius:var(--rsm);padding:.8rem 1.1rem;margin:.8rem auto;
       max-width:480px;text-align:left;font-size:.9rem;color:var(--sl);font-style:italic;}
.cg{color:#10b981;font-family:'JetBrains Mono';font-size:.85rem;}
.ca{color:#f59e0b;font-family:'JetBrains Mono';font-size:.85rem;}
.cr{color:#ef4444;font-family:'JetBrains Mono';font-size:.85rem;}

/* Pills */
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 12px;border-radius:999px;
      font-size:10.5px;font-weight:600;letter-spacing:.04em;}
.p-k{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3);}
.p-u{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
.p-a{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3);}
.p-p{background:rgba(124,92,252,.15);color:#a78bfa;border:1px solid rgba(124,92,252,.3);}
.p-tg{background:rgba(34,158,217,.15);color:#229ED9;border:1px solid rgba(34,158,217,.3);}
.p-c{background:rgba(6,182,212,.15);color:#06b6d4;border:1px solid rgba(6,182,212,.3);}
.p-v{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3);}

/* Dual recognition scores */
.dual-wrap{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:.8rem 0;}
.dual-card{background:rgba(255,255,255,.04);border:1px solid var(--bo);
           border-radius:var(--rsm);padding:.75rem .9rem;text-align:center;}
.dual-icon{font-size:1.4rem;display:block;margin-bottom:4px;}
.dual-label{font-size:9.5px;color:var(--sl);letter-spacing:.1em;text-transform:uppercase;
            font-family:'JetBrains Mono';}
.dual-val{font-family:'JetBrains Mono';font-size:1.2rem;font-weight:600;margin-top:3px;}
.dual-name{font-size:11px;color:var(--wh);margin-top:2px;font-weight:500;}

/* Combined trust */
.trust-combined{background:linear-gradient(135deg,rgba(124,92,252,.1),rgba(6,182,212,.1));
                border:1px solid rgba(124,92,252,.3);border-radius:var(--rsm);
                padding:.9rem 1.1rem;margin-top:.6rem;}
.trust-title{font-size:9px;color:var(--sl);letter-spacing:.12em;text-transform:uppercase;
             font-family:'JetBrains Mono';margin-bottom:.5rem;}
.trust-bar-bg{background:rgba(255,255,255,.07);border-radius:999px;height:10px;
              overflow:hidden;border:1px solid var(--bo);}
.trust-bar{height:10px;border-radius:999px;transition:width .6s ease;}

/* Step bar */
.sbar{display:flex;border-radius:var(--rsm);overflow:hidden;border:1px solid var(--bo);margin:.8rem 0;}
.stb{flex:1;padding:.5rem .3rem;text-align:center;font-size:10px;font-weight:600;
     letter-spacing:.03em;color:var(--sl);background:var(--bg2);
     border-right:1px solid var(--bo);font-family:'JetBrains Mono';}
.stb:last-child{border-right:none;}
.stb.act{background:rgba(124,92,252,.2);color:var(--pu2);}
.stb.dn{background:rgba(16,185,129,.15);color:#10b981;}

/* XAI */
.xai-wrap{background:var(--bg2);border:1px solid var(--bo);border-radius:var(--rsm);padding:1rem 1.2rem;margin-top:.6rem;}
.xai-lbl{font-size:9px;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--sl);margin-bottom:.65rem;font-family:'JetBrains Mono';}
.xr{display:flex;align-items:center;gap:9px;margin-bottom:6px;}
.xn{font-size:12px;font-weight:500;min-width:100px;color:var(--wh);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.xbw{flex:1;background:rgba(255,255,255,.07);border-radius:999px;height:6px;overflow:hidden;}
.xb{height:6px;border-radius:999px;}
.xv{font-family:'JetBrains Mono';font-size:10px;color:var(--sl);min-width:36px;text-align:right;}
.xth{font-size:9.5px;color:rgba(255,255,255,.25);margin-top:5px;}

/* Section label */
.slb{font-size:9.5px;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--sl);
     display:flex;align-items:center;gap:8px;margin:1rem 0 .5rem;font-family:'JetBrains Mono';}
.slb::after{content:'';flex:1;height:1px;background:var(--bo);}

/* Notification banners */
.nb-g{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);border-radius:var(--rsm);padding:.6rem 1rem;margin-top:.5rem;font-size:12.5px;color:#10b981;}
.nb-r{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);border-radius:var(--rsm);padding:.6rem 1rem;margin-top:.5rem;font-size:12.5px;color:#ef4444;}
.nb-a{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);border-radius:var(--rsm);padding:.6rem 1rem;margin-top:.5rem;font-size:12.5px;color:#f59e0b;}
.nb-tg{background:rgba(34,158,217,.12);border:1px solid rgba(34,158,217,.3);border-radius:var(--rsm);padding:.6rem 1rem;margin-top:.5rem;font-size:12.5px;color:#229ED9;}
.nb-p{background:rgba(124,92,252,.12);border:1px solid rgba(124,92,252,.3);border-radius:var(--rsm);padding:.6rem 1rem;margin-top:.5rem;font-size:12.5px;color:#a78bfa;}

/* Chat */
.chat-ai{background:rgba(124,92,252,.12);border:1px solid rgba(124,92,252,.22);
         border-radius:14px 14px 14px 2px;padding:.65rem 1rem;margin:.35rem 0;
         font-size:.87rem;color:var(--wh);max-width:90%;}
.chat-user{background:rgba(6,182,212,.12);border:1px solid rgba(6,182,212,.22);
           border-radius:14px 14px 2px 14px;padding:.65rem 1rem;margin:.35rem 0 .35rem auto;
           font-size:.87rem;color:var(--wh);max-width:90%;text-align:right;}

/* Log row */
.log-row{display:flex;align-items:center;gap:12px;padding:.55rem .8rem;border-radius:var(--rsm);
         background:var(--bg2);margin-bottom:5px;border:1px solid var(--bo);font-size:12.5px;}
.log-ts{font-family:'JetBrains Mono';font-size:10px;color:var(--sl);min-width:70px;}

/* Mic pulse / spin */
.mic{display:inline-block;font-size:2.5rem;animation:mp 1s ease-in-out infinite;}
@keyframes mp{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.18);opacity:.6}}
.spin{display:inline-block;animation:sp 1.4s linear infinite;}
@keyframes sp{to{transform:rotate(360deg)}}

/* Metric tile */
.mt{background:var(--bg2);border:1px solid var(--bo);border-radius:var(--rsm);padding:.85rem 1rem;text-align:center;}
.mt .mv2{font-family:'Orbitron',sans-serif;font-size:1.9rem;color:var(--wh);}
.mt .ml2{font-size:9.5px;color:var(--sl);letter-spacing:.06em;text-transform:uppercase;margin-top:1px;}

/* Streamlit overrides */
.stButton>button{border-radius:var(--rsm)!important;font-family:'Inter'!important;font-weight:500!important;
    background:var(--bg2)!important;border:1px solid var(--bo)!important;color:var(--wh)!important;transition:all .2s!important;}
.stButton>button:hover{border-color:var(--pu2)!important;box-shadow:var(--gp)!important;color:var(--pu2)!important;}
.stButton>button[kind="primary"]{background:var(--grad)!important;border-color:transparent!important;
    color:#fff!important;font-weight:600!important;}
div[data-testid="stTextInput"] input,div[data-testid="stTextArea"] textarea{
    background:var(--bg2)!important;border:1px solid var(--bo)!important;
    border-radius:var(--rsm)!important;color:var(--wh)!important;}
div[data-testid="stTextInput"] input:focus{border-color:var(--pu2)!important;box-shadow:var(--gp)!important;}
.stProgress>div>div{background:var(--grad)!important;}
[data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid var(--bo)!important;border-radius:var(--rsm)!important;}
hr{border-color:var(--bo)!important;margin:1rem 0!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--bo);border-radius:4px;}
</style>
""", unsafe_allow_html=True)


# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR       = os.path.join(FACE_PKG,  "models")
KNOWN_DIR        = os.path.join(MODELS_DIR, "known")
PENDING_DIR      = os.path.join(MODELS_DIR, "pending")
PROTOTYPE_PATH   = os.path.join(MODELS_DIR, "prototypes.pkl")
VOICE_MODELS_DIR = os.path.join(VOICE_PKG,  "models")          # voice_recognition/models/
VOICE_PROTO_PATH = os.path.join(VOICE_MODELS_DIR, "voice_prototypes.pkl")
NOTIFY_CFG       = os.path.join(ROOT, "notify_config.json")
DATA_DIR         = os.path.join(ROOT, "data")
LOG_PATH         = os.path.join(DATA_DIR, "recognition_log.json")

for _d in [KNOWN_DIR, PENDING_DIR, DATA_DIR, VOICE_MODELS_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_face_model():
    try:
        from face.fsl_model import RelativeRecognitionModel
        m = RelativeRecognitionModel()
        m.load_prototypes(PROTOTYPE_PATH)
        return m, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_incremental_model():
    try:
        from face.incremental_fsl_model import IncrementalFSLModel
        return IncrementalFSLModel(), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_voice_model():
    try:
        from voice_recognition_system.voice_model import SpeakerRecognitionModel
        vm = SpeakerRecognitionModel()
        if os.path.exists(VOICE_PROTO_PATH):
            vm.load_prototypes(VOICE_PROTO_PATH)
        return vm, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_recogniser(_face_model):
    try:
        from face.new_person_handler import OpenSetRecogniser
        return OpenSetRecogniser(_face_model) if _face_model else None
    except Exception:
        return None


@st.cache_resource
def get_speaker():
    try:
        from face.voice_output import VoiceSpeaker
        return VoiceSpeaker()
    except Exception:
        return None


@st.cache_resource
def load_tg():
    try:
        from telegram_notifier import TelegramNotifier
        return TelegramNotifier(NOTIFY_CFG), None
    except Exception as e:
        return None, str(e)


def load_prototypes_raw() -> dict:
    if os.path.exists(PROTOTYPE_PATH):
        with open(PROTOTYPE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def load_voice_prototypes_raw() -> dict:
    if os.path.exists(VOICE_PROTO_PATH):
        with open(VOICE_PROTO_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def load_cfg() -> dict:
    if os.path.exists(NOTIFY_CFG):
        with open(NOTIFY_CFG) as f:
            return json.load(f)
    return {}


def load_log() -> list:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return []


def append_log(entry: dict):
    logs = load_log()
    logs.insert(0, entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs[:500], f, indent=2)


def get_face_model_live():
    """
    Returns a face model whose prototypes are ALWAYS fresh from disk.
    Called every recognition tick — never uses stale cache.
    The model object is cached but prototypes are reloaded each call.
    """
    m, err = load_face_model()
    if m is None:
        return None
    # Always reload prototypes from disk so deletions take effect immediately
    try:
        m.load_prototypes(PROTOTYPE_PATH)
    except Exception:
        pass
    return m


def get_recogniser_live():
    """Always builds a fresh OpenSetRecogniser from the live model."""
    m = get_face_model_live()
    if m is None:
        return None
    try:
        from face_recogniton.new_person_handler import OpenSetRecogniser
        return OpenSetRecogniser(m)
    except Exception:
        return None


# ── Singletons ────────────────────────────────────────────────────────────────
# face_model and voice_model are cached (heavy — encoder, detector)
# recogniser is rebuilt live each recognition tick (lightweight wrapper)
face_model,  face_err   = load_face_model()
incr_model,  incr_err   = load_incremental_model()
voice_model, voice_err  = load_voice_model()
speaker                 = get_speaker()
tg_notifier, tg_err    = load_tg()

# NOTE: recogniser is NOT stored here — get_recogniser_live() is called
# inside the recognition loop so prototypes are always fresh from disk.

protos_now = load_prototypes_raw()
cfg        = load_cfg()
tg_ok      = tg_notifier is not None and getattr(tg_notifier, "enabled", False)


# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def save_frame_jpg(frame, folder, label="img"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    path = os.path.join(folder, f"{label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def speak_async(text, spk=None):
    spk = spk or speaker
    if spk:
        threading.Thread(target=spk.speak, args=(text,), daemon=True).start()


def gender_word(rel):
    for w in ["son","brother","father","husband","uncle","grandfather","grandson","nephew"]:
        if w in rel.lower(): return "He"
    return "She"


def conf_cls(c):
    return "cg" if c >= 0.75 else ("ca" if c >= 0.5 else "cr")


def combined_trust(face_conf, voice_conf):
    """
    Weighted combination: face=60%, voice=40%.
    If voice not available, use face only.
    """
    if voice_conf <= 0:
        return face_conf
    return 0.60 * face_conf + 0.40 * voice_conf


def trust_color(c):
    if c >= 0.75: return "linear-gradient(90deg,#10b981,#06b6d4)"
    if c >= 0.5:  return "linear-gradient(90deg,#f59e0b,#ef8c0b)"
    return "linear-gradient(90deg,#ef4444,#dc2626)"


def render_trust(face_c, voice_c, label="Combined Trust"):
    ct  = combined_trust(face_c, voice_c)
    col = trust_color(ct)
    pct = int(ct * 100)
    v_row = ""
    if voice_c > 0:
        v_row = f"""
        <div style="display:flex;justify-content:space-between;margin-top:6px;">
            <span style="font-size:9.5px;color:var(--sl);font-family:'JetBrains Mono';">
                📷 Face: {face_c*100:.0f}% &nbsp;&nbsp; 🎙 Voice: {voice_c*100:.0f}%
            </span>
        </div>"""
    return f"""
    <div class="trust-combined">
        <div class="trust-title">{label}</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:10px;color:var(--sl);font-family:'JetBrains Mono';">
                Weighted confidence (face 60% · voice 40%)
            </span>
            <span style="font-family:'JetBrains Mono';font-size:11px;">{pct}%</span>
        </div>
        <div class="trust-bar-bg">
            <div class="trust-bar" style="width:{pct}%;background:{col};"></div>
        </div>
        {v_row}
    </div>"""


def render_dual_scores(face_conf, face_name, voice_conf, voice_name):
    """Two side-by-side score cards — face and voice."""
    fc_col = "#10b981" if face_conf>=0.75 else ("#f59e0b" if face_conf>=0.5 else "#ef4444")
    vc_col = "#10b981" if voice_conf>=0.75 else ("#f59e0b" if voice_conf>=0.5 else "#94a3b8")
    v_name_disp = voice_name if voice_name else "—"
    v_conf_disp = f"{voice_conf*100:.0f}%" if voice_conf > 0 else "—"
    return f"""
    <div class="dual-wrap">
        <div class="dual-card">
            <span class="dual-icon">📷</span>
            <div class="dual-label">Face FSL</div>
            <div class="dual-val" style="color:{fc_col};">{face_conf*100:.0f}%</div>
            <div class="dual-name">{face_name or '—'}</div>
        </div>
        <div class="dual-card">
            <span class="dual-icon">🎙</span>
            <div class="dual-label">Voice ID</div>
            <div class="dual-val" style="color:{vc_col};">{v_conf_disp}</div>
            <div class="dual-name">{v_name_disp}</div>
        </div>
    </div>"""


def xai_bar_color(s, best):
    if best and s >= 0.75: return "#10b981"
    if best and s >= 0.50: return "#f59e0b"
    if best:               return "#ef4444"
    return "rgba(255,255,255,.14)"


def render_xai(scores, title="Face similarity"):
    if not scores: return ""
    best = max(scores, key=scores.get)
    rows = ""
    for n, s in sorted(scores.items(), key=lambda x: -x[1])[:5]:
        pct = min(100, max(0, s*100))
        col = xai_bar_color(s, n == best)
        rows += (f'<div class="xr"><span class="xn">{n}</span>'
                 f'<div class="xbw"><div class="xb" style="width:{pct:.1f}%;background:{col};"></div></div>'
                 f'<span class="xv">{s:.3f}</span></div>')
    return (f'<div class="xai-wrap"><div class="xai-lbl">🔍 {title}</div>{rows}'
            f'<div class="xth">Cosine · '
            f'<span style="color:#10b981;">≥0.80 known</span> · '
            f'<span style="color:#f59e0b;">≥0.60 uncertain</span></div></div>')


def step_bar_html(active):
    labels = ["1·Face+Voice", "2·Listen", "3·Telegram", "4·Train"]
    html = '<div class="sbar">'
    for i, lbl in enumerate(labels, 1):
        cls = "dn" if i < active else ("act" if i == active else "")
        html += f'<div class="stb {cls}">{lbl}</div>'
    return html + "</div>"


def start_voice_listener():
    """Start the RealtimeListener background thread once per session."""
    if not voice_model:
        return None
    if "v_listener" not in st.session_state or st.session_state.v_listener is None:
        try:
            from voice_recognition_system.realtime_listener import RealtimeListener
            lsn = RealtimeListener(voice_model)
            lsn.start()
            st.session_state.v_listener = lsn
        except Exception as e:
            print(f"[App] Voice listener start failed: {e}")
            st.session_state.v_listener = None
    return st.session_state.get("v_listener")


def stop_voice_listener():
    lsn = st.session_state.get("v_listener")
    if lsn:
        try: lsn.stop()
        except Exception: pass
        st.session_state.v_listener = None


def voice_result_is_real(vr: dict) -> bool:
    """
    Returns True only if we have a genuine speaker ID result.
    Filters out no_profiles, empty, and error states.
    """
    if not vr:
        return False
    state = vr.get("state", "")
    return state in ("known", "uncertain")   # NOT "no_profiles" or "unknown"


def listen_once_async(result_q: queue.Queue, duration: float = 6.0):
    """
    Run listen_once() in a daemon thread.
    listen_once() already cleans the name via extract_name_from_transcript().
    Result is placed into result_q as a cleaned name string.
    """
    def _worker():
        try:
            from voice_recognition.realtime_listener import listen_once
            name = listen_once(int(duration))
            result_q.put(name or "")
        except Exception as e:
            print(f"[listen_once_async] Error: {e}")
            result_q.put("")
    threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
face_dot  = "#10b981" if face_model  else "#ef4444"
voice_dot = "#10b981" if voice_model else "#f59e0b"
incr_dot  = "#10b981" if incr_model  else "#f59e0b"
tg_dot    = "#229ED9" if tg_ok       else "#444466"

st.markdown(f"""
<div class="topbar">
    <span class="brand">🧠 MindBridge AI</span>
    <div class="srow">
        <span>
            <span class="sdot" style="background:{face_dot};box-shadow:0 0 6px {face_dot};"></span>
            <span class="slbl">Face FSL</span>
        </span>
        <span>
            <span class="sdot" style="background:{voice_dot};box-shadow:0 0 6px {voice_dot};"></span>
            <span class="slbl">Voice ID</span>
        </span>
        <span>
            <span class="sdot" style="background:{incr_dot};box-shadow:0 0 6px {incr_dot};"></span>
            <span class="slbl">Incremental</span>
        </span>
        <span>
            <span class="sdot" style="background:{tg_dot};box-shadow:0 0 6px {tg_dot};"></span>
            <span class="slbl">Telegram</span>
        </span>
        <span class="slbl" style="color:rgba(255,255,255,.25);">
            👤 {len(protos_now)} registered
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:.8rem 0 .3rem;">
        <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:700;
                    background:linear-gradient(135deg,#7c5cfc,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:.06em;">MindBridge AI</div>
        <div style="font-size:10px;color:#475569;letter-spacing:.1em;margin-top:2px;">
            FACE + VOICE RECOGNITION
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠  Dashboard",
        "📷  Live Recognition",
        "📲  Telegram Setup",
        "📋  Logs",
        "🔬  Explainability",
        "⚙️  Settings",
    ], label_visibility="collapsed")

    st.divider()
    pending_files = list(Path(PENDING_DIR).glob("*.jpg"))
    st.markdown(f"""
    <div style="font-size:11px;color:#475569;line-height:2.1;font-family:'JetBrains Mono';">
        <div>👤 <span style="color:#e2e8f0;">{len(protos_now)}</span> faces</div>
        <div>🎙 <span style="color:#e2e8f0;">
            {len(voice_model.prototypes) if voice_model and hasattr(voice_model,'prototypes') else 0}
        </span> voices</div>
        <div>⏳ <span style="color:#f59e0b;">{len(pending_files)}</span> pending</div>
        <div>📲 TG: <span style="color:{'#229ED9' if tg_ok else '#475569'};">
            {'active' if tg_ok else 'off'}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="wrap">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown('<div class="ptitle">System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Face + Voice dual recognition overview</div>', unsafe_allow_html=True)

    logs       = load_log()
    today_str  = datetime.date.today().isoformat()
    t_logs     = [l for l in logs if l.get("time","").startswith(today_str)]
    known_t    = [l for l in t_logs if l.get("state") == "known"]
    unk_t      = [l for l in t_logs if l.get("state") == "unknown"]
    avg_conf   = sum(l.get("conf",0) for l in t_logs) / max(len(t_logs),1)
    v_profiles = len(voice_model.prototypes) if voice_model and hasattr(voice_model,"prototypes") else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, icon, val, lbl, clr in [
        (c1,"👤",str(len(protos_now)),       "Face Profiles",     "#7c5cfc"),
        (c2,"🎙",str(v_profiles),            "Voice Profiles",    "#a78bfa"),
        (c3,"⏳",str(len(pending_files)),    "Pending TG",        "#f59e0b"),
        (c4,"🎯",f"{avg_conf*100:.0f}%",     "Avg Trust Today",   "#10b981"),
        (c5,"✅",str(len(known_t)),          "Known Today",       "#06b6d4"),
        (c6,"❓",str(len(unk_t)),            "Unknown Today",     "#ef4444"),
    ]:
        with col:
            st.markdown(f"""
            <div class="mcard">
                <span class="mi">{icon}</span>
                <div class="mv" style="color:{clr};">{val}</div>
                <div class="ml">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)

    # Activity chart
    if PLOTLY and logs:
        st.markdown('<div class="slb">📈 Recognition timeline</div>', unsafe_allow_html=True)
        times  = [l.get("time","")[:16] for l in logs[:20]][::-1]
        confs  = [l.get("conf",0) for l in logs[:20]][::-1]
        states = [l.get("state","unknown") for l in logs[:20]][::-1]
        colors = ["#10b981" if s=="known" else "#f59e0b" if s=="uncertain" else "#ef4444" for s in states]
        fig = go.Figure(go.Bar(x=times, y=confs, marker_color=colors,
                               text=[f"{c:.0%}" for c in confs],
                               textposition="outside",
                               textfont=dict(size=9, color="#94a3b8")))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8",size=8)),
                          yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,.06)',
                                     tickfont=dict(color="#94a3b8",size=9), range=[0,1]),
                          margin=dict(l=10,r=10,t=10,b=40), height=180)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="slb">🕐 Recent events</div>', unsafe_allow_html=True)
    for log in logs[:8]:
        state = log.get("state","unknown")
        icon  = "✅" if state=="known" else ("⚠️" if state=="uncertain" else "❓")
        col_s = "#10b981" if state=="known" else ("#f59e0b" if state=="uncertain" else "#ef4444")
        mode  = log.get("mode","face")
        mode_icon = "📷🎙" if mode=="dual" else ("🎙" if mode=="voice" else "📷")
        st.markdown(f"""
        <div class="log-row">
            <span class="log-ts">{log.get('time','')[:16]}</span>
            <span>{icon}</span>
            <span style="color:{col_s};font-weight:600;min-width:80px;">{state.upper()}</span>
            <span style="flex:1;color:var(--wh);">{log.get('name','—')}</span>
            <span style="color:var(--sl);font-size:11px;">{mode_icon}</span>
            <span style="color:var(--sl);font-family:'JetBrains Mono';font-size:10px;">
                {log.get('conf',0)*100:.0f}%
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE RECOGNITION  (face + voice dual)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📷  Live Recognition":

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION STATE INIT — MUST be first, before any UI or function definitions
    # so that st.session_state.flow is always available everywhere on this page
    # ══════════════════════════════════════════════════════════════════════════
    _defs = {
        "flow":           "recognising",
        "last_check":     0.0,
        "last_spoken":    None,
        "last_scores":    {},
        "conv_log":       [],
        # voice
        "v_listener":     None,
        "last_v_result":  None,
        # listening (name capture)
        "stt_q":          None,
        "stt_started":    False,
        "spoken_prompt":  False,
        # telegram
        "heard_name":     "",
        "heard_rel":      "",
        "tg_photo":       None,
        "tg_sent":        False,
        "tg_verified":    set(),
        "tg_rejected":    set(),
        "train_frames":   [],
    }
    for _k, _v in _defs.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.markdown('<div class="ptitle">Live Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Face FSL + Voice ID running simultaneously · Telegram 2-step for new visitors</div>', unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns(5)
    with ctl1: run_sys    = st.toggle("▶ Start",        value=True,  key="run_sys")
    with ctl2: en_voice_r = st.toggle("🎙 Voice ID",    value=True,  key="en_voice_r")
    with ctl3: show_xai   = st.toggle("🔍 XAI",        value=True,  key="show_xai")
    with ctl4: en_tts     = st.toggle("🔊 TTS output",  value=True,  key="en_tts")
    with ctl5: rinterval  = st.slider("Interval (s)", 2.0, 8.0, 3.0, 0.5,
                                       label_visibility="collapsed", key="recog_iv")

    st.markdown('<hr style="margin:.4rem 0 1rem;">', unsafe_allow_html=True)

    cam_col, ai_col = st.columns([3, 2], gap="large")

    with cam_col:
        st.markdown('<div class="slb">📷 Camera + 🎙 Voice</div>', unsafe_allow_html=True)
        cam_ph  = st.empty()
        step_ph = st.empty()

    with ai_col:
        st.markdown('<div class="slb">🤖 AI result panel</div>', unsafe_allow_html=True)
        # Show voice profile status once
        if en_voice_r and voice_model:
            _v_protos = getattr(voice_model, "prototypes", {})
            if not _v_protos:
                st.markdown(
                    '<div class="nb-a">🎙 Voice running in <b>transcription-only mode</b> — '
                    'no voice profiles registered. Speaker ID disabled.<br>'
                    f'Register voices in: <code>{VOICE_PROTO_PATH}</code></div>',
                    unsafe_allow_html=True,
                )
        result_ph = st.empty()
        dual_ph   = st.empty()
        trust_ph  = st.empty()
        notif_ph  = st.empty()
        xai_ph    = st.empty()
        st.markdown('<div class="slb">💬 Conversation</div>', unsafe_allow_html=True)
        conv_ph   = st.empty()

    def add_conv(role, text):
        st.session_state.conv_log.insert(0, {
            "role": role, "text": text,
            "time": datetime.datetime.now().strftime("%H:%M")
        })
        st.session_state.conv_log = st.session_state.conv_log[:12]

    def render_conv():
        html = ""
        for msg in reversed(st.session_state.conv_log[-6:]):
            cls = "chat-ai" if msg["role"] == "ai" else "chat-user"
            html += (f'<div class="{cls}">'
                     f'<span style="font-size:9px;color:var(--sl);">{msg["time"]}</span>'
                     f'<br>{msg["text"]}</div>')
        conv_ph.markdown(html, unsafe_allow_html=True)

    # ── Start / stop voice listener when toggle changes ───────────────────────
    if en_voice_r:
        v_listener = start_voice_listener()
    else:
        stop_voice_listener()
        v_listener = None

    # ══════════════════════════════════════════════════════════════════════════
    # STATIC PANELS  — ALL interactive widgets here, NEVER inside while loop
    # ══════════════════════════════════════════════════════════════════════════
    stop_btn     = st.button("⏹ Stop", key="stop_cam", type="secondary")
    listen_panel = st.empty()
    tg_panel     = st.empty()

    def render_static_panels():
        flow = st.session_state.get("flow", "recognising")

        # ── Listening panel ───────────────────────────────────────────────────
        if flow == "listening":
            with listen_panel.container():
                st.markdown('<div style="height:.3rem;"></div>', unsafe_allow_html=True)
                st.caption("🎤 Mic is listening… or type your name:")
                m_name = st.text_input("Name",         key="man_name_s",
                                       placeholder="e.g. Skanda",
                                       label_visibility="collapsed")
                m_rel  = st.text_input("Relationship", key="man_rel_s",
                                       placeholder="e.g. Son, Nurse",
                                       label_visibility="collapsed")
                if st.button("➜ Continue", key="man_btn_s",
                             use_container_width=True, type="primary"):
                    if m_name.strip():
                        st.session_state.heard_name    = m_name.strip()
                        st.session_state.heard_rel     = m_rel.strip()
                        st.session_state.spoken_prompt = False
                        st.session_state.stt_started   = False
                        add_conv("user", m_name.strip())
                        st.session_state.flow = "telegram_step"
                        st.rerun()
        else:
            listen_panel.empty()

        # ── Telegram approval panel ───────────────────────────────────────────
        if flow == "telegram_step":
            hn = st.session_state.get("heard_name", "")
            with tg_panel.container():
                st.markdown('<div style="height:.3rem;"></div>', unsafe_allow_html=True)
                st.caption("Manual override (or wait for Telegram /yes · /no):")
                ta, tb = st.columns(2)
                if ta.button("✅ Approve", key="tg_yes_s",
                             type="primary", use_container_width=True):
                    st.session_state.tg_verified.add(hn)
                    st.session_state.flow = "training"
                    st.rerun()
                if tb.button("❌ Reject", key="tg_no_s",
                             use_container_width=True):
                    st.session_state.tg_rejected.add(hn)
                    add_conv("ai", "Identity not confirmed. Please try again.")
                    speak_async("Sorry, I could not verify your identity.")
                    st.session_state.stt_started   = False
                    st.session_state.spoken_prompt = False
                    st.session_state.heard_name    = ""
                    st.session_state.heard_rel     = ""
                    st.session_state.tg_sent       = False
                    st.session_state.flow          = "listening"
                    st.rerun()
        else:
            tg_panel.empty()

    render_static_panels()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = None
    if run_sys:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None
            cam_ph.markdown("""
            <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
                        border-radius:12px;padding:2rem;text-align:center;color:#ef4444;">
                <div style="font-size:2rem;margin-bottom:.5rem;">📷</div>
                <div style="font-weight:600;margin-bottom:.3rem;">Cannot open webcam</div>
                <div style="font-size:.85rem;color:#94a3b8;">
                    Check camera is connected and not used by another app.<br>
                    Try: <code>ls /dev/video*</code> to list available cameras.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP  — only st.empty() placeholders updated, no new widgets
    # ══════════════════════════════════════════════════════════════════════════
    try:
        while run_sys and not stop_btn:
            now   = time.time()
            frame = None

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cam_ph.image(frame_to_pil(frame), use_container_width=True)

            # If no camera, still run the AI/voice logic — just without face recognition
            if cap is None and st.session_state.flow == "recognising":
                step_ph.markdown(step_bar_html(1), unsafe_allow_html=True)
                render_conv()
                result_ph.markdown("""
                <div class="hero gc-a">
                    <span class="pill p-a">⚠ Camera unavailable</span>
                    <div class="hn-l">No camera feed</div>
                    <div style="color:var(--sl);font-size:.85rem;margin-top:.5rem;">
                        Voice recognition is still active.<br>
                        Connect a camera and restart to enable face recognition.
                    </div>
                </div>""", unsafe_allow_html=True)
                time.sleep(1.0)
                continue

            # Poll voice listener result every tick
            v_listener = st.session_state.get("v_listener")
            if v_listener and en_voice_r:
                vr = v_listener.get_latest_result()
                if vr and voice_result_is_real(vr):
                    st.session_state.last_v_result = vr

            vr_now   = st.session_state.get("last_v_result") or {}
            # Only use voice result if it's a real speaker ID (not no_profiles/unknown)
            _v_real  = voice_result_is_real(vr_now)
            v_state  = vr_now.get("state",      "unknown") if _v_real else "unknown"
            v_name   = (vr_now.get("name") or vr_now.get("best_guess") or "") if _v_real else ""
            v_conf   = vr_now.get("confidence", 0.0) if _v_real else 0.0
            v_txt    = vr_now.get("transcript",  "")
            v_scores = vr_now.get("all_scores",  {}) if _v_real else {}

            flow = st.session_state.flow

            # ══════════════════════════════════════════════════════════════
            # STATE 1 — RECOGNISING  (face + voice simultaneously)
            # ══════════════════════════════════════════════════════════════
            if flow == "recognising":
                step_ph.markdown(step_bar_html(1), unsafe_allow_html=True)
                render_conv()

                if (frame is not None
                        and (now - st.session_state.last_check) >= rinterval):

                    st.session_state.last_check = now

                    # Always get a fresh recogniser (reloads prototypes.pkl from disk)
                    recogniser = get_recogniser_live()
                    if recogniser is None:
                        time.sleep(0.1)
                        continue

                    f_result = recogniser.recognise(frame)

                    f_state  = f_result.get("state",      "unknown")
                    f_name   = f_result.get("name")  or f_result.get("best_guess") or ""
                    f_rel    = f_result.get("relation",   "")
                    f_conf   = f_result.get("confidence", 0.0)
                    f_hint   = f_result.get("hint",       "")
                    f_scores = f_result.get("all_scores", {})
                    st.session_state.last_scores = f_scores

                    # ── DECIDE combined state ─────────────────────────────
                    # Both known and agree on same person → CONFIRMED
                    # Face known, voice unknown/different → FACE ONLY
                    # Voice known, face unknown → VOICE ONLY
                    # Both unknown → trigger listening flow

                    names_match = (f_name and v_name and
                                   f_name.lower() == v_name.lower())

                    if f_state == "known" and v_state == "known" and names_match:
                        combined_state = "dual_confirmed"
                        display_name   = f_name
                        display_rel    = f_rel
                        display_conf   = combined_trust(f_conf, v_conf)
                        display_hint   = f_hint
                        mode_tag       = "dual"

                    elif f_state == "known":
                        combined_state = "face_only"
                        display_name   = f_name
                        display_rel    = f_rel
                        display_conf   = f_conf
                        display_hint   = f_hint
                        mode_tag       = "face"

                    elif v_state == "known":
                        combined_state = "voice_only"
                        display_name   = v_name
                        display_rel    = ""
                        display_conf   = v_conf
                        display_hint   = f"Voice recognised: {v_name}"
                        mode_tag       = "voice"

                    else:
                        combined_state = "unknown"
                        display_name   = ""
                        display_rel    = ""
                        display_conf   = max(f_conf, v_conf)
                        display_hint   = ""
                        mode_tag       = "none"

                    # Log
                    append_log({
                        "time": datetime.datetime.now().isoformat(),
                        "state": "known" if combined_state in ("dual_confirmed","face_only","voice_only") else "unknown",
                        "name": display_name, "conf": display_conf,
                        "relation": display_rel, "mode": mode_tag,
                    })

                    # ── DUAL CONFIRMED ────────────────────────────────────
                    if combined_state == "dual_confirmed":
                        pro = gender_word(display_rel)
                        tts = (f"Hi, is this {display_name}? "
                               f"{pro} is your {display_rel}. "
                               f"Face and voice both confirmed.")
                        if display_name != st.session_state.last_spoken:
                            if en_tts: speak_async(tts)
                            add_conv("ai", tts)
                            st.session_state.last_spoken = display_name

                        result_ph.markdown(f"""
                        <div class="hero gc-g">
                            <span class="pill p-k">✓ FACE + VOICE CONFIRMED</span>
                            <div class="hn-k">{display_name}</div>
                            <div class="hrel">{display_rel}</div>
                            <div class="hhint">"{display_hint or tts}"</div>
                        </div>""", unsafe_allow_html=True)
                        dual_ph.markdown(
                            render_dual_scores(f_conf, f_name, v_conf, v_name),
                            unsafe_allow_html=True)
                        trust_ph.markdown(
                            render_trust(f_conf, v_conf, "Combined Trust"),
                            unsafe_allow_html=True)
                        notif_ph.empty()

                    # ── FACE ONLY ─────────────────────────────────────────
                    elif combined_state == "face_only":
                        pro = gender_word(display_rel)
                        tts = f"Hi, is this {display_name}? {pro} is your {display_rel}."
                        if display_name != st.session_state.last_spoken:
                            if en_tts: speak_async(tts)
                            add_conv("ai", tts)
                            st.session_state.last_spoken = display_name

                        result_ph.markdown(f"""
                        <div class="hero gc-c">
                            <span class="pill p-c">📷 FACE RECOGNISED</span>
                            <div class="hn-k">{display_name}</div>
                            <div class="hrel">{display_rel}</div>
                            <div class="hhint">"{display_hint or tts}"</div>
                            <div class="ca">Voice not matched · Face only</div>
                        </div>""", unsafe_allow_html=True)
                        dual_ph.markdown(
                            render_dual_scores(f_conf, f_name, v_conf, v_name),
                            unsafe_allow_html=True)
                        trust_ph.markdown(
                            render_trust(f_conf, v_conf, "Trust Score"),
                            unsafe_allow_html=True)
                        notif_ph.empty()

                    # ── VOICE ONLY ────────────────────────────────────────
                    elif combined_state == "voice_only":
                        tts = f"I recognise the voice of {display_name}."
                        if display_name != st.session_state.last_spoken:
                            if en_tts: speak_async(tts)
                            add_conv("ai", tts)
                            st.session_state.last_spoken = display_name

                        result_ph.markdown(f"""
                        <div class="hero gc-p">
                            <span class="pill p-v">🎙 VOICE RECOGNISED</span>
                            <div class="hn-k">{display_name}</div>
                            <div class="ca">Face not matched · Voice only</div>
                            {"<div class='hhint'>\"" + v_txt + "\"</div>" if v_txt else ""}
                        </div>""", unsafe_allow_html=True)
                        dual_ph.markdown(
                            render_dual_scores(f_conf, f_name, v_conf, v_name),
                            unsafe_allow_html=True)
                        trust_ph.markdown(
                            render_trust(f_conf, v_conf, "Trust Score"),
                            unsafe_allow_html=True)
                        notif_ph.empty()

                    # ── UNKNOWN → go to listening ─────────────────────────
                    else:
                        st.session_state.last_spoken   = None
                        st.session_state.stt_q         = queue.Queue()
                        st.session_state.stt_started   = False
                        st.session_state.spoken_prompt = False
                        st.session_state.heard_name    = ""
                        st.session_state.heard_rel     = ""
                        st.session_state.tg_sent       = False
                        st.session_state.tg_photo      = None
                        st.session_state.train_frames  = []
                        st.session_state.last_v_result = None
                        st.session_state.flow          = "listening"
                        st.rerun()

                    # XAI bars
                    if show_xai:
                        xai_html = render_xai(f_scores, "Face similarity")
                        if v_scores:
                            xai_html += render_xai(v_scores, "Voice similarity")
                        xai_ph.markdown(xai_html, unsafe_allow_html=True)
                    else:
                        xai_ph.empty()

            # ══════════════════════════════════════════════════════════════
            # STATE 2 — LISTENING  (ask name via TTS, capture via listen_once)
            # ══════════════════════════════════════════════════════════════
            elif flow == "listening":
                step_ph.markdown(step_bar_html(2), unsafe_allow_html=True)
                xai_ph.empty()
                dual_ph.empty()
                trust_ph.empty()
                notif_ph.empty()

                # Speak prompt once
                if not st.session_state.spoken_prompt:
                    msg = "Sorry, may I know your name?"
                    if en_tts: speak_async(msg)
                    add_conv("ai", msg)
                    st.session_state.spoken_prompt = True

                render_conv()

                result_ph.markdown("""
                <div class="hero gc-a">
                    <span class="pill p-a">🎙 LISTENING FOR NAME</span>
                    <div class="hn-l">May I know<br>your name?</div>
                    <div style="margin:.9rem 0;"><span class="mic">🎤</span></div>
                    <div style="color:var(--sl);font-size:.85rem;">
                        Speak your name clearly…
                    </div>
                </div>""", unsafe_allow_html=True)

                # Launch listen_once() in thread once
                if not st.session_state.stt_started and st.session_state.stt_q is not None:
                    st.session_state.stt_started = True
                    listen_once_async(st.session_state.stt_q, duration=6.0)

                # Check result
                if st.session_state.stt_q is not None:
                    try:
                        heard = st.session_state.stt_q.get_nowait()
                        if heard.strip():
                            st.session_state.heard_name    = heard.strip()
                            st.session_state.stt_started   = False
                            st.session_state.spoken_prompt = False
                            add_conv("user", f"My name is {heard.strip()}")
                            add_conv("ai",   f"Got it — {heard.strip()}! Let me verify with your family…")
                            st.session_state.flow = "telegram_step"
                            st.rerun()
                        else:
                            # Nothing heard — re-prompt
                            st.session_state.stt_started   = False
                            st.session_state.spoken_prompt = False
                    except queue.Empty:
                        pass

            # ══════════════════════════════════════════════════════════════
            # STATE 3 — TELEGRAM  (capture 3 photos, send, poll for yes/no)
            # ══════════════════════════════════════════════════════════════
            elif flow == "telegram_step":
                step_ph.markdown(step_bar_html(3), unsafe_allow_html=True)
                xai_ph.empty()
                dual_ph.empty()
                trust_ph.empty()
                render_conv()

                heard_name = st.session_state.heard_name
                heard_rel  = st.session_state.heard_rel
                patient_n  = cfg.get("patient_name", "the patient")

                # Capture 3 photos + send once
                if not st.session_state.tg_sent:
                    captured = []
                    for shot_i in range(3):
                        if cap and cap.isOpened():
                            for _ in range(3): cap.read()   # flush buffer
                            ret_s, shot_f = cap.read()
                            if ret_s:
                                p = save_frame_jpg(shot_f, PENDING_DIR,
                                                   f"{heard_name}_s{shot_i+1}")
                                captured.append(p)
                                cam_ph.image(frame_to_pil(shot_f),
                                             use_container_width=True)
                        time.sleep(0.8)

                    primary = captured[-1] if captured else None
                    st.session_state.tg_photo     = primary
                    st.session_state.train_frames = captured

                    if tg_notifier and captured:
                        try:
                            tg_notifier.send_new_person_request(
                                name=heard_name, relation_hint=heard_rel,
                                image_path=primary, patient_name=patient_n,
                                extra_photos=captured[:-1],
                            )
                        except Exception as ex:
                            print(f"[TG] {ex}")

                    add_conv("ai",
                             f"📸 3 photos sent to family for verification of {heard_name}…")
                    st.session_state.tg_sent = True

                result_ph.markdown(f"""
                <div class="hero gc-tg">
                    <span class="pill p-tg">📲 TELEGRAM VERIFICATION</span>
                    <div class="hn-tg">{heard_name}</div>
                    <div style="color:var(--sl);font-size:.88rem;margin:.5rem 0;">
                        3 photos sent · Awaiting caregiver reply<br>
                        <b>/yes</b> → train FSL &nbsp;·&nbsp; <b>/no</b> → ask again
                    </div>
                    <div class="spin">⏳</div>
                </div>""", unsafe_allow_html=True)

                # Show thumbnails
                if st.session_state.train_frames:
                    cols_t = notif_ph.columns(
                        min(3, len(st.session_state.train_frames)))
                    for _i, _p in enumerate(st.session_state.train_frames):
                        if os.path.exists(_p):
                            cols_t[_i].image(_p, use_container_width=True,
                                              caption=f"Shot {_i+1}")

                # Poll Telegram
                if tg_notifier:
                    try:
                        resp = tg_notifier.poll_response(heard_name)
                        if resp == "yes":
                            st.session_state.tg_verified.add(heard_name)
                            st.session_state.flow = "training"
                            st.rerun()
                        elif resp == "no":
                            st.session_state.tg_rejected.add(heard_name)
                            add_conv("ai", "Identity not confirmed. Please try again.")
                            speak_async("Sorry, I could not verify your identity.")
                            st.session_state.stt_started   = False
                            st.session_state.spoken_prompt = False
                            st.session_state.heard_name    = ""
                            st.session_state.heard_rel     = ""
                            st.session_state.tg_sent       = False
                            st.session_state.flow          = "listening"
                            st.rerun()
                    except Exception:
                        pass

            # ══════════════════════════════════════════════════════════════
            # STATE 4 — TRAINING  (FSL trains on all 3 photos)
            # ══════════════════════════════════════════════════════════════
            elif flow == "training":
                step_ph.markdown(step_bar_html(4), unsafe_allow_html=True)
                xai_ph.empty(); dual_ph.empty(); notif_ph.empty()
                render_conv()

                heard_name = st.session_state.heard_name
                heard_rel  = st.session_state.heard_rel or "visitor"
                hint       = f"{heard_name} is your {heard_rel}."
                frames     = st.session_state.train_frames
                pro        = gender_word(heard_rel)

                result_ph.markdown(f"""
                <div class="hero gc-p">
                    <span class="pill p-p">🧠 TRAINING FSL</span>
                    <div class="hn-k">{heard_name}</div>
                    <div style="color:var(--sl);font-size:.88rem;margin:.5rem 0;">
                        Caregiver approved — teaching the AI now…
                    </div>
                    <div class="spin">⚙️</div>
                </div>""", unsafe_allow_html=True)
                trust_ph.empty()

                # Move all 3 photos from pending → known
                real_dir = os.path.join(KNOWN_DIR, heard_name.replace(" ", "_"))
                Path(real_dir).mkdir(parents=True, exist_ok=True)
                renamed = []
                for i, old_p in enumerate(frames):
                    if not (old_p and os.path.exists(old_p)): continue
                    ts    = datetime.datetime.now().strftime("%H%M%S_%f")[:16]
                    new_p = os.path.join(real_dir, f"shot_{i+1:02d}_{ts}.jpg")
                    try:
                        shutil.move(old_p, new_p)
                        renamed.append(new_p)
                    except Exception:
                        renamed.append(old_p)

                trained = False
                summary = {}

                # Incremental FSL first, fall back to base
                for model_obj, method in [
                    (incr_model, "register_person"),
                    (face_model, "register_relative"),
                ]:
                    if trained or not (model_obj and renamed): continue
                    try:
                        fn = getattr(model_obj, method)
                        summary = fn(heard_name, renamed,
                                     relation=heard_rel, hint=hint)
                        (incr_model if method == "register_person"
                         else face_model).save_prototypes(PROTOTYPE_PATH)
                        trained = True
                    except Exception:
                        pass

                if trained:
                    est   = summary.get("estimated_confidence", 0.0)
                    shots = summary.get("shots", len(renamed))
                    tts_w = (f"Welcome {heard_name}. "
                             f"{pro} is your {heard_rel}. "
                             f"I will remember you from now on.")
                    if en_tts: speak_async(tts_w)
                    add_conv("ai", tts_w)
                    render_conv()

                    result_ph.markdown(f"""
                    <div class="hero gc-g">
                        <span class="pill p-k">✅ REGISTERED & WELCOMED</span>
                        <div class="hn-k">{heard_name}</div>
                        <div class="hrel">{heard_rel}</div>
                        <div class="hhint">"{tts_w}"</div>
                        <div class="cg">
                            est. {est:.0%} · {shots}-shot FSL ·
                            {len(renamed)} photos used
                        </div>
                    </div>""", unsafe_allow_html=True)

                    notif_ph.markdown(
                        f'<div class="nb-g">✅ {heard_name} trained — '
                        f'recognised automatically next time.</div>',
                        unsafe_allow_html=True)
                    trust_ph.markdown(render_trust(est, 0, "New profile trust"),
                                      unsafe_allow_html=True)
                    append_log({"time": datetime.datetime.now().isoformat(),
                                "state": "registered", "name": heard_name,
                                "conf": est, "relation": heard_rel, "mode": "dual"})
                    time.sleep(5)
                else:
                    notif_ph.markdown(
                        '<div class="nb-r">⚠ Training failed — check model files.</div>',
                        unsafe_allow_html=True)
                    time.sleep(3)

                # Reset back to recognising
                st.session_state.flow          = "recognising"
                st.session_state.last_spoken   = None
                st.session_state.stt_started   = False
                st.session_state.spoken_prompt = False
                st.session_state.heard_name    = ""
                st.session_state.heard_rel     = ""
                st.session_state.tg_sent       = False
                st.session_state.train_frames  = []
                st.session_state.tg_photo      = None
                st.session_state.last_check    = 0.0
                st.session_state.last_v_result = None
                st.rerun()

            time.sleep(0.04)

    finally:
        if cap:
            cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TELEGRAM SETUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📲  Telegram Setup":
    st.markdown('<div class="ptitle">Telegram Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Configure 2-step identity verification via Telegram bot</div>', unsafe_allow_html=True)

    if tg_ok:
        st.markdown('<div class="nb-g">✅ Telegram bot is active.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="nb-r">⚠ Not configured — manual Approve/Reject buttons used.</div>', unsafe_allow_html=True)

    with st.expander("📖 Setup guide", expanded=not tg_ok):
        st.markdown("""
**Step 1** — Open Telegram → search `@BotFather` → `/newbot` → copy **Bot Token**

**Step 2** — Each caregiver messages your bot once, then opens:
`https://api.telegram.org/bot<TOKEN>/getUpdates`
Find `"chat":{"id": 123456789}` — that number is their Chat ID.

**Step 3** — Fill below → Save → Test.

**What caregivers receive:**
```
🔔 Visitor verification needed

Someone named Skanda (says: Son) is with Grandma.
3 photos attached.

/yes → AI learns their face permanently
/no  → AI resets and asks again
```
        """)

    cfg = load_cfg()
    f1, f2 = st.columns(2)
    with f1:
        new_tok = st.text_input("Bot Token",    value=cfg.get("bot_token",""), type="password", key="tg_tok")
        new_pt  = st.text_input("Patient name", value=cfg.get("patient_name",""), key="tg_pt")
    with f2:
        new_cd = st.number_input("Unknown alert cooldown (min)", 1, 60, value=cfg.get("unknown_cooldown_min",3), key="tg_cd")
        new_pi = st.number_input("Poll interval (s)", 2, 30, value=cfg.get("poll_interval_secs",5), key="tg_pi")

    st.markdown('<div class="slb">Caregivers</div>', unsafe_allow_html=True)
    caregivers = cfg.get("caregivers", [])
    updated_cg = []
    for i, cg in enumerate(caregivers):
        with st.expander(f"👤 {cg.get('name','Caregiver')} — {cg.get('chat_id','')}"):
            ca1, ca2, ca3 = st.columns([2,2,1])
            cgn  = ca1.text_input("Name",    value=cg.get("name",""),         key=f"cgn_{i}")
            cgid = ca2.text_input("Chat ID", value=str(cg.get("chat_id","")), key=f"cgid_{i}")
            cgr  = ca3.text_input("Role",    value=cg.get("role",""),          key=f"cgr_{i}")
            if not st.button("🗑 Remove", key=f"rm_cg_{i}"):
                updated_cg.append({"name": cgn, "chat_id": cgid, "role": cgr})

    st.markdown("**Add caregiver**")
    cb1, cb2, cb3 = st.columns([2,2,1])
    ncgn  = cb1.text_input("Name",    key="ncgn",  placeholder="e.g. Priya")
    ncgid = cb2.text_input("Chat ID", key="ncgid", placeholder="123456789")
    ncgr  = cb3.text_input("Role",    key="ncgr",  placeholder="Daughter")
    if st.button("➕ Add", key="addcg"):
        if ncgn and ncgid:
            updated_cg.append({"name": ncgn, "chat_id": ncgid, "role": ncgr})

    st.divider()
    sv1, sv2 = st.columns(2)
    if sv1.button("💾 Save", type="primary", use_container_width=True, key="save_tg"):
        new_cfg = {**cfg, "bot_token": new_tok, "patient_name": new_pt,
                   "unknown_cooldown_min": int(new_cd), "poll_interval_secs": int(new_pi),
                   "caregivers": updated_cg}
        with open(NOTIFY_CFG, "w") as f:
            json.dump(new_cfg, f, indent=2)
        if tg_notifier:
            try: tg_notifier.reload_config()
            except Exception: pass
        st.success("✅ Saved!")
        st.rerun()
    if sv2.button("📲 Test", use_container_width=True, key="test_tg"):
        if tg_ok:
            with st.spinner("Sending…"):
                try:
                    ok, msg = tg_notifier.test_connection()
                    st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Save a valid bot token first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LOGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Logs":
    st.markdown('<div class="ptitle">Recognition Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Full audit trail — face, voice, dual, and registration events</div>', unsafe_allow_html=True)

    logs = load_log()
    f1, f2, f3 = st.columns(3)
    sf = f1.selectbox("State", ["All","known","unknown","uncertain","registered"], key="lf_state")
    df = f2.date_input("Date",  value=None, key="lf_date")
    mf = f3.selectbox("Mode",  ["All","face","voice","dual","none"],              key="lf_mode")

    if f3.button("🗑 Clear logs", key="clear_logs"):
        with open(LOG_PATH,"w") as f: json.dump([], f)
        st.rerun()

    filtered = logs
    if sf != "All": filtered = [l for l in filtered if l.get("state") == sf]
    if df:          filtered = [l for l in filtered if l.get("time","").startswith(str(df))]
    if mf != "All": filtered = [l for l in filtered if l.get("mode","face") == mf]

    st.markdown(f'<div class="slb">{len(filtered)} events</div>', unsafe_allow_html=True)
    for log in filtered[:50]:
        state = log.get("state","unknown")
        icon  = "✅" if state in ("known","registered") else ("⚠️" if state=="uncertain" else "❓")
        col_s = "#10b981" if state in ("known","registered") else ("#f59e0b" if state=="uncertain" else "#ef4444")
        mode  = log.get("mode","face")
        mode_i= "📷🎙" if mode=="dual" else ("🎙" if mode=="voice" else "📷")
        st.markdown(f"""
        <div class="log-row">
            <span class="log-ts">{log.get('time','')[:16]}</span>
            <span>{icon}</span>
            <span style="color:{col_s};font-weight:600;min-width:90px;font-size:11px;">{state.upper()}</span>
            <span style="flex:1;font-weight:500;">{log.get('name','—')}</span>
            <span style="color:var(--sl);font-size:10px;">{log.get('relation','')}</span>
            <span style="font-size:12px;">{mode_i}</span>
            <span style="font-family:'JetBrains Mono';font-size:11px;color:{col_s};">
                {log.get('conf',0)*100:.0f}%
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Explainability":
    st.markdown('<div class="ptitle">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Why did the AI make each decision?</div>', unsafe_allow_html=True)

    logs         = load_log()
    last_scores  = st.session_state.get("last_scores", {})
    last_log     = logs[0] if logs else {}

    x1, x2 = st.columns([2,3], gap="large")

    with x1:
        st.markdown('<div class="slb">🎯 Last decision</div>', unsafe_allow_html=True)
        if last_log:
            state = last_log.get("state","unknown")
            conf  = last_log.get("conf", 0.0)
            name  = last_log.get("name","—")
            mode  = last_log.get("mode","face")
            col_s = "#10b981" if state in ("known","registered") else ("#f59e0b" if state=="uncertain" else "#ef4444")
            mode_i= "📷🎙" if mode=="dual" else ("🎙" if mode=="voice" else "📷")

            st.markdown(f"""
            <div class="gc">
                <div style="font-size:10px;color:var(--sl);font-family:'JetBrains Mono';">{last_log.get('time','')[:19]}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{col_s};margin:.3rem 0;">{name}</div>
                <div style="font-size:.85rem;color:var(--cy);">{last_log.get('relation','')}</div>
                <div style="font-size:11px;color:var(--sl);margin-top:5px;">
                    Mode: {mode_i} · State: <b style="color:{col_s};">{state.upper()}</b>
                </div>
                {render_trust(conf, 0, "Confidence")}
            </div>""", unsafe_allow_html=True)

            # Reasoning
            st.markdown('<div class="slb">💡 Why this decision?</div>', unsafe_allow_html=True)
            cfg_s = load_cfg().get("settings", {})
            th_k  = cfg_s.get("threshold_known",    0.80)
            th_u  = cfg_s.get("threshold_uncertain", 0.60)

            if state in ("known","registered"):
                st.markdown(f"""
                <div class="gc gc-g">
                    <b style="color:#10b981;">Recognised because:</b><br>
                    <span style="font-size:.88rem;color:var(--sl);">
                    • Cosine similarity {conf:.3f} exceeded threshold ({th_k})<br>
                    • Mode: {mode_i} — {'both face and voice matched' if mode=='dual'
                      else 'face matched' if mode=='face' else 'voice matched'}<br>
                    • FSL prototype found in known/ folder<br>
                    • Combined trust score: {conf*100:.0f}%
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="gc gc-r">
                    <b style="color:#ef4444;">Not recognised because:</b><br>
                    <span style="font-size:.88rem;color:var(--sl);">
                    • Best similarity {conf:.3f} below threshold ({th_u})<br>
                    • No matching FSL prototype found<br>
                    • Voice ID also returned no match<br>
                    • Triggered name prompt → Telegram verification
                    </span>
                </div>""", unsafe_allow_html=True)

        if PLOTLY and last_log:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last_log.get("conf",0)*100,
                number=dict(suffix="%", font=dict(color="#e2e8f0", size=22, family="JetBrains Mono")),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#94a3b8", tickfont=dict(color="#94a3b8",size=9)),
                    bar=dict(color="#7c5cfc"),
                    bgcolor='rgba(255,255,255,.05)',
                    steps=[
                        dict(range=[0,60],  color='rgba(239,68,68,.15)'),
                        dict(range=[60,80], color='rgba(245,158,11,.15)'),
                        dict(range=[80,100],color='rgba(16,185,129,.15)'),
                    ],
                    threshold=dict(line=dict(color="#10b981",width=2),thickness=.75,value=80),
                ),
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                 margin=dict(l=20,r=20,t=20,b=10), height=160)
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

    with x2:
        st.markdown('<div class="slb">🔍 Face similarity scores</div>', unsafe_allow_html=True)
        if last_scores:
            st.markdown(render_xai(last_scores, "Top face matches"), unsafe_allow_html=True)

            # Threshold visualiser
            if PLOTLY:
                cfg_s = load_cfg().get("settings", {})
                th_k  = cfg_s.get("threshold_known",    0.80)
                th_u  = cfg_s.get("threshold_uncertain", 0.60)
                cv    = last_log.get("conf",0) if last_log else 0
                fig_th = go.Figure()
                fig_th.add_vrect(x0=0,    x1=th_u, fillcolor="rgba(239,68,68,.1)", line_width=0,
                                  annotation_text="REJECT", annotation_position="top left",
                                  annotation_font=dict(color="#ef4444",size=9))
                fig_th.add_vrect(x0=th_u, x1=th_k, fillcolor="rgba(245,158,11,.1)", line_width=0,
                                  annotation_text="UNCERTAIN", annotation_position="top left",
                                  annotation_font=dict(color="#f59e0b",size=9))
                fig_th.add_vrect(x0=th_k, x1=1.0,  fillcolor="rgba(16,185,129,.1)",  line_width=0,
                                  annotation_text="KNOWN", annotation_position="top left",
                                  annotation_font=dict(color="#10b981",size=9))
                fig_th.add_vline(x=cv, line_color="#a78bfa", line_width=2,
                                  annotation_text=f"score={cv:.2f}",
                                  annotation_font=dict(color="#a78bfa",size=10))
                fig_th.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis=dict(range=[0,1], tickfont=dict(color="#94a3b8",size=9),
                                                 title=dict(text="Cosine Similarity",font=dict(color="#94a3b8",size=10))),
                                      yaxis=dict(visible=False), margin=dict(l=10,r=10,t=30,b=30), height=120)
                st.plotly_chart(fig_th, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Run Live Recognition to populate XAI data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown('<div class="ptitle">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Thresholds, feature toggles, profile management</div>', unsafe_allow_html=True)

    cfg   = load_cfg()
    s_cfg = cfg.get("settings", {})

    st.markdown('<div class="slb">🎯 Recognition thresholds</div>', unsafe_allow_html=True)
    sg1, sg2 = st.columns(2)
    with sg1:
        th_k  = st.slider("Known threshold",     0.50, 0.99, float(s_cfg.get("threshold_known",    0.80)), 0.01, key="th_k")
        th_u  = st.slider("Uncertain threshold", 0.30, 0.79, float(s_cfg.get("threshold_uncertain", 0.60)), 0.01, key="th_u")
    with sg2:
        rv_wt = st.slider("Voice weight in combined trust", 0.0, 0.6, float(s_cfg.get("voice_weight", 0.40)), 0.05, key="rv_wt")
        ri    = st.slider("Recognition interval (s)", 1.0, 8.0, float(s_cfg.get("recognition_interval", 3.0)), 0.5, key="ri")

    st.markdown('<div class="slb">🔧 Feature toggles</div>', unsafe_allow_html=True)
    fg1, fg2, fg3 = st.columns(3)
    ev  = fg1.toggle("🔊 TTS voice output",     value=s_cfg.get("voice_enabled",     True), key="sv")
    etg = fg1.toggle("📲 Telegram 2FA",         value=s_cfg.get("telegram_enabled",  True), key="stg")
    ex  = fg2.toggle("🔍 XAI scores",           value=s_cfg.get("xai_enabled",       True), key="sx")
    ei  = fg2.toggle("🧠 Incremental FSL",      value=s_cfg.get("incremental_enabled",True),key="si")
    el  = fg3.toggle("📋 Activity logging",     value=s_cfg.get("logging_enabled",   True), key="sl")
    evr = fg3.toggle("🎙 Voice recognition",    value=s_cfg.get("voice_recog_enabled",True),key="svr")

    st.markdown('<div class="slb">👥 Registered face profiles</div>', unsafe_allow_html=True)
    protos = load_prototypes_raw()
    if protos:
        for name, data in protos.items():
            with st.expander(f"👤 {name} · {data.get('relation','')}"):
                d1, d2 = st.columns([3,1])
                est = min(0.98, 0.70 + data.get("shots",1)*0.06)
                d1.markdown(f"**Relation:** {data.get('relation','—')}  \n"
                            f"**Hint:** {data.get('hint','—')}  \n"
                            f"**Shots:** {data.get('shots','—')} · est. {est:.0%}")
                if d2.button(f"🗑 Remove", key=f"rm_{name}"):
                    try:
                        # 1. Load fresh from disk, remove, save back
                        raw = load_prototypes_raw()
                        if name in raw:
                            del raw[name]
                            with open(PROTOTYPE_PATH, "wb") as _f:
                                pickle.dump(raw, _f)

                        # 2. Also update the cached model object in memory
                        m = load_face_model()[0]
                        if m and hasattr(m, "prototypes") and name in m.prototypes:
                            del m.prototypes[name]

                        # 3. Clear Streamlit's resource cache so next
                        #    get_recogniser_live() picks up the deletion
                        st.cache_resource.clear()

                        # 4. Also remove photos from known/ folder
                        person_folder = os.path.join(KNOWN_DIR, name.replace(" ", "_"))
                        if os.path.isdir(person_folder):
                            shutil.rmtree(person_folder)

                        st.success(f"✅ Removed {name} — model cache cleared.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
    else:
        st.info("No profiles yet.")

    st.divider()
    if st.button("💾 Save all settings", type="primary", key="save_set"):
        cfg["settings"] = {
            "threshold_known": th_k, "threshold_uncertain": th_u,
            "voice_weight": rv_wt, "recognition_interval": ri,
            "voice_enabled": ev, "telegram_enabled": etg,
            "xai_enabled": ex, "incremental_enabled": ei,
            "logging_enabled": el, "voice_recog_enabled": evr,
        }
        with open(NOTIFY_CFG, "w") as f:
            json.dump(cfg, f, indent=2)
        st.success("✅ Settings saved!")

st.markdown('</div>', unsafe_allow_html=True)
