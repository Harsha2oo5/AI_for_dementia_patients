"""
app.py
------
Dementia AI Recognition System — Streamlit UI (v3)

Folder structure:
    face_recognition/
        fsl_model.py
        new_person_handler.py
        voice_output.py
        caregiver_setup.py
        camera_recognition_v2.py
    voice_recognition/
        realtime_listener.py
        voice_model.py
    models/
        prototypes.pkl
        voice_prototypes.pkl
    data/
        relatives/
        unknown_faces/
        voice_recordings/

Run:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
import threading
import datetime
import pickle
import base64
import json
import math
from pathlib import Path
from PIL import Image

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
FACE_REC_DIR  = os.path.join(ROOT, "face_recognition")
VOICE_REC_DIR = os.path.join(ROOT, "voice_recognition")

for d in [FACE_REC_DIR, VOICE_REC_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBridge — Dementia AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --cream:    #FAF7F2;
    --warm-100: #F5EFE6;
    --warm-200: #E8D9C5;
    --warm-300: #C9A97A;
    --teal:     #2D6A6A;
    --teal-lt:  #3D8B8B;
    --teal-pale:#E8F4F4;
    --amber:    #D97706;
    --amber-lt: #FEF3C7;
    --rose:     #BE185D;
    --rose-lt:  #FCE7F3;
    --slate:    #1E293B;
    --slate-60: #64748B;
    --slate-30: #CBD5E1;
    --green:    #059669;
    --green-lt: #D1FAE5;
    --red:      #DC2626;
    --red-lt:   #FEE2E2;
    --shadow-sm:0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow:   0 4px 16px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);
    --shadow-lg:0 12px 40px rgba(0,0,0,.12), 0 4px 12px rgba(0,0,0,.06);
    --r: 14px;
    --r-sm: 8px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream) !important;
}
.block-container { padding: 1.2rem 2rem 2rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"] {
    background: var(--slate) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stRadio > label { font-size: 13px !important; color: var(--slate-30) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem; color: var(--slate); letter-spacing: -.02em;
    margin: 0 0 0.2rem; line-height: 1.1;
}
.page-sub {
    font-size: 0.9rem; color: var(--slate-60);
    margin-bottom: 1.6rem; font-weight: 300;
}
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: var(--slate-60);
    margin: 1.4rem 0 0.6rem;
}

.card {
    background: #ffffff;
    border: 1px solid var(--warm-200);
    border-radius: var(--r);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
}
.card-teal {
    background: var(--teal-pale);
    border: 1px solid #B2D8D8;
    border-radius: var(--r);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 600; letter-spacing: .04em;
}
.pill-known     { background: var(--green-lt); color: #065f46; }
.pill-uncertain { background: var(--amber-lt); color: #92400e; }
.pill-unknown   { background: var(--red-lt);   color: #991b1b; }
.pill-voice     { background: var(--teal-pale); color: var(--teal); }

.patient-hero {
    background: #ffffff;
    border: 1px solid var(--warm-200);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin-top: 0.6rem;
}
.patient-name {
    font-family: 'DM Serif Display', serif;
    font-size: 3.8rem; color: var(--slate); line-height: 1;
    margin: 0.5rem 0;
}
.patient-relation {
    font-size: 1.2rem; color: var(--teal);
    font-weight: 500; margin-bottom: 0.6rem;
}
.patient-hint {
    font-size: 1rem; color: var(--slate-60);
    background: var(--warm-100);
    border-left: 3px solid var(--warm-300);
    border-radius: var(--r-sm);
    padding: 0.9rem 1.2rem;
    margin: 1rem auto; max-width: 520px;
    text-align: left; font-style: italic;
}
.conf-high { color: var(--green);  font-weight: 600; }
.conf-med  { color: var(--amber);  font-weight: 600; }
.conf-low  { color: var(--red);    font-weight: 600; }

.xai-wrap {
    background: var(--warm-100);
    border: 1px solid var(--warm-200);
    border-radius: var(--r);
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
}
.xai-title {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--slate-60);
    margin-bottom: 0.8rem;
}
.score-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 6px;
}
.score-name { font-size: 13px; font-weight: 500; min-width: 90px; color: var(--slate); }
.score-bar-wrap {
    flex: 1; background: var(--warm-200);
    border-radius: 999px; height: 8px; overflow: hidden;
}
.score-bar { height: 8px; border-radius: 999px; transition: width .4s ease; }
.score-val { font-family: 'JetBrains Mono'; font-size: 11px; color: var(--slate-60); min-width: 40px; text-align: right; }

.voice-score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.voice-score-item {
    background: #fff; border: 1px solid var(--warm-200);
    border-radius: var(--r-sm); padding: 0.7rem 0.9rem;
}
.voice-score-item .name { font-size: 12px; font-weight: 500; color: var(--slate); }
.voice-score-item .score {
    font-family: 'JetBrains Mono'; font-size: 1.1rem;
    font-weight: 500; color: var(--teal); margin-top: 2px;
}
.voice-score-item .bar { background: var(--warm-200); border-radius: 4px; height: 5px; margin-top: 5px; }
.voice-score-item .bar-fill { height: 5px; border-radius: 4px; background: var(--teal); }

.metric-tile {
    background: #fff;
    border: 1px solid var(--warm-200);
    border-radius: var(--r);
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.metric-tile .val { font-family: 'DM Serif Display'; font-size: 2rem; color: var(--slate); }
.metric-tile .lbl { font-size: 11px; color: var(--slate-60); font-weight: 500; margin-top: 2px; }

.alert-banner {
    background: var(--amber-lt); border: 1px solid #FCD34D;
    border-radius: var(--r-sm); padding: 0.75rem 1rem;
    margin-bottom: 1rem; font-size: 13.5px; color: #78350F;
    display: flex; align-items: center; gap: 8px;
}

.recorder-idle {
    background: var(--teal-pale); border: 2px dashed #7dd3fc;
    border-radius: var(--r); padding: 1.8rem; text-align: center;
    color: var(--teal); font-size: 14px;
}
.recorder-active {
    background: #FFF0F0; border: 2px solid var(--red);
    border-radius: var(--r); padding: 1.8rem; text-align: center;
    color: var(--red); font-size: 14px;
    animation: pulse-red 1.2s ease-in-out infinite;
}
@keyframes pulse-red { 0%,100%{opacity:1; border-color: var(--red);} 50%{opacity:.7; border-color: #F87171;} }

[data-testid="stTabs"] [role="tablist"] { gap: 4px; }
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'DM Sans' !important; font-size: 13px !important;
    font-weight: 500 !important; border-radius: var(--r-sm) !important;
    padding: 6px 16px !important;
}

.stButton > button {
    border-radius: var(--r-sm) !important;
    font-family: 'DM Sans' !important; font-weight: 500 !important;
}
.stButton > button[kind="primary"] {
    background: var(--teal) !important; border-color: var(--teal) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--teal-lt) !important; border-color: var(--teal-lt) !important;
}

hr { border-color: var(--warm-200) !important; margin: 1.2rem 0 !important; }

.camera-wrap {
    border-radius: var(--r); overflow: hidden;
    border: 2px solid var(--warm-200);
    background: #0a0a0a;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROTOTYPE_PATH       = os.path.join(ROOT, "models", "prototypes.pkl")
VOICE_PROTOTYPE_PATH = os.path.join(ROOT, "models", "voice_prototypes.pkl")
CAPTURE_DIR          = os.path.join(ROOT, "data", "relatives")
UNKNOWN_DIR          = os.path.join(ROOT, "data", "unknown_faces")
VOICE_DIR            = os.path.join(ROOT, "data", "voice_recordings")
ALERT_LOG_PATH       = os.path.join(ROOT, "data", "unknown_alerts.json")

for d in [CAPTURE_DIR, UNKNOWN_DIR, VOICE_DIR,
          os.path.join(ROOT, "models"), os.path.join(ROOT, "data")]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_model():
    try:
        from face_recogniton.fsl_model import RelativeRecognitionModel
        m = RelativeRecognitionModel()
        m.load_prototypes(PROTOTYPE_PATH)
        return m, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_voice_model():
    try:
        from voice_recognition.voice_model import SpeakerRecognitionModel
        vm = SpeakerRecognitionModel()
        vm.load_prototypes(VOICE_PROTOTYPE_PATH)
        return vm, None
    except Exception as e:
        return None, str(e)


def load_prototypes_raw() -> dict:
    if os.path.exists(PROTOTYPE_PATH):
        with open(PROTOTYPE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def load_voice_prototypes_raw() -> dict:
    if os.path.exists(VOICE_PROTOTYPE_PATH):
        with open(VOICE_PROTOTYPE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def conf_class(c: float) -> str:
    if c >= 0.75: return "conf-high"
    if c >= 0.50: return "conf-med"
    return "conf-low"


def state_pill(state: str, extra: str = "") -> str:
    icons  = {"known": "✓", "uncertain": "⚠", "unknown": "✗"}
    labels = {"known": "Recognised", "uncertain": "Uncertain", "unknown": "Unknown"}
    cls    = {"known": "pill-known", "uncertain": "pill-uncertain", "unknown": "pill-unknown"}
    ic  = icons.get(state, "?")
    lb  = labels.get(state, state)
    cl  = cls.get(state, "pill-unknown")
    txt = f"{ic} {lb}" + (f" — {extra}" if extra else "")
    return f'<span class="pill {cl}">{txt}</span>'


def bar_color(score: float, is_best: bool) -> str:
    if is_best and score >= 0.75: return "#059669"
    if is_best and score >= 0.50: return "#D97706"
    if is_best: return "#DC2626"
    return "#94A3B8"


def render_xai_face(all_scores: dict, title: str = "Face similarity scores") -> str:
    if not all_scores:
        return ""
    best = max(all_scores, key=all_scores.get)
    rows = ""
    for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        pct = min(100, max(0, score * 100))
        col = bar_color(score, name == best)
        rows += f"""
        <div class="score-row">
            <span class="score-name">{name}</span>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width:{pct:.1f}%;background:{col};"></div>
            </div>
            <span class="score-val">{score:.2f}</span>
        </div>"""
    return f"""
    <div class="xai-wrap">
        <div class="xai-title">🔍 XAI — {title}</div>
        {rows}
        <div style="margin-top:8px;font-size:11px;color:#94A3B8;">
            Cosine similarity · Threshold: <b style="color:#059669">≥0.80</b> known · <b style="color:#D97706">≥0.60</b> uncertain
        </div>
    </div>"""


def render_xai_voice(all_scores: dict, title: str = "Voice similarity scores") -> str:
    if not all_scores:
        return ""
    best = max(all_scores, key=all_scores.get)
    items = ""
    for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        pct = min(100, max(0, score * 100))
        col = bar_color(score, name == best)
        items += f"""
        <div class="voice-score-item">
            <div class="name">{name}</div>
            <div class="score" style="color:{col};">{score:.3f}</div>
            <div class="bar"><div class="bar-fill" style="width:{pct:.1f}%;background:{col};"></div></div>
        </div>"""
    return f"""
    <div class="xai-wrap">
        <div class="xai-title">🎙 XAI — {title}</div>
        <div class="voice-score-grid">{items}</div>
        <div style="margin-top:8px;font-size:11px;color:#94A3B8;">
            D-vector cosine similarity · Threshold: <b style="color:#059669">≥0.75</b> known · <b style="color:#D97706">≥0.50</b> uncertain
        </div>
    </div>"""


def build_face_result_html(state, name, rel, conf, hint) -> str:
    """
    FIX: Build the entire result card as a single HTML string.
    This ensures unsafe_allow_html is never lost across multiple st.markdown calls
    inside a placeholder container.
    """
    pill = state_pill(state)
    cc   = conf_class(conf)

    if state == "known":
        return f"""
        {pill}
        <div class="patient-hero">
            <div class="patient-name">{name}</div>
            <div class="patient-relation">{rel}</div>
            <div class="patient-hint">{hint}</div>
            <div class="{cc}">{conf*100:.0f}% confidence</div>
        </div>"""

    elif state == "uncertain":
        return f"""
        {pill}
        <div class="patient-hero">
            <div class="patient-name" style="color:var(--amber);">{name}?</div>
            <div class="patient-hint">I'm not fully confident. Please wait a moment.</div>
            <div class="{cc}">{conf*100:.0f}% confidence</div>
        </div>"""

    else:
        return f"""
        {pill}
        <div class="patient-hero">
            <div class="patient-name" style="color:var(--red);">Not recognised</div>
            <div class="patient-hint">Please call your caregiver.</div>
        </div>"""


def build_voice_result_html(vstate, vname, vconf, vtxt) -> str:
    """
    FIX: Build voice result as single HTML string for reliable rendering.
    """
    pill = state_pill(vstate, f"{vconf*100:.0f}%")
    cc   = conf_class(vconf)

    transcript_block = ""
    if vtxt:
        transcript_block = f"""
        <div style="
            background:var(--warm-100);
            border-radius:10px;
            padding:.8rem 1rem;
            margin-top:1rem;
            font-size:14px;
            color:var(--slate-60);
            font-style:italic;
        ">"{vtxt}"</div>"""

    if vstate == "known":
        return f"""
        {pill} <span class="pill pill-voice">🎙 Voice</span>
        <div class="patient-hero">
            <div class="patient-name">{vname}</div>
            <div class="patient-relation">Voice identified</div>
            <div class="{cc}">{vconf*100:.0f}% confidence</div>
            {transcript_block}
        </div>"""
    else:
        return f"""
        {pill} <span class="pill pill-voice">🎙 Voice</span>
        <div class="patient-hero">
            <div class="patient-name" style="color:var(--red);">Voice not recognised</div>
            {transcript_block}
        </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#F8FAFC;">MindBridge</div>
        <div style="font-size:11px;color:#64748B;letter-spacing:.08em;text-transform:uppercase;">Dementia AI · v3</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mode = st.radio(
        "Mode",
        ["🟣  Patient view", "🔵  Caregiver panel"],
        index=0,
    )
    st.divider()

    protos       = load_prototypes_raw()
    voice_protos = load_voice_prototypes_raw()

    st.markdown(f"""
    <div style="font-size:12px;color:#94A3B8;line-height:1.9;">
        <div>👤 <b style="color:#E2E8F0;">{len(protos)}</b> face(s) registered</div>
        <div>🎙 <b style="color:#E2E8F0;">{len(voice_protos)}</b> voice(s) registered</div>
    </div>
    """, unsafe_allow_html=True)

    if protos:
        st.markdown('<div style="margin-top:.8rem;"></div>', unsafe_allow_html=True)
        for name, data in protos.items():
            has_voice = name in voice_protos
            v_icon    = "🎙" if has_voice else "  "
            st.markdown(
                f'<div style="font-size:12px;color:#CBD5E1;padding:2px 0;">👤{v_icon} {name} · <i style="color:#64748B;">{data.get("relation","")}</i></div>',
                unsafe_allow_html=True,
            )
    st.divider()
    st.caption("All data is stored locally.\nNo cloud. No internet after setup.")


face_model,  face_model_err  = load_face_model()
voice_model, voice_model_err = load_voice_model()


# ══════════════════════════════════════════════════════════════════════════════
# 🟣 PATIENT VIEW
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🟣  Patient view":

    st.markdown('<div class="page-title">🧠 Who is here?</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">The camera and microphone will help recognise your loved ones.</div>', unsafe_allow_html=True)

    if face_model_err and not face_model:
        st.error(f"Face model could not load: {face_model_err}")
        st.stop()

    if face_model and not face_model.prototypes:
        st.warning("⚠️ No relatives registered yet — ask your caregiver to set this up.")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1: run_cam   = st.toggle("📷 Camera recognition", value=True)
    with c2: run_voice = st.toggle("🎙 Voice recognition",  value=True)
    with c3: show_xai  = st.toggle("🔍 Show XAI scores",    value=True)
    with c4: interval  = st.slider("Check every (s)", 1.5, 6.0, 2.5, 0.5, label_visibility="collapsed")

    st.divider()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_cam, col_result = st.columns([3, 2], gap="large")

    with col_cam:
        st.markdown('<div class="section-label">📷 Live camera</div>', unsafe_allow_html=True)
        cam_ph = st.empty()

    with col_result:
        st.markdown('<div class="section-label">🧠 Recognition result</div>', unsafe_allow_html=True)
        result_ph = st.empty()
        xai_ph    = st.empty()

        st.markdown('<div class="section-label">🎙 Voice result</div>', unsafe_allow_html=True)
        # FIX: always define voice placeholders regardless of run_voice toggle
        voice_ph     = st.empty()
        voice_xai_ph = st.empty()

        if not run_voice:
            voice_ph.markdown(
                '<div style="color:var(--slate-60);font-size:13px;padding:.5rem 0;">Voice recognition is off.</div>',
                unsafe_allow_html=True,
            )

    # ── Session state ─────────────────────────────────────────────────────────
    for k, v in {
        "last_result": None,
        "last_spoken": None,
        "last_voice_result": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Speaker ───────────────────────────────────────────────────────────────
    speaker = None
    try:
        from face_recogniton.voice_output import VoiceSpeaker
        speaker = VoiceSpeaker()
    except Exception:
        pass

    # ── Face recogniser ───────────────────────────────────────────────────────
    recogniser = None
    if face_model:
        try:
            from face_recogniton.new_person_handler import OpenSetRecogniser
            recogniser = OpenSetRecogniser(face_model)
        except Exception as e:
            st.warning(f"Could not load OpenSetRecogniser: {e}")

    # ── Voice listener ────────────────────────────────────────────────────────
    LIVE_VOICE_DIR = os.path.join(ROOT, "data", "live_voice_logs")
    listener = None

    if run_voice and voice_model:
        try:
            from voice_recognition.realtime_listener import RealtimeListener
            if "listener" not in st.session_state:
                st.session_state.listener = RealtimeListener(voice_model, save_dir=LIVE_VOICE_DIR)
            listener = st.session_state.listener
            if "voice_listener_started" not in st.session_state:
                listener.start()
                st.session_state.voice_listener_started = True
        except Exception as e:
            voice_ph.warning(f"Voice listener error: {e}")

    if not run_voice:
        if "listener" in st.session_state:
            try:
                st.session_state.listener.stop()
            except Exception:
                pass
            st.session_state.listener = None
        st.session_state.pop("voice_listener_started", None)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = None
    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
            st.stop()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stop_btn   = st.button("⏹ Stop system", type="secondary")
    last_check = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while (run_cam or run_voice) and not stop_btn:

            # ── Camera / face recognition ──────────────────────────────────
            if run_cam and cap:
                ret, frame = cap.read()
                if ret:
                    now = time.time()
                    cam_ph.image(frame_to_pil(frame), use_container_width=True)

                    if now - last_check >= interval and recogniser:
                        last_check = now
                        result     = recogniser.recognise(frame)
                        st.session_state.last_result = result

                        state  = result.get("state", "unknown")
                        name   = result.get("name") or result.get("best_guess") or "Unknown"
                        rel    = result.get("relation", "")
                        conf   = result.get("confidence", 0.0)
                        hint   = result.get("hint", "")
                        scores = result.get("all_scores", {})

                        # ── FIX: single markdown call on the placeholder directly ──
                        result_ph.markdown(
                            build_face_result_html(state, name, rel, conf, hint),
                            unsafe_allow_html=True,
                        )

                        # Announce via speaker (face)
                        if state == "known" and speaker and name != st.session_state.last_spoken:
                            threading.Thread(
                                target=speaker.speak,
                                args=(hint or f"{name} is here with you",),
                                daemon=True,
                            ).start()
                            st.session_state.last_spoken = name

                        if state == "unknown" and recogniser:
                            recogniser.handle_unknown(frame, result)

                        if show_xai and scores:
                            xai_ph.markdown(render_xai_face(scores), unsafe_allow_html=True)
                        elif not scores:
                            xai_ph.empty()

            # ── Voice recognition ──────────────────────────────────────────
            if run_voice and listener:
                vr = listener.get_latest_result()
                if vr:
                    st.session_state.last_voice_result = vr

                    vstate  = vr.get("state", "unknown")
                    vname   = vr.get("name") or vr.get("best_guess") or "Unknown"
                    vconf   = vr.get("confidence", 0.0)
                    vtxt    = vr.get("transcript", "")
                    vscores = vr.get("all_scores", {})

                    # ── FIX: single markdown call on placeholder directly ──
                    voice_ph.markdown(
                        build_voice_result_html(vstate, vname, vconf, vtxt),
                        unsafe_allow_html=True,
                    )

                    # ── FIX: announce voice speaker with dedup key ──
                    if vstate == "known" and speaker:
                        voice_dedup_key = f"voice__{vname}"
                        if voice_dedup_key != st.session_state.last_spoken:
                            threading.Thread(
                                target=speaker.speak,
                                args=(f"{vname} is speaking to you",),
                                daemon=True,
                            ).start()
                            st.session_state.last_spoken = voice_dedup_key

                    if show_xai and vscores:
                        voice_xai_ph.markdown(render_xai_voice(vscores), unsafe_allow_html=True)
                    elif not vscores:
                        voice_xai_ph.empty()

            time.sleep(0.05)

    finally:
        if cap:
            cap.release()
        if listener:
            try:
                listener.stop()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 🔵 CAREGIVER PANEL
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-title">🔵 Caregiver Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Register relatives, record voices, monitor recognition, and review alerts.</div>', unsafe_allow_html=True)

    if face_model_err:
        st.warning(f"⚠️ Face model issue: {face_model_err}")
    if voice_model_err:
        st.warning(f"⚠️ Voice model issue: {voice_model_err}")

    tabs = st.tabs([
        "👤  Register face",
        "🎙️  Register voice",
        "👁️  Live recognition",
        "📋  Manage people",
        "🔔  Unknown alerts",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Register face
    # ══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Register a new relative — Face")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-label">Person details</div>', unsafe_allow_html=True)
            reg_name     = st.text_input("Full name",         placeholder="e.g. Sarah Patel",           key="freg_name")
            reg_relation = st.text_input("Relationship",      placeholder="e.g. Daughter, Son, Nurse",  key="freg_rel")
            reg_hint     = st.text_area(
                "Voice hint for patient",
                placeholder="e.g. Sarah is your daughter. She visits every Sunday.",
                height=90, key="freg_hint",
            )
            reg_shots = st.slider("Number of photos to capture", 1, 5, 3, key="freg_shots")
            if not reg_hint and reg_name and reg_relation:
                st.caption(f'Default hint: *"{reg_name} is your {reg_relation}."*')

        with col2:
            st.markdown('<div class="section-label">Camera capture</div>', unsafe_allow_html=True)
            cam_reg_ph  = st.empty()
            shot_status = st.empty()
            cb1, cb2    = st.columns(2)
            capture_btn = cb1.button("📸 Capture photo", use_container_width=True, type="primary", key="cap_btn")
            auto_btn    = cb2.button("⚡ Auto-capture",  use_container_width=True, key="auto_btn")
            retake_btn  = st.button("🔄 Retake all",    use_container_width=True, key="retake_btn")

            if "reg_shots_taken" not in st.session_state:
                st.session_state.reg_shots_taken = []

            n    = len(st.session_state.reg_shots_taken)
            dots = "🟢" * n + "⚪" * (reg_shots - n)
            shot_status.markdown(f"**Shots captured:** {dots} &nbsp;({n}/{reg_shots})")

            if retake_btn:
                st.session_state.reg_shots_taken = []
                st.rerun()

        if capture_btn or auto_btn:
            if not reg_name:
                st.warning("Enter the person's name first.")
            else:
                save_dir = os.path.join(CAPTURE_DIR, reg_name.replace(" ", "_"))
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                cap_reg = cv2.VideoCapture(0)
                if cap_reg.isOpened():
                    if auto_btn:
                        for i in range(reg_shots):
                            ret, frame = cap_reg.read()
                            if ret:
                                ts   = datetime.datetime.now().strftime("%H%M%S_%f")[:10]
                                path = os.path.join(save_dir, f"shot_{i+1:02d}_{ts}.jpg")
                                cv2.imwrite(path, frame)
                                st.session_state.reg_shots_taken.append(path)
                                cam_reg_ph.image(frame_to_pil(frame), use_container_width=True)
                                time.sleep(1.0)
                    else:
                        ret, frame = cap_reg.read()
                        if ret:
                            ts   = datetime.datetime.now().strftime("%H%M%S_%f")[:10]
                            idx  = len(st.session_state.reg_shots_taken) + 1
                            path = os.path.join(save_dir, f"shot_{idx:02d}_{ts}.jpg")
                            cv2.imwrite(path, frame)
                            st.session_state.reg_shots_taken.append(path)
                            cam_reg_ph.image(frame_to_pil(frame), use_container_width=True)
                    cap_reg.release()
                    st.rerun()
                else:
                    st.error("Cannot open webcam.")

        st.divider()
        sc1, _ = st.columns([1, 2])
        with sc1:
            if st.button("✅ Register face", type="primary", use_container_width=True, key="reg_face_btn"):
                if not reg_name:
                    st.error("Name is required.")
                elif not st.session_state.reg_shots_taken:
                    st.error("Capture at least 1 photo first.")
                elif face_model is None:
                    st.error("Face model not loaded.")
                else:
                    hint = reg_hint or f"{reg_name} is your {reg_relation}."
                    with st.spinner(f"Building FSL prototype for {reg_name}..."):
                        try:
                            summary = face_model.register_relative(
                                reg_name,
                                st.session_state.reg_shots_taken,
                                relation=reg_relation,
                                hint=hint,
                            )
                            face_model.save_prototypes(PROTOTYPE_PATH)
                            st.session_state.reg_shots_taken = []
                            est = summary.get("estimated_confidence", 0.0)
                            st.success(f"✅ **{reg_name}** registered — {summary['shots']}-shot · est. {est:.0%} confidence")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Registration failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Register voice
    # ══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Register voice — Speaker identification")
        st.caption("Record 1–5 voice samples for each person. Longer samples give better accuracy.")

        col_vr, col_vlist = st.columns([1, 1], gap="large")

        with col_vr:
            st.markdown('<div class="section-label">Record voice sample</div>', unsafe_allow_html=True)

            protos_now       = load_prototypes_raw()
            registered_names = list(protos_now.keys())
            voice_protos_now = load_voice_prototypes_raw()

            if registered_names:
                voice_target = st.selectbox("Person to register voice for", registered_names, key="voice_target")
                if voice_target in voice_protos_now:
                    st.info(f"ℹ️ {voice_target} already has a voice registered. New recording will update it.")
            else:
                st.warning("Register at least one face profile first (Tab 1).")
                voice_target = None

            st.markdown('<div class="section-label">Recording settings</div>', unsafe_allow_html=True)
            rec_duration = st.slider("Duration (seconds)", 5, 30, 8, key="vrec_dur")
            rec_rate     = st.selectbox("Sample rate", [16000, 22050], index=0, key="vrec_rate")

            if "v_recording"       not in st.session_state: st.session_state.v_recording       = False
            if "v_last_recording"  not in st.session_state: st.session_state.v_last_recording  = None
            if "v_recording_paths" not in st.session_state: st.session_state.v_recording_paths = []

            rec_col1, rec_col2 = st.columns(2)

            if not st.session_state.v_recording:
                if rec_col1.button("🎙 Record sample",    type="primary", use_container_width=True, key="rec_start"):
                    st.session_state.v_recording = True
                    st.rerun()
                if rec_col2.button("🔄 Clear all samples", use_container_width=True, key="rec_clear"):
                    st.session_state.v_recording_paths = []
                    st.rerun()
            else:
                st.markdown('<div class="recorder-active">🔴 Recording — speak naturally...</div>', unsafe_allow_html=True)
                prog  = st.progress(0)
                tleft = st.empty()
                try:
                    import sounddevice as sd
                    from scipy.io.wavfile import write as wav_write

                    audio = sd.rec(int(rec_duration * rec_rate), samplerate=rec_rate, channels=1, dtype="float32")
                    for i in range(rec_duration):
                        time.sleep(1)
                        prog.progress((i + 1) / rec_duration)
                        tleft.markdown(f"**{rec_duration - i - 1}s** remaining")
                    sd.wait()
                    prog.progress(1.0)

                    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    tgt_name = (voice_target or "unknown").replace(" ", "_")
                    out_path = os.path.join(VOICE_DIR, f"{tgt_name}_{ts}.wav")
                    wav_write(out_path, rec_rate, audio)

                    st.session_state.v_last_recording = out_path
                    st.session_state.v_recording_paths.append(out_path)
                    st.session_state.v_recording = False
                    st.success(f"✅ Sample {len(st.session_state.v_recording_paths)} saved.")
                    st.rerun()
                except ImportError:
                    st.error("sounddevice / scipy not installed. Run: `pip install sounddevice scipy`")
                    st.session_state.v_recording = False
                except Exception as e:
                    st.error(f"Recording failed: {e}")
                    st.session_state.v_recording = False

            n_samples = len(st.session_state.v_recording_paths)
            if n_samples > 0:
                st.markdown(f'<div class="section-label">{n_samples} sample(s) captured</div>', unsafe_allow_html=True)
                st.markdown("🟢" * n_samples + "⚪" * max(0, 3 - n_samples))
                if st.session_state.v_last_recording and os.path.exists(st.session_state.v_last_recording):
                    st.audio(st.session_state.v_last_recording)

            st.divider()
            if st.button("✅ Register voice", type="primary", use_container_width=True, key="reg_voice_btn"):
                if not voice_target:
                    st.error("Select a person first.")
                elif not st.session_state.v_recording_paths:
                    st.error("Record at least 1 voice sample.")
                elif voice_model is None:
                    st.error("Voice model not loaded.")
                else:
                    pdata = protos_now.get(voice_target, {})
                    with st.spinner(f"Building voice prototype for {voice_target}..."):
                        try:
                            summary = voice_model.register_speaker(
                                voice_target,
                                st.session_state.v_recording_paths,
                                relation=pdata.get("relation", ""),
                                hint=pdata.get("hint", f"{voice_target} is someone you know."),
                            )
                            voice_model.save_prototypes(VOICE_PROTOTYPE_PATH)
                            st.session_state.v_recording_paths = []
                            est = summary.get("estimated_confidence", 0.0)
                            st.success(f"✅ Voice registered for **{voice_target}** — {summary['shots']}-shot · est. {est:.0%} confidence")
                        except Exception as e:
                            st.error(f"Voice registration failed: {e}")

        with col_vlist:
            st.markdown('<div class="section-label">Registered voice profiles</div>', unsafe_allow_html=True)
            vp = load_voice_prototypes_raw()
            if not vp:
                st.info("No voice profiles yet.")
            else:
                for name, data in vp.items():
                    with st.expander(f"🎙 {name} · *{data.get('relation','')}*"):
                        shots = data.get("shots", 1)
                        est   = min(0.98, 0.68 + shots * 0.07)
                        st.markdown(f"**Samples:** {shots}-shot  \n**Est. confidence:** {est:.0%}  \n**Hint:** {data.get('hint','—')}")
                        for wav_path in list(Path(VOICE_DIR).glob(f"{name.replace(' ','_')}*.wav"))[:3]:
                            st.audio(str(wav_path))
                        if st.button(f"🗑 Remove voice of {name}", key=f"rm_v_{name}"):
                            if voice_model:
                                voice_model.remove_speaker(name)
                                voice_model.save_prototypes(VOICE_PROTOTYPE_PATH)
                            st.success(f"Removed voice of {name}.")
                            st.rerun()

            st.markdown('<div class="section-label">All voice recordings</div>', unsafe_allow_html=True)
            wavs = sorted(Path(VOICE_DIR).glob("*.wav"), reverse=True)
            if not wavs:
                st.caption("No recordings saved.")
            else:
                for wav in wavs[:8]:
                    with st.expander(wav.stem, expanded=False):
                        st.audio(str(wav))
                        st.caption(f"{wav.stat().st_size // 1024} KB")
                        if st.button(f"🗑 Delete", key=f"del_wav_{wav.stem}"):
                            wav.unlink()
                            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Live recognition
    # ══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Live recognition — Face & Voice with XAI")

        if face_model is None:
            st.error("Face model not loaded.")
        else:
            lv_c1, lv_c2, lv_c3, lv_c4 = st.columns(4)
            with lv_c1: live_run      = st.toggle("▶ Run recognition", value=False, key="live_run_tog")
            with lv_c2: live_voice    = st.toggle("🎙 Voice ID",        value=False, key="live_voice_tog")
            with lv_c3: live_interval = st.slider("Interval (s)", 1.0, 5.0, 2.0, 0.5, key="live_int")
            with lv_c4: show_box      = st.toggle("Face box",           value=True,  key="live_box")

            lv_cam_col, lv_res_col = st.columns([3, 2], gap="large")
            with lv_cam_col:
                st.markdown('<div class="section-label">Camera feed</div>', unsafe_allow_html=True)
                live_cam = st.empty()
            with lv_res_col:
                st.markdown('<div class="section-label">Face result + XAI</div>', unsafe_allow_html=True)
                live_face_ph     = st.empty()
                live_xai_ph      = st.empty()
                st.markdown('<div class="section-label">Voice result + XAI</div>', unsafe_allow_html=True)
                live_voice_ph    = st.empty()
                live_voice_xai_ph = st.empty()

            if "live_log_entries" not in st.session_state:
                st.session_state.live_log_entries = []

            if live_run:
                from face_recogniton.new_person_handler import OpenSetRecogniser
                rec2 = OpenSetRecogniser(face_model)

                live_listener = None
                if live_voice and voice_model:
                    try:
                        from voice_recognition.realtime_listener import RealtimeListener
                        live_listener = RealtimeListener(voice_model)
                        live_listener.start()
                    except Exception as e:
                        live_voice_ph.warning(f"Voice listener: {e}")

                cap2  = cv2.VideoCapture(0)
                last2 = 0.0
                stop2 = st.button("⏹ Stop live", key="stop_live_btn")

                try:
                    while live_run and not stop2:
                        ret, frame = cap2.read()
                        if not ret:
                            break

                        now     = time.time()
                        display = frame.copy()

                        if show_box:
                            try:
                                from PIL import Image as PILImg
                                pil = PILImg.fromarray(frame[:, :, ::-1])
                                boxes, probs = face_model.detector.detect(pil)
                                if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                                    c = float(probs[0])
                                    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                                    col = (80, 200, 80) if c >= 0.85 else (60, 60, 220)
                                    cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                                    cv2.putText(display, f"{c*100:.0f}%", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                            except Exception:
                                pass

                        live_cam.image(frame_to_pil(display), use_container_width=True)

                        if now - last2 >= live_interval:
                            last2  = now
                            result = rec2.recognise(frame)
                            state  = result.get("state", "unknown")
                            name   = result.get("name") or result.get("best_guess", "?")
                            conf   = result.get("confidence", 0.0)
                            rel    = result.get("relation", "")
                            hint   = result.get("hint", "")
                            scores = result.get("all_scores", {})

                            if state == "unknown":
                                rec2.handle_unknown(frame, result)

                            # FIX: use single-call helpers here too
                            live_face_ph.markdown(
                                build_face_result_html(state, name, rel, conf, hint),
                                unsafe_allow_html=True,
                            )
                            if scores:
                                live_xai_ph.markdown(render_xai_face(scores), unsafe_allow_html=True)

                            if live_listener:
                                vr = live_listener.get_latest_result()
                                if vr:
                                    vstate  = vr.get("state", "unknown")
                                    vname   = vr.get("name") or vr.get("best_guess", "?")
                                    vconf   = vr.get("confidence", 0.0)
                                    vtxt    = vr.get("transcript", "")
                                    vscores = vr.get("all_scores", {})

                                    live_voice_ph.markdown(
                                        build_voice_result_html(vstate, vname, vconf, vtxt),
                                        unsafe_allow_html=True,
                                    )
                                    if vscores:
                                        live_voice_xai_ph.markdown(render_xai_voice(vscores), unsafe_allow_html=True)

                            st.session_state.live_log_entries.insert(0, {
                                "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                                "state": state,
                                "name":  name,
                                "conf":  f"{conf*100:.0f}%",
                            })
                            st.session_state.live_log_entries = st.session_state.live_log_entries[:25]

                        time.sleep(0.04)
                finally:
                    cap2.release()
                    if live_listener:
                        live_listener.stop()

            if st.session_state.live_log_entries:
                st.markdown('<div class="section-label">Recognition log</div>', unsafe_allow_html=True)
                import pandas as pd
                st.dataframe(pd.DataFrame(st.session_state.live_log_entries), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — Manage people
    # ══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Registered people")

        protos  = load_prototypes_raw()
        vprotos = load_voice_prototypes_raw()

        if not protos:
            st.info("No relatives registered yet. Use the 'Register face' tab.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-tile"><div class="val">{len(protos)}</div><div class="lbl">Face profiles</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><div class="val">{len(vprotos)}</div><div class="lbl">Voice profiles</div></div>', unsafe_allow_html=True)
            avg_s  = sum(v.get("shots", 0) for v in protos.values()) / max(len(protos), 1)
            avg_vs = sum(v.get("shots", 0) for v in vprotos.values()) / max(len(vprotos), 1) if vprotos else 0
            m3.markdown(f'<div class="metric-tile"><div class="val">{avg_s:.1f}</div><div class="lbl">Avg face shots</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-tile"><div class="val">{avg_vs:.1f}</div><div class="lbl">Avg voice shots</div></div>', unsafe_allow_html=True)

            st.divider()

            for name, data in protos.items():
                has_voice = name in vprotos
                vdata     = vprotos.get(name, {})
                with st.expander(f"{'👤🎙' if has_voice else '👤'} {name} · *{data.get('relation','')}*"):
                    pc1, pc2 = st.columns([2, 1])
                    with pc1:
                        st.markdown(f"**Relationship:** {data.get('relation','—')}")
                        st.markdown(f"**Voice hint:** {data.get('hint','—')}")
                        face_conf = min(0.98, 0.70 + data.get("shots", 1) * 0.06)
                        st.markdown(f"**Face shots:** {data.get('shots','—')} · est. {face_conf:.0%} confidence")
                        if has_voice:
                            vs = vdata.get("shots", 1)
                            vc = min(0.98, 0.68 + vs * 0.07)
                            st.markdown(f"**Voice shots:** {vs} · est. {vc:.0%} confidence")
                        else:
                            st.markdown("**Voice:** ⚪ *not registered — go to Register voice tab*")
                    with pc2:
                        person_dir = os.path.join(CAPTURE_DIR, name.replace(" ", "_"))
                        if os.path.isdir(person_dir):
                            for img_path in list(Path(person_dir).glob("*.jpg"))[:3]:
                                st.image(str(img_path), width=80)
                        rb1, rb2 = st.columns(2)
                        if rb1.button(f"🗑 Face",  key=f"rm_f_{name}"):
                            if face_model:
                                face_model.remove_relative(name)
                                face_model.save_prototypes(PROTOTYPE_PATH)
                            st.success(f"Removed face of {name}.")
                            st.rerun()
                        if rb2.button(f"🗑 Voice", key=f"rm_v2_{name}", disabled=not has_voice):
                            if voice_model and has_voice:
                                voice_model.remove_speaker(name)
                                voice_model.save_prototypes(VOICE_PROTOTYPE_PATH)
                            st.success(f"Removed voice of {name}.")
                            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — Unknown alerts
    # ══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Unknown face alerts")
        st.caption("Faces the system could not recognise — review and register if needed.")

        if not os.path.exists(ALERT_LOG_PATH):
            st.info("No unknown face alerts yet. The system will log unrecognised faces here.")
        else:
            with open(ALERT_LOG_PATH) as f:
                alerts = json.load(f)

            unreviewed = [a for a in alerts if not a.get("reviewed")]
            reviewed   = [a for a in alerts if a.get("reviewed")]

            if unreviewed:
                st.markdown(f'<div class="alert-banner">⚠️ <b>{len(unreviewed)}</b> unreviewed alert(s) need your attention.</div>', unsafe_allow_html=True)
            else:
                st.success(f"✅ All {len(reviewed)} alert(s) reviewed.")

            for alert in unreviewed:
                ts     = alert.get("timestamp", "")
                conf   = alert.get("confidence", 0.0)
                img    = alert.get("image_path", "")
                scores = alert.get("all_scores", {})

                with st.expander(f"🔴 {ts[:19]}  — best score: {conf*100:.0f}%"):
                    ac1, ac2 = st.columns([1, 2])
                    with ac1:
                        if img and os.path.exists(img):
                            st.image(img, width=130, caption="Unrecognised face")
                        else:
                            st.caption("Image not available")
                    with ac2:
                        st.markdown(f"**Time:** {ts[:19]}")
                        st.markdown(f"**Best similarity score:** {conf*100:.0f}%")
                        if scores:
                            st.markdown(render_xai_face(scores, "Why was this unknown?"), unsafe_allow_html=True)

                        new_name = st.text_input("Name this person", key=f"aname_{ts}")
                        new_rel  = st.text_input("Relationship",     key=f"arel_{ts}")
                        ab1, ab2 = st.columns(2)
                        if ab1.button("✅ Register & close", key=f"areg_{ts}"):
                            if new_name and img and os.path.exists(img) and face_model:
                                try:
                                    face_model.register_relative(new_name, [img], relation=new_rel, hint=f"{new_name} is your {new_rel}.")
                                    face_model.save_prototypes(PROTOTYPE_PATH)
                                    for a in alerts:
                                        if a.get("timestamp") == ts:
                                            a["reviewed"] = True
                                    with open(ALERT_LOG_PATH, "w") as f:
                                        json.dump(alerts, f, indent=2)
                                    st.success(f"Registered {new_name}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        if ab2.button("✓ Dismiss", key=f"adismiss_{ts}"):
                            for a in alerts:
                                if a.get("timestamp") == ts:
                                    a["reviewed"] = True
                            with open(ALERT_LOG_PATH, "w") as f:
                                json.dump(alerts, f, indent=2)
                            st.rerun()

            if reviewed:
                with st.expander(f"✅ {len(reviewed)} reviewed alert(s)"):
                    for alert in reviewed[-8:]:
                        st.caption(f"{alert.get('timestamp','')[:19]} · reviewed")