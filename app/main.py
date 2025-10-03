# main.py

import os
import uuid
import shutil
import random
import re
import tempfile
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gtts import gTTS
import soundfile as sf
import numpy as np
from pydub import AudioSegment, effects
import pyrubberband as pyrb  # uses system rubberband binary/library

# ---------- Basic config ----------
ROOT_TMP = tempfile.gettempdir()
RAW_ROOT = os.path.join(ROOT_TMP, "kids_tts_raw")
OUT_ROOT = os.path.join(ROOT_TMP, "kids_tts_out")
os.makedirs(RAW_ROOT, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

# Limit concurrent heavy work (avoid overloading your laptop)
MAX_CONCURRENT_TASKS = 2
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kids-tts-minimal")

app = FastAPI(title="Kids TTS Minimal")

# ---------- Request model ----------
class SynthesisRequest(BaseModel):
    text: str
    lang: str  # 'en', 'hi', 'mr'


# ---------- Utilities ----------
def chunk_text(text: str):
    """Split into smaller phrases on punctuation or every ~6 words."""
    parts = re.split(r'(?<=[,;:.!?])\s+', text.strip())
    if len(parts) == 1:
        words = parts[0].split()
        if len(words) <= 8:
            return parts
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+6])
            chunks.append(chunk)
            i += 6
        return chunks
    return parts

def rb_pitch_and_time(y: np.ndarray, sr: int, n_steps: float = 0.0, speed: float = 1.0):
    """Formant-preserving pitch shift + time-stretch via pyrubberband."""
    if abs(n_steps) > 1e-6:
        y = pyrb.pitch_shift(y, sr, n_steps)
    if abs(speed - 1.0) > 1e-3:
        stretch_factor = 1.0 / float(speed)
        y = pyrb.time_stretch(y, sr, stretch_factor)
    return y

def add_light_reverb_and_normalize(seg: AudioSegment, reverb_db: float = 8.0):
    seg = effects.normalize(seg)
    delayed = seg - int(reverb_db)
    res = seg.overlay(delayed, position=40)  # 40ms delayed overlay
    return effects.normalize(res)

def generate_tmp_dir():
    d = os.path.join(RAW_ROOT, f"req_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d

# ---------- Hardcoded presets (eaxh language) ----------
# These are intentionally conservative and editable here.
HARDCODED_PRESETS = {
    # ENG can be in Roman script
    "en": {"base_semitones": 3.6, "base_speed": 1.3, "semitone_jitter": 0.7, "speed_jitter": 0.05, "reverb_db": 9},
    # Marathi & hindi needs gentler changes and DEVANAGRI text input for best results
    "hi": {"base_semitones": 3.0, "base_speed": 1.06, "semitone_jitter": 0.4, "speed_jitter": 0.04, "reverb_db": 8},
    "mr": {"base_semitones": 2.4, "base_speed": 1.05, "semitone_jitter": 0.35, "speed_jitter": 0.03, "reverb_db": 7},
}

def clamp_settings(s: dict):
    s["base_semitones"] = float(max(-6.0, min(6.0, s["base_semitones"])))
    s["base_speed"] = float(max(0.6, min(1.6, s["base_speed"])))
    s["semitone_jitter"] = float(max(0.0, min(2.5, s["semitone_jitter"])))
    s["speed_jitter"] = float(max(0.0, min(0.25, s["speed_jitter"])))
    s["reverb_db"] = float(max(0.0, min(18.0, s["reverb_db"])))
    return s

# ---------- Core expressive pipeline ----------
def expressive_pipeline_to_file(text: str, lang: str, out_path: str, settings: dict):
    tmp_dir = generate_tmp_dir()
    try:
        chunks = chunk_text(text)
        segs = []
        sr = 22050
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            # 1) synth chunk using gTTS
            tmp_in = os.path.join(tmp_dir, f"g_{uuid.uuid4().hex[:6]}.wav")
            tts = gTTS(text=ch, lang=lang, slow=False)
            tts.save(tmp_in)

            # 2) load array
            y, _ = sf.read(tmp_in, dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # 3) jitter + transform
            sem = settings["base_semitones"] + random.uniform(-settings["semitone_jitter"], settings["semitone_jitter"])
            spd = settings["base_speed"] + random.uniform(-settings["speed_jitter"], settings["speed_jitter"])
            try:
                y2 = rb_pitch_and_time(y, sr, n_steps=sem, speed=spd)
            except Exception as e:
                # fallback to original if rubberband hiccups
                y2 = y

            tmp_proc = tmp_in.replace(".wav", ".proc.wav")
            sf.write(tmp_proc, y2, sr, subtype='PCM_16')
            seg = AudioSegment.from_wav(tmp_proc)
            segs.append(seg)
            segs.append(AudioSegment.silent(duration=60))

        final = sum(segs) if segs else AudioSegment.silent(duration=300)
        final = add_light_reverb_and_normalize(final, reverb_db=settings["reverb_db"])
        final.export(out_path, format="wav")
        return out_path
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

# ---------- API endpoints ----------
@app.get("/health")
def health():
    return {"status": "200 ok", "tmp_raw": RAW_ROOT, "tmp_out": OUT_ROOT}

@app.post("/synthesize")
async def synth(req: SynthesisRequest):
    text = (req.text or "").strip()
    lang = (req.lang or "").strip().lower()
    if not text:
        raise HTTPException(status_code=400, detail="text must be provided for synthesis")
    if lang not in HARDCODED_PRESETS:
        raise HTTPException(status_code=400, detail=f"lang must be one of {list(HARDCODED_PRESETS.keys())}")

    # pick hardcoded settings for lang
    settings = HARDCODED_PRESETS[lang].copy()
    settings = clamp_settings(settings)

    out_file = os.path.join(OUT_ROOT, f"{uuid.uuid4().hex}.wav")

    async with task_semaphore:
        # run blocking, heavy work in threadpool
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, expressive_pipeline_to_file, text, lang, out_file, settings)
        except Exception as e:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {str(e)}")

    return FileResponse(out_file, media_type="audio/wav", filename="speech.wav")
