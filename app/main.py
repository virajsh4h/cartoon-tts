# app/main.py
"""
Kids TTS API (original character-style voices only).
Pipeline: gTTS -> chunking -> pyrubberband pitch+time -> pydub reverb+normalize -> return WAV
Notes:
 - Languages supported: 'en', 'hi', 'mr'
 - Keep voices original and do NOT attempt to impersonate copyrighted characters.
"""

import os
import uuid
import shutil
import random
import re
import tempfile
import logging
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# audio libs
from gtts import gTTS
import soundfile as sf
import numpy as np
from pydub import AudioSegment, effects

# formant-preserving transforms
import pyrubberband as pyrb

# Basic config
ROOT = tempfile.gettempdir()
RAW_ROOT = os.path.join(ROOT, "kids_tts_raw")
OUT_ROOT = os.path.join(ROOT, "kids_tts_out")
os.makedirs(RAW_ROOT, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

# concurrency limit (avoid CPU exhaustion)
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "2"))
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Simple API key support (set API_KEY env var for production)
API_KEY = os.environ.get("API_KEY", None)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kids-tts")

app = FastAPI(title="Kids TTS API (original character-style)")

# ---- request/response models ----
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    lang: str = Field("en", description="Language code: 'en', 'hi', 'mr'")
    preset: Optional[str] = Field(None, description="Named preset to use")
    # optional overrides (fine-grained control)
    semitones: Optional[float] = None
    base_speed: Optional[float] = None
    semitone_jitter: Optional[float] = None
    speed_jitter: Optional[float] = None
    reverb_db: Optional[float] = None

class PresetInfo(BaseModel):
    name: str
    settings: Dict[str, Any]
    notes: str

# ---- default presets ----
PRESETS = {
    "sparky": {"base_semitones": 3.8, "base_speed": 1.10, "semitone_jitter": 0.8, "speed_jitter": 0.06, "reverb_db": 9},
    "mellow": {"base_semitones": 1.6, "base_speed": 0.98, "semitone_jitter": 0.35, "speed_jitter": 0.02, "reverb_db": 6},
    "zippy":  {"base_semitones": 4.5, "base_speed": 1.16, "base_semitone_jitter": 0.9, "speed_jitter": 0.08, "reverb_db": 10},
    "gentle": {"base_semitones": 1.2, "base_speed": 0.95, "semitone_jitter": 0.4, "speed_jitter": 0.025, "reverb_db": 5},
}

# Marathi-specific safer presets (gentler shifts)
MARATHI_PRESETS = {
    "sparky_mr": {"base_semitones": 2.6, "base_speed": 1.06, "semitone_jitter": 0.4, "speed_jitter": 0.03, "reverb_db": 7},
    "mellow_mr": {"base_semitones": 1.2, "base_speed": 0.98, "semitone_jitter": 0.25, "speed_jitter": 0.02, "reverb_db": 5},
}

# helper: validated languages
ALLOWED_LANGS = {"en", "hi", "mr"}

# ---- audio utilities ----
def chunk_text(text: str):
    """Split text into chunks around punctuation for expressive prosody."""
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
    """Use pyrubberband to apply formant-preserving pitch shift and time stretch."""
    if abs(n_steps) > 1e-6:
        y = pyrb.pitch_shift(y, sr, n_steps)
    if abs(speed - 1.0) > 1e-3:
        stretch_factor = 1.0 / float(speed)
        y = pyrb.time_stretch(y, sr, stretch_factor)
    return y

def add_light_reverb_and_normalize(seg: AudioSegment, reverb_db: float = 8):
    seg = effects.normalize(seg)
    delayed = seg - int(reverb_db)
    res = seg.overlay(delayed, position=40)
    return effects.normalize(res)

def generate_unique_dir(prefix: str):
    d = os.path.join(RAW_ROOT, f"{prefix}_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d

# This function runs the heavy work in a thread (pyrb/pydub are blocking)
def expressive_pipeline_to_file(text: str, lang: str, out_path: str, settings: Dict[str, float]):
    """
    text -> generate WAV at out_path
    settings: expected keys: base_semitones, base_speed, semitone_jitter, speed_jitter, reverb_db
    """
    tmp_dir = generate_unique_dir("req")
    try:
        chunks = chunk_text(text)
        segs = []
        sr = 22050
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            # 1) synth chunk with gTTS
            tmp_in = os.path.join(tmp_dir, f"g_{uuid.uuid4().hex[:6]}.wav")
            tts = gTTS(text=ch, lang=lang, slow=False)
            tts.save(tmp_in)

            # 2) load numeric array
            y, _ = sf.read(tmp_in, dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # 3) jittered transform
            sem = settings["base_semitones"] + random.uniform(-settings["semitone_jitter"], settings["semitone_jitter"])
            spd = settings["base_speed"] + random.uniform(-settings["speed_jitter"], settings["speed_jitter"])

            try:
                y2 = rb_pitch_and_time(y, sr, n_steps=sem, speed=spd)
            except Exception as e:
                # on rare pyrb issues, fallback to using original audio and slightly resampling
                y2 = y

            tmp_proc = tmp_in.replace(".wav", ".proc.wav")
            sf.write(tmp_proc, y2, sr, subtype='PCM_16')
            seg = AudioSegment.from_wav(tmp_proc)
            segs.append(seg)
            # natural silence after a chunk
            segs.append(AudioSegment.silent(duration=60))

        final = sum(segs) if segs else AudioSegment.silent(duration=300)
        final = add_light_reverb_and_normalize(final, reverb_db=settings["reverb_db"])
        final.export(out_path, format="wav")
        return out_path
    finally:
        # cleanup temporary chunk files & folder
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

def pick_settings(lang: str, preset: Optional[str], overrides: Dict[str, Optional[float]]):
    """Resolve settings: use preset (with Marathi-specific fallback) then apply explicit overrides."""
    settings = {}
    # start from base
    if lang == "mr":
        # prefer marathi presets if requested
        if preset and preset in MARATHI_PRESETS:
            settings = MARATHI_PRESETS[preset].copy()
        elif preset and preset in PRESETS:
            # if user selected general preset, adapt it gently for mr
            base = PRESETS[preset].copy()
            base["base_semitones"] = max(1.2, base["base_semitones"] - 1.2)
            settings = base
        else:
            # default marathi
            settings = MARATHI_PRESETS["sparky_mr"].copy()
    else:
        # non-marathi languages
        if preset and preset in PRESETS:
            settings = PRESETS[preset].copy()
        else:
            settings = PRESETS["sparky"].copy()

    # apply overrides if provided
    if overrides.get("semitones") is not None:
        settings["base_semitones"] = float(overrides["semitones"])
    if overrides.get("base_speed") is not None:
        settings["base_speed"] = float(overrides["base_speed"])
    if overrides.get("semitone_jitter") is not None:
        settings["semitone_jitter"] = float(overrides["semitone_jitter"])
    if overrides.get("speed_jitter") is not None:
        settings["speed_jitter"] = float(overrides["speed_jitter"])
    if overrides.get("reverb_db") is not None:
        settings["reverb_db"] = float(overrides["reverb_db"])

    # safety clamps
    settings["base_semitones"] = float(max(-6.0, min(6.0, settings["base_semitones"])))
    settings["base_speed"] = float(max(0.6, min(1.6, settings["base_speed"])))
    settings["semitone_jitter"] = float(max(0.0, min(2.5, settings.get("semitone_jitter", 0.5))))
    settings["speed_jitter"] = float(max(0.0, min(0.25, settings.get("speed_jitter", 0.05))))
    settings["reverb_db"] = float(max(0.0, min(18.0, settings.get("reverb_db", 8.0))))

    return settings

# ---- API endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "max_concurrent_tasks": MAX_CONCURRENT_TASKS}

@app.get("/presets")
def list_presets():
    presets = []
    for k, v in PRESETS.items():
        presets.append(PresetInfo(name=k, settings=v, notes="generic"))
    for k, v in MARATHI_PRESETS.items():
        presets.append(PresetInfo(name=k, settings=v, notes="marathi-tuned"))
    return {"presets": presets}

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest, x_api_key: Optional[str] = Header(None)):
    # Optional API key check
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="missing/invalid API key")

    # Basic validation
    if not (req.text and req.text.strip()):
        raise HTTPException(status_code=400, detail="text must be non-empty")
    lang = req.lang.strip().lower()
    if lang not in ALLOWED_LANGS:
        raise HTTPException(status_code=400, detail=f"lang must be one of {list(ALLOWED_LANGS)}")

    # Prepare settings (preset + overrides)
    overrides = {
        "semitones": req.semitones,
        "base_speed": req.base_speed,
        "semitone_jitter": req.semitone_jitter,
        "speed_jitter": req.speed_jitter,
        "reverb_db": req.reverb_db,
    }
    # if user requested preset use it, else None -> pick default
    settings = pick_settings(lang, req.preset, overrides)

    # create output path
    out_file = os.path.join(OUT_ROOT, f"{uuid.uuid4().hex}.wav")

    # throttle concurrency and run heavy blocking work safely
    async with task_semaphore:
        try:
            # run pipeline in threadpool to avoid blocking asyncio loop
            await run_in_threadpool(expressive_pipeline_to_file, req.text, lang, out_file, settings)
        except Exception as e:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {str(e)}")

    # Return file response (caller should download or stream). We don't delete out_file immediately
    return FileResponse(out_file, media_type="audio/wav", filename="speech.wav")
