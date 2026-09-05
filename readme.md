# Real-Time Face Mesh + Gemini Face Analysis

## What changed from the original version

1. **MediaPipe: migrated off the removed legacy API.**
   `mp.solutions.face_mesh` was deprecated back in 2023, and current
   `mediapipe` wheels (0.10.x) don't ship `mediapipe.solutions` at all
   anymore — the original script would fail with
   `ImportError: cannot import name 'solutions' from 'mediapipe'`.
   This version uses the current Tasks API
   (`mediapipe.tasks.python.vision.FaceLandmarker`, `VIDEO` running mode),
   with drawing helpers from their new home
   (`mediapipe.tasks.python.vision.drawing_utils` / `drawing_styles`).
   The face mesh model bundle is downloaded once and cached in `models/`.

2. **Gemini: migrated off a retired model, and off manual JSON parsing.**
   `gemini-1.5-flash` has been shut down by Google (all 1.0/1.5 models
   return 404 now), and `gemini-2.5-flash` is scheduled for shutdown in
   October 2026. The default model is now `gemini-3.5-flash` (GA,
   current as of this writing) — override it any time with the
   `GEMINI_MODEL` env var if it's retired by the time you run this.
   Instead of `json.loads(response.text)` (which throws on any
   malformed output), this uses the SDK's structured-output support —
   a Pydantic `FaceAttributes` schema is passed as `response_schema`,
   and the SDK validates and parses it for you (`response.parsed`).

3. **Concurrency cleanup.** The manual `threading.Thread(daemon=True)` +
   boolean flag was replaced with a single-worker `ThreadPoolExecutor` and
   a `Future`, which serializes analysis calls and cleans up
   deterministically on shutdown (`executor.shutdown(cancel_futures=True)`).

4. **Minor fixes:** removed a pointless `int(getattr(cv2, ...))` dance for
   the font constant, added a `close()` method usable from any exit path,
   and made the Face Landmarker's model path/URL and confidence knobs
   configurable in one place.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and paste in your Gemini API key
```

## Run

```bash
python face_analysis_app.py
```

Press `q` in the video window to quit. The face mesh model
(`face_landmarker.task`, ~4 MB) downloads automatically to `models/` the
first time you run it — that needs an internet connection once.

## Notes

- Get a Gemini API key from https://aistudio.google.com/apikey.
- If Google retires `gemini-3.5-flash` after this was written, set
  `GEMINI_MODEL=<current-flash-model>` in your `.env` — check
  https://ai.google.dev/gemini-api/docs/models for the current list.
- `API_COOLDOWN_SECONDS` in `Config` controls how often the app calls
  Gemini (default: every 4s per detected face) — lower it for more
  frequent updates at higher API cost.