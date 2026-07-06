# AccentCoach

AccentCoach is a pronunciation coaching app. Users record themselves speaking a prompt, and the app transcribes the audio, scores it against the target sentence, and gives per-phoneme feedback on pronunciation and accent.

## Project Structure

- `src/` — React (TypeScript) frontend. Key pieces: `Recorder.tsx` (audio capture), `PhonemeAnalyzer.tsx` / `PhonemeFeedback.tsx` (results UI), `firebase/` (auth), `services/phonemeApi.ts` (backend client).
- `server/` — Flask backend (`accent_server.py`). Exposes `/predict-accent` and `/verify-sentence`, plus phoneme analysis routes in `phoneme_blueprint.py` (`/analyze-pronunciation`). Uses `faster-whisper` for transcription, `librosa` for audio features, and a scikit-learn model (`model.joblib`) for accent prediction.
- `training/train.py` — trains the accent classification model and produces `model.joblib`.

## Setup

### Frontend

```bash
npm install
npm start
```

Runs the React app at [http://localhost:3000](http://localhost:3000).

### Backend

```bash
cd server
pip install -r requirements.txt
python accent_server.py
```

Runs the Flask API at `http://localhost:5001`. Requires `ffmpeg` on your PATH (used by `pydub` for audio conversion).

### Model training

```bash
cd training
python train.py
```

Trains a new model and writes it to `model.joblib`.

## Scripts

- `npm start` — run the frontend in development mode
- `npm test` — run the frontend test suite
- `npm run build` — build the frontend for production
