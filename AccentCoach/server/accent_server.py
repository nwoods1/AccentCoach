from flask import Flask, request, jsonify
from flask_cors import CORS
import os, tempfile, uuid, traceback, json, re
import numpy as np
import joblib
import librosa
import difflib
STRICT_VERIFY = os.environ.get("STRICT_VERIFY", "1") == "1"  # default ON


# --- Optional: transcode webm->wav (requires pydub + ffmpeg) ---
try:
    from pydub import AudioSegment
    PYDUB = True
except Exception:
    PYDUB = False

# --- Optional: pronunciation blueprint (if you ever add it later) ---
try:
    from phoneme_blueprint import bp as phoneme_bp  # safe-if-present
    HAVE_PHONEME = True
except Exception:
    HAVE_PHONEME = False

# --- Optional: transcription (faster-whisper) ---
try:
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False

# --- Env toggles ---
DEBUG_ACCENT   = os.environ.get("DEBUG_ACCENT", "0") == "1"
CONVERT_TO_WAV = os.environ.get("CONVERT_TO_WAV", "1") == "1"  # default ON for consistency
FLIP_LABELS    = os.environ.get("FLIP_LABELS", "0") == "1"     # flips 0<->1 after argmax

# Verification thresholds
MIN_COVERAGE = float(os.environ.get("MIN_COVERAGE", "0.8"))  # fraction of expected words present (order-aware)
MAX_WER      = float(os.environ.get("MAX_WER", "0.5"))       # max allowed word error rate

# --- App ---
app = Flask(__name__)
CORS(app)

if HAVE_PHONEME:
    app.register_blueprint(phoneme_bp)

# --- Load model ---
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
model = joblib.load(MODEL_PATH)
print("MODEL TYPE:", type(model))
classes_attr = getattr(model, "classes_", None)
print("MODEL CLASSES:", classes_attr)
for attr in ("n_features_in_",):
    if hasattr(model, attr):
        print(f"MODEL expected {attr}:", getattr(model, attr))

# --- Feature extractors for /predict-accent ---
SR = 16000

def extract_basic(path: str) -> np.ndarray:
    """13 MFCC mean — your original RF used this."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    min_len = int(0.5 * SR)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0).astype(np.float32)
    if DEBUG_ACCENT:
        print(f"[features/basic] len={feat.shape[0]} first3={feat[:3]}")
    return feat

def _stats(v: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean(v), np.std(v), np.median(v),
        np.percentile(v, 5), np.percentile(v, 95)
    ], dtype=np.float32)

def _agg(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([_stats(c) for c in mat], axis=0).astype(np.float32)

def extract_rich(path: str, n_mfcc: int = 20) -> np.ndarray:
    """
    Rich feature used by the SVC pipeline: 320 dims
    (5 stats * 20 coeffs * 3 [mfcc,d1,d2]) + 4 bands * 5 stats
    """
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    min_len = int(0.5 * SR)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n_mfcc)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    zcr  = librosa.feature.zero_crossing_rate(y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.95)[0]

    feat = np.concatenate([
        _agg(mfcc), _agg(d1), _agg(d2),
        _stats(zcr), _stats(cent), _stats(bw), _stats(roll)
    ], axis=0).astype(np.float32)

    bad = ~np.isfinite(feat)
    if bad.any():
        feat[bad] = 0.0
    if DEBUG_ACCENT:
        print(f"[features/rich] len={feat.shape[0]} first3={feat[:3]}")
    return feat

def _is_pipeline_estimator(m) -> bool:
    return hasattr(m, "named_steps") and isinstance(getattr(m, "named_steps"), dict)

def _predict_with(feat: np.ndarray) -> dict:
    x = feat.reshape(1, -1)
    info = {"proba": None, "classes": None, "label": None, "confidence": None}
    try:
        proba = model.predict_proba(x)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None and hasattr(classes, "tolist"):
            classes = classes.tolist()
        info["proba"] = [float(p) for p in proba]
        info["classes"] = classes if classes is not None else [0, 1]

        idx = int(np.argmax(proba))
        pred_class = info["classes"][idx] if info["classes"] else idx
        if FLIP_LABELS:
            pred_class = 1 - int(pred_class)
        info["label"] = "Native" if int(pred_class) == 0 else "Non-Native"
        info["confidence"] = float(proba[idx])
        return info
    except Exception as e:
        if DEBUG_ACCENT:
            print("[_predict_with] predict_proba failed; fallback to predict:", repr(e))

    raw = int(model.predict(x)[0])
    if FLIP_LABELS:
        raw = 1 - raw
    info["label"] = "Native" if raw == 0 else "Non-Native"
    info["confidence"] = None
    return info

# ---------------------------
# /predict-accent (existing)
# ---------------------------
@app.route("/predict-accent", methods=["POST"])
def predict_accent():
    audio_path = None
    wav_path = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No 'audio' file provided"}), 400

        f = request.files["audio"]
        tmpdir = tempfile.gettempdir()
        uid = uuid.uuid4().hex
        audio_path = os.path.join(tmpdir, f"{uid}.webm")
        f.save(audio_path)

        path_for_librosa = audio_path
        if CONVERT_TO_WAV:
            if not PYDUB:
                return jsonify({"error": "CONVERT_TO_WAV=1 but pydub/ffmpeg not available"}), 500
            wav_path = os.path.join(tmpdir, f"{uid}.wav")
            AudioSegment.from_file(audio_path).set_frame_rate(SR).set_channels(1).export(wav_path, format="wav")
            path_for_librosa = wav_path

        use_rich_first = _is_pipeline_estimator(model)
        tried = []
        result = None
        feat_len = None
        chosen = None

        for choice in ([extract_rich, extract_basic] if use_rich_first else [extract_basic, extract_rich]):
            name = "rich" if choice is extract_rich else "basic"
            tried.append(name)
            try:
                feat = choice(path_for_librosa)
                feat_len = int(feat.shape[0])
                result = _predict_with(feat)
                chosen = name
                break
            except ValueError as ve:
                if DEBUG_ACCENT:
                    print(f"[predict] extractor '{name}' failed ({repr(ve)}). Trying other extractor...")
                continue

        if result is None:
            feat = extract_basic(path_for_librosa)
            feat_len = int(feat.shape[0])
            result = _predict_with(feat)
            chosen = "basic(last-ditch)"

        resp = {"result": result["label"], "confidence": result["confidence"]}

        if DEBUG_ACCENT or request.args.get("debug") == "1":
            resp["_debug"] = {
                "chosen_extractor": chosen,
                "tried_order": tried,
                "feature_length": feat_len,
                "classes": result.get("classes"),
                "proba": result.get("proba"),
                "flip_labels": FLIP_LABELS,
                "convert_to_wav": CONVERT_TO_WAV,
                "pydub_available": PYDUB,
                "upload": {
                    "filename": getattr(f, "filename", None),
                    "mimetype": getattr(f, "mimetype", None),
                    "size": os.path.getsize(audio_path) if audio_path and os.path.exists(audio_path) else None
                }
            }
        return jsonify(resp)

    except Exception as e:
        print("ERROR:", str(e))
        if DEBUG_ACCENT:
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass



# lazy model handle (so startup stays fast)
_WHISPER = None
_WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "tiny")  # tiny | base | small (CPU-friendly first)
_WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

def _load_whisper():
    global _WHISPER
    if _WHISPER is None:
        if not HAVE_WHISPER:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper")
        # you can change compute_type to "int8_float16" if you have AVX2
        _WHISPER = WhisperModel(_WHISPER_SIZE, device=_WHISPER_DEVICE, compute_type="int8")
    return _WHISPER

_WORD_RE = re.compile(r"[a-z]+'?[a-z]+|[a-z]+")

def _norm_text(t: str) -> str:
    t = t.lower()
    # remove punctuation, keep apostrophes within words
    return " ".join(_WORD_RE.findall(t))

def _tokens(t: str):
    return _norm_text(t).split()

def word_error_rate(ref_tokens, hyp_tokens) -> float:
    # standard DP WER (insertions+deletions+subs)/N
    n = len(ref_tokens); m = len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m] / max(1, n)

def coverage_in_order(ref_tokens, hyp_tokens) -> float:
    # how many ref tokens appear in order in hyp (LCS / len(ref))
    sm = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    matches = sum(tr.size for tr in sm.get_matching_blocks())
    if len(ref_tokens) == 0:
        return 1.0
    # SequenceMatcher adds a zero-size sentinel block, hence min(...)
    matches = min(matches, len(ref_tokens))
    return matches / len(ref_tokens)

def diff_words(ref_tokens, hyp_tokens):
    # return (missing, extra) lists
    sm = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    missing = []
    extra = []
    i = j = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            missing.extend(ref_tokens[i1:i2])
        elif tag == "insert":
            extra.extend(hyp_tokens[j1:j2])
        elif tag == "replace":
            missing.extend(ref_tokens[i1:i2])
            extra.extend(hyp_tokens[j1:j2])
        elif tag == "equal":
            pass
    return missing, extra

@app.route("/verify-sentence", methods=["POST"])
def verify_sentence():
    audio_path = None
    wav_path = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No 'audio' file provided"}), 400
        expected = request.form.get("expected", "")
        if not expected.strip():
            return jsonify({"error": "No 'expected' sentence provided"}), 400

        if not HAVE_WHISPER:
            return jsonify({"error": "faster-whisper not installed. pip install faster-whisper"}), 500

        f = request.files["audio"]
        tmpdir = tempfile.gettempdir()
        uid = uuid.uuid4().hex
        audio_path = os.path.join(tmpdir, f"{uid}.webm")
        f.save(audio_path)

        path_for_asr = audio_path
        if CONVERT_TO_WAV:
            if not PYDUB:
                return jsonify({"error": "CONVERT_TO_WAV=1 but pydub/ffmpeg not available"}), 500
            wav_path = os.path.join(tmpdir, f"{uid}.wav")
            AudioSegment.from_file(audio_path).set_frame_rate(SR).set_channels(1).export(wav_path, format="wav")
            path_for_asr = wav_path

        # Transcribe
        model = _load_whisper()
        segments, info = model.transcribe(path_for_asr, beam_size=1, vad_filter=True)
        hyp_text = "".join(seg.text for seg in segments) if segments else ""
        hyp_norm = _norm_text(hyp_text)
        ref_norm = _norm_text(expected)

        ref_toks = ref_norm.split()
        hyp_toks = hyp_norm.split()

        wer = word_error_rate(ref_toks, hyp_toks)
        cov = coverage_in_order(ref_toks, hyp_toks)
        missing, extra = diff_words(ref_toks, hyp_toks)

        
        if STRICT_VERIFY:
            # Exact match after normalization: same words, same order, no extras.
            spoken_ok = (ref_toks == hyp_toks)
        else:
            # Fuzzy fallback (old behavior)
            spoken_ok = (cov >= MIN_COVERAGE) and (wer <= MAX_WER) and (len(missing) == 0)


        resp = {
            "transcript": hyp_text.strip(),
            "normalizedTranscript": hyp_norm,
            "expectedNormalized": ref_norm,
            "coverage": round(cov, 3),
            "wer": round(wer, 3),
            "missing": missing,
            "extra": extra,
            "spoken_ok": bool(spoken_ok),
        }

        if DEBUG_ACCENT or request.args.get("debug") == "1":
            resp["_debug"] = {
                "model_size": _WHISPER_SIZE,
                "device": _WHISPER_DEVICE,
                "convert_to_wav": CONVERT_TO_WAV,
                "pydub_available": PYDUB,
                "upload": {
                    "filename": getattr(f, "filename", None),
                    "mimetype": getattr(f, "mimetype", None),
                    "size": os.path.getsize(audio_path) if audio_path and os.path.exists(audio_path) else None
                }
            }

        return jsonify(resp)

    except Exception as e:
        print("ERROR(/verify-sentence):", str(e))
        if DEBUG_ACCENT:
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(port=5001, host="0.0.0.0", debug=True)


# from flask import Flask, request, jsonify
# import os, io, uuid, tempfile, traceback
# import numpy as np
# import joblib
# import librosa
# from flask_cors import CORS
# from phoneme_blueprint import bp as phoneme_bp



# try:
#     from pydub import AudioSegment
#     HAVE_PYDUB = True
# except Exception:
#     HAVE_PYDUB = False

# app = Flask(__name__)
# CORS(app)

# #app.register_blueprint(phoneme_bp)

# # Config
# SR_TARGET = 16000
# N_MFCC = 20  # must match your current training
# CONVERT_TO_WAV = os.environ.get("CONVERT_TO_WAV", "1") == "1"
# RETURN_DEBUG = os.environ.get("DEBUG_ACCENT", "0") == "1"


# NATIVE_CLASS_ID = 1
# NON_NATIVE_CLASS_ID = 0

# # Load model
# model = joblib.load("model.joblib")

# def get_final_estimator(m):
#     try:
#         from sklearn.pipeline import Pipeline
#         if isinstance(m, Pipeline):
#             return m.steps[-1][1]
#     except Exception:
#         pass
#     return m

# final_est = get_final_estimator(model)
# classes_attr = getattr(final_est, "classes_", None)
# print("MODEL TYPE:", type(model))
# print("FINAL ESTIMATOR TYPE:", type(final_est))
# print("MODEL CLASSES:", classes_attr)

# # Feature extractor
# def _stats(vec: np.ndarray) -> np.ndarray:
#     return np.array(
#         [np.mean(vec), np.std(vec), np.median(vec),
#          np.percentile(vec, 5), np.percentile(vec, 95)],
#         dtype=np.float32
#     )

# def _agg(matrix: np.ndarray) -> np.ndarray:
#     return np.concatenate([_stats(row) for row in matrix], axis=0)

# def extract_features_320(y: np.ndarray, sr: int) -> np.ndarray:
#     y, _ = librosa.effects.trim(y, top_db=30)
#     min_len = int(0.5 * SR_TARGET)
#     if len(y) < min_len:
#         y = np.pad(y, (0, min_len - len(y)))
#     peak = np.max(np.abs(y)) if y.size else 0.0
#     if peak > 0:
#         y = y / peak

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
#     d1   = librosa.feature.delta(mfcc)
#     d2   = librosa.feature.delta(mfcc, order=2)

#     zcr  = librosa.feature.zero_crossing_rate(y)[0]
#     cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
#     bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
#     roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]

#     feat = np.concatenate([
#         _agg(mfcc),  # 5*N_MFCC
#         _agg(d1),    # 5*N_MFCC
#         _agg(d2),    # 5*N_MFCC
#         _stats(zcr), _stats(cent), _stats(bw), _stats(roll)  # 4*5
#     ], axis=0).astype(np.float32)

#     if not np.all(np.isfinite(feat)):
#         feat[~np.isfinite(feat)] = 0.0
#     return feat  # N_MFCC=20 -> 320 dims

# def decode_audio_to_16k_mono(tmp_path: str) -> np.ndarray:
#     if CONVERT_TO_WAV and HAVE_PYDUB:
#         a = AudioSegment.from_file(tmp_path).set_channels(1).set_frame_rate(SR_TARGET)
#         buf = io.BytesIO()
#         a.export(buf, format="wav")
#         buf.seek(0)
#         b = AudioSegment.from_file(buf, format="wav").set_channels(1).set_frame_rate(SR_TARGET)
#         arr = np.array(b.get_array_of_samples()).astype(np.float32)
#         if b.sample_width == 2:
#             arr /= 32768.0
#         elif b.sample_width == 4:
#             arr /= 2147483648.0
#         return np.clip(arr, -1.0, 1.0)
#     else:
#         y, _ = librosa.load(tmp_path, sr=SR_TARGET, mono=True)
#         return y.astype(np.float32)

# # API
# @app.route("/predict-accent", methods=["POST"])
# def predict_accent():
#     audio_path = None
#     try:
#         if "audio" not in request.files:
#             return jsonify({"error": "No 'audio' file provided"}), 400

#         f = request.files["audio"]
#         audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{f.filename or 'clip'}")
#         f.save(audio_path)

#         y = decode_audio_to_16k_mono(audio_path)
#         feat = extract_features_320(y, SR_TARGET).reshape(1, -1)

#         try:
#             proba = model.predict_proba(feat)[0]
#         except Exception:
#             pred_only = int(getattr(model, "predict")(feat)[0])
#             if classes_attr is not None:
#                 proba = np.zeros(len(classes_attr), dtype=np.float32)
#                 if pred_only in classes_attr:
#                     proba[list(classes_attr).index(pred_only)] = 1.0
#                 else:
#                     proba[0] = 1.0
#             else:
#                 proba = np.array([1.0, 0.0], dtype=np.float32)

#         classes_list = list(classes_attr) if classes_attr is not None else [0, 1]

#         # index for "Native" is class id 1; "Non-Native" is class id 0
#         try:
#             p_native     = float(proba[classes_list.index(NATIVE_CLASS_ID)])
#             p_non_native = float(proba[classes_list.index(NON_NATIVE_CLASS_ID)])
#         except Exception:
#             # fallback assume ordering [0,1] == [Non-Native, Native]
#             p_non_native = float(proba[0]) if len(proba) > 0 else 0.0
#             p_native     = float(proba[1]) if len(proba) > 1 else 0.0

#         if p_native >= p_non_native:
#             predicted_class = NATIVE_CLASS_ID
#             label = "Native"
#             confidence = p_native
#         else:
#             predicted_class = NON_NATIVE_CLASS_ID
#             label = "Non-Native"
#             confidence = p_non_native

#         resp = {"result": label, "confidence": float(confidence)}

#         want_debug = RETURN_DEBUG or request.args.get("debug") == "1"
#         if want_debug:
#             resp["_debug"] = {
#                 "upload_filename": f.filename,
#                 "upload_mimetype": f.mimetype,
#                 "upload_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else None,
#                 "pydub_available": HAVE_PYDUB,
#                 "convert_to_wav": CONVERT_TO_WAV,
#                 "classes_raw": classes_list,
#                 "mapped_native_class_id": NATIVE_CLASS_ID,
#                 "mapped_non_native_class_id": NON_NATIVE_CLASS_ID,
#                 "proba_raw": [float(x) for x in proba.tolist()],
#                 "p_native": p_native,
#                 "p_non_native": p_non_native,
#                 "feat_len": int(feat.shape[1]),
#                 "y_len": int(len(y)),
#                 "y_peak": float(np.max(np.abs(y))) if y.size else 0.0,
#             }

#         return jsonify(resp)

#     except Exception as e:
#         print("ERROR:", e)
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
#     finally:
#         try:
#             if audio_path and os.path.exists(audio_path):
#                 os.remove(audio_path)
#         except Exception:
#             pass

# if __name__ == "__main__":
#     app.run(port=5001, host="0.0.0.0", debug=True)

