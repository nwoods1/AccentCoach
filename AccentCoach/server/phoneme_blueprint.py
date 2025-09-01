# server/phoneme_blueprint.py
import os, tempfile, uuid, json
from typing import Any, Dict, List
from flask import Blueprint, request, jsonify

# ⬇️ absolute import (no leading dot)
from phoneme_utils import (
    webm_to_wav_16k_mono,
    sample_word_formants,
    arpabet_seq,
    arpabet_to_ipa_seq,
    first_vowel_ipa,
    nearest_vowel,
    vowel_spell,
    VOWEL_TARGETS,
    VOWEL_TIPS,
)


bp = Blueprint("phoneme", __name__)

@bp.route("/analyze-pronunciation", methods=["POST"])
def analyze_pronunciation():
    """
    Prototype analysis:
      - expects form fields: audio (webm), sentence (text)
      - converts to wav 16k mono
      - splits audio into N equal chunks (N = words) and samples F1/F2 per chunk
      - maps each chunk to nearest target vowel (IPA)
      - g2p the reference words -> expected IPA
      - returns per-word 'expected vs heard' + simple tips based on the vowel shift
    """
    temp_webm = None
    temp_wav  = None
    try:
        if "audio" not in request.files:
            return jsonify({"available": False, "error": "No audio"}), 400
        sentence = request.form.get("sentence", "").strip()
        if not sentence:
            return jsonify({"available": False, "error": "No sentence"}), 400

        # Save upload
        f = request.files["audio"]
        tempdir = tempfile.gettempdir()
        temp_webm = os.path.join(tempdir, f"{uuid.uuid4().hex}.webm")
        f.save(temp_webm)

        # Convert to wav 16k mono
        temp_wav = webm_to_wav_16k_mono(temp_webm)

        # Tokenize words (very simple split)
        words = [w for w in sentence.split() if w.strip()]
        if not words:
            return jsonify({"available": False, "error": "No words parsed"}), 400

        # Sample formants per word
        formants = sample_word_formants(temp_wav, len(words))

        items: List[Dict[str, Any]] = []
        for idx, word in enumerate(words):
            arpas = arpabet_seq(word)
            expected_ipa_seq = arpabet_to_ipa_seq(arpas)
            expected_vowel = first_vowel_ipa(arpas) or ""

            F1, F2 = formants[idx]
            heard_ipa, distance = nearest_vowel(F1, F2)

            items.append({
                "word_index": idx,
                "word": word,
                "expected": {
                    "arpabet": arpas,
                    "ipa": "".join(expected_ipa_seq),
                    "spelling": vowel_spell(expected_vowel) if expected_vowel else "",
                },
                "heard": {
                    "arpabet": [],   # not deriving a full phone seq here
                    "ipa": heard_ipa,
                    "spelling": vowel_spell(heard_ipa),
                },
                "vowel_notes": [{
                    "phoneme_expected": expected_vowel,
                    "phoneme_heard": heard_ipa,
                    "measured": {"F1": round(F1,2), "F2": round(F2,2)},
                    "target":  {"F1": VOWEL_TARGETS[heard_ipa][0], "F2": VOWEL_TARGETS[heard_ipa][1]},
                    "distance": round(distance, 1),
                }],
            })

        resp = {
            "available": True,
            "word_feedback": {
                "available": True,
                "items": items,
                "words": words,
            }
        }

        if request.args.get("debug") == "1":
            resp["_debug"] = {
                "words": words,
                "formants": formants,
            }

        return jsonify(resp)

    except Exception as e:
        return jsonify({"available": False, "error": str(e)}), 500
    finally:
        for p in (temp_webm, temp_wav):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
