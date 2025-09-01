# server/phoneme_utils.py
import os, tempfile, uuid
import numpy as np
from typing import List, Tuple, Dict, Any

# Audio / analysis libs
from pydub import AudioSegment       # requires ffmpeg
import parselmouth                   # praat-parselmouth
import librosa

# -----------------------------
# Audio helpers
# -----------------------------
def webm_to_wav_16k_mono(src_path: str) -> str:
    """Convert WEBM/OPUS (or anything ffmpeg can read) to 16kHz mono WAV."""
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.wav")
    AudioSegment.from_file(src_path).set_frame_rate(16000).set_channels(1).export(out_path, format="wav")
    return out_path

def load_duration_seconds(wav_path: str) -> float:
    snd = parselmouth.Sound(wav_path)
    return snd.get_total_duration()

# -----------------------------
# Simple vowel targets (F1/F2)
# values are approximate averages for American English
# -----------------------------
VOWEL_TARGETS = {
    "i":  (300, 2200),  # FLEECE
    "ɪ":  (400, 1900),  # KIT
    "e":  (450, 2000),  # FACE (monophthong approx)
    "ɛ":  (600, 1800),  # DRESS
    "æ":  (700, 1700),  # TRAP
    "ɝ":  (500, 1500),  # NURSE (r-colored)
    "ʌ":  (600, 1300),  # STRUT
    "ɑ":  (750, 1100),  # PALM
    "ɔ":  (600,  900),  # THOUGHT
    "o":  (500,  900),  # GOAT (monophthong approx)
    "ʊ":  (450, 1100),  # FOOT
    "u":  (350,  900),  # GOOSE
}

VOWEL_TIPS = {
    "i":  "Say “ee” as in ‘see’. Keep the tongue high and front.",
    "ɪ":  "Say “ih” as in ‘sit’. Slightly lower tongue than “ee”.",
    "e":  "Say “ay” as in ‘say’ (monophthong).",
    "ɛ":  "Say “eh” as in ‘bed’.",
    "æ":  "Say “a” as in ‘cat’. Open the jaw a bit.",
    "ɝ":  "Say “er” as in ‘bird’ (American).",
    "ʌ":  "Say “uh” as in ‘cup’.",
    "ɑ":  "Say “ah” as in ‘father’.",
    "ɔ":  "Say “aw” as in ‘thought’.",
    "o":  "Say “oh” as in ‘go’.",
    "ʊ":  "Say “u” as in ‘book’.",
    "u":  "Say “oo” as in ‘goose’.",
}

def vowel_spell(ipa: str) -> str:
    return {
        "i": "“ee”",
        "ɪ": "“ih”",
        "e": "“ay”",
        "ɛ": "“eh”",
        "æ": "“a” (cat)",
        "ɝ": "“er”",
        "ʌ": "“uh”",
        "ɑ": "“ah”",
        "ɔ": "“aw”",
        "o": "“oh”",
        "ʊ": "“u” (book)",
        "u": "“oo”",
    }.get(ipa, ipa)

def nearest_vowel(F1: float, F2: float) -> Tuple[str, float]:
    """Return (ipa, distance) of nearest target in (F1,F2) space."""
    best = None
    best_d = float("inf")
    for ipa, (t1, t2) in VOWEL_TARGETS.items():
        d = (F1 - t1) ** 2 + (F2 - t2) ** 2
        if d < best_d:
            best_d = d
            best = ipa
    return best, float(np.sqrt(best_d))

# -----------------------------
# G2P (ARPAbet) -> IPA
# -----------------------------
from g2p_en import G2p

G2P = G2p()

ARPABET_TO_IPA = {
    # vowels
    "IY":"i", "IH":"ɪ", "EY":"e", "EH":"ɛ", "AE":"æ",
    "ER":"ɝ", "AH":"ʌ", "AA":"ɑ", "AO":"ɔ", "OW":"o",
    "UH":"ʊ", "UW":"u",
    # rimes/diphthongs simplified to monophthongs above
    # consonants (not exhaustively used here)
    "P":"p","B":"b","T":"t","D":"d","K":"k","G":"ɡ",
    "F":"f","V":"v","TH":"θ","DH":"ð","S":"s","Z":"z",
    "SH":"ʃ","ZH":"ʒ","HH":"h","M":"m","N":"n","NG":"ŋ",
    "CH":"tʃ","JH":"dʒ","L":"l","R":"ɹ","W":"w","Y":"j",
}

VOWEL_TAGS = set(["IY","IH","EY","EH","AE","ER","AH","AA","AO","OW","UH","UW"])

def strip_stress(p: str) -> str:
    # AE1 -> AE
    if len(p) >= 3 and p[-1].isdigit():
        return p[:-1]
    return p

def arpabet_seq(word: str) -> List[str]:
    out = []
    for tok in G2P(word):
        t = tok.strip()
        if not t or t == " ":
            continue
        # G2p returns words and arpabet tokens; we keep only arpabet (all caps)
        if t.isalpha() and t.isupper():
            out.append(t)
    return out

def arpabet_to_ipa_seq(arpas: List[str]) -> List[str]:
    ipa = []
    for a in arpas:
        base = strip_stress(a)
        ipa.append(ARPABET_TO_IPA.get(base, base))
    return ipa

def first_vowel_ipa(arpas: List[str]) -> str:
    for a in arpas:
        if strip_stress(a) in VOWEL_TAGS:
            return ARPABET_TO_IPA.get(strip_stress(a), "")
    return ""

# -----------------------------
# Formant sampling per word
# -----------------------------
def sample_word_formants(wav_path: str, n_words: int) -> List[Tuple[float,float]]:
    """Split duration into equal segments per word and sample F1/F2 at each midpoint."""
    snd = parselmouth.Sound(wav_path)
    total = snd.get_total_duration()
    formant = snd.to_formant_burg(time_step=0.01)
    vals = []
    if n_words <= 0:
        return vals
    for i in range(n_words):
        t = (i + 0.5) * (total / n_words)
        F1 = formant.get_value_at_time(1, t) or np.nan
        F2 = formant.get_value_at_time(2, t) or np.nan
        vals.append((float(F1), float(F2)))
    # replace NaNs with median of finite values
    arr = np.array(vals, dtype=float)
    for col in [0,1]:
        col_vals = arr[:, col]
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size:
            med = float(np.median(finite))
            col_vals[~np.isfinite(col_vals)] = med
            arr[:, col] = col_vals
    return [(float(a), float(b)) for a,b in arr.tolist()]
