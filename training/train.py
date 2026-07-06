import os
import random
import time
import joblib
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NATIVE_DIRS = ["cmu_us_bdl_arctic/wav", "cmu_us_slt_arctic/wav"]
NONNATIVE_ROOT = "nonnative"
NATIVE_FILES_PER_SPK = 80
NONNATIVE_FILES_PER_SPK = 10
AUGMENT_COPIES = 3
SR = 16000
N_MFCC = 20
MODEL_OUT = "server/model.joblib"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def list_wavs(directory, limit=None):
    if not os.path.isdir(directory):
        return []
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    )
    return files[:limit] if limit else files


def _stats(v):
    return np.array(
        [np.mean(v), np.std(v), np.median(v), np.percentile(v, 5), np.percentile(v, 95)],
        dtype=np.float32,
    )


def _agg(mat):
    return np.concatenate([_stats(row) for row in mat])


def extract_features(y):
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.95)[0]

    feat = np.concatenate([
        _agg(mfcc), _agg(d1), _agg(d2),
        _stats(zcr), _stats(cent), _stats(bw), _stats(roll),
    ])
    feat[~np.isfinite(feat)] = 0.0
    return feat.astype(np.float32)


def augment(y):
    y = y * np.random.uniform(0.6, 1.4)
    if np.random.rand() < 0.6:
        y = y + (np.random.randn(len(y)) * 0.008).astype(np.float32)
    try:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.85, 1.15))
    except Exception:
        pass
    if np.random.rand() < 0.4:
        shift = int(np.random.uniform(-0.1, 0.1) * SR)
        y = np.roll(y, shift)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def build_dataset():
    native_items = []
    for i, d in enumerate(NATIVE_DIRS):
        for p in list_wavs(d, limit=NATIVE_FILES_PER_SPK):
            native_items.append((p, f"native_{i}"))

    nonnative_items = []
    if os.path.isdir(NONNATIVE_ROOT):
        for spk in sorted(os.listdir(NONNATIVE_ROOT)):
            wav_dir = os.path.join(NONNATIVE_ROOT, spk, "wav")
            for p in list_wavs(wav_dir, limit=NONNATIVE_FILES_PER_SPK):
                nonnative_items.append((p, spk))

    if not native_items:
        raise RuntimeError(f"No native files found. Check NATIVE_DIRS: {NATIVE_DIRS}")
    if not nonnative_items:
        raise RuntimeError(f"No non-native files found. Check NONNATIVE_ROOT: {NONNATIVE_ROOT}")

    random.shuffle(nonnative_items)
    nonnative_items = nonnative_items[:len(native_items)]

    print(f"Native:     {len(native_items):>4} files, {len({s for _, s in native_items})} speakers")
    print(f"Non-native: {len(nonnative_items):>4} files, {len({s for _, s in nonnative_items})} speakers")
    return native_items, nonnative_items


def compute_features(items, label):
    X, y, groups = [], [], []
    for path, spk in items:
        try:
            raw = load_audio(path)
            X.append(extract_features(raw))
            y.append(label)
            groups.append(spk)
            for _ in range(AUGMENT_COPIES):
                X.append(extract_features(augment(raw)))
                y.append(label)
                groups.append(spk)
        except Exception as e:
            print(f"  skipped {path}: {e}")
    return X, y, groups


def speaker_disjoint_folds(groups, y):
    groups, y = np.asarray(groups), np.asarray(y)
    native_spks = list(dict.fromkeys(g for g, l in zip(groups, y) if l == 0))
    non_spks = list(dict.fromkeys(g for g, l in zip(groups, y) if l == 1))
    random.shuffle(non_spks)
    half = len(non_spks) // 2

    val_A = set(native_spks[:1]) | set(non_spks[:half])
    val_B = set(native_spks[1:2]) | set(non_spks[half:])

    folds = []
    for val_set in [val_A, val_B]:
        train_idx = np.where(~np.isin(groups, list(val_set)))[0]
        val_idx = np.where(np.isin(groups, list(val_set)))[0]
        folds.append((train_idx, val_idx))
    return folds


def train_model(X, y, folds):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED,
        )),
    ])
    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.001, 0.005, 0.01, 0.05, 0.1]},
        cv=folds,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    grid.fit(X, y)
    print(f"Best C={grid.best_params_['clf__C']}  CV macro-F1={grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_["clf__C"]


def evaluate(model, X, y, folds):
    for i, (_, val_idx) in enumerate(folds):
        y_pred = model.predict(X[val_idx])
        print(f"\nFold {i + 1}:")
        print(classification_report(y[val_idx], y_pred, target_names=["Native", "Non-Native"]))
        print(confusion_matrix(y[val_idx], y_pred))


def main():
    t0 = time.perf_counter()

    native_items, nonnative_items = build_dataset()

    print("\nExtracting features (with augmentation)...")
    X_nat, y_nat, g_nat = compute_features(native_items, label=0)
    X_non, y_non, g_non = compute_features(nonnative_items, label=1)

    X = np.vstack(X_nat + X_non).astype(np.float32)
    y = np.concatenate([y_nat, y_non]).astype(np.int64)
    groups = np.array(g_nat + g_non)

    print(f"\n{len(y)} samples in {time.perf_counter() - t0:.1f}s")
    print(f"Native(0): {(y == 0).sum()}  Non-Native(1): {(y == 1).sum()}")

    folds = speaker_disjoint_folds(groups, y)

    print("\nGrid search...")
    best_model, best_C = train_model(X, y, folds)

    evaluate(best_model, X, y, folds)

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED,
            C=best_C,
        )),
    ])
    final.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(MODEL_OUT)), exist_ok=True)
    joblib.dump(final, MODEL_OUT)
    print(f"\nSaved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
