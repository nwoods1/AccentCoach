import { useEffect, useRef, useState, useMemo } from 'react';
import { auth, db } from '../firebase/firebase';
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  serverTimestamp,
  increment,
  addDoc,
  collection,
} from 'firebase/firestore';
import Parrot from '../img/mascot-parrot.png';

type PredictResponse = {
  result: 'Native' | 'Non-Native' | string;
  confidence?: number;
  _debug?: any;
};

type ResultLabel = 'Native' | 'Non-Native';

const SENTENCES: string[] = [
  "Hello, my name is John.",
  "The quick brown fox jumps over the lazy dog.",
  "Please place the blue book on the small table.",   
  "The meeting starts at 10 in the morning.",      
  "Please speak clearly and at a steady pace."
];
const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

/** Save an attempt + update accentProgress in Firestore */
async function saveAttemptAndProgress(opts: {
  uid: string;
  level: number;
  passed: boolean;
  result: ResultLabel;
  confidence: number | null;
  sentence: string;
  totalLevels: number;
}) {
  const { uid, level, passed, result, confidence, sentence, totalLevels } = opts;

  const userRef = doc(db, 'users', uid);
  const snap = await getDoc(userRef);

  let levels: boolean[] = Array(totalLevels).fill(false);
  let highestLevel = 0;

  if (snap.exists()) {
    const data = snap.data() as any;
    const ap = data?.accentProgress || {};
    if (Array.isArray(ap.levels)) {
      const existing: boolean[] = ap.levels.slice(0, totalLevels);
      levels =
        existing.length < totalLevels
          ? [...existing, ...Array(totalLevels - existing.length).fill(false)]
          : existing;
    }
    highestLevel = Number((data?.accentProgress?.highestLevel) ?? 0);
  } else {
    // First-time init: note lastLevel is null (not 0)
    await setDoc(
      userRef,
      {
        createdAt: serverTimestamp(),
        accentProgress: {
          levels,
          highestLevel: 0,
          completed: false,
          lastLevel: null,
          lastResult: null,
          lastConfidence: null,
          attemptsCount: 0,
        },
      },
      { merge: true }
    );
  }

  // mark this level as passed/failed (passed -> true, else stays/sets false)
  levels[level] = passed;
  const newHighest = passed ? Math.max(highestLevel, level) : highestLevel;
  const newCompleted = levels.every(Boolean);

  // Update summary progress
  await updateDoc(userRef, {
    'accentProgress.levels': levels,
    'accentProgress.highestLevel': newHighest,
    'accentProgress.completed': newCompleted,
    'accentProgress.lastLevel': level,
    'accentProgress.lastResult': result,
    'accentProgress.lastConfidence': confidence ?? null,
    'accentProgress.attemptsCount': increment(1),
    updatedAt: serverTimestamp(),
  });

  // Log attempt
  await addDoc(collection(userRef, 'attempts'), {
    level,
    sentence,
    result,
    confidence: confidence ?? null,
    createdAt: serverTimestamp(),
  });
}

const Recorder = () => {
  // derive initial level from URL (?level=0..4); if missing, we’ll compute from Firestore in effect
  const levelsTotal = SENTENCES.length;
  const urlLevelInit = (() => {
    try {
      const sp = new URLSearchParams(window.location.search);
      const raw = sp.get('level');
      if (raw == null) return null;
      const n = Number(raw);
      if (Number.isNaN(n)) return null;
      return clamp(n, 0, levelsTotal - 1);
    } catch {
      return null;
    }
  })();

  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [level, setLevel] = useState<number>(urlLevelInit ?? 0); // will be overridden once if URL missing
  const [status, setStatus] = useState<'idle'|'pass'|'fail'>('idle');
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const initializedFromUrlRef = useRef<boolean>(urlLevelInit !== null); // if true, don't override from Firestore

  const currentSentence = SENTENCES[level];

  // One-time: if no ?level=, select level from Firestore (lastLevel → highestLevel → firstIncomplete)
  useEffect(() => {
    if (initializedFromUrlRef.current) return; // URL wins; don't override later
    const pickLevel = async () => {
      const uid = auth.currentUser?.uid;
      if (!uid) return;
      const snap = await getDoc(doc(db, 'users', uid));
      const ap = (snap.data()?.accentProgress) || {};
      const levels = Array.from({ length: levelsTotal }, (_, i) => Boolean(ap?.levels?.[i]));
      const firstIncompleteIdx = (() => {
        const i = levels.findIndex(v => !v);
        return i === -1 ? levelsTotal - 1 : i;
      })();

      let chosen: number;
      if (typeof ap?.lastLevel === 'number') chosen = clamp(ap.lastLevel, 0, levelsTotal - 1);
      else if (typeof ap?.highestLevel === 'number') chosen = clamp(ap.highestLevel, 0, levelsTotal - 1);
      else chosen = firstIncompleteIdx;

      setLevel(chosen);
      initializedFromUrlRef.current = true;
    };
    pickLevel();
  }, [levelsTotal]);

  useEffect(() => {
    const saved = localStorage.getItem('theme');
    const prefersDark =
      window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialDark = saved ? saved === 'dark' : prefersDark;
    setDark(initialDark);
    document.documentElement.classList.toggle('dark', initialDark);
  }, []);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  const resetClipState = () => {
    setAudioURL(null);
    setResult(null);
    setConfidence(null);
    setStatus('idle');
  };

  const startRecording = async () => {
    try {
      resetClipState();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Make onstop async so we can await the two server calls
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);

        setLoading(true);
        try {
          // 1) Verify the user actually read the sentence
          const verifyForm = new FormData();
          verifyForm.append('audio', audioBlob, 'recording.webm');
          verifyForm.append('expected', currentSentence);

          const verifyRes = await fetch('http://localhost:5001/verify-sentence', {
            method: 'POST',
            body: verifyForm,
          });
          const verify = await verifyRes.json();

          if (!verifyRes.ok) {
            console.error('verify-sentence error:', verify);
            setResult('Could not verify the sentence.');
            setStatus('fail');
            return; // stop here; don't classify
          }

          if (!verify.spoken_ok) {
            const heard = verify?.transcript ? `We heard: “${verify.transcript}”.` : '';
            const miss  = Array.isArray(verify?.missing) && verify.missing.length ? ` Missing: ${verify.missing.join(', ')}.` : '';
            const extra = Array.isArray(verify?.extra) && verify.extra.length ? ` Extra: ${verify.extra.join(', ')}.` : '';
            setResult(`Please read exactly: “${currentSentence}”. ${heard}${miss}${extra}`);
            setStatus('fail');
            return; // DO NOT classify if the sentence doesn't match
          }

          // 2) Passed verification → now classify accent
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.webm');

          const res = await fetch('http://localhost:5001/predict-accent', {
            method: 'POST',
            body: formData,
          });
          const data: PredictResponse = await res.json();

          const native = data.result === 'Native';
          const conf = typeof data.confidence === 'number' ? data.confidence : null;

          setResult(data.result);
          setConfidence(conf);

          const passed = native;
          setStatus(passed ? 'pass' : 'fail');

          // 3) Save attempt to Firestore (if signed in)
          const user = auth.currentUser;
          if (user && !user.isAnonymous) {
            const uid = user.uid; 
            try {
              await saveAttemptAndProgress({
                uid,
                level,
                passed,
                result: native ? 'Native' : 'Non-Native',
                confidence: conf,
                sentence: currentSentence,
                totalLevels: SENTENCES.length,
              });
            } catch (e) {
              console.error('saveAttemptAndProgress error:', e);
            }
          }
        } catch (err) {
          console.error('Upload/analysis error:', err);
          setResult('Error analyzing audio.');
          setStatus('fail');
        } finally {
          setLoading(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (e) {
      console.error('Microphone permission or recording error:', e);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }
    setIsRecording(false);
  };

  const nextLevel = async () => {
    if (level < levelsTotal - 1) {
      const newLevel = level + 1;
      setLevel(newLevel);
      resetClipState();
      // keep lastLevel in Firestore aligned with local navigation
      const uid = auth.currentUser?.uid;
      if (uid) {
        try {
          await updateDoc(doc(db, 'users', uid), {
            'accentProgress.lastLevel': newLevel,
          });
        } catch {}
      }
    }
  };

  const restartCourse = async () => {
    setLevel(0);
    resetClipState();
    const uid = auth.currentUser?.uid;
    if (uid) {
      try {
        await updateDoc(doc(db, 'users', uid), {
          'accentProgress.lastLevel': 0,
        });
      } catch {}
    }
  };

  const allDone = level === levelsTotal - 1 && status === 'pass';

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-950 dark:to-slate-900 transition-colors">
      {/* Top bar */}
      <div className="max-w-4xl mx-auto px-4 pt-6 pb-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img
            src={Parrot}
            alt="Accent Coach mascot"
            className="h-14 md:h-40 w-auto drop-shadow"
            loading="eager"
          />
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100">Accent Coach</h1>
        </div>
      </div>

      {/* Card */}
      <div className="max-w-2xl mx-auto px-4 pb-12">
        <div className="card">
          <p className="text-slate-600 dark:text-slate-300">
            Say the sentence exactly as shown. Pass to unlock the next level.
          </p>

          {/* Progress */}
          <div className="mt-6 flex items-center gap-2">
            {SENTENCES.map((_, idx) => {
              const reached = idx < level || (idx === level && status !== 'idle');
              const passed = idx < level || (idx === level && status === 'pass');
              return (
                <div
                  key={idx}
                  className={[
                    'h-2 flex-1 rounded-full transition-colors',
                    passed
                      ? 'bg-emerald-500'
                      : reached
                      ? 'bg-amber-400'
                      : 'bg-slate-200 dark:bg-slate-700',
                  ].join(' ')}
                  title={`Level ${idx + 1}`}
                />
              );
            })}
          </div>

          {/* Current level / sentence */}
          <div className="mt-6">
            <div className="text-sm text-slate-500 dark:text-slate-400">
              Level {level + 1} of {SENTENCES.length}
            </div>
            <div
              className="mt-2 p-4 rounded-xl border bg-white/70 dark:bg-slate-900/50 backdrop-blur
                          border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100"
            >
              <span className="font-semibold">Say:</span>{' '}
              <span className="italic">“{currentSentence}”</span>
            </div>
          </div>

          {/* Controls */}
          <div className="mt-6 flex gap-3">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={isRecording ? 'btn-danger flex-1' : 'btn-primary flex-1'}
              disabled={loading}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>
            <button onClick={restartCourse} className="btn-outline" disabled={isRecording || loading}>
              Reset
            </button>
          </div>

          {/* Playback & result */}
          {audioURL && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Playback</h3>
              <audio className="w-full" controls src={audioURL}></audio>

              {loading && <div className="mt-3 text-sm text-slate-500">Analyzing…</div>}

              {!loading && result && (
                <div
                  className={[
                    'mt-4 border rounded-xl p-4 text-center',
                    status === 'pass'
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-900 dark:bg-emerald-900/30 dark:border-emerald-700 dark:text-emerald-200'
                      : status === 'fail'
                      ? 'bg-rose-50 border-rose-200 text-rose-900 dark:bg-rose-900/30 dark:border-rose-700 dark:text-rose-200'
                      : 'bg-slate-50 border-slate-200 text-slate-900 dark:bg-slate-900/50 dark:border-slate-700 dark:text-slate-100',
                  ].join(' ')}
                >
                  <p className="text-lg font-semibold">
                    You sound like a{' '}
                    <span
                      className={
                        status === 'pass'
                          ? 'text-emerald-700 dark:text-emerald-300'
                          : 'text-indigo-700 dark:text-indigo-300'
                      }
                    >
                      {result}
                    </span>
                    {typeof confidence === 'number' && (
                      <span className="ml-2 text-sm opacity-70">
                        ({Math.round(confidence * 100)}%)
                      </span>
                    )}
                    .
                  </p>

                  {status === 'pass' && (
                    <div className="mt-3">
                      {level === levelsTotal - 1 ? (
                        <div className="flex flex-col gap-3 items-center">
                          <div className="text-emerald-700 dark:text-emerald-300 font-medium">
                            🎉 You passed all {levelsTotal} levels!
                          </div>
                          <button onClick={restartCourse} className="btn-success">
                            Restart Course
                          </button>
                        </div>
                      ) : (
                        <button onClick={nextLevel} className="btn-success">
                          Go to Level {level + 2}
                        </button>
                      )}
                    </div>
                  )}

                  {status === 'fail' && (
                    <div className="mt-3">
                      <div className="mb-2">Let’s try that sentence again.</div>
                      <button onClick={startRecording} className="btn-danger">
                        Retry Level {level + 1}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Recorder;
