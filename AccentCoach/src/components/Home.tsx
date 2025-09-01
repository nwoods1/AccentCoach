import React, { useEffect, useState, useMemo } from 'react'; 
import { Link } from 'react-router-dom';
import { auth, db } from '../firebase/firebase';
import { doc, onSnapshot, updateDoc } from 'firebase/firestore';
import Parrot from '../img/mascot-parrot.png';

const LEVELS_TOTAL = 5;

type Progress = {
  levels: boolean[];
  highestLevel: number;
  completed: boolean;
  lastLevel?: number | null;
  lastResult?: string | null;
  lastConfidence?: number | null;
  attemptsCount?: number | null;
};

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

const Home: React.FC = () => {
  const user = auth.currentUser;
  const isGuest = !!user?.isAnonymous;
  const uid = isGuest ? null : (user?.uid ?? null);

  const [progress, setProgress] = useState<Progress>({
    levels: Array(LEVELS_TOTAL).fill(false),
    highestLevel: 0,
    completed: false,
  });
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    if (!uid) {                 // <- guests skip Firestore and use defaults
      setLoading(false);
      return;
    }
    const ref = doc(db, 'users', uid);
    const unsub = onSnapshot(
      ref,
      (snap) => {
        const ap = (snap.data()?.accentProgress) || {};
        setProgress({
          levels: Array.from({ length: LEVELS_TOTAL }, (_, i) => Boolean(ap?.levels?.[i])),
          highestLevel: Number(ap?.highestLevel ?? 0),
          completed: Boolean(ap?.completed ?? false),
          lastLevel: ap?.lastLevel ?? null,
          lastResult: ap?.lastResult ?? null,
          lastConfidence: ap?.lastConfidence ?? null,
          attemptsCount: ap?.attemptsCount ?? null,
        });
        setLoading(false);
      },
      () => setLoading(false)
    );
    return () => unsub();
  }, [uid]);

  const completedCount = useMemo(
    () => progress.levels.filter(Boolean).length,
    [progress.levels]
  );

  const firstUnpassed = useMemo(() => {
    const i = progress.levels.findIndex((v) => !v);
    return i === -1 ? LEVELS_TOTAL - 1 : i;
  }, [progress.levels]);

  const resumeLevel = useMemo(
    () => clamp(firstUnpassed, 0, LEVELS_TOTAL - 1),
    [firstUnpassed]
  );

  // Persist lastLevel when signed-in (not guest)
  const handleStart = async () => {
    if (!uid) return; // guest: do nothing (no saving)
    try {
      await updateDoc(doc(db, 'users', uid), {
        'accentProgress.lastLevel': resumeLevel,
      });
    } catch (e) {
      console.warn('failed to set lastLevel:', e);
    }
  };

  // Reset progress (only for signed-in users)
  const handleReset = async () => {
    if (!uid || resetting) return;
    const ok = window.confirm('Reset your progress? This will mark all 5 levels as not completed.');
    if (!ok) return;

    try {
      setResetting(true);
      await updateDoc(doc(db, 'users', uid), {
        'accentProgress.levels': Array(LEVELS_TOTAL).fill(false),
        'accentProgress.highestLevel': 0,
        'accentProgress.completed': false,
        'accentProgress.lastLevel': null,
        'accentProgress.lastResult': null,
        'accentProgress.lastConfidence': null,
        'accentProgress.attemptsCount': 0,
      });
    } catch (e) {
      console.warn('failed to reset progress:', e);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-950">
      <div className="max-w-5xl mx-auto px-4 py-10">

        {/* Guest banner */}
        {isGuest && (
          <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 text-amber-900 px-4 py-2
                          dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200">
            Guest mode: progress won’t be saved.
          </div>
        )}

        {/* Hero row with mascot */}
        <div className="mb-8 flex items-center gap-4 md:gap-6">
          <img
            src={Parrot}
            alt="Accent Coach mascot"
            className="hidden sm:block w-24 md:w-32 lg:w-40 h-auto drop-shadow-[0_8px_24px_rgba(0,0,0,0.25)]"
            loading="eager"
          />
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold text-slate-900 dark:text-slate-100">
              Welcome back <span className="align-middle">👋</span>
            </h1>
            <p className="text-slate-600 dark:text-slate-300">
              Track your progress and jump back into practice.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Summary card */}
          <div className="md:col-span-1 rounded-2xl border bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 p-6">
            <div className="text-sm text-slate-500 dark:text-slate-400 mb-2">Overall</div>
            <div className="text-4xl font-bold text-slate-900 dark:text-slate-100">
              {completedCount}/{LEVELS_TOTAL}
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400 mt-1">levels completed</div>

            {progress.lastResult && !isGuest && (
              <div className="mt-4 text-sm text-slate-600 dark:text-slate-300">
                Last result:{' '}
                <span className="font-medium">{progress.lastResult}</span>
                {typeof progress.lastConfidence === 'number' &&
                  ` (${Math.round(progress.lastConfidence * 100)}%)`}
              </div>
            )}

            <div className="mt-6 space-y-3">
              <Link
                to={`/practice?level=${resumeLevel}`}
                onClick={handleStart}
                className="btn-primary w-full inline-flex items-center justify-center"
              >
                {completedCount === 0 ? 'Start practicing' : `Continue at Level ${resumeLevel + 1}`}
              </Link>

              {!isGuest && (
                <button
                  onClick={handleReset}
                  disabled={resetting}
                  className="w-full inline-flex items-center justify-center rounded-md border border-slate-300 dark:border-slate-700 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100/50 dark:hover:bg-slate-800/50 disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {resetting ? 'Resetting…' : 'Reset progress'}
                </button>
              )}
            </div>
          </div>

          {/* Roadmap (5 levels) */}
          <div className="md:col-span-2 rounded-2xl border bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 p-6">
            <div className="text-sm text-slate-500 dark:text-slate-400 mb-4">Your roadmap</div>

            <ol className="relative ms-4">
              {Array.from({ length: LEVELS_TOTAL }).map((_, i) => {
                const passed = progress.levels[i];
                const isUpNext = !passed && !progress.completed && i === firstUnpassed;
                const isLocked = !passed && i > firstUnpassed;

                return (
                  <li key={i} className="mb-8">
                    <div className="absolute -left-4 mt-1 h-5 w-5 rounded-full flex items-center justify-center ring-2 ring-slate-300 dark:ring-slate-700 bg-slate-200 dark:bg-slate-800">
                      {passed ? (
                        <svg viewBox="0 0 24 24" className="h-4 w-4 text-emerald-500">
                          <path fill="currentColor" d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" />
                        </svg>
                      ) : isUpNext ? (
                        <span className="block h-2.5 w-2.5 rounded-full bg-yellow-500" />
                      ) : (
                        <span className="block h-2.5 w-2.5 rounded-full bg-slate-400 dark:bg-slate-600" />
                      )}
                    </div>
                    <div className="ms-4">
                      <div className="flex items-center justify-between">
                        <div className="font-medium text-slate-900 dark:text-slate-100">
                          Level {i + 1}
                        </div>
                        <div
                          className={
                            passed
                              ? 'text-emerald-600 dark:text-emerald-400 text-sm'
                              : isUpNext
                              ? 'text-yellow-500 text-sm'
                              : 'text-slate-500 dark:text-slate-400 text-sm'
                          }
                        >
                          {passed ? 'Completed' : isUpNext ? 'Up next' : isLocked ? 'Locked' : ''}
                        </div>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>

        {loading && <div className="mt-6 text-sm text-slate-500">Loading progress…</div>}
      </div>
    </div>
  );
};

export default Home;
