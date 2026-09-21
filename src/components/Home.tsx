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
    <div className="page">
      <div className="page-container">

        {/* Guest banner */}
        {isGuest && (
          <div className="alert-warning mb-6 flex items-center gap-2">
            <span aria-hidden>👤</span>
            Guest mode: progress won&rsquo;t be saved.
          </div>
        )}

        {/* Hero row with mascot */}
        <div className="mb-8 flex items-center gap-4 md:gap-6">
          <img
            src={Parrot}
            alt="Accent Coach mascot"
            className="hidden sm:block w-16 md:w-20 h-16 md:h-20 rounded-2xl object-cover object-top shadow-soft ring-1 ring-slate-200 dark:ring-slate-800"
            loading="eager"
          />
          <div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
              Welcome back <span className="align-middle">👋</span>
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-0.5">
              Track your progress and jump back into practice.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Summary card */}
          <div className="md:col-span-1 card flex flex-col">
            <div className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">Overall</div>
            <div className="text-4xl font-bold text-slate-900 dark:text-slate-100">
              {completedCount}<span className="text-slate-400 dark:text-slate-500">/{LEVELS_TOTAL}</span>
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400 mt-1">levels completed</div>

            <div className="mt-3 h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <div
                className="h-full rounded-full bg-brand-500 transition-all"
                style={{ width: `${(completedCount / LEVELS_TOTAL) * 100}%` }}
              />
            </div>

            {progress.lastResult && !isGuest && (
              <div className="mt-4 text-sm text-slate-600 dark:text-slate-300">
                Last result:{' '}
                <span className="font-semibold">{progress.lastResult}</span>
                {typeof progress.lastConfidence === 'number' &&
                  ` (${Math.round(progress.lastConfidence * 100)}%)`}
              </div>
            )}

            <div className="mt-6 space-y-2.5">
              <Link
                to={`/practice?level=${resumeLevel}`}
                onClick={handleStart}
                className="btn-primary w-full"
              >
                {completedCount === 0 ? 'Start practicing' : `Continue at Level ${resumeLevel + 1}`}
              </Link>

              <Link to="/analyze" className="btn-outline w-full">
                Try phoneme analyzer
              </Link>

              {!isGuest && (
                <button
                  onClick={handleReset}
                  disabled={resetting}
                  className="btn-ghost w-full"
                >
                  {resetting ? 'Resetting…' : 'Reset progress'}
                </button>
              )}
            </div>
          </div>

          {/* Roadmap (5 levels) */}
          <div className="md:col-span-2 card">
            <div className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-5">Your roadmap</div>

            <ol className="relative">
              {Array.from({ length: LEVELS_TOTAL }).map((_, i) => {
                const passed = progress.levels[i];
                const isUpNext = !passed && !progress.completed && i === firstUnpassed;
                const isLocked = !passed && i > firstUnpassed;
                const isLast = i === LEVELS_TOTAL - 1;

                return (
                  <li key={i} className="relative pl-10 pb-8 last:pb-0">
                    {!isLast && (
                      <span
                        className={[
                          'absolute left-[15px] top-7 bottom-0 w-0.5',
                          passed ? 'bg-emerald-400 dark:bg-emerald-600' : 'bg-slate-200 dark:bg-slate-700',
                        ].join(' ')}
                        aria-hidden
                      />
                    )}
                    <div
                      className={[
                        'absolute left-0 top-0 h-8 w-8 rounded-full flex items-center justify-center ring-4 ring-white dark:ring-slate-900',
                        passed
                          ? 'bg-emerald-500'
                          : isUpNext
                          ? 'bg-brand-500'
                          : 'bg-slate-200 dark:bg-slate-700',
                      ].join(' ')}
                    >
                      {passed ? (
                        <svg viewBox="0 0 24 24" className="h-4 w-4 text-white">
                          <path fill="currentColor" d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" />
                        </svg>
                      ) : (
                        <span
                          className={[
                            'text-xs font-bold',
                            isUpNext ? 'text-white' : 'text-slate-500 dark:text-slate-400',
                          ].join(' ')}
                        >
                          {i + 1}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      <div className="font-semibold text-slate-900 dark:text-slate-100">
                        Level {i + 1}
                      </div>
                      <span
                        className={
                          passed
                            ? 'badge-success'
                            : isUpNext
                            ? 'badge-warning'
                            : 'badge-neutral'
                        }
                      >
                        {passed ? 'Completed' : isUpNext ? 'Up next' : isLocked ? 'Locked' : ''}
                      </span>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>

        {loading && (
          <div className="mt-6 flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            <span className="h-3.5 w-3.5 rounded-full border-2 border-slate-300 border-t-brand-500 animate-spin" />
            Loading progress…
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;
