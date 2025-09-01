import React from "react";
import { PhonemeWordItem, VowelNote } from "../types/phoneme";

function tipFromNote(n: VowelNote): string {
  const dF1 = n.measured.F1 - n.target.F1; // + more open
  const dF2 = n.measured.F2 - n.target.F2; // + fronter
  const moves: string[] = [];

  if (Math.abs(dF1) > 80) {
    moves.push(dF1 > 0 ? "close your jaw slightly" : "open your jaw slightly");
  }
  if (Math.abs(dF2) > 120) {
    moves.push(dF2 > 0 ? "pull the tongue slightly back" : "move the tongue slightly forward");
  }
  if (!moves.length) return "Keep the vowel steady; aim for a relaxed placement.";
  return moves.join("; ") + ".";
}

export default function PhonemeFeedback({ items }: { items: PhonemeWordItem[] }) {
  if (!items?.length) {
    return (
      <div className="mt-4 rounded-xl border bg-slate-50/70 dark:bg-slate-900/40 border-slate-200 dark:border-slate-700 p-4">
        <div className="text-slate-700 dark:text-slate-300">
          No specific mispronunciations detected in this clip.
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 space-y-4">
      {items.map((it) => {
        const changed =
          it.expected.ipa !== it.heard.ipa || it.expected.spelling !== it.heard.spelling;

        return (
          <div
            key={it.word_index}
            className="rounded-xl border bg-white/70 dark:bg-slate-900/50 border-slate-200 dark:border-slate-700 p-4"
          >
            <div className="flex items-baseline justify-between">
              <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {it.word}
              </h4>
              <span
                className={
                  "text-xs px-2 py-1 rounded-full " +
                  (changed
                    ? "bg-amber-100 text-amber-900 dark:bg-amber-900/30 dark:text-amber-200"
                    : "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/30 dark:text-emerald-200")
                }
              >
                {changed ? "Adjusted" : "Matched"}
              </span>
            </div>

            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="rounded-lg p-3 bg-slate-50 dark:bg-slate-900/40">
                <div className="text-xs uppercase text-slate-500 dark:text-slate-400">
                  Expected
                </div>
                <div className="mt-1 text-slate-900 dark:text-slate-100">
                  <div><span className="font-medium">IPA:</span> {it.expected.ipa}</div>
                  <div><span className="font-medium">Sounds like:</span> {it.expected.spelling}</div>
                </div>
              </div>

              <div className="rounded-lg p-3 bg-slate-50 dark:bg-slate-900/40">
                <div className="text-xs uppercase text-slate-500 dark:text-slate-400">
                  Heard (estimated)
                </div>
                <div className="mt-1 text-slate-900 dark:text-slate-100">
                  <div><span className="font-medium">IPA:</span> {it.heard.ipa}</div>
                  <div><span className="font-medium">Sounds like:</span> {it.heard.spelling}</div>
                </div>
              </div>
            </div>

            {!!it.vowel_notes?.length && (
              <div className="mt-3 rounded-lg border border-slate-200 dark:border-slate-700 p-3">
                <div className="text-xs uppercase text-slate-500 dark:text-slate-400 mb-1">
                  Vowel notes
                </div>
                <ul className="space-y-1 text-sm text-slate-800 dark:text-slate-200">
                  {it.vowel_notes.map((n, idx) => (
                    <li key={idx} className="leading-relaxed">
                      Expected <span className="font-mono">{n.phoneme_expected || "—"}</span>,
                      heard <span className="font-mono">{n.phoneme_heard}</span>.{" "}
                      <span className="italic">{tipFromNote(n)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
