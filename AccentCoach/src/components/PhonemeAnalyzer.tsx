import React, { useRef, useState } from "react";
import { analyzePronunciation } from "../services/phonemeApi";
import PhonemeFeedback from "./PhonemeFeedback";
import { AnalyzeResponse, PhonemeWordItem } from "../types/phoneme";

const PRESET_SENTENCES = [
  "Hello, my name is John.",
  "The quick brown fox jumps over the lazy dog.",
  "She sells seashells by the seashore.",
  "The sixth sick sheik's sixth sheep's sick.",
  "Irish wristwatch, Swiss wristwatch.",
];

export default function PhonemeAnalyzer() {
  const [sentence, setSentence] = useState(PRESET_SENTENCES[0]);
  const [custom, setCustom] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<AnalyzeResponse | null>(null);
  const [items, setItems] = useState<PhonemeWordItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mr = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const effectiveSentence = (custom || "").trim().length ? custom.trim() : sentence;

  const start = async () => {
    try {
      setResp(null);
      setItems(null);
      setError(null);
      setAudioURL(null);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const m = new MediaRecorder(stream);
      mr.current = m;
      chunks.current = [];

      m.ondataavailable = (e: any) => {
        if (e.data && e.data.size > 0) chunks.current.push(e.data);
      };

      m.onstop = async () => {
        try {
          const blob = new Blob(chunks.current, { type: "audio/webm" });
          setAudioURL(URL.createObjectURL(blob));
          setLoading(true);
          const data = await analyzePronunciation(blob, effectiveSentence, false);
          setResp(data);
          setItems(data?.word_feedback?.items ?? null);
        } catch (err: any) {
          setError(err?.message || "Analyze failed");
        } finally {
          setLoading(false);
        }
      };

      m.start();
      setIsRecording(true);
    } catch (e: any) {
      setError(e?.message || "Microphone permission error");
    }
  };

  const stop = () => {
    const m = mr.current;
    if (!m) return;
    if (m.state !== "inactive") {
      m.stop();
      m.stream.getTracks().forEach((t) => t.stop());
    }
    setIsRecording(false);
  };

  return (
    <div className="max-w-3xl mx-auto p-4">
      <div className="rounded-2xl shadow-md border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-slate-900/50 p-6">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">Phoneme Analyzer</h2>
        <p className="text-slate-600 dark:text-slate-300 mt-2">
          Record a sentence to see which words and vowel sounds might need adjustment.
        </p>

        {/* Sentence picker */}
        <div className="mt-4 space-y-3">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Choose a preset
          </label>
          <select
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/60 px-3 py-2"
          >
            {PRESET_SENTENCES.map((s, i) => (
              <option key={i} value={s}>{s}</option>
            ))}
          </select>

          <div className="text-xs text-slate-500 dark:text-slate-400">or enter your own</div>
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder="Type your custom sentence (optional)"
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/60 px-3 py-2"
          />
        </div>

        {/* Controls */}
        <div className="mt-5 flex gap-3">
          <button
            onClick={isRecording ? stop : start}
            className={
              "flex-1 rounded-xl px-4 py-2 font-medium text-white transition " +
              (isRecording
                ? "bg-red-500 hover:bg-red-600"
                : "bg-indigo-600 hover:bg-indigo-700")
            }
            disabled={loading}
          >
            {isRecording ? "Stop Recording" : "Start Recording"}
          </button>

          <button
            onClick={() => {
              setResp(null);
              setItems(null);
              setAudioURL(null);
              setError(null);
            }}
            className="rounded-xl px-4 py-2 font-medium border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/60"
            disabled={loading || isRecording}
          >
            Clear
          </button>
        </div>

        {/* Playback */}
        {audioURL && (
          <div className="mt-4">
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Playback</h3>
            <audio controls className="w-full" src={audioURL} />
          </div>
        )}

        {/* Status / Errors */}
        {loading && <div className="mt-3 text-sm text-slate-500">Analyzing…</div>}
        {error && (
          <div className="mt-3 text-sm rounded-lg bg-rose-50 dark:bg-rose-900/30 border border-rose-200 dark:border-rose-700 text-rose-800 dark:text-rose-200 p-3">
            {error}
          </div>
        )}

        {/* Results */}
        {resp && !loading && (
          <>
            {!resp.available || resp.word_feedback?.available === false ? (
              <div className="mt-4 rounded-xl border bg-amber-50/70 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700 p-4 text-amber-900 dark:text-amber-100">
                Detailed phoneme analysis isn’t available. Make sure the server has
                <code className="mx-1">g2p_en</code>,
                <code className="mx-1">praat-parselmouth</code>, and
                <code className="mx-1">ffmpeg</code> set up.
              </div>
            ) : (
              <PhonemeFeedback items={items || []} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
