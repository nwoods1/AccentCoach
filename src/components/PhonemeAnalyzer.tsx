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
    <div className="page">
      <div className="page-container max-w-3xl">
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Phoneme Analyzer
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-0.5">
            Record a sentence to see which words and vowel sounds might need adjustment.
          </p>
        </div>

        <div className="card">
          {/* Sentence picker */}
          <div className="space-y-3">
            <div>
              <label className="field-label">Choose a preset</label>
              <select
                value={sentence}
                onChange={(e) => setSentence(e.target.value)}
                className="input"
              >
                {PRESET_SENTENCES.map((s, i) => (
                  <option key={i} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="field-label">Or enter your own</label>
              <input
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                placeholder="Type your custom sentence (optional)"
                className="input"
              />
            </div>
          </div>

          {/* Controls */}
          <div className="mt-5 flex gap-3">
            <button
              onClick={isRecording ? stop : start}
              className={isRecording ? 'btn-danger flex-1' : 'btn-primary flex-1'}
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
              className="btn-outline"
              disabled={loading || isRecording}
            >
              Clear
            </button>
          </div>

          {/* Playback */}
          {audioURL && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Playback</h3>
              <audio controls className="w-full" src={audioURL} />
            </div>
          )}

          {/* Status / Errors */}
          {loading && (
            <div className="mt-4 flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
              <span className="h-3.5 w-3.5 rounded-full border-2 border-slate-300 border-t-brand-500 animate-spin" />
              Analyzing…
            </div>
          )}
          {error && <div className="alert-danger mt-4">{error}</div>}

          {/* Results */}
          {resp && !loading && (
            <>
              {!resp.available || resp.word_feedback?.available === false ? (
                <div className="alert-warning mt-4">
                  Detailed phoneme analysis isn&rsquo;t available. Make sure the server has{' '}
                  <code className="mx-1 px-1 py-0.5 rounded bg-black/5 dark:bg-white/10">g2p_en</code>,
                  <code className="mx-1 px-1 py-0.5 rounded bg-black/5 dark:bg-white/10">praat-parselmouth</code>, and
                  <code className="mx-1 px-1 py-0.5 rounded bg-black/5 dark:bg-white/10">ffmpeg</code> set up.
                </div>
              ) : (
                <PhonemeFeedback items={items || []} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
