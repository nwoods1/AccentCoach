import { AnalyzeResponse } from "../types/phoneme";

const BASE_URL = "http://localhost:5001";

export async function analyzePronunciation(
  audioBlob: Blob,
  sentence: string,
  debug: boolean = false
): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("audio", audioBlob, "recording.webm");
  fd.append("sentence", sentence);

  const url = `${BASE_URL}/analyze-pronunciation${debug ? "?debug=1" : ""}`;
  const res = await fetch(url, { method: "POST", body: fd });

  if (!res.ok) {
    let msg = "Server error";
    try {
      const j = await res.json();
      msg = j?.error || msg;
    } catch {}
    throw new Error(msg);
  }
  return (await res.json()) as AnalyzeResponse;
}
