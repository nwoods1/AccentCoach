export type VowelNote = {
  phoneme_expected: string;
  phoneme_heard: string;
  measured: { F1: number; F2: number };
  target: { F1: number; F2: number };
  distance: number;
};

export type PhonemeWordItem = {
  word_index: number;
  word: string;
  expected: {
    arpabet: string[];
    ipa: string;
    spelling: string;
  };
  heard: {
    arpabet: string[];
    ipa: string;
    spelling: string;
  };
  vowel_notes: VowelNote[];
};

export type AnalyzeResponse = {
  available: boolean;
  word_feedback?: {
    available: boolean;
    items?: PhonemeWordItem[];
    words?: string[];
  };
  _debug?: Record<string, any>;
  error?: string;
};
