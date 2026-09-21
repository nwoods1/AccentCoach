# AccentCoach

AccentCoach is a speech-practice web app that analyzes recorded English pronunciation using machine learning and provides accent classification, confidence scores, and targeted vowel feedback.

## Overview

AccentCoach helps users practice spoken English by turning pronunciation practice into a guided, measurable workflow. Users read prompted sentences into their microphone, the app verifies that the correct sentence was spoken, and a trained audio classifier returns a Native or Non-Native classification with a confidence score.

A separate pronunciation-analysis mode provides word-level feedback by comparing expected and detected vowel sounds and generating targeted coaching suggestions.

## Key Features

- Guided 5-level practice course with progress saved across sessions
- Sentence verification before scoring using speech-to-text to confirm the user read the assigned sentence
- Native / Non-Native accent classification with a confidence score from a trained audio classifier
- Pronunciation analyzer that provides per-word expected-versus-heard IPA feedback and vowel-adjustment tips
- Multiple authentication options including email/password, Google sign-in, and anonymous guest access
- Progress tracking with completed levels, locked levels, recent results, and attempt history stored in Firestore
- Light and dark themes in the practice experience

## Tech Stack

**Frontend:** React 19, TypeScript, React Router, Tailwind CSS, Firebase JS SDK

**Backend:** Python, Flask, Flask-CORS

**Machine Learning & Audio:** scikit-learn, librosa, NumPy, joblib, pydub, ffmpeg, faster-whisper, g2p_en, praat-parselmouth

**Authentication & Database:** Firebase Authentication, Cloud Firestore

## How It Works

1. The user signs in with email/password, Google, or guest mode.
2. The user selects a practice level and records the prompted sentence in the browser.
3. The recording is transcribed and compared with the target sentence.
4. If verification succeeds, the recording is passed to the accent classifier.
5. The model returns a Native or Non-Native prediction with a confidence score.
6. Signed-in users have their result, progress, and attempt history stored in Firestore.
7. In the separate pronunciation-analysis flow, users can record any sentence and receive word-level vowel feedback without affecting course progress.

## Machine Learning

### Dataset

The classifier is trained using native-speaker recordings from the CMU ARCTIC corpus (bdl and slt) together with a non-native speaker corpus organized by speaker. The training set is balanced by subsampling the larger class.

### Audio Processing

Recordings are silence-trimmed, peak-normalized, and converted to 16 kHz mono WAV before feature extraction. Training audio is also augmented with gain changes, additive noise, time stretching, and small time shifts to improve robustness across recording conditions and speaking rates.

### Feature Extraction

The model uses MFCC-based acoustic features, including:

- 20 MFCCs
- First- and second-order MFCC derivatives
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff

Each feature series is reduced to summary statistics such as mean, standard deviation, median, and percentile values, producing a fixed-length feature vector for each recording.

### Model

The current classifier uses a scikit-learn pipeline:

```
StandardScaler -> LogisticRegression
```

Regularization strength is tuned with `GridSearchCV`. Cross-validation folds are speaker-disjoint so recordings from the same speaker do not appear in both training and validation data, reducing the risk of the model learning individual voices rather than broader acoustic patterns.

At inference time, the backend extracts the same features from the uploaded recording and uses `predict_proba` to return the predicted class and confidence score.

## Architecture

- **Frontend:** React and TypeScript handle authentication, recording, course progress, and pronunciation feedback.
- **Backend:** Flask exposes endpoints for accent prediction, sentence verification, and pronunciation analysis.
- **Machine Learning:** The trained scikit-learn model is loaded by the Flask server and used to score uploaded recordings.
- **Pronunciation Analysis:** Speech-to-text, grapheme-to-phoneme conversion, and acoustic formant analysis are combined to generate word-level vowel feedback.
- **Authentication:** Firebase Authentication manages email/password, Google, and anonymous sign-in.
- **Database:** Cloud Firestore stores user profiles, course progress, recent results, and attempt history.

## Technical Challenges & Engineering Decisions

### Preventing Incorrect Sentences From Being Scored

Scoring accent alone would allow a user to submit any clearly pronounced sentence instead of the assigned prompt. AccentCoach uses a two-step verify-then-classify workflow so speech-to-text verification must succeed before the accent model runs.

### Avoiding Speaker Leakage

Because the dataset contains a limited number of speakers, a random train-validation split could allow the model to recognize individual voices instead of learning broader pronunciation patterns. Speaker-disjoint cross-validation keeps each speaker entirely within either training or validation for a given fold.

### Handling Variable-Length Audio

Audio recordings vary in duration, while the classifier requires fixed-size input. Frame-level MFCC and spectral features are summarized using statistical measures to create a consistent-length feature vector for every recording.

### Providing Word-Level Feedback Without Forced Alignment

Accurate phoneme-level timing normally requires a dedicated forced aligner. AccentCoach uses a simpler approximation that divides recordings into word-level time segments and analyzes acoustic formants within those segments, trading some timing precision for a lighter-weight implementation.

## Limitations

- Accent scoring is currently binary rather than graded or accent-specific.
- The training dataset contains a limited number of speakers, so the model may not generalize equally well across all voices, accents, microphones, and recording environments.
- Word-level pronunciation timing is approximate and can become less accurate when speech contains long pauses, filler sounds, or uneven pacing.

## Future Improvements

- Expand the training dataset with more speakers, accents, microphones, and recording environments.
- Replace approximate word segmentation with forced alignment for more accurate phoneme-level feedback.
- Move from binary classification toward graded pronunciation scoring and more detailed accent-specific feedback.