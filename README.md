# Sign Language Detector 🤟

A real-time American Sign Language (ASL) detector that reads hand poses from your webcam,
recognises letters A–Z, assembles them into words, and speaks the result aloud — all without
blocking the camera feed.

## Features

- 🎥 **Real-time hand tracking** – MediaPipe detects your hand skeleton at up to 30 fps
- 🔤 **ASL A–Z recognition** – rule-based geometric classifier, no training data needed
- 💬 **Smart word formation** – consecutive letters merge into words; a 1.5-second pause
  inserts a space and triggers speech for the completed word
- 🔊 **Non-blocking TTS** – `pyttsx3` speaks in a daemon thread so the UI never freezes
- 📊 **Live HUD** – shows detected letter, streak progress, word buffer, completed words,
  and the full sentence in a clean overlay

## Project Structure

```
sign-language-detector/
├── main.py            # Main application loop (entry point)
├── hand_detector.py   # MediaPipe hand-landmark wrapper
├── sign_recognizer.py # ASL letter classifier (A-Z geometric rules)
├── text_processor.py  # Letter → word → sentence state machine
├── speech_engine.py   # Non-blocking pyttsx3 text-to-speech
├── config.py          # All tunable parameters in one place
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.10 or later
- A webcam accessible at index `0` (configurable in `config.py`)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sign-language-detector.git
cd sign-language-detector

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Keyboard Controls

| Key   | Action                              |
|-------|-------------------------------------|
| `Q`   | Quit the application                |
| `C`   | Clear all text and start over       |
| `SPACE` | Manually speak the current text   |

### Example session

```
You sign: H → I  →  (1.5 s pause)  →  B → R → O  →  (pause)
Display : "HI BRO"
Speech  : "HI" … "BRO"
```

## Configuration

Edit `config.py` to tune behaviour:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam index |
| `CONFIDENCE_THRESHOLD` | `3` | Frames required to confirm a letter |
| `MIN_LETTER_INTERVAL` | `0.4` | Minimum seconds between accepted letters |
| `PAUSE_DURATION` | `1.5` | Seconds of silence before a word boundary |
| `TTS_RATE` | `150` | Speech rate (words per minute) |
| `SPEAK_ON_WORD_COMPLETE` | `True` | Auto-speak each completed word |

## How It Works

1. **HandDetector** captures frames from the webcam and runs MediaPipe Hands to extract
   21 3-D landmarks per detected hand.
2. **SignRecognizer** applies geometric rules (finger extension, relative distances,
   angles) to map the landmark pattern to an ASL letter.
3. **TextProcessor** tracks the streak of identical letters across consecutive frames.
   Once the streak reaches `CONFIDENCE_THRESHOLD` and enough time has passed since the
   last accepted letter, the letter is appended to the word buffer.  When no sign is
   detected for `PAUSE_DURATION` seconds the word is committed to the sentence.
4. **SpeechEngine** dispatches `pyttsx3` speech jobs to daemon threads — the camera
   loop never blocks waiting for audio to finish.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Camera not found | Change `CAMERA_INDEX` in `config.py` |
| No audio on Linux | Install `espeak`: `sudo apt-get install espeak` |
| Low recognition accuracy | Ensure good lighting and keep your hand centred in the frame |
| Letters added too quickly | Increase `MIN_LETTER_INTERVAL` in `config.py` |

## License

This project is licensed under the MIT License.
