"""
main.py – Entry point for the Sign Language Detector application.

Run with:
    python main.py

Controls:
    Q      – quit
    C      – clear text and start over
    SPACE  – manually trigger speech for the current sentence
"""

import sys
import time
import logging

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_FPS,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    SPEAK_ON_WORD_COMPLETE,
    FONT_SCALE,
    FONT_THICKNESS,
    UI_PANEL_HEIGHT,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_BLACK,
    COLOR_YELLOW,
    COLOR_CYAN,
    COLOR_PANEL_BG,
)
from hand_detector import HandDetector
from sign_recognizer import SignRecognizer
from text_processor import TextProcessor
from speech_engine import SpeechEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = COLOR_WHITE,
    scale: float = FONT_SCALE,
    thickness: int = FONT_THICKNESS,
) -> None:
    cv2.putText(
        img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, COLOR_BLACK, thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _draw_ui_panel(
    frame: np.ndarray,
    processor: TextProcessor,
    speech: SpeechEngine,
    fps: float,
) -> np.ndarray:
    """Append a dark info panel below the camera frame."""
    panel = np.full(
        (UI_PANEL_HEIGHT, frame.shape[1], 3),
        COLOR_PANEL_BG,
        dtype=np.uint8,
    )

    # ── Row 1: FPS + speaking indicator ────────────────────────────────
    fps_label = f"FPS: {fps:.1f}"
    speak_label = "  [Speaking...]" if speech.is_speaking else ""
    _put_text(panel, fps_label + speak_label, (10, 30), COLOR_GREEN, 0.65, 1)

    # ── Row 2: Current detection + streak progress ──────────────────────
    if processor.current_letter:
        streak_pct = min(processor.streak / CONFIDENCE_THRESHOLD, 1.0)
        bar_len = int(200 * streak_pct)
        cv2.rectangle(panel, (200, 10), (400, 30), (60, 60, 60), -1)
        cv2.rectangle(panel, (200, 10), (200 + bar_len, 30), COLOR_GREEN, -1)
        _put_text(
            panel,
            f"Detecting: {processor.current_letter}  [{processor.streak}/{CONFIDENCE_THRESHOLD}]",
            (10, 60),
            COLOR_YELLOW,
            0.75,
            2,
        )
    else:
        _put_text(panel, "Detecting: --", (10, 60), (150, 150, 150), 0.75, 1)

    # ── Row 3: Current word buffer ──────────────────────────────────────
    buf = processor.letter_buffer or "..."
    _put_text(panel, f"Buffer : {buf}", (10, 95), COLOR_CYAN, 0.75, 2)

    # ── Pause progress bar ──────────────────────────────────────────────
    pause_frac = processor.pause_progress
    if pause_frac > 0:
        bar_w = int(frame.shape[1] * pause_frac)
        cv2.rectangle(panel, (0, 100), (bar_w, 108), COLOR_BLUE, -1)
        _put_text(panel, f"Pause: {pause_frac * 100:.0f}%", (10, 125), COLOR_BLUE, 0.6, 1)

    # ── Row 4: Completed words ──────────────────────────────────────────
    words_str = " ".join(processor.words) if processor.words else "---"
    _put_text(panel, f"Words  : {words_str}", (10, 150), COLOR_WHITE, 0.75, 2)

    # ── Row 5: Full sentence ────────────────────────────────────────────
    full = processor.get_display_text() or "---"
    # Truncate long text for display
    if len(full) > 55:
        full = "..." + full[-52:]
    _put_text(panel, f"Text   : {full}", (10, 185), COLOR_GREEN, 0.8, 2)

    return np.vstack([frame, panel])


def _draw_controls_overlay(frame: np.ndarray) -> None:
    """Draw a semi-transparent controls legend in the top-right corner."""
    h, w = frame.shape[:2]
    controls = ["Q=Quit", "C=Clear", "SPACE=Speak"]
    for i, ctrl in enumerate(controls):
        _put_text(
            frame, ctrl,
            (w - 160, 30 + i * 28),
            COLOR_YELLOW, 0.6, 1,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Sign Language Detector…")

    # Initialise components
    detector = HandDetector(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    recognizer = SignRecognizer()
    processor = TextProcessor()
    speech = SpeechEngine()

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", CAMERA_INDEX)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    logger.info("Camera opened.  Press Q to quit, C to clear, SPACE to speak.")

    prev_time = time.time()
    prev_word_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame – retrying…")
                time.sleep(0.05)
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Detect hands and extract landmarks
            frame, results = detector.find_hands(frame, draw=True)
            landmarks = detector.get_landmark_list(results, frame.shape)

            # Recognise letter
            letter = recognizer.recognize(landmarks) if landmarks else None

            # Update text state
            processor.update(letter)

            # Auto-speak newly completed words
            current_word_count = len(processor.words)
            if (
                SPEAK_ON_WORD_COMPLETE
                and current_word_count > prev_word_count
                and not speech.is_speaking
            ):
                speech.speak(processor.get_last_word())
            prev_word_count = current_word_count

            # FPS calculation
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Overlay controls
            _draw_controls_overlay(frame)

            # Compose final display
            display = _draw_ui_panel(frame, processor, speech, fps)

            cv2.imshow("Sign Language Detector", display)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Quit requested.")
                break
            elif key == ord("c") or key == ord("C"):
                processor.clear()
                logger.info("Text cleared.")
            elif key == ord(" "):
                text_to_speak = processor.get_display_text()
                if text_to_speak:
                    speech.speak(text_to_speak)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Resources released.  Goodbye!")


if __name__ == "__main__":
    main()
