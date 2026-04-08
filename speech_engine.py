"""
speech_engine.py – Non-blocking text-to-speech using pyttsx3 and threading.

Each speech request is dispatched to a dedicated daemon thread so the camera
feed and UI remain fully responsive during playback.
"""

import threading
import logging
from typing import Optional

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False

from config import TTS_RATE, TTS_VOLUME

logger = logging.getLogger(__name__)


class SpeechEngine:
    """Converts text to speech in a background thread.

    Falls back gracefully (prints to console) when pyttsx3 is unavailable
    or when the audio subsystem cannot be initialised.
    """

    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME) -> None:
        self._rate = rate
        self._volume = volume
        self._lock = threading.Lock()
        self._speaking = False
        self._available = _PYTTSX3_AVAILABLE
        self._engine: Optional[object] = None

        if self._available:
            self._init_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Speak *text* in a background thread (non-blocking).

        If the engine is already speaking the request is ignored to avoid
        overlapping audio.
        """
        if not text.strip():
            return

        with self._lock:
            if self._speaking:
                return
            self._speaking = True

        thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
        )
        thread.start()

    @property
    def is_speaking(self) -> bool:
        """True while a speech job is running."""
        with self._lock:
            return self._speaking

    def stop(self) -> None:
        """Stop any ongoing speech."""
        if self._available and self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        """Try to create the pyttsx3 engine once during construction."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            self._engine = engine
            logger.info("pyttsx3 TTS engine initialised successfully.")
        except Exception as exc:
            logger.warning("Could not initialise pyttsx3: %s – falling back to console output.", exc)
            self._available = False

    def _speak_worker(self, text: str) -> None:
        """Target function for the speech thread."""
        try:
            if self._available and self._engine is not None:
                # pyttsx3 engines are not thread-safe; create a fresh one per job.
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            else:
                # Graceful degradation – just print to stdout
                print(f"[SPEECH] {text}")
        except Exception as exc:
            logger.warning("TTS error: %s", exc)
            print(f"[SPEECH] {text}")
        finally:
            with self._lock:
                self._speaking = False
