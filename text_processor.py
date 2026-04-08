"""
text_processor.py – Handles letter-to-word-to-sentence formation.

Logic overview
--------------
* Letters are added one at a time via :meth:`add_letter`.
* A letter is only accepted if it is the same letter for
  ``CONFIDENCE_THRESHOLD`` consecutive frames AND at least
  ``MIN_LETTER_INTERVAL`` seconds have elapsed since the previous accepted
  letter (avoids duplicates while holding a sign).
* If no sign is detected for ``PAUSE_DURATION`` seconds the current *word*
  is finalised and a space is inserted before the next letter.
* The full sentence is available via :attr:`full_text`.
"""

import time
from config import (
    CONFIDENCE_THRESHOLD,
    MIN_LETTER_INTERVAL,
    PAUSE_DURATION,
    MAX_TEXT_LENGTH,
)


class TextProcessor:
    """Accumulates letters into words and words into sentences."""

    def __init__(self) -> None:
        self._current_letter: str = ""          # letter seen this frame
        self._letter_streak: int = 0             # consecutive frames with same letter
        self._last_accepted_letter: str = ""     # last letter added to word
        self._last_accepted_time: float = 0.0   # when last letter was accepted
        self._last_sign_time: float = time.time()  # last frame with *any* sign

        self.letter_buffer: str = ""            # current word being built
        self.words: list[str] = []              # completed words
        self._space_pending: bool = False       # True after a pause
        self.full_text: str = ""               # complete sentence so far

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detected_letter: str | None) -> None:
        """Call once per frame with the currently detected letter (or None).

        Parameters
        ----------
        detected_letter:
            Single uppercase letter or *None* when no sign is visible.
        """
        now = time.time()

        if detected_letter:
            self._last_sign_time = now
            self._space_pending = False

            # Track consecutive-frame streak
            if detected_letter == self._current_letter:
                self._letter_streak += 1
            else:
                self._current_letter = detected_letter
                self._letter_streak = 1

            # Accept letter when streak meets threshold AND enough time has passed
            if (
                self._letter_streak >= CONFIDENCE_THRESHOLD
                and (now - self._last_accepted_time) >= MIN_LETTER_INTERVAL
                and detected_letter != self._last_accepted_letter
            ):
                self._accept_letter(detected_letter, now)
        else:
            # No sign detected
            self._current_letter = ""
            self._letter_streak = 0

            # Check for pause → word boundary
            elapsed = now - self._last_sign_time
            if elapsed >= PAUSE_DURATION and self.letter_buffer and not self._space_pending:
                self._finalise_word()

    def clear(self) -> None:
        """Reset all state (bound to keyboard 'C')."""
        self._current_letter = ""
        self._letter_streak = 0
        self._last_accepted_letter = ""
        self._last_accepted_time = 0.0
        self._last_sign_time = time.time()
        self._space_pending = False
        self.letter_buffer = ""
        self.words = []
        self.full_text = ""

    @property
    def current_letter(self) -> str:
        """The letter currently being tracked (not yet confirmed)."""
        return self._current_letter

    @property
    def streak(self) -> int:
        """Current consecutive-frame streak for the tracked letter."""
        return self._letter_streak

    @property
    def pause_progress(self) -> float:
        """Fraction (0.0-1.0) of the pause duration elapsed since last sign."""
        if self.letter_buffer:
            elapsed = time.time() - self._last_sign_time
            return min(elapsed / PAUSE_DURATION, 1.0)
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accept_letter(self, letter: str, now: float) -> None:
        """Add a confirmed letter to the current word buffer."""
        self._last_accepted_letter = letter
        self._last_accepted_time = now
        # Guard against exceeding MAX_TEXT_LENGTH across both committed text
        # and the in-progress buffer (the two are joined by a space on finalise).
        separator_len = 1 if self.full_text and self.letter_buffer else 0
        total_len = len(self.full_text) + separator_len + len(self.letter_buffer) + 1
        if total_len <= MAX_TEXT_LENGTH:
            self.letter_buffer += letter

    def _finalise_word(self) -> None:
        """Commit the current word and prepare for the next one."""
        if self.letter_buffer:
            self.words.append(self.letter_buffer)
            if self.full_text:
                self.full_text += " " + self.letter_buffer
            else:
                self.full_text = self.letter_buffer
            self.letter_buffer = ""
            self._last_accepted_letter = ""
            self._space_pending = True

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_display_text(self) -> str:
        """Return completed words plus the in-progress buffer."""
        base = self.full_text
        if self.letter_buffer:
            separator = " " if base else ""
            base = base + separator + self.letter_buffer
        return base

    def get_last_word(self) -> str:
        """Return the most recently completed word (or empty string)."""
        return self.words[-1] if self.words else ""
