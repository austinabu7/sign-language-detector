"""
sign_recognizer.py – Classifies ASL (American Sign Language) letters A-Z
from a list of 21 MediaPipe hand landmarks.

The classifier uses geometric rules derived from finger angles and relative
landmark positions.  This rule-based approach requires no training data,
runs entirely on CPU, and has zero external ML dependencies beyond MediaPipe
itself.

Landmark index reference (MediaPipe):
    0  – WRIST
    1  – THUMB_CMC
    2  – THUMB_MCP
    3  – THUMB_IP
    4  – THUMB_TIP
    5  – INDEX_FINGER_MCP
    6  – INDEX_FINGER_PIP
    7  – INDEX_FINGER_DIP
    8  – INDEX_FINGER_TIP
    9  – MIDDLE_FINGER_MCP
    10 – MIDDLE_FINGER_PIP
    11 – MIDDLE_FINGER_DIP
    12 – MIDDLE_FINGER_TIP
    13 – RING_FINGER_MCP
    14 – RING_FINGER_PIP
    15 – RING_FINGER_DIP
    16 – RING_FINGER_TIP
    17 – PINKY_MCP
    18 – PINKY_PIP
    19 – PINKY_DIP
    20 – PINKY_TIP
"""

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------

def _dist(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two landmarks (x, y only)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_extended(landmarks: list[list[float]], tip: int, pip: int) -> bool:
    """Return True when *tip* is above (smaller y) its *pip* joint."""
    return landmarks[tip][1] < landmarks[pip][1]


def _thumb_extended(landmarks: list[list[float]]) -> bool:
    """Heuristic for thumb extension based on x-distance from wrist."""
    # Works for right hand facing camera; sufficient for landmark geometry
    return landmarks[4][0] < landmarks[3][0] or _dist(landmarks[4], landmarks[0]) > _dist(landmarks[3], landmarks[0])


def _all_fingers_extended(landmarks: list[list[float]]) -> bool:
    return (
        _finger_extended(landmarks, 8, 6)
        and _finger_extended(landmarks, 12, 10)
        and _finger_extended(landmarks, 16, 14)
        and _finger_extended(landmarks, 20, 18)
    )


def _no_fingers_extended(landmarks: list[list[float]]) -> bool:
    return not (
        _finger_extended(landmarks, 8, 6)
        or _finger_extended(landmarks, 12, 10)
        or _finger_extended(landmarks, 16, 14)
        or _finger_extended(landmarks, 20, 18)
    )


def _fingers_extended_list(landmarks: list[list[float]]) -> list[bool]:
    """Return [index, middle, ring, pinky] extension flags."""
    return [
        _finger_extended(landmarks, 8, 6),
        _finger_extended(landmarks, 12, 10),
        _finger_extended(landmarks, 16, 14),
        _finger_extended(landmarks, 20, 18),
    ]


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class SignRecognizer:
    """Classifies a hand pose as an ASL letter (A-Z) or None."""

    def recognize(self, landmarks: list[list[float]]) -> Optional[str]:
        """Return the recognised letter or *None* if no confident match.

        Parameters
        ----------
        landmarks:
            21 landmarks as ``[[x, y, z], ...]`` in MediaPipe normalised space.
        """
        if len(landmarks) != 21:
            return None

        ext = _fingers_extended_list(landmarks)
        idx_ext, mid_ext, rng_ext, pnk_ext = ext

        thumb_ext = _thumb_extended(landmarks)

        # Convenience shorthands
        wrist     = landmarks[0]
        thumb_tip = landmarks[4]
        idx_tip   = landmarks[8]
        mid_tip   = landmarks[12]
        rng_tip   = landmarks[16]
        pnk_tip   = landmarks[20]
        idx_pip   = landmarks[6]
        mid_pip   = landmarks[10]
        rng_pip   = landmarks[14]
        pnk_pip   = landmarks[18]
        idx_mcp   = landmarks[5]
        mid_mcp   = landmarks[9]

        # ----------------------------------------------------------------
        # A – closed fist, thumb resting on side (not tucked inside)
        # ----------------------------------------------------------------
        if _no_fingers_extended(landmarks) and not thumb_ext:
            if thumb_tip[1] < landmarks[2][1]:   # thumb above thumb MCP
                return "A"

        # ----------------------------------------------------------------
        # B – all 4 fingers extended straight up, thumb tucked across palm
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and rng_ext and pnk_ext and not thumb_ext:
            # Fingers should be roughly together (close x spread)
            spread = abs(landmarks[8][0] - landmarks[20][0])
            if spread < 0.15:
                return "B"

        # ----------------------------------------------------------------
        # C – curved hand (all fingers and thumb curved, forming a 'C')
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # Tips should be spread out and roughly same height as MCP joints
            # Thumb tip x should be to one side, index tip x to the other
            if abs(thumb_tip[0] - idx_tip[0]) > 0.1 and abs(thumb_tip[1] - idx_tip[1]) < 0.12:
                return "C"

        # ----------------------------------------------------------------
        # D – index extended, middle/ring/pinky curled, thumb touches middle
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if _dist(thumb_tip, mid_tip) < 0.07:
                return "D"

        # ----------------------------------------------------------------
        # E – all fingers curled/bent, fingertips near palm
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # All tips relatively close to wrist
            avg_tip_y = (idx_tip[1] + mid_tip[1] + rng_tip[1] + pnk_tip[1]) / 4
            if avg_tip_y > wrist[1] - 0.05:
                return "E"

        # ----------------------------------------------------------------
        # F – index and thumb touch (OK-like), other 3 fingers up
        # ----------------------------------------------------------------
        if mid_ext and rng_ext and pnk_ext and not idx_ext:
            if _dist(thumb_tip, idx_tip) < 0.06:
                return "F"

        # ----------------------------------------------------------------
        # G – index pointing sideways, thumb parallel (gun-like shape)
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thumb_ext:
            # Index should be pointing horizontally (small y difference)
            if abs(idx_tip[1] - idx_mcp[1]) < 0.07:
                return "G"

        # ----------------------------------------------------------------
        # H – index and middle extended horizontally side by side
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            # Both fingers pointing roughly sideways
            if abs(idx_tip[1] - idx_mcp[1]) < 0.08 and abs(mid_tip[1] - mid_mcp[1]) < 0.08:
                return "H"

        # ----------------------------------------------------------------
        # I – only pinky extended
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and pnk_ext:
            return "I"

        # ----------------------------------------------------------------
        # J – like I but with motion (we detect the static pose only – 'I')
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # K – index and middle up, thumb between them
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and not rng_ext and not pnk_ext and thumb_ext:
            # Thumb tip between index and middle finger tips
            if idx_tip[0] < thumb_tip[0] < mid_tip[0] or mid_tip[0] < thumb_tip[0] < idx_tip[0]:
                return "K"

        # ----------------------------------------------------------------
        # L – L-shape: index up, thumb out horizontal
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thumb_ext:
            # Index pointing up, thumb pointing sideways
            if idx_tip[1] < idx_mcp[1] and abs(thumb_tip[1] - wrist[1]) < 0.10:
                return "L"

        # ----------------------------------------------------------------
        # M – three fingers folded over thumb
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # Thumb tucked, three finger tips roughly at same level above thumb
            if (
                abs(idx_tip[1] - mid_tip[1]) < 0.04
                and abs(mid_tip[1] - rng_tip[1]) < 0.04
                and thumb_tip[1] > idx_tip[1]
            ):
                return "M"

        # ----------------------------------------------------------------
        # N – index and middle folded over thumb
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if (
                abs(idx_tip[1] - mid_tip[1]) < 0.05
                and thumb_tip[1] > idx_tip[1]
                and rng_tip[1] > idx_tip[1]
            ):
                return "N"

        # ----------------------------------------------------------------
        # O – all fingers and thumb form a circle (tips close together)
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            avg_tip_y = (idx_tip[1] + mid_tip[1] + rng_tip[1] + pnk_tip[1]) / 4
            if _dist(thumb_tip, idx_tip) < 0.06 and avg_tip_y < wrist[1]:
                return "O"

        # ----------------------------------------------------------------
        # P – index pointing down, thumb horizontal (like K tilted)
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thumb_ext:
            if idx_tip[1] > idx_mcp[1]:    # pointing downward
                return "P"

        # ----------------------------------------------------------------
        # Q – like G but pointing downward
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thumb_ext:
            if idx_tip[1] > wrist[1]:
                return "Q"

        # ----------------------------------------------------------------
        # R – index and middle crossed (close x distance)
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thumb_ext:
            # Fingers close together / crossed
            if abs(idx_tip[0] - mid_tip[0]) < 0.03:
                return "R"

        # ----------------------------------------------------------------
        # S – fist with thumb over fingers
        # ----------------------------------------------------------------
        if _no_fingers_extended(landmarks) and not thumb_ext:
            if thumb_tip[1] < idx_tip[1]:   # thumb crosses over index
                return "S"

        # ----------------------------------------------------------------
        # T – thumb between index and middle (tucked)
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if (
                landmarks[4][1] < landmarks[8][1]
                and landmarks[4][0] > landmarks[5][0]
            ):
                return "T"

        # ----------------------------------------------------------------
        # U – index and middle extended together pointing up
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thumb_ext:
            if abs(idx_tip[0] - mid_tip[0]) < 0.05:
                return "U"

        # ----------------------------------------------------------------
        # V – index and middle extended in a V (spread apart)
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thumb_ext:
            if abs(idx_tip[0] - mid_tip[0]) > 0.05:
                return "V"

        # ----------------------------------------------------------------
        # W – index, middle, ring extended, spread out
        # ----------------------------------------------------------------
        if idx_ext and mid_ext and rng_ext and not pnk_ext and not thumb_ext:
            return "W"

        # ----------------------------------------------------------------
        # X – index finger hooked/bent
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # Index tip closer to palm than PIP suggests a hooked finger
            if landmarks[8][1] > landmarks[6][1] and landmarks[6][1] < landmarks[5][1]:
                return "X"

        # ----------------------------------------------------------------
        # Y – thumb and pinky extended (hang-loose / shaka)
        # ----------------------------------------------------------------
        if not idx_ext and not mid_ext and not rng_ext and pnk_ext and thumb_ext:
            return "Y"

        # ----------------------------------------------------------------
        # Z – index finger traces a Z (we detect the pointing pose only)
        # ----------------------------------------------------------------
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thumb_ext:
            return "Z"

        return None
