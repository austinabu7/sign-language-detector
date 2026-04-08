"""
hand_detector.py – MediaPipe-based hand landmark detection.

Wraps mediapipe.solutions.hands to provide a clean interface
for detecting hand landmarks from a BGR OpenCV frame.
"""

import cv2

try:
    from mediapipe.solutions import hands as _mp_hands_module
    from mediapipe.solutions import drawing_utils as _mp_draw_module
    from mediapipe.solutions import drawing_styles as _mp_draw_styles_module
except ImportError as exc:
    raise ImportError(
        "MediaPipe is not installed or the 'solutions' submodule is missing. "
        "Install a compatible version with: pip install 'mediapipe>=0.9.0,<0.11.0'"
    ) from exc


class HandDetector:
    """Detects hand landmarks in a video frame using MediaPipe Hands."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = _mp_hands_module
        self._mp_draw = _mp_draw_module
        self._mp_draw_styles = _mp_draw_styles_module

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, frame: "np.ndarray", draw: bool = True):  # noqa: F821
        """Process *frame* (BGR) and return (annotated_frame, results).

        Parameters
        ----------
        frame:
            BGR image array from OpenCV.
        draw:
            When True, draw landmark skeleton on the returned frame.

        Returns
        -------
        tuple[np.ndarray, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList | None]
            Annotated frame and raw MediaPipe results object.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw_styles.get_default_hand_landmarks_style(),
                    self._mp_draw_styles.get_default_hand_connections_style(),
                )

        return frame, results

    def get_landmark_list(self, results, frame_shape: tuple) -> list[list[float]]:
        """Extract a flat list of (x, y, z) landmark coordinates.

        Coordinates are normalised to [0, 1] by MediaPipe and kept as floats.

        Parameters
        ----------
        results:
            Raw results object returned by :meth:`find_hands`.
        frame_shape:
            (height, width, channels) of the source frame – used to derive
            pixel coordinates when needed by the recogniser.

        Returns
        -------
        list[list[float]]
            List of 21 landmarks, each ``[x, y, z]``.  Empty list when no
            hand is detected.
        """
        if not results.multi_hand_landmarks:
            return []

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = frame_shape[:2]
        landmark_list: list[list[float]] = []
        for lm in hand_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])

        return landmark_list

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
