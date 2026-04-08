"""
Configuration parameters for the Sign Language Detector application.
Adjust these values to tune detection sensitivity and performance.
"""

# Camera settings
CAMERA_INDEX = 0          # Default camera (0 = built-in webcam)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# MediaPipe hand detection settings
MAX_NUM_HANDS = 1                   # Only process one hand at a time
MIN_DETECTION_CONFIDENCE = 0.7     # Minimum confidence to detect a hand
MIN_TRACKING_CONFIDENCE = 0.6      # Minimum confidence to track a hand

# Letter recognition settings
CONFIDENCE_THRESHOLD = 3           # Number of consecutive frames required to confirm a letter
MIN_LETTER_INTERVAL = 0.4          # Minimum seconds between accepted letters (prevents duplicates)

# Word / sentence formation settings
PAUSE_DURATION = 1.5               # Seconds of no sign before adding a space between words
WORD_COMPLETE_PAUSE = 1.5          # Alias for clarity – same as PAUSE_DURATION
MAX_TEXT_LENGTH = 200              # Maximum characters in the full sentence

# Speech settings
TTS_RATE = 150                     # Words per minute for text-to-speech
TTS_VOLUME = 1.0                   # Speech volume (0.0 – 1.0)
SPEAK_ON_WORD_COMPLETE = True      # Automatically speak each completed word

# UI / Display settings
FONT_SCALE = 0.8
FONT_THICKNESS = 2
UI_PANEL_HEIGHT = 200              # Height of the info panel below the camera feed

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 100, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_PANEL_BG = (30, 30, 30)
