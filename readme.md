# Multi-Person Pose Screenshot & Landmark Saver

This project uses **MediaPipe BlazePose** (modern Tasks API) to detect **multiple humans** on the screen, save their pose landmarks, and take screenshots **on key press**.

---

## Features

- Detect up to 5 humans in a single screen capture.
- Press `s` to capture the current screen:
  - Saves **annotated screenshot** with landmarks.
  - Saves **landmark TXT file** with `x, y, z, visibility`.
- Session folder automatically created with timestamp.
- Press `q` to quit the program.

---

## Requirements

- Python 3.14+
- MediaPipe latest version
- OpenCV
- NumPy
- MSS (screen capture library)

Install dependencies:

```bash
pip install mediapipe opencv-python numpy mss