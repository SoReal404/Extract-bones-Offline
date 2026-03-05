import mss
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import keyboard  # pip install keyboard

# -----------------------------
# 1️⃣ Model Setup
# -----------------------------
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
session_name = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=5  # detect up to 5 people
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# -----------------------------
# 2️⃣ Dataset Folder
# -----------------------------
base_folder = "dataset"
session_folder = os.path.join(base_folder, session_name)
os.makedirs(session_folder, exist_ok=True)

# -----------------------------
# 3️⃣ Screen Capture Setup
# -----------------------------
with mss.mss() as sct:
    monitor = sct.monitors[1]
    print("Press 's' anywhere to take a screenshot, 'q' to quit.")

    while True:
        frame = np.array(sct.grab(monitor))  # BGRA

        # Optional live preview
        cv2.imshow("Live Preview", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
        cv2.waitKey(1)  # required to refresh window

        # -----------------------------
        # 4️⃣ Global hotkeys
        # -----------------------------
        if keyboard.is_pressed("s"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                print(f"{len(result.pose_landmarks)} person(s) detected.")
                for person_idx, pose in enumerate(result.pose_landmarks):
                    person_frame = frame.copy()
                    for landmark in pose:
                        x = int(landmark.x * monitor["width"])
                        y = int(landmark.y * monitor["height"])
                        cv2.circle(person_frame, (x, y), 5, (0, 255, 0), -1)

                    # Save image
                    image_path = os.path.join(session_folder, f"person_{person_idx+1}_{timestamp}.png")
                    cv2.imwrite(image_path, person_frame)

                    # Save landmarks
                    txt_path = os.path.join(session_folder, f"person_{person_idx+1}_{timestamp}.txt")
                    with open(txt_path, "w") as f:
                        for idx, landmark in enumerate(pose):
                            f.write(f"{idx} {landmark.x:.6f} {landmark.y:.6f} {landmark.z:.6f} {landmark.visibility:.6f}\n")
                print(f"Screenshot taken at {timestamp}!")

            else:
                print("No poses detected.")

            # Wait a bit to avoid multiple triggers
            keyboard.wait("s", suppress=False)  # wait until released

        elif keyboard.is_pressed("q"):
            print("Quitting...")
            break

cv2.destroyAllWindows()
landmarker.close()