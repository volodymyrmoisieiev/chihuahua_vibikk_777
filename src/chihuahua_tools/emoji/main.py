from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
TONGUE_OUT_THRESHOLD = 0.03

MEME_NEUTRAL = "chihuahua.png"
MEME_TONGUE = "withtongue.png"


def _assets_dir() -> Path:
    """Return the path to the package `assets/images` directory."""
    return Path(str(files("chihuahua_tools").joinpath("assets", "images")))


ASSETS_DIR = _assets_dir()
NEUTRAL_PATH = ASSETS_DIR / MEME_NEUTRAL
TONGUE_PATH = ASSETS_DIR / MEME_TONGUE

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
)


def is_tongue_out(face_landmarks: Any) -> bool:
    """Return True if the mouth opening exceeds the configured threshold."""
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    mouth_opening = abs(upper_lip.y - lower_lip.y)
    return mouth_opening > TONGUE_OUT_THRESHOLD


def main() -> None:
    """Run webcam loop and switch the meme image based on mouth opening."""
    print("=" * 60)
    print("Tongue Detection Meme Display")
    print("=" * 60)

    if not NEUTRAL_PATH.exists():
        print(f"\n[ERROR] {MEME_NEUTRAL} not found at: {NEUTRAL_PATH}")
        return
    if not TONGUE_PATH.exists():
        print(f"\n[ERROR] {MEME_TONGUE} not found at: {TONGUE_PATH}")
        return

    apple_img = cv2.imread(str(NEUTRAL_PATH))
    appletongue_img = cv2.imread(str(TONGUE_PATH))
    if apple_img is None or appletongue_img is None:
        print("\n[ERROR] Could not load meme images. Check PNG files.")
        return
    print("[OK] Meme images loaded successfully!")

    apple_img = cv2.resize(apple_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    appletongue_img = cv2.resize(appletongue_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    print("[OK] Webcam initialized successfully!")

    cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Meme Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Input", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow("Meme Output", WINDOW_WIDTH, WINDOW_HEIGHT)

    print("\n[CAMERA] Windows opened")
    print("[TONGUE] Stick your tongue out to change the meme! (press 'q' to quit)\n")

    current_meme = apple_img.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[ERROR] Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if is_tongue_out(face_landmarks):
                    current_meme = appletongue_img.copy()
                    cv2.putText(
                        frame,
                        "TONGUE OUT!",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                    )
                else:
                    current_meme = apple_img.copy()
                    cv2.putText(
                        frame,
                        "No tongue detected",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
        else:
            current_meme = apple_img.copy()
            cv2.putText(
                frame,
                "No face detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Camera Input", frame)
        cv2.imshow("Meme Output", current_meme)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[QUIT] Quitting application...")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[OK] Application closed successfully.")


if __name__ == "__main__":
    main()
