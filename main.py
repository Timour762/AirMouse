import cv2

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    STATUS_TEXT_COLOR,
    WINDOW_NAME,
)
from hand_tracker import HandTracker


def open_camera():
    if hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture

    return cv2.VideoCapture(CAMERA_INDEX)


def main():
    cap = open_camera()
    tracker = HandTracker()

    if not cap.isOpened():
        raise RuntimeError("Camera is not opening. Check CAMERA_INDEX and camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)

            status_text = tracker.error or "Show your hand to the camera"
            if tracker.available and results and results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    tracker.draw_landmarks(frame, hand_landmarks)

                    handedness = tracker.get_handedness_label(results, hand_index)
                    if handedness:
                        cv2.putText(
                            frame,
                            handedness,
                            (20, 80 + hand_index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            STATUS_TEXT_COLOR,
                            2,
                        )
                status_text = "Hand detected"

            cv2.putText(
                frame,
                "AirMouse hand tracking - press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                STATUS_TEXT_COLOR,
                2,
            )
            cv2.putText(
                frame,
                status_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                STATUS_TEXT_COLOR,
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
