import cv2

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONTROL_POINT_COLOR,
    CONTROL_POINT_RADIUS,
    STATUS_TEXT_COLOR,
    WINDOW_NAME,
)
from hand_tracker import HandTracker
from mouse_controller import MouseController
from mouse_mapper import MouseMapper


def open_camera():
    if hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture

    return cv2.VideoCapture(CAMERA_INDEX)


def main():
    cap = open_camera()
    tracker = HandTracker()
    mouse = MouseController()

    if not cap.isOpened():
        raise RuntimeError("Camera is not opening. Check CAMERA_INDEX and camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame from the camera.")

    frame_height, frame_width = frame.shape[:2]
    mouse_mapper = None
    if mouse.available:
        mouse_mapper = MouseMapper(
            frame_width,
            frame_height,
            mouse.screen_size[0],
            mouse.screen_size[1],
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)

            status_text = tracker.error or mouse.error or "Show your hand to the camera"
            index_point = None
            screen_point = None
            if tracker.available and results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                tracker.draw_landmarks(frame, hand_landmarks)
                index_point = tracker.get_index_finger_tip(hand_landmarks, frame_width, frame_height)

                if index_point is not None:
                    cv2.circle(frame, index_point, CONTROL_POINT_RADIUS, CONTROL_POINT_COLOR, -1)
                    if mouse.available and mouse_mapper is not None:
                        screen_point = mouse_mapper.map_point(index_point)
                        mouse.move_to(screen_point[0], screen_point[1])

                handedness = tracker.get_handedness_label(results, 0)
                if handedness:
                    cv2.putText(
                        frame,
                        handedness,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        STATUS_TEXT_COLOR,
                        2,
                    )

                if screen_point is not None:
                    status_text = f"Cursor: {screen_point[0]}, {screen_point[1]}"
                elif index_point is not None:
                    status_text = f"Index tip: {index_point[0]}, {index_point[1]}"
                else:
                    status_text = "Hand detected"

            cv2.putText(
                frame,
                "AirMouse cursor control - press Q to quit",
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
            if index_point is not None:
                cv2.putText(
                    frame,
                    f"Finger: {index_point[0]}, {index_point[1]}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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
