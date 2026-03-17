import time

import cv2

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONTROL_POINT_COLOR,
    CONTROL_POINT_RADIUS,
    DRAG_HOLD_SECONDS,
    SMOOTHING_POINTS,
    STATUS_TEXT_COLOR,
    WINDOW_NAME,
)
from hand_tracker import HandTracker
from mouse_controller import MouseController
from mouse_mapper import MouseMapper
from point_smoother import PointSmoother


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
    smoother = PointSmoother(SMOOTHING_POINTS)
    pinch_gesture_active = False
    right_click_gesture_active = False
    pinch_started_at = None
    drag_active = False

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
            thumb_point = None
            middle_point = None
            smoothed_index_point = None
            screen_point = None
            pinch_ratio = None
            right_click_ratio = None
            pinch_duration = None
            if tracker.available and results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                tracker.draw_landmarks(frame, hand_landmarks)
                index_point = tracker.get_index_finger_tip(hand_landmarks, frame_width, frame_height)
                thumb_point = tracker.get_thumb_tip(hand_landmarks, frame_width, frame_height)
                middle_point = tracker.get_middle_finger_tip(hand_landmarks, frame_width, frame_height)
                pinch_ratio = tracker.get_pinch_distance_ratio(hand_landmarks)
                right_click_ratio = tracker.get_right_click_distance_ratio(hand_landmarks)
                is_left_click_gesture = tracker.is_left_click_gesture(hand_landmarks)
                is_right_click_gesture = tracker.is_right_click_gesture(hand_landmarks)

                if index_point is not None:
                    cv2.circle(frame, index_point, CONTROL_POINT_RADIUS, CONTROL_POINT_COLOR, -1)
                    smoothed_index_point = smoother.smooth(index_point)
                    if smoothed_index_point is not None:
                        cv2.circle(frame, smoothed_index_point, CONTROL_POINT_RADIUS + 4, STATUS_TEXT_COLOR, 2)
                    if mouse.available and mouse_mapper is not None:
                        screen_point = mouse_mapper.map_point(smoothed_index_point or index_point)
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

                if thumb_point is not None:
                    cv2.circle(frame, thumb_point, CONTROL_POINT_RADIUS, CONTROL_POINT_COLOR, 2)

                if thumb_point is not None and index_point is not None:
                    cv2.line(frame, thumb_point, index_point, CONTROL_POINT_COLOR, 2)
                if middle_point is not None:
                    cv2.circle(frame, middle_point, CONTROL_POINT_RADIUS, STATUS_TEXT_COLOR, 2)
                if thumb_point is not None and middle_point is not None:
                    cv2.line(frame, thumb_point, middle_point, STATUS_TEXT_COLOR, 2)

                current_time = time.monotonic()
                if is_left_click_gesture:
                    if not pinch_gesture_active:
                        pinch_gesture_active = True
                        pinch_started_at = current_time
                    pinch_duration = current_time - pinch_started_at if pinch_started_at is not None else 0.0

                    if not drag_active and pinch_duration >= DRAG_HOLD_SECONDS and mouse.available:
                        mouse.left_down()
                        drag_active = True

                elif pinch_gesture_active:
                    if drag_active and mouse.available:
                        mouse.left_up()
                        status_text = "Drag released"
                    elif mouse.available:
                        mouse.left_click()
                        status_text = "Left click"

                    pinch_gesture_active = False
                    pinch_started_at = None
                    drag_active = False
                elif is_right_click_gesture and not right_click_gesture_active and not pinch_gesture_active:
                    if mouse.available:
                        mouse.right_click()
                        status_text = "Right click"

                right_click_gesture_active = is_right_click_gesture

                if drag_active:
                    status_text = "Dragging"
                elif pinch_gesture_active and pinch_duration is not None:
                    status_text = f"Pinch hold: {pinch_duration:.2f}s"
                elif right_click_gesture_active:
                    status_text = "Right click gesture"
                elif screen_point is not None:
                    status_text = f"Cursor: {screen_point[0]}, {screen_point[1]} (smoothed)"
                elif index_point is not None:
                    status_text = f"Index tip: {index_point[0]}, {index_point[1]}"
                else:
                    status_text = "Hand detected"
            else:
                smoother.reset()
                tracker.reset()
                pinch_gesture_active = False
                right_click_gesture_active = False
                pinch_started_at = None
                if drag_active and mouse.available:
                    mouse.left_up()
                drag_active = False

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
            if smoothed_index_point is not None:
                cv2.putText(
                    frame,
                    f"Smoothed: {smoothed_index_point[0]}, {smoothed_index_point[1]}",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    STATUS_TEXT_COLOR,
                    2,
                )
            if pinch_ratio is not None:
                cv2.putText(
                    frame,
                    f"Pinch ratio: {pinch_ratio:.2f}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    STATUS_TEXT_COLOR,
                    2,
                )
            if right_click_ratio is not None:
                cv2.putText(
                    frame,
                    f"Right pinch ratio: {right_click_ratio:.2f}",
                    (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    STATUS_TEXT_COLOR,
                    2,
                )
            if pinch_duration is not None:
                cv2.putText(
                    frame,
                    f"Pinch hold: {pinch_duration:.2f}s / {DRAG_HOLD_SECONDS:.2f}s",
                    (20, 280),
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
        if drag_active and mouse.available:
            mouse.left_up()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
