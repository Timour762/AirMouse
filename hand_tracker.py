from collections import deque

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

from config import (
    FINGER_Y_MARGIN,
    HAND_CONNECTION_COLOR,
    HAND_DRAW_RADIUS,
    HAND_DRAW_THICKNESS,
    HAND_LANDMARK_COLOR,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    PINCH_DISTANCE_RATIO,
    PINCH_SMOOTHING_FRAMES,
    RIGHT_CLICK_DISTANCE_RATIO,
    SCROLL_GESTURE_SMOOTHING_FRAMES,
)


class HandTracker:
    def __init__(self):
        self.available = False
        self.error = None
        self.mp_hands = None
        self.mp_draw = None
        self.hands = None
        self.landmark_drawing_spec = None
        self.connection_drawing_spec = None
        self.left_click_gesture_history = deque(maxlen=PINCH_SMOOTHING_FRAMES)
        self.right_click_gesture_history = deque(maxlen=PINCH_SMOOTHING_FRAMES)
        self.scroll_gesture_history = deque(maxlen=SCROLL_GESTURE_SMOOTHING_FRAMES)

        if mp is None:
            self.error = "MediaPipe is not installed."
            return

        if not hasattr(mp, "solutions"):
            self.error = "MediaPipe hands API is unavailable in this Python. Use Python 3.12."
            return

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(
            color=HAND_LANDMARK_COLOR,
            thickness=HAND_DRAW_THICKNESS,
            circle_radius=HAND_DRAW_RADIUS,
        )
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(
            color=HAND_CONNECTION_COLOR,
            thickness=HAND_DRAW_THICKNESS,
            circle_radius=HAND_DRAW_RADIUS,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.available = True

    def process(self, frame):
        if not self.available or self.hands is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        if not self.available or self.mp_draw is None or self.mp_hands is None:
            return

        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_drawing_spec,
            self.connection_drawing_spec,
        )

    def get_landmark_point(self, hand_landmarks, landmark_index, width, height):
        if not self.available:
            return None

        landmark = hand_landmarks.landmark[landmark_index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return (x, y)

    def get_index_finger_tip(self, hand_landmarks, width, height):
        if not self.available or self.mp_hands is None:
            return None

        return self.get_landmark_point(
            hand_landmarks,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            width,
            height,
        )

    def get_thumb_tip(self, hand_landmarks, width, height):
        if not self.available or self.mp_hands is None:
            return None

        return self.get_landmark_point(
            hand_landmarks,
            self.mp_hands.HandLandmark.THUMB_TIP,
            width,
            height,
        )

    def get_middle_finger_tip(self, hand_landmarks, width, height):
        if not self.available or self.mp_hands is None:
            return None

        return self.get_landmark_point(
            hand_landmarks,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            width,
            height,
        )

    def get_pinch_distance_ratio(self, hand_landmarks):
        if not self.available or self.mp_hands is None:
            return None

        return self.get_landmark_distance_ratio(
            hand_landmarks,
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
        )

    def get_right_click_distance_ratio(self, hand_landmarks):
        if not self.available or self.mp_hands is None:
            return None

        return self.get_landmark_distance_ratio(
            hand_landmarks,
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        )

    def get_scroll_control_point(self, hand_landmarks, width, height):
        index_point = self.get_index_finger_tip(hand_landmarks, width, height)
        middle_point = self.get_middle_finger_tip(hand_landmarks, width, height)
        if index_point is None or middle_point is None:
            return None

        return (
            int((index_point[0] + middle_point[0]) / 2),
            int((index_point[1] + middle_point[1]) / 2),
        )

    def get_landmark_distance_ratio(self, hand_landmarks, first_landmark_index, second_landmark_index):
        if not self.available or self.mp_hands is None:
            return None

        hand_landmark = self.mp_hands.HandLandmark
        first_tip = hand_landmarks.landmark[first_landmark_index]
        second_tip = hand_landmarks.landmark[second_landmark_index]
        index_mcp = hand_landmarks.landmark[hand_landmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[hand_landmark.PINKY_MCP]

        pinch_distance = self._distance(first_tip, second_tip)
        palm_width = max(self._distance(index_mcp, pinky_mcp), 1e-6)
        return pinch_distance / palm_width

    def is_left_click_gesture(self, hand_landmarks):
        pinch_ratio = self.get_pinch_distance_ratio(hand_landmarks)
        if pinch_ratio is None:
            return False

        raw_gesture = pinch_ratio < PINCH_DISTANCE_RATIO
        self.left_click_gesture_history.append(raw_gesture)
        required_votes = max(1, len(self.left_click_gesture_history) // 2 + 1)
        return sum(self.left_click_gesture_history) >= required_votes

    def is_right_click_gesture(self, hand_landmarks):
        right_click_ratio = self.get_right_click_distance_ratio(hand_landmarks)
        left_click_ratio = self.get_pinch_distance_ratio(hand_landmarks)
        if right_click_ratio is None or left_click_ratio is None:
            return False

        raw_gesture = (
            right_click_ratio < RIGHT_CLICK_DISTANCE_RATIO
            and left_click_ratio >= PINCH_DISTANCE_RATIO
        )
        self.right_click_gesture_history.append(raw_gesture)
        required_votes = max(1, len(self.right_click_gesture_history) // 2 + 1)
        return sum(self.right_click_gesture_history) >= required_votes

    def is_scroll_gesture(self, hand_landmarks):
        if not self.available or self.mp_hands is None:
            return False

        hand_landmark = self.mp_hands.HandLandmark
        left_click_ratio = self.get_pinch_distance_ratio(hand_landmarks)
        right_click_ratio = self.get_right_click_distance_ratio(hand_landmarks)
        if left_click_ratio is None or right_click_ratio is None:
            return False

        raw_gesture = (
            self._is_finger_extended(
                hand_landmarks,
                hand_landmark.INDEX_FINGER_TIP,
                hand_landmark.INDEX_FINGER_PIP,
            )
            and self._is_finger_extended(
                hand_landmarks,
                hand_landmark.MIDDLE_FINGER_TIP,
                hand_landmark.MIDDLE_FINGER_PIP,
            )
            and self._is_finger_folded(
                hand_landmarks,
                hand_landmark.RING_FINGER_TIP,
                hand_landmark.RING_FINGER_PIP,
            )
            and self._is_finger_folded(
                hand_landmarks,
                hand_landmark.PINKY_TIP,
                hand_landmark.PINKY_PIP,
            )
            and left_click_ratio >= PINCH_DISTANCE_RATIO
            and right_click_ratio >= RIGHT_CLICK_DISTANCE_RATIO
        )
        self.scroll_gesture_history.append(raw_gesture)
        required_votes = max(1, len(self.scroll_gesture_history) // 2 + 1)
        return sum(self.scroll_gesture_history) >= required_votes

    def get_handedness_label(self, results, hand_index=0):
        if results is None or not results.multi_handedness:
            return None

        if hand_index >= len(results.multi_handedness):
            return None

        classification = results.multi_handedness[hand_index].classification
        if not classification:
            return None

        return classification[0].label

    def reset(self):
        self.left_click_gesture_history.clear()
        self.right_click_gesture_history.clear()
        self.scroll_gesture_history.clear()

    def _is_finger_extended(self, hand_landmarks, tip_index, pip_index):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        return tip.y < pip.y - FINGER_Y_MARGIN

    def _is_finger_folded(self, hand_landmarks, tip_index, pip_index):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        return tip.y > pip.y + FINGER_Y_MARGIN

    def _distance(self, first_landmark, second_landmark):
        return (
            (first_landmark.x - second_landmark.x) ** 2
            + (first_landmark.y - second_landmark.y) ** 2
        ) ** 0.5

    def close(self):
        if self.hands is not None:
            self.hands.close()
