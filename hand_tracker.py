import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

from config import (
    HAND_CONNECTION_COLOR,
    HAND_DRAW_RADIUS,
    HAND_DRAW_THICKNESS,
    HAND_LANDMARK_COLOR,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
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

        if mp is None:
            self.error = "MediaPipe is not installed."
            return

        if not hasattr(mp, "solutions"):
            self.error = "MediaPipe hands API is unavailable."
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

    def get_handedness_label(self, results, hand_index=0):
        if results is None or not results.multi_handedness:
            return None

        if hand_index >= len(results.multi_handedness):
            return None

        classification = results.multi_handedness[hand_index].classification
        if not classification:
            return None

        return classification[0].label

    def close(self):
        if self.hands is not None:
            self.hands.close()
