import cv2
import mediapipe as mp
import math

class GestureEngine:
    """
    Central engine for processing hand landmarks and recognizing gestures.
    """
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        """Processes an RGB frame and returns the results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    @staticmethod
    def calculate_distance(point1, point2, width, height):
        """Calculates Euclidean distance between two points."""
        x1, y1 = int(point1.x * width), int(point1.y * height)
        x2, y2 = int(point2.x * width), int(point2.y * height)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
        """Checks if a finger is up based on its tip and PIP (Proximal Interphalangeal) joint."""
        tip = hand_landmarks.landmark[finger_tip_id]
        pip = hand_landmarks.landmark[finger_pip_id]
        return tip.y < pip.y

    def count_fingers_up(self, hand_landmarks):
        """Counts how many fingers are up."""
        fingers = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        # Thumb (special case, checking horizontal movement relative to IP joint)
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        # Using a simple heuristic for thumb (horizontal distance on flipped frame)
        # Note: This may need calibration depending on hand orientation
        fingers['thumb'] = abs(thumb_tip.x - thumb_ip.x) > 0.03
        
        # Other fingers
        fingers['index'] = self.is_finger_up(hand_landmarks, 
                                             self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                             self.mp_hands.HandLandmark.INDEX_FINGER_PIP)
        fingers['middle'] = self.is_finger_up(hand_landmarks,
                                              self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                              self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        fingers['ring'] = self.is_finger_up(hand_landmarks,
                                            self.mp_hands.HandLandmark.RING_FINGER_TIP,
                                            self.mp_hands.HandLandmark.RING_FINGER_PIP)
        fingers['pinky'] = self.is_finger_up(hand_landmarks,
                                             self.mp_hands.HandLandmark.PINKY_TIP,
                                             self.mp_hands.HandLandmark.PINKY_PIP)
        
        return fingers

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()
