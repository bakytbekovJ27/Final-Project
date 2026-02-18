import unittest
from core.gesture_engine import GestureEngine

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TestGestureEngine(unittest.TestCase):
    def setUp(self):
        # We don't initialize the full engine to avoid MediaPipe overhead in simple tests
        # but we can test static methods
        pass

    def test_calculate_distance(self):
        p1 = MockLandmark(0.1, 0.1)
        p2 = MockLandmark(0.4, 0.5)
        width, height = 1000, 1000
        # x distances: (0.4-0.1)*1000 = 300
        # y distances: (0.5-0.1)*1000 = 400
        # dist = sqrt(300^2 + 400^2) = 500
        dist = GestureEngine.calculate_distance(p1, p2, width, height)
        self.assertEqual(dist, 500.0)

    def test_is_finger_up(self):
        class MockLandmarks:
            def __init__(self, tip_y, pip_y):
                self.landmark = {
                    8: MockLandmark(0.5, tip_y),
                    6: MockLandmark(0.5, pip_y)
                }
        
        # Case: Finger is up (tip is above PIP, so tip.y < pip.y)
        up_landmarks = MockLandmarks(0.2, 0.4)
        self.assertTrue(GestureEngine.is_finger_up(up_landmarks, 8, 6))
        
        # Case: Finger is down
        down_landmarks = MockLandmarks(0.6, 0.4)
        self.assertFalse(GestureEngine.is_finger_up(down_landmarks, 8, 6))

if __name__ == '__main__':
    unittest.main()
