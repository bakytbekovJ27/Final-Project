import cv2
import numpy as np
import math
import time

# Constants for Modes
MODE_DRAW = "DRAW"
MODE_ERASE = "ERASE"
MODE_CLEAR = "CLEAR"
MODE_SHAPE = "SHAPE"
MODE_IDLE = "IDLE"

# Constants for Shapes
SHAPE_CIRCLE = "CIRCLE"
SHAPE_RECT = "RECT"
SHAPE_SQUARE = "SQUARE"
SHAPE_TRIANGLE = "TRIANGLE"

class Painter:
    """
    Virtual drawing module with gesture control and shape support.
    """
    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Drawing Settings
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'pink': (255, 0, 255),
            'orange': (0, 165, 255),
            'cyan': (255, 255, 0),
            'purple': (255, 51, 153),
            'black': (0, 0, 0)
        }
        self.current_color = self.colors['blue']
        self.brush_thickness = 5
        self.eraser_thickness = 30
        
        # State
        self.current_mode = MODE_IDLE
        self.current_shape = None
        self.prev_x, self.prev_y = None, None
        
        # Clearing logic
        self.clear_start_time = None
        self.clear_delay = 1.5
        
        # Shape logic
        self.shape_start = None
        self.shape_end = None

    def detect_mode(self, fingers):
        """Maps fingers up to modes."""
        fingers_up_count = sum(fingers.values())
        
        if fingers_up_count == 5:
            return MODE_CLEAR
        elif fingers['thumb'] and fingers_up_count == 1:
            return MODE_ERASE
        elif fingers['index'] and fingers_up_count == 1:
            return MODE_DRAW
        else:
            return MODE_IDLE

    def draw_shapes_preview(self, frame, start, end, shape, color):
        """Draws a preview of the selected shape on the current frame."""
        if shape == SHAPE_CIRCLE:
            radius = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2))
            cv2.circle(frame, start, radius, color, 2)
            cv2.circle(frame, start, 5, (0, 255, 0), -1)
        elif shape == SHAPE_RECT:
            cv2.rectangle(frame, start, end, color, 2)
            cv2.circle(frame, start, 5, (0, 255, 0), -1)
        elif shape == SHAPE_SQUARE:
            dx, dy = end[0] - start[0], end[1] - start[1]
            side = max(abs(dx), abs(dy))
            end_x = start[0] + side * (1 if dx > 0 else -1)
            end_y = start[1] + side * (1 if dy > 0 else -1)
            cv2.rectangle(frame, start, (end_x, end_y), color, 2)
            cv2.circle(frame, start, 5, (0, 255, 0), -1)
        elif shape == SHAPE_TRIANGLE:
            cv2.line(frame, start, (end[0], start[1]), color, 2)
            cv2.line(frame, (end[0], start[1]), end, color, 2)
            cv2.line(frame, end, start, color, 2)
            cv2.circle(frame, start, 5, (0, 255, 0), -1)

    def draw_shapes_final(self, start, end, shape, color, thickness):
        """Draws the final shape on the persistent canvas."""
        if shape == SHAPE_CIRCLE:
            radius = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2))
            cv2.circle(self.canvas, start, radius, color, thickness)
        elif shape == SHAPE_RECT:
            cv2.rectangle(self.canvas, start, end, color, thickness)
        elif shape == SHAPE_SQUARE:
            dx, dy = end[0] - start[0], end[1] - start[1]
            side = max(abs(dx), abs(dy))
            end_x = start[0] + side * (1 if dx > 0 else -1)
            end_y = start[1] + side * (1 if dy > 0 else -1)
            cv2.rectangle(self.canvas, start, (end_x, end_y), color, thickness)
        elif shape == SHAPE_TRIANGLE:
            cv2.line(self.canvas, start, (end[0], start[1]), color, thickness)
            cv2.line(self.canvas, (end[0], start[1]), end, color, thickness)
            cv2.line(self.canvas, end, start, color, thickness)

    def update(self, frame, results, gesture_engine):
        """Processes a frame, updates the canvas, and overlays UI elements."""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture_engine.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, gesture_engine.mp_hands.HAND_CONNECTIONS
                )
                
                fingers = gesture_engine.count_fingers_up(hand_landmarks)
                detected_mode = self.detect_mode(fingers)
                
                # Index finger for drawing/shapes
                index_tip = hand_landmarks.landmark[gesture_engine.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * self.frame_width)
                y = int(index_tip.y * self.frame_height)
                
                # CLEAR MODE
                if detected_mode == MODE_CLEAR:
                    self.current_mode = MODE_CLEAR
                    self.shape_start = None
                    if self.clear_start_time is None:
                        self.clear_start_time = time.time()
                    elapsed = time.time() - self.clear_start_time
                    if elapsed > self.clear_delay:
                        self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                        self.clear_start_time = None
                    else:
                        remaining = self.clear_delay - elapsed
                        cv2.putText(frame, f"Clearing in {remaining:.1f} sec", 
                                    (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 0, 255), 2)
                    self.prev_x, self.prev_y = None, None
                
                # SHAPE MODE
                elif self.current_shape is not None and detected_mode == MODE_DRAW:
                    self.current_mode = MODE_SHAPE
                    self.clear_start_time = None
                    if self.shape_start is None:
                        self.shape_start = (x, y)
                    else:
                        self.draw_shapes_preview(frame, self.shape_start, (x, y), self.current_shape, self.current_color)
                        self.shape_end = (x, y)
                    self.prev_x, self.prev_y = None, None
                
                # DRAW MODE (Lines)
                elif detected_mode == MODE_DRAW and self.current_shape is None:
                    self.current_mode = MODE_DRAW
                    self.clear_start_time = None
                    cv2.circle(frame, (x, y), self.brush_thickness + 2, self.current_color, 2)
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                                 self.current_color, self.brush_thickness)
                    self.prev_x, self.prev_y = x, y
                
                # ERASE MODE
                elif detected_mode == MODE_ERASE:
                    self.current_mode = MODE_ERASE
                    self.clear_start_time = None
                    self.shape_start = None
                    thumb_tip = hand_landmarks.landmark[gesture_engine.mp_hands.HandLandmark.THUMB_TIP]
                    tx, ty = int(thumb_tip.x * self.frame_width), int(thumb_tip.y * self.frame_height)
                    cv2.circle(frame, (tx, ty), self.eraser_thickness, (200, 200, 200), 2)
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (tx, ty), (0, 0, 0), self.eraser_thickness)
                    self.prev_x, self.prev_y = tx, ty
                
                # IDLE / FINALIZE SHAPE
                else:
                    if self.shape_start is not None and self.shape_end is not None:
                        self.draw_shapes_final(self.shape_start, self.shape_end, self.current_shape, self.current_color, self.brush_thickness)
                        self.shape_start = None
                        self.shape_end = None
                    self.current_mode = MODE_IDLE
                    self.clear_start_time = None
                    self.prev_x, self.prev_y = None, None
        else:
            # Handle final shape if hand lost
            if self.shape_start is not None and self.shape_end is not None:
                self.draw_shapes_final(self.shape_start, self.shape_end, self.current_shape, self.current_color, self.brush_thickness)
            self.prev_x, self.prev_y = None, None
            self.clear_start_time = None
            self.shape_start = None
            self.shape_end = None
            self.current_mode = MODE_IDLE

        # UI Overlay
        return self._render(frame)

    def _render(self, frame):
        """Combines the video frame with the drawing canvas and UI panel."""
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)
        
        # Info panel
        info_panel = np.zeros((140, self.frame_width, 3), dtype=np.uint8)
        mode_display = f"Mode: {self.current_shape if self.current_shape else self.current_mode}"
        cv2.putText(info_panel, mode_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(info_panel, (10, 50), (70, 110), self.current_color, -1)
        cv2.rectangle(info_panel, (10, 50), (70, 110), (255, 255, 255), 2)
        cv2.putText(info_panel, f"Brush: {self.brush_thickness}px | Eraser: {self.eraser_thickness}px", 
                    (90, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_panel, "Q-Exit | S-Save | X-Clear | 1-0:Colors | C,R,V,T:Shapes | D:Draw",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return np.vstack([combined, info_panel])

    def save_canvas(self):
        """Saves the current canvas to a file."""
        white_bg = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        output = np.where(mask[:, :, None] != 0, self.canvas, white_bg)
        filename = f"paint_{int(time.time())}.png"
        cv2.imwrite(filename, output)
        return filename
