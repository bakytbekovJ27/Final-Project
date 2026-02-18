import cv2
from core.gesture_engine import GestureEngine
from modules.painter import Painter, SHAPE_CIRCLE, SHAPE_RECT, SHAPE_SQUARE, SHAPE_TRIANGLE

def main():
    print("Initializing Refactored Painter...")
    
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize Engine and Module
    engine = GestureEngine()
    painter = Painter(1280, 720)
    
    print("Painter Ready. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process Frame
        results = engine.process_frame(frame)
        
        # Update Painter
        display_frame = painter.update(frame, results, engine)
        
        # Keyboard handling (transferred from old painter.py)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'): painter.current_color = painter.colors['red']
        elif key == ord('2'): painter.current_color = painter.colors['green']
        elif key == ord('3'): painter.current_color = painter.colors['blue']
        elif key == ord('4'): painter.current_color = painter.colors['yellow']
        elif key == ord('5'): painter.current_color = painter.colors['white']
        elif key == ord('6'): painter.current_color = painter.colors['pink']
        elif key == ord('7'): painter.current_color = painter.colors['orange']
        elif key == ord('8'): painter.current_color = painter.colors['cyan']
        elif key == ord('9'): painter.current_color = painter.colors['purple']
        elif key == ord('0'): painter.current_color = painter.colors['black']
        
        elif key == ord('c'): painter.current_shape = SHAPE_CIRCLE
        elif key == ord('r'): painter.current_shape = SHAPE_RECT
        elif key == ord('v'): painter.current_shape = SHAPE_SQUARE
        elif key == ord('t'): painter.current_shape = SHAPE_TRIANGLE
        elif key == ord('d'): painter.current_shape = None
        
        elif key == ord('s'): 
            fname = painter.save_canvas()
            print(f"Saved to {fname}")
        elif key == ord('x'): 
            painter.canvas.fill(0)
            print("Canvas Cleared")
        elif key == ord('+') or key == ord('='):
            painter.brush_thickness = min(painter.brush_thickness + 2, 50)
        elif key == ord('-') or key == ord('_'):
            painter.brush_thickness = max(painter.brush_thickness - 2, 1)
        elif key == ord('['):
            painter.eraser_thickness = max(painter.eraser_thickness - 5, 10)
        elif key == ord(']'):
            painter.eraser_thickness = min(painter.eraser_thickness + 5, 100)

        cv2.imshow('Refactored Gesture Painter', display_frame)

    cap.release()
    cv2.destroyAllWindows()
    engine.close()
    print("Finished.")

if __name__ == "__main__":
    main()
