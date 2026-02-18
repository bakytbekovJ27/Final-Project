import cv2
import sys
from core.gesture_engine import GestureEngine
from modules.painter import Painter

def main():
    print("=" * 50)
    print("GESTUREPRO: Professional Gesture Control System")
    print("=" * 50)
    
    # Initialization
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Get frame dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {width}x{height}")

        engine = GestureEngine()
        painter = Painter(width, height)
        
        print("\nSystem started. Press 'Q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the frame BEFORE processing to align landmarks with display
            frame = cv2.flip(frame, 1)
            
            # Process hand landmarks
            results = engine.process_frame(frame)
            
            # Current active module: Painter
            # (In the future, StateMachine will handle switching)
            display_frame = painter.update(frame, results, engine)
            
            # Show output
            cv2.imshow('GesturePro', display_frame)
            
            # Global keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                painter.save_canvas()
            elif key == ord('x'):
                painter.canvas.fill(0)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'engine' in locals():
            engine.close()
        print("\nSystem shutdown gracefully.")

if __name__ == "__main__":
    main()
