from ultralytics import YOLO
import cv2
import numpy as np
import time
import json
import shutil
from datetime import datetime
from pathlib import Path

class SessionManager:
    def __init__(self, saved_data_dir='saved_data'):
        self.saved_data_dir = Path(saved_data_dir)
        self.treatment_records_path = self.saved_data_dir / 'treatment_records.json'
        self.current_session = None
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.saved_data_dir.mkdir(exist_ok=True, parents=True)
        (self.saved_data_dir / 'images').mkdir(exist_ok=True)

    def start_new_session(self):
        """Start a new session with a unique ID and temporary directory."""
        session_id = f"COL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        temp_dir = self.saved_data_dir / 'temp' / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'medications': {},
            'image_paths': [],
            'temp_dir': str(temp_dir),
            'validated': False
        }
        return self.current_session

    def _save_image(self, frame, points, prefix):
        """Save a cropped image with high quality, without any detection labels."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.jpg"
            img_path = Path(self.current_session['temp_dir']) / filename
            
            # Get the bounding coordinates for cropping
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = max(0, int(min(x_coords))), min(frame.shape[1], int(max(x_coords)))
            y_min, y_max = max(0, int(min(y_coords))), min(frame.shape[0], int(max(y_coords)))
            
            if x_max > x_min and y_max > y_min:
                # Get clean crop from original frame, not the annotated one
                cropped = frame[y_min:y_max, x_min:x_max].copy()
                cv2.imwrite(str(img_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
                return str(img_path)
        except Exception as e:
            print(f"Error saving image: {e}")
        return None

    def add_medication(self, frame, points, name, confidence):
        """Add a detected medication with its cropped image."""
        if not self.current_session:
            return
        name = name.strip().lower()
        if name not in self.current_session['medications']:
            img_path = self._save_image(frame, points, 'med')
            if img_path:
                self.current_session['medications'][name] = {
                    'detected_at': datetime.now().isoformat(),
                    'confidence': float(confidence),
                    'image_path': img_path
                }
                self.current_session['image_paths'].append(img_path)

    def validate_session(self):
        """Validate and save the current session, moving images to permanent storage."""
        if not self.current_session or self.current_session['validated']:
            return False
        try:
            dest_dir = self.saved_data_dir / 'images' / self.current_session['session_id']
            dest_dir.mkdir(exist_ok=True)
            
            # Move all session images
            for img_path in self.current_session['image_paths']:
                shutil.move(img_path, dest_dir / Path(img_path).name)
            
            # Update session data
            self.current_session['end_time'] = datetime.now().isoformat()
            self.current_session['validated'] = True
            self.current_session['image_paths'] = [str(dest_dir / Path(p).name) for p in self.current_session['image_paths']]
            self.current_session['medications'] = [{'name': k, **v} for k, v in self.current_session['medications'].items()]
            
            # Save to records
            records = []
            if self.treatment_records_path.exists():
                with open(self.treatment_records_path, 'r') as f:
                    records = json.load(f)
            records.append(self.current_session)
            with open(self.treatment_records_path, 'w') as f:
                json.dump(records, f, indent=4)
            
            self.current_session = None
            return True
        except Exception as e:
            print(f"Error validating session: {e}")
            return False

    def clear_history(self):
        """Clear all saved session data and reset the system."""
        try:
            shutil.rmtree(self.saved_data_dir / 'images')
            (self.saved_data_dir / 'images').mkdir()
            if self.treatment_records_path.exists():
                self.treatment_records_path.unlink()
            self.current_session = None
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False

class PharmaDetectionSystem:
    def __init__(self):
        self.model = YOLO("runs/obb/train9/weights/best.pt")
        self.cap = cv2.VideoCapture(1)
        
        # Camera settings for better focus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        
        self.lot_number_class = 15
        self.confidence_threshold = 0.9
        self.blur_threshold = 500  # Adjust this based on your camera and environment
        self.session_manager = SessionManager()
        self.frame_count = 0
        self.start_time = time.time()
        self.ui_config = {
            'colors': {
                'background': (245, 245, 245),
                'lot_number': (0, 204, 102),
                'medication': (51, 153, 255),
                'warning': (0, 0, 255),
                'text': (51, 51, 51),
                'session_active': (102, 255, 102),
                'session_inactive': (255, 102, 102)
            },
            'fonts': {
                'primary': cv2.FONT_HERSHEY_SIMPLEX,
                'secondary': cv2.FONT_HERSHEY_DUPLEX
            }
        }
        self.current_blur_value = 0

    def detect_blur(self, image, threshold=None):
        """Detect if an image is blurry using Laplacian variance."""
        if threshold is None:
            threshold = self.blur_threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.current_blur_value = lap_var
        return lap_var < threshold, lap_var  # Returns (is_blurry, blur_value)

    def get_stable_frame(self, num_frames=3):
        """Capture and average multiple frames to reduce motion blur."""
        frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.03)  # Short delay between captures
        
        if not frames:
            return None
            
        # Average the frames
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        return avg_frame

    def process_frame(self, frame):
        """Process a frame using the YOLO model."""
        is_blurry, _ = self.detect_blur(frame)
        if is_blurry:
            # Skip detection for blurry frames
            return None
        
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        return results[0].obb if results[0].obb and len(results[0].obb) > 0 else None

    def draw_detections(self, frame, detections):
        """Draw detection boxes and save medications only if lot number is detected."""
        lot_detected = False
        original_frame = frame.copy()  # Save a clean copy of the frame for cropping
        
        if detections:
            classes = detections.cls.cpu().numpy().astype(int)
            lot_detected = any(c == self.lot_number_class for c in classes)
            boxes = detections.xyxyxyxy.cpu().numpy()
            confidences = detections.conf.cpu().numpy()
            for cls, box, conf in zip(classes, boxes, confidences):
                points = box.reshape((-1, 2)).astype(np.int32)
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                if cls == self.lot_number_class:
                    color = self.ui_config['colors']['lot_number']
                    self._draw_detection(frame, points, label, color)
                elif lot_detected:
                    color = self.ui_config['colors']['medication']
                    self._draw_detection(frame, points, label, color)
                    # Only add medication if image quality is good
                    if self.session_manager.current_session and self.current_blur_value >= self.blur_threshold:
                        med_name = self.model.names[int(cls)].strip().lower()
                        # Pass the original clean frame for saving
                        self.session_manager.add_medication(original_frame, points, med_name, conf)
        
        return frame, lot_detected

    def _draw_detection(self, frame, points, label, color):
        """Helper to draw a single detection."""
        cv2.polylines(frame, [points], True, color, 2)
        cv2.putText(frame, label, (points[0][0], points[0][1]-10),
                    self.ui_config['fonts']['primary'], 0.7, color, 2)

    def draw_ui(self, frame, lot_detected):
        """Draw the user interface."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), self.ui_config['colors']['background'], -1)
        cv2.putText(frame, "PharmaQC - Production System", (20, 40), 
                   self.ui_config['fonts']['secondary'], 1, self.ui_config['colors']['text'], 2)
        
        # Add blur indicator
        blur_status = "Clear" if self.current_blur_value >= self.blur_threshold else "Blurry"
        blur_color = self.ui_config['colors']['session_active'] if self.current_blur_value >= self.blur_threshold else self.ui_config['colors']['warning']
        cv2.putText(frame, f"Image Quality: {blur_status} ({self.current_blur_value:.0f})", 
                   (20, 110), self.ui_config['fonts']['primary'], 0.6, blur_color, 1)
        
        status_color = self.ui_config['colors']['lot_number'] if lot_detected else self.ui_config['colors']['warning']
        status_text = "Lot Number: Detected" if lot_detected else "Lot Number: Missing"
        cv2.putText(frame, status_text, (frame.shape[1]-300, 40), 
                   self.ui_config['fonts']['primary'], 0.8, status_color, 2)
        
        session_color = self.ui_config['colors']['session_active'] if self.session_manager.current_session else self.ui_config['colors']['session_inactive']
        session_text = "Session Active" if self.session_manager.current_session else "Session Inactive"
        cv2.putText(frame, session_text, (frame.shape[1]-300, 80), 
                   self.ui_config['fonts']['primary'], 0.7, session_color, 2)
        
        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {self.confidence_threshold:.2f} | Blur: {self.blur_threshold}",
                   (20, frame.shape[0]-20), self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['text'], 1)
        
        if self.session_manager.current_session:
            sid = self.session_manager.current_session['session_id'][-8:]
            cv2.putText(frame, f"Session ID: {sid}", (20, 80), 
                       self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['text'], 1)
            y = 140  # Moved down to make room for blur indicator
            meds = self.session_manager.current_session['medications']
            names = meds.keys() if isinstance(meds, dict) else [m['name'] for m in meds]
            if names:
                cv2.putText(frame, "Medications Detected:", (20, y), 
                           self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['text'], 1)
                y += 30
                for n in names:
                    cv2.putText(frame, f"- {n.title()}", (30, y), 
                               self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['medication'], 1)
                    y += 25
        
        cv2.putText(frame, "[N] New Session  [V] Validate  [C] Clear  [Q] Quit  [B/M] Blur Threshold",
                   (20, frame.shape[0]-60), self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['text'], 1)
        return frame

    def run(self):
        """Run the main detection loop."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Get frame and detect blur
                self.detect_blur(frame)
                
                # Process frame only if not blurry
                if self.current_blur_value >= self.blur_threshold:
                    detections = self.process_frame(frame)
                else:
                    detections = None
                    
                frame_copy = frame.copy()
                frame_copy, lot = self.draw_detections(frame_copy, detections)
                ui_frame = self.draw_ui(frame_copy, lot)
                cv2.imshow("PharmaQC System", ui_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.session_manager.start_new_session()
                elif key == ord('v'):
                    if self.session_manager.validate_session():
                        print("Session validated")
                    else:
                        print("No session to validate")
                elif key == ord('c'):
                    if self.session_manager.clear_history():
                        print("History cleared")
                    else:
                        print("Error clearing history")
                elif key == ord('+'):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                elif key == ord('b'):
                    # Decrease blur threshold (more sensitive to blur)
                    self.blur_threshold = max(20, self.blur_threshold - 10)
                elif key == ord('m'):
                    # Increase blur threshold (less sensitive to blur)
                    self.blur_threshold = min(1000, self.blur_threshold + 10)
                elif key == ord('s'):
                    # Take a stable averaged frame
                    stable_frame = self.get_stable_frame(5)
                    if stable_frame is not None:
                        cv2.imshow("Stable Frame", stable_frame)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.session_manager.current_session and not self.session_manager.current_session['validated']:
                shutil.rmtree(self.session_manager.current_session['temp_dir'], ignore_errors=True)

if __name__ == "__main__":
    print("Starting PharmaQC System...")
    system = PharmaDetectionSystem()
    system.run()
    print("Shutdown complete.")