from ultralytics import YOLO
import cv2
import numpy as np
import time
import sqlite3
import csv
from datetime import datetime

class DetectionLogger:
    def __init__(self, db_name='pharma_detections.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()
        print(f"Database connection established: {db_name}")

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             medication_name TEXT,
                             lot_number_present BOOLEAN,
                             confidence REAL,
                             timestamp DATETIME,
                             frame_width INTEGER,
                             frame_height INTEGER,
                             valid_detection BOOLEAN)''')
        self.conn.commit()

    def log_detection(self, medication_name, confidence, frame_width, frame_height, 
                     lot_number_present, valid_detection):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('''INSERT INTO detections 
                            (medication_name, lot_number_present, confidence, 
                             timestamp, frame_width, frame_height, valid_detection)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (medication_name, lot_number_present, confidence,
                          timestamp, frame_width, frame_height, valid_detection))
        self.conn.commit()

    def get_recent_detections(self, limit=10):
        self.cursor.execute('''SELECT * FROM detections 
                            ORDER BY timestamp DESC LIMIT ?''', (limit,))
        return self.cursor.fetchall()

    def export_to_csv(self, filename='detections_export.csv'):
        detections = self.cursor.execute('''SELECT * FROM detections''').fetchall()
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Medication', 'Lot Present', 'Confidence',
                            'Timestamp', 'Width', 'Height', 'Valid'])
            writer.writerows(detections)
        print(f"Exported {len(detections)} records to {filename}")

    def __del__(self):
        self.conn.close()
        print("Database connection closed")

class PharmaDetectionSystem:
    def __init__(self):
        self.model = YOLO("runs/obb/train8/weights/best.pt")
        self.cap = cv2.VideoCapture(1)
        self.lot_number_class = 14
        self.confidence_threshold = 0.7
        self.logger = DetectionLogger()
        self.frame_count = 0
        self.start_time = time.time()

        # UI Configuration
        self.ui_config = {
            'colors': {
                'background': (245, 245, 245),
                'lot_number': (0, 204, 102),
                'medication': (51, 153, 255),
                'warning': (0, 0, 255),
                'text': (51, 51, 51)
            },
            'fonts': {
                'primary': cv2.FONT_HERSHEY_SIMPLEX,
                'secondary': cv2.FONT_HERSHEY_DUPLEX
            }
        }

    def process_frame(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        return results[0].obb if len(results[0].obb) > 0 else None

    def draw_detections(self, frame, detections):
        lot_detected = False
        class_counts = {}
        h, w = frame.shape[:2]

        if detections:
            classes = detections.cls.cpu().numpy()
            boxes = detections.xyxyxyxy.cpu().numpy()
            confidences = detections.conf.cpu().numpy()

            lot_detected = any(cls == self.lot_number_class for cls in classes)

            for i, (cls, box, conf) in enumerate(zip(classes, boxes, confidences)):
                class_id = int(cls)
                label = f"{self.model.names[class_id]} {conf:.2f}"
                points = box.reshape((-1, 2)).astype(np.int32)

                # Logique de journalisation
                if class_id == self.lot_number_class:
                    self.logger.log_detection(
                        medication_name="lot_number",
                        confidence=float(conf),
                        frame_width=w,
                        frame_height=h,
                        lot_number_present=True,
                        valid_detection=True
                    )
                    color = self.ui_config['colors']['lot_number']
                    self._draw_detection(frame, points, label, color)
                elif lot_detected:
                    self.logger.log_detection(
                        medication_name=self.model.names[class_id],
                        confidence=float(conf),
                        frame_width=w,
                        frame_height=h,
                        lot_number_present=True,
                        valid_detection=True
                    )
                    color = self.ui_config['colors']['medication']
                    self._draw_detection(frame, points, label, color)
                    class_counts[self.model.names[class_id]] = class_counts.get(self.model.names[class_id], 0) + 1

        return frame, lot_detected, class_counts

    def _draw_detection(self, frame, points, label, color):
        cv2.polylines(frame, [points], True, color, 2)
        cv2.putText(frame, label, (points[0][0], points[0][1]-10),
                    self.ui_config['fonts']['primary'], 0.7, color, 2)

    def draw_ui(self, frame, lot_detected, class_counts):
        # Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), self.ui_config['colors']['background'], -1)
        cv2.putText(frame, "PharmaQC - Production System", (20, 40), 
                   self.ui_config['fonts']['secondary'], 1, self.ui_config['colors']['text'], 2)

        # Status panel
        status_text = "Lot Number: Detected" if lot_detected else "Lot Number: Missing"
        status_color = self.ui_config['colors']['lot_number'] if lot_detected else self.ui_config['colors']['warning']
        cv2.putText(frame, status_text, (frame.shape[1]-300, 40), 
                   self.ui_config['fonts']['primary'], 0.8, status_color, 2)

        # Statistics
        fps = self.frame_count / (time.time() - self.start_time)
        stats_text = f"FPS: {fps:.1f} | Confidence: {self.confidence_threshold:.2f} | DB: {self.logger.conn}"
        cv2.putText(frame, stats_text, (20, frame.shape[0]-20), 
                   self.ui_config['fonts']['primary'], 0.6, self.ui_config['colors']['text'], 1)

        # Class counts
        if lot_detected and class_counts:
            y_offset = 80
            for med, count in class_counts.items():
                cv2.putText(frame, f"{med}: {count}", (frame.shape[1]-300, y_offset), 
                          self.ui_config['fonts']['primary'], 0.7, self.ui_config['colors']['medication'], 1)
                y_offset += 30

        return frame

    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                self.frame_count += 1
                
                # Processing pipeline
                detections = self.process_frame(frame)
                processed_frame, lot_detected, class_counts = self.draw_detections(frame, detections)
                final_frame = self.draw_ui(processed_frame, lot_detected, class_counts)

                cv2.imshow("chimio System", final_frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("+"):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                elif key == ord("-"):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                elif key == ord("e"):
                    self.logger.export_to_csv()
                    print("Exported data to CSV")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.logger.export_to_csv()

if __name__ == "__main__":
    print("Initializing PharmaQC System...")
    system = PharmaDetectionSystem()
    system.run()
    print("System shutdown complete. Data saved to database.")