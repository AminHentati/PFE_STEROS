# client.py (modified image saving only)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import cv2
import json
import numpy as np
import time
import re
import shutil
from datetime import datetime
from pathlib import Path
import requests
import base64
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QHBoxLayout, QVBoxLayout, QSplitter,
    QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QThreadPool, QRunnable, QObject
from PyQt5.QtGui import QImage, QPixmap, QColor

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
        """Save a cropped image with high quality."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.jpg"
            img_path = Path(self.current_session['temp_dir']) / filename
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min = max(0, int(min(x_coords)))
            x_max = min(frame.shape[1], int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(frame.shape[0], int(max(y_coords)))
            
            if x_max > x_min and y_max > y_min:
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

    def add_text(self, frame, points, text, confidence):
        """Add detected text with its cropped image."""
        if not self.current_session:
            return
        img_path = self._save_image(frame, points, 'text')
        if img_path:
            self.current_session['image_paths'].append(img_path)
            return img_path
        return None

    def validate_session(self):
        """Validate and save the current session."""
        if not self.current_session or self.current_session['validated']:
            return False
        try:
            dest_dir = self.saved_data_dir / 'images' / self.current_session['session_id']
            dest_dir.mkdir(exist_ok=True)
            
            for img_path in self.current_session['image_paths']:
                shutil.move(img_path, dest_dir / Path(img_path).name)
            
            self.current_session['end_time'] = datetime.now().isoformat()
            self.current_session['validated'] = True
            self.current_session['image_paths'] = [str(dest_dir / Path(p).name) for p in self.current_session['image_paths']]
            self.current_session['medications'] = [{'name': k, **v} for k, v in self.current_session['medications'].items()]
            
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

class WorkerSignals(QObject):
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

class ServerRequestRunnable(QRunnable):
    def __init__(self, frame, server_url, confidences, session_manager):
        super().__init__()
        self.frame = frame
        self.server_url = server_url
        self.confidences = confidences
        self.session_manager = session_manager
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            _, buffer = cv2.imencode('.jpg', self.frame)
            jpg_as_text = base64.b64encode(buffer).decode()
            data = {
                'image': jpg_as_text,
                'med_confidence': self.confidences['med'],
                'lot_confidence': self.confidences['lot'],
                'ocr_confidence': self.confidences['ocr']
            }
            response = requests.post(self.server_url, json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                
                # Process medications
                for med in result['medications']:
                    points = np.array(med['box'], dtype=np.int32)
                    self.session_manager.add_medication(
                        self.frame, points, 
                        med['name'], med['confidence']
                    )
                
                # Process texts
                for text in result['texts']:
                    x, y, x2, y2 = text['box']
                    points = np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.int32)
                    img_path = self.session_manager.add_text(
                        self.frame, points,
                        text['text'], text['confidence']
                    )
                    if img_path:
                        text['image_path'] = img_path
                
                self.signals.result.emit(result)
            else:
                self.signals.error.emit(f"Error: {response.status_code}")
        except Exception as e:
            self.signals.error.emit(str(e))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    process_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True
        self.last_send_time = 0
        self.send_interval = 0.2

    def run(self):
        while self._run_flag:
            ret, frame = self.parent.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    self.process_frame_signal.emit(frame.copy())
                    self.last_send_time = current_time

    def stop(self):
        self._run_flag = False
        self.wait()

class PharmaQCSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.thread_pool = QThreadPool()
        self.server_url = "http://localhost:8000/process_frame"
        self.blur_threshold = 100
        self.med_confidence = 0.7
        self.lot_confidence = 0.7
        self.ocr_confidence = 0.5
        self.session_manager = SessionManager()
        self.displayed_pairs = set()
        self.unique_texts = set()
        self.is_processing = False
        self.initUI()
        self.start_video_thread()

    def initUI(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QLabel { font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px; }
            QListWidget { background-color: #ffffff; border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px; font-size: 12px; }
            QPushButton { background-color: #3498db; color: white; border: none; border-radius: 5px; padding: 8px 15px; 
                          font-size: 13px; min-width: 100px; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f6aa5; }
        """)
        self.setWindowTitle("PharmaQC System")
        self.setGeometry(100, 100, 1280, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header_label = QLabel("PharmaQC Analysis System")
        header_label.setStyleSheet("font-size: 22px; color: #2c3e50; padding: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        threshold_panel = QWidget()
        threshold_layout = QHBoxLayout(threshold_panel)
        self.btn_blur = QPushButton(f"Blur Threshold: {self.blur_threshold}")
        self.btn_med_conf = QPushButton(f"Med Confidence: {self.med_confidence:.2f}")
        self.btn_lot_conf = QPushButton(f"Lot Confidence: {self.lot_confidence:.2f}")
        self.btn_ocr_conf = QPushButton(f"OCR Confidence: {self.ocr_confidence:.2f}")
        for btn in [self.btn_blur, self.btn_med_conf, self.btn_lot_conf, self.btn_ocr_conf]:
            btn.setStyleSheet("background-color: #95a5a6;")
            btn.clicked.connect(self.update_thresholds)
            threshold_layout.addWidget(btn)
        main_layout.addWidget(threshold_panel)

        content_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.detection_list = QListWidget()
        self.detection_list.setAlternatingRowColors(True)
        left_layout.addWidget(QLabel("Detected Items:"))
        left_layout.addWidget(self.detection_list)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; min-height: 480px;")
        self.video_label.setFixedSize(640, 480)
        right_layout.addWidget(QLabel("Live Camera Feed:"))
        right_layout.addWidget(self.video_label)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        content_layout.addWidget(splitter)
        main_layout.addLayout(content_layout)

        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        self.btn_new = QPushButton("New Session")
        self.btn_validate = QPushButton("Validate Session")
        self.btn_clear = QPushButton("Clear History")
        for btn in [self.btn_new, self.btn_validate, self.btn_clear]:
            control_layout.addWidget(btn)
        main_layout.addWidget(control_panel)

        self.statusBar().showMessage("Ready")
        self.btn_new.clicked.connect(self.new_session)
        self.btn_validate.clicked.connect(self.validate_session)
        self.btn_clear.clicked.connect(self.clear_history)

    def update_thresholds(self):
        btn = self.sender()
        if btn == self.btn_blur:
            value, ok = QInputDialog.getInt(self, "Blur Threshold", 
                                           "Enter blur threshold (0-200):", 
                                           self.blur_threshold, 0, 200)
            if ok: 
                self.blur_threshold = value
                btn.setText(f"Blur Threshold: {value}")
        elif btn == self.btn_med_conf:
            value, ok = QInputDialog.getDouble(self, "Medication Confidence", 
                                              "Enter confidence (0-1):", 
                                              self.med_confidence, 0.0, 1.0, 2)
            if ok:
                self.med_confidence = value
                btn.setText(f"Med Confidence: {value:.2f}")
        elif btn == self.btn_lot_conf:
            value, ok = QInputDialog.getDouble(self, "Lot Number Confidence", 
                                              "Enter confidence (0-1):", 
                                              self.lot_confidence, 0.0, 1.0, 2)
            if ok:
                self.lot_confidence = value
                btn.setText(f"Lot Confidence: {value:.2f}")
        elif btn == self.btn_ocr_conf:
            value, ok = QInputDialog.getDouble(self, "OCR Confidence", 
                                              "Enter confidence (0-1):", 
                                              self.ocr_confidence, 0.0, 1.0, 2)
            if ok: 
                self.ocr_confidence = value
                btn.setText(f"OCR Confidence: {value:.2f}")

    def start_video_thread(self):
        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.process_frame_signal.connect(self.start_server_request)
        self.video_thread.start()

    @pyqtSlot(QImage)
    def update_video(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(np.ndarray)
    def start_server_request(self, frame):
        if not self.is_processing:
            self.is_processing = True
            runnable = ServerRequestRunnable(
                frame, self.server_url, {
                    'med': self.med_confidence,
                    'lot': self.lot_confidence,
                    'ocr': self.ocr_confidence
                },
                self.session_manager
            )
            runnable.signals.result.connect(self.handle_server_response)
            runnable.signals.error.connect(self.handle_server_error)
            self.thread_pool.start(runnable)

    @pyqtSlot(dict)
    def handle_server_response(self, result):
        self.is_processing = False
        for med in result.get('medications', []):
            self.add_medication(med)
        for text in result.get('texts', []):
            self.add_text(text)
        self.try_match_medication_text()

    @pyqtSlot(str)
    def handle_server_error(self, error):
        self.is_processing = False
        print(f"Server error: {error}")

    def add_medication(self, med):
        item = QListWidgetItem(f"Medication: {med['name'].title()} ({med['confidence']:.2f})")
        item.setBackground(QColor(220, 255, 220))
        self.detection_list.addItem(item)

    def add_text(self, text):
        if text['text'] not in self.unique_texts:
            self.unique_texts.add(text['text'])
            item = QListWidgetItem(f"Lot: {text['text']} ({text['confidence']:.2f})")
            item.setBackground(QColor(220, 220, 255))
            self.detection_list.addItem(item)

    def try_match_medication_text(self):
        pass  # Keep your existing matching logic here

    def new_session(self):
        self.session_manager.start_new_session()
        self.displayed_pairs.clear()
        self.unique_texts.clear()
        self.detection_list.clear()
        self.statusBar().showMessage("New session started")

    def validate_session(self):
        if self.session_manager.validate_session():
            QMessageBox.information(self, "Success", "Session validated and saved")
        else:
            QMessageBox.warning(self, "Error", "No valid session to save")
        self.new_session()

    def clear_history(self):
        try:
            shutil.rmtree(self.session_manager.saved_data_dir / 'images')
            (self.session_manager.saved_data_dir / 'images').mkdir()
            if self.session_manager.treatment_records_path.exists():
                self.session_manager.treatment_records_path.unlink()
            QMessageBox.information(self, "Cleared", "All history cleared")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error clearing history: {e}")

    def closeEvent(self, event):
        self.video_thread.stop()
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PharmaQCSystem()
    window.show()
    sys.exit(app.exec_())