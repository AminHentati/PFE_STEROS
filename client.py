import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import cv2
import json
import numpy as np
import time
import re
from datetime import datetime
from collections import deque
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

# Signal class for background worker
class WorkerSignals(QObject):
    result = pyqtSignal(dict)  # Emits server response
    error = pyqtSignal(str)    # Emits error messages

# Background task for server requests
class ServerRequestRunnable(QRunnable):
    def __init__(self, frame, server_url, confidences):
        super().__init__()
        self.frame = frame
        self.server_url = server_url
        self.confidences = confidences
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', self.frame)
            jpg_as_text = base64.b64encode(buffer).decode()
            data = {
                'image': jpg_as_text,
                'med_confidence': self.confidences['med'],
                'lot_confidence': self.confidences['lot'],
                'ocr_confidence': self.confidences['ocr']
            }
            # Send request to server
            response = requests.post(self.server_url, json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Process medications
                for med in result['medications']:
                    med_img_path = f"med_{timestamp}_{med['name']}.jpg"
                    cv2.imwrite(med_img_path, self.frame)
                    med['image_path'] = med_img_path
                    med['timestamp'] = datetime.now().isoformat()
                # Process texts
                for text in result['texts']:
                    x, y, x2, y2 = text['box']
                    crop = self.frame[y:y2, x:x2]
                    text_img_path = f"text_{timestamp}_{text['text']}.jpg"
                    cv2.imwrite(text_img_path, crop)
                    text['image_path'] = text_img_path
                    text['timestamp'] = datetime.now().isoformat()
                self.signals.result.emit(result)
            else:
                self.signals.error.emit(f"Error: {response.status_code}")
        except Exception as e:
            self.signals.error.emit(str(e))

# Video capture thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)     # For displaying frames
    process_frame_signal = pyqtSignal(np.ndarray) # For sending frames to server

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True
        self.last_send_time = 0
        self.send_interval = 0.5 # Send frames every 0.5 seconds

    def run(self):
        while self._run_flag:
            ret, frame = self.parent.cap.read()
            if ret:
                # Convert frame for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
                # Send frame for processing if interval has passed
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    self.process_frame_signal.emit(frame.copy())
                    self.last_send_time = current_time

    def stop(self):
        self._run_flag = False
        self.wait()

# Main application window
class PharmaQCSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)  # Open default camera
        self.thread_pool = QThreadPool()  # Pool for background tasks
        self.server_url = "http://localhost:8000/process_frame"
        self.blur_threshold = 100
        self.med_confidence = 0.7
        self.lot_confidence = 0.7
        self.ocr_confidence = 0.5
        self.session_data = {
            'medications': [],
            'texts': [],
            'start_time': None,
            'end_time': None
        }
        self.displayed_pairs = set()
        self.unique_texts = set()
        self.initUI()
        self.start_video_thread()

    def initUI(self):
        """Initialize the user interface."""
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

        # Header
        header_label = QLabel("PharmaQC Analysis System")
        header_label.setStyleSheet("font-size: 22px; color: #2c3e50; padding: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        # Threshold controls
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

        # Main content
        content_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Detection list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.detection_list = QListWidget()
        self.detection_list.setAlternatingRowColors(True)
        left_layout.addWidget(QLabel("Detected Items:"))
        left_layout.addWidget(self.detection_list)
        
        # Right panel: Video feed
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; min-height: 480px;")
        self.video_label.setFixedSize(640, 480)  # Optimize GUI updates
        right_layout.addWidget(QLabel("Live Camera Feed:"))
        right_layout.addWidget(self.video_label)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        content_layout.addWidget(splitter)
        main_layout.addLayout(content_layout)

        # Control buttons
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        self.btn_new = QPushButton("New Session")
        self.btn_validate = QPushButton("Validate Session")
        self.btn_clear = QPushButton("Clear History")
        for btn in [self.btn_new, self.btn_validate, self.btn_clear]:
            control_layout.addWidget(btn)
        main_layout.addWidget(control_panel)

        # Status bar
        self.statusBar().showMessage("Ready")
        self.btn_new.clicked.connect(self.new_session)
        self.btn_validate.clicked.connect(self.validate_session)
        self.btn_clear.clicked.connect(self.clear_history)

    def update_thresholds(self):
        """Update detection thresholds via dialog."""
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
        """Start the video capture thread."""
        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.process_frame_signal.connect(self.start_server_request)
        self.video_thread.start()

    @pyqtSlot(QImage)
    def update_video(self, image):
        """Update the video display."""
        self.video_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(np.ndarray)
    def start_server_request(self, frame):
        """Start a server request in the background."""
        runnable = ServerRequestRunnable(frame, self.server_url, {
            'med': self.med_confidence,
            'lot': self.lot_confidence,
            'ocr': self.ocr_confidence
        })
        runnable.signals.result.connect(self.handle_server_response)
        runnable.signals.error.connect(self.handle_server_error)
        self.thread_pool.start(runnable)

    @pyqtSlot(dict)
    def handle_server_response(self, result):
        """Handle successful server response."""
        for med in result.get('medications', []):
            self.add_medication(med)
        for text in result.get('texts', []):
            self.add_text(text)

    @pyqtSlot(str)
    def handle_server_error(self, error):
        """Handle server errors."""
        print(f"Server error: {error}")

    def add_medication(self, med):
        """Add detected medication to session data."""
        self.session_data['medications'].append({
            'name': med['name'],
            'confidence': med['confidence'],
            'timestamp': med['timestamp'],
            'image_path': med['image_path'],
            'displayed': False
        })
        self.try_match_medication_text()

    def add_text(self, text):
        """Add detected text to session data."""
        if text['text'] not in self.unique_texts:
            self.unique_texts.add(text['text'])
            self.session_data['texts'].append({
                'text': text['text'],
                'type': 'lot_number',
                'confidence': text['confidence'],
                'timestamp': text['timestamp'],
                'image_path': text['image_path'],
                'displayed': False
            })
            self.try_match_medication_text()

    def try_match_medication_text(self):
        """Match medications with texts based on timestamp proximity."""
        undisplayed_meds = [med for med in self.session_data['medications'] if not med['displayed']]
        undisplayed_texts = [text for text in self.session_data['texts'] if not text['displayed']]
        for med in undisplayed_meds:
            med_timestamp = datetime.fromisoformat(med['timestamp'])
            best_text = None
            min_time_diff = float('inf')
            for text in undisplayed_texts:
                text_timestamp = datetime.fromisoformat(text['timestamp'])
                time_diff = abs((text_timestamp - med_timestamp).total_seconds())
                if time_diff < min_time_diff and time_diff < 10:
                    min_time_diff = time_diff
                    best_text = text
            if best_text:
                med['displayed'] = True
                best_text['displayed'] = True
                pair_key = (med['name'], best_text['text'])
                if pair_key not in self.displayed_pairs:
                    self.displayed_pairs.add(pair_key)
                    self.add_detection_pair(med, best_text)

    def add_detection_pair(self, med, text):
        """Display matched medication-text pair in the list."""
        med_item = QListWidgetItem(f"Medication: {med['name'].title()} ({med['confidence']:.2f})")
        med_item.setBackground(QColor(220, 255, 220))
        self.detection_list.addItem(med_item)
        text_item = QListWidgetItem(f"Lot: {text['text']} ({text['confidence']:.2f})")
        text_item.setBackground(QColor(220, 220, 255))
        self.detection_list.addItem(text_item)
        self.detection_list.scrollToBottom()

    def new_session(self):
        """Start a new session."""
        self.session_data = {
            'medications': [],
            'texts': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        self.displayed_pairs.clear()
        self.unique_texts.clear()
        self.detection_list.clear()
        self.statusBar().showMessage("New session started")

    def validate_session(self):
        """Validate and save the current session."""
        valid_pairs = []
        for med in self.session_data['medications']:
            for text in self.session_data['texts']:
                time_diff = abs((datetime.fromisoformat(text['timestamp']) - 
                              datetime.fromisoformat(med['timestamp'])).total_seconds())
                if time_diff < 10:
                    valid_pairs.append(f"{med['name']} - {text['text']}")
        if valid_pairs:
            self.session_data['end_time'] = datetime.now().isoformat()
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.session_data, f, indent=4)
            QMessageBox.information(self, "Success", f"Session saved to {filename}")
        else:
            QMessageBox.warning(self, "Error", "No valid medication-text pairs found")
        self.new_session()

    def clear_history(self):
        """Clear the detection history."""
        self.detection_list.clear()
        self.new_session()
        QMessageBox.information(self, "Cleared", "All history cleared")

    def closeEvent(self, event):
        """Handle window close event."""
        self.video_thread.stop()
        self.cap.release()
        event.accept()

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PharmaQCSystem()
    window.show()
    sys.exit(app.exec_())