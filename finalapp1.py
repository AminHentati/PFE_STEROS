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
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QPushButton, QListWidget, QListWidgetItem,  
    QHBoxLayout, QVBoxLayout, QSplitter,
    QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QColor

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

try:
    import easyocr
    USE_EASYOCR = True
except ImportError:
    USE_EASYOCR = False

class PharmaQCSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO("runs/obb/train9/weights/best.pt")
        self.model.to(DEVICE)
        self.cap = cv2.VideoCapture(1)
        
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
        
        self.statusBar().showMessage(f"Device: {DEVICE} | Ready")
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
        self.video_thread.medication_signal.connect(self.add_medication)
        self.video_thread.text_signal.connect(self.add_text)
        self.video_thread.start()

    @pyqtSlot(QImage)
    def update_video(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(dict)
    def add_medication(self, med):
        self.session_data['medications'].append({
            'name': med['name'],
            'confidence': med['confidence'],
            'timestamp': datetime.now().isoformat(),
            'image_path': med['image_path'],
            'displayed': False
        })
        self.try_match_medication_text()

    @pyqtSlot(dict)
    def add_text(self, text):
        if text['text'] not in self.unique_texts:
            self.unique_texts.add(text['text'])
            self.session_data['texts'].append({
                'text': text['text'],
                'type': text['type'],
                'confidence': text['confidence'],
                'timestamp': datetime.now().isoformat(),
                'image_path': text['image_path'],
                'displayed': False
            })
            self.try_match_medication_text()

    def try_match_medication_text(self):
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
        med_item = QListWidgetItem(f"Medication: {med['name'].title()} ({med['confidence']:.2f})")
        med_item.setBackground(QColor(220, 255, 220))
        self.detection_list.addItem(med_item)
        
        text_item = QListWidgetItem(f"Lot: {text['text']} ({text['confidence']:.2f})")
        text_item.setBackground(QColor(220, 220, 255))
        self.detection_list.addItem(text_item)
        self.detection_list.scrollToBottom()

    def new_session(self):
        self.session_data = {
            'medications': [],
            'texts': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        self.displayed_pairs.clear()
        self.unique_texts.clear()
        self.detection_list.clear()
        self.video_thread.reset_session_data()
        self.statusBar().showMessage("New session started")

    def validate_session(self):
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
        self.detection_list.clear()
        self.new_session()
        QMessageBox.information(self, "Cleared", "All history cleared")

    def closeEvent(self, event):
        self.video_thread.stop()
        self.cap.release()
        event.accept()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    medication_signal = pyqtSignal(dict)
    text_signal = pyqtSignal(dict)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True
        self.lot_class_id = 15
        self.cooling_period = {}
        self.min_detection_interval = 2
        self.text_history = deque(maxlen=5)
        self.text_candidates = {}
        
        if USE_EASYOCR:
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def run(self):
        while self._run_flag:
            ret, frame = self.parent.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.parent.blur_threshold:
            return frame
            
        min_conf = min(self.parent.med_confidence, self.parent.lot_confidence)
        results = self.parent.model(frame, conf=min_conf, device=DEVICE)
        current_texts = set()
        
        if results[0].obb:
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()
            classes = results[0].obb.cls.cpu().numpy().astype(int)
            confidences = results[0].obb.conf.cpu().numpy()
            
            for cls, box, conf in zip(classes, boxes, confidences):
                points = box.reshape((-1, 2)).astype(np.int32)
                if cls == self.lot_class_id:
                    if conf < self.parent.lot_confidence:
                        continue
                    text_data = self.process_text_region(frame, points)
                    if text_data:
                        current_texts.add(text_data['text'])
                else:
                    if conf < self.parent.med_confidence:
                        continue
                    self.process_medication(frame, points, cls, conf)
        
        self.text_history.append(current_texts)
        self.validate_text_consistency()
        return frame

    def process_text_region(self, frame, points):
        x, y, w, h = cv2.boundingRect(points)
        crop = frame[y:y+h, x:x+w]
        
        if USE_EASYOCR and self.reader:
            results = self.reader.readtext(crop)
            text = ' '.join([res[1] for res in results]).strip()
            confidence = sum(res[2] for res in results)/len(results) if results else 0
        else:
            text = "OCR_UNAVAILABLE"
            confidence = 0
        
        if confidence < self.parent.ocr_confidence:
            return None
            
        clean_text = self.normalize_text(text)
        if not clean_text:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = f"text_{timestamp}.jpg"
        cv2.imwrite(img_path, crop)
        
        if clean_text in self.text_candidates:
            if confidence > self.text_candidates[clean_text][0]:
                self.text_candidates[clean_text] = (confidence, img_path)
        else:
            self.text_candidates[clean_text] = (confidence, img_path)
        
        return {'text': clean_text, 'confidence': confidence, 'image_path': img_path}

    def normalize_text(self, text):
        clean_text = re.sub(r'[^A-Z0-9-]', '', text.upper().strip())
        return clean_text if len(clean_text) >= 4 else None

    def validate_text_consistency(self):
        for text in list(self.text_candidates.keys()):
            conf, path = self.text_candidates[text]
            count = sum(1 for frame_texts in self.text_history if text in frame_texts)
            
            if count >= 3 and text not in self.parent.unique_texts:
                self.text_signal.emit({
                    'text': text,
                    'type': 'lot_number',
                    'confidence': conf,
                    'image_path': path
                })
                del self.text_candidates[text]

    def process_medication(self, frame, points, cls, conf):
        med_name = self.parent.model.names[int(cls)]
        current_time = time.time()
        
        if med_name in self.cooling_period and current_time - self.cooling_period[med_name] < self.min_detection_interval:
            return
        
        self.cooling_period[med_name] = current_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = f"med_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)
        
        self.medication_signal.emit({
            'name': med_name,
            'confidence': float(conf),
            'image_path': img_path
        })
        
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        cv2.putText(frame, f"{med_name} {conf:.2f}", tuple(points[0]), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def reset_session_data(self):
        self.text_history.clear()
        self.text_candidates.clear()

    def stop(self):
        self._run_flag = False
        self.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PharmaQCSystem()
    window.show()
    sys.exit(app.exec_())