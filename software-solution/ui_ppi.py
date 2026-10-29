import sys
import random
import numpy as np
import pydicom
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer


class InteractiveImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.min_zoom = 1.0

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.scene.itemsBoundingRect())
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.min_zoom = self.transform().m11()

    def wheelEvent(self, event):
        if not self.pixmap_item.pixmap() or self.pixmap_item.pixmap().isNull():
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        current_scale = self.transform().m11()

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            if current_scale * zoom_factor < self.min_zoom:
                zoom_factor = self.min_zoom / current_scale

        self.scale(zoom_factor, zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            current_scale = self.transform().m11()
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.min_zoom = self.transform().m11()
            if current_scale > self.min_zoom:
                self.resetTransform()
                self.scale(current_scale, current_scale)


class PACSInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PACS Diagnostic Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.image_loaded = False

        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QFrame#sidebar {
                background-color: #1e1e1e;
                border-right: 1px solid #333;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #555;
                border: 1px solid #333;
            }
            QLabel#result_label {
                font-size: 17px;
                padding: 10px;
                border-radius: 4px;
            }
            QLabel#disclaimer {
                color: #888;
                font-size: 15px;
                font-style: italic;
            }
            QLabel#status_label {
                color: #50fa7b;
                font-size: 13px;
                font-weight: bold;
            }
        """)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)

        title_label = QLabel("DIAGNOSTIC TOOLS")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #888; font-size: 12px; letter-spacing: 1px;")

        self.btn_upload = QPushButton("Upload .dcm Image")
        self.btn_upload.clicked.connect(self.upload_image)

        self.btn_analyse = QPushButton("Run AI Metastasis Analysis")
        self.btn_analyse.clicked.connect(self.start_analysis)
        self.btn_analyse.setEnabled(False)

        info_text = "Patient: JOHN DOE\nID: 987654321\nDOB: 01/01/1970\nStudy: MRI Routine"
        self.lbl_patient_info = QLabel(info_text)
        self.lbl_patient_info.setStyleSheet("color: #aaa; font-size: 17px; line-height: 1.5;")
        self.lbl_patient_info.hide()

        # AI Status / Waiting State
        self.lbl_status = QLabel("Analyzing your image...")
        self.lbl_status.setObjectName("status_label")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.hide()

        # AI Result Container
        self.result_container = QWidget()
        result_layout = QVBoxLayout(self.result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_disclaimer = QLabel("AI suggestion • may be wrong")
        self.lbl_disclaimer.setObjectName("disclaimer")

        self.lbl_result = QLabel("")
        self.lbl_result.setObjectName("result_label")
        self.lbl_result.setWordWrap(True)

        feedback_layout = QHBoxLayout()
        self.btn_thumbs_up = QPushButton("Good")
        self.btn_thumbs_down = QPushButton("Bad")
        self.btn_regenerate = QPushButton("Regenerate")

        self.btn_thumbs_up.clicked.connect(lambda: self.submit_feedback("positive"))
        self.btn_thumbs_down.clicked.connect(lambda: self.submit_feedback("negative"))
        self.btn_regenerate.clicked.connect(self.start_analysis)

        feedback_layout.addWidget(self.btn_thumbs_up)
        feedback_layout.addWidget(self.btn_thumbs_down)
        feedback_layout.addWidget(self.btn_regenerate)

        result_layout.addWidget(self.lbl_disclaimer)
        result_layout.addWidget(self.lbl_result)
        result_layout.addLayout(feedback_layout)
        self.result_container.hide()

        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(self.btn_upload)
        sidebar_layout.addWidget(self.btn_analyse)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self.lbl_patient_info)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.lbl_status)
        sidebar_layout.addWidget(self.result_container)

        self.image_viewer = InteractiveImageViewer()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.image_viewer)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)",
                                                   options=options)

        if file_name:
            try:
                dicom_data = pydicom.dcmread(file_name)
                pixel_array = dicom_data.pixel_array.astype(float)

                if pixel_array.max() > 0:
                    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0

                pixel_array = np.uint8(pixel_array)
                height, width = pixel_array.shape
                bytes_per_line = width

                q_img = QImage(pixel_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_img)

                self.image_viewer.set_image(pixmap)

                self.image_loaded = True
                self.btn_analyse.setEnabled(True)
                self.lbl_patient_info.show()
                self.result_container.hide()
                self.lbl_status.hide()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM image.\n\nDetails: {str(e)}")

    def start_analysis(self):
        if not self.image_loaded:
            return

        # disable buttons and show waiting state
        self.btn_analyse.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.result_container.hide()
        self.lbl_status.show()

        # simulate mock AI network delay
        QTimer.singleShot(2000, self.finish_analysis)

    def finish_analysis(self):
        self.lbl_status.hide()
        self.btn_analyse.setEnabled(True)
        self.btn_upload.setEnabled(True)

        # mocked AI data generation
        probability = random.uniform(2.5, 98.5)

        mock_reasons = [
            "Irregular tissue density detected in upper quadrant",
            "Calcification patterns align with historical anomalies",
            "Asymmetrical morphological features observed",
            "No significant focal lesions identified"
        ]
        reason = random.choice(mock_reasons) if probability > 30 else mock_reasons[3]

        urgency = "HIGH" if probability > 75 else ("MEDIUM" if probability > 30 else "LOW")

        result_text = (f"<b>Urgency: {urgency}</b><br><br>"
                       f"Probability: {probability:.1f}%<br>"
                       f"Why: {reason}<br>"
                       f"Source: Radiology Triage Model v1.2")

        self.lbl_result.setText(result_text)

        if probability > 75:
            self.lbl_result.setStyleSheet("color: #ff5555; border: 1px solid #ff5555; background-color: #2a1111;")
        elif probability > 30:
            self.lbl_result.setStyleSheet("color: #ffb86c; border: 1px solid #ffb86c; background-color: #2a2011;")
        else:
            self.lbl_result.setStyleSheet("color: #50fa7b; border: 1px solid #50fa7b; background-color: #112a18;")

        self.result_container.show()

    def submit_feedback(self, feedback_type):
        QMessageBox.information(self, "Feedback Recorded",
                                f"Thank you for your {feedback_type} feedback. This helps improve the model.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PACSInterface()
    window.show()
    sys.exit(app.exec_())