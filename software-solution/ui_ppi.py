import sys
import os
import json
import datetime
import numpy as np
import cv2
import pydicom
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from ai import ai_service, config


class AnalysisThread(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, pixels, laterality, view, case_id, parent=None):
        super().__init__(parent)
        self.pixels = pixels
        self.laterality = laterality
        self.view = view
        self.case_id = case_id

    def run(self):
        try:
            result = ai_service.analyse(self.pixels, self.laterality, self.view, self.case_id)
            study = {
                "laterality": self.laterality,
                "view": self.view,
                "accession": self.case_id,
                "_bbox": ai_service.breast_bbox(self.pixels.astype(np.float32)),
            }
            result.explanation, result.explanation_source = ai_service.explain(result, study)
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class InteractiveImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.heatmap_item = QGraphicsPixmapItem()
        self.heatmap_item.setOpacity(0.55)
        self.heatmap_item.setZValue(1)
        self.scene.addItem(self.heatmap_item)
        self.roi_items = []

        self.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.min_zoom = 1.0

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.clear_overlay()
        self.setSceneRect(self.scene.itemsBoundingRect())
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.min_zoom = self.transform().m11()

    def set_overlay(self, heatmap, rois):
        self.clear_overlay()
        if heatmap is not None and heatmap.max() > 1e-6:
            normalised = heatmap / heatmap.max()
            colored = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_JET)
            rgba = np.zeros((*heatmap.shape, 4), np.uint8)
            rgba[..., :3] = colored[..., ::-1]
            rgba[..., 3] = (np.clip((normalised - 0.25) / 0.75, 0, 1) * 235).astype(np.uint8)
            rgba = np.ascontiguousarray(rgba)
            h, w = heatmap.shape
            image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            self.heatmap_item.setPixmap(QPixmap.fromImage(image))
        for rank, roi in enumerate(rois[:3]):
            rect = QGraphicsRectItem(roi["x"], roi["y"], roi["w"], roi["h"])
            pen = QPen(QColor("#ffb86c") if rank == 0 else QColor("#e0e0e0"))
            pen.setCosmetic(True)
            pen.setWidthF(2.0 if rank == 0 else 1.0)
            pen.setStyle(Qt.SolidLine if rank == 0 else Qt.DashLine)
            rect.setPen(pen)
            rect.setZValue(2)
            self.scene.addItem(rect)
            self.roi_items.append(rect)

    def clear_overlay(self):
        self.heatmap_item.setPixmap(QPixmap())
        for item in self.roi_items:
            self.scene.removeItem(item)
        self.roi_items = []

    def set_overlay_visible(self, visible):
        self.heatmap_item.setVisible(visible)
        for item in self.roi_items:
            item.setVisible(visible)

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
        self.pixels = None
        self.laterality = "L"
        self.view = "CC"
        self.case_id = ""
        self.last_result = None
        self.worker = None
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self.analysis_timeout)

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
            QPushButton:checked {
                background-color: #2a3a4a;
                border: 1px solid #6aa0d8;
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

        self.btn_upload = QPushButton("Upload Image (.dcm / .png)")
        self.btn_upload.clicked.connect(self.upload_image)

        self.btn_analyse = QPushButton("Run AI Analysis")
        self.btn_analyse.clicked.connect(self.start_analysis)
        self.btn_analyse.setEnabled(False)

        self.btn_heatmap = QPushButton("Show Heatmap")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setChecked(True)
        self.btn_heatmap.toggled.connect(self.toggle_heatmap)
        self.btn_heatmap.hide()

        self.lbl_patient_info = QLabel("")
        self.lbl_patient_info.setStyleSheet("color: #aaa; font-size: 17px; line-height: 1.5;")
        self.lbl_patient_info.setWordWrap(True)
        self.lbl_patient_info.hide()

        self.lbl_status = QLabel("Analyzing your image...")
        self.lbl_status.setObjectName("status_label")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.hide()

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
        sidebar_layout.addWidget(self.btn_heatmap)
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
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Mammogram", "",
            "Mammograms (*.dcm *.png *.jpg *.jpeg *.tif *.tiff);;DICOM Files (*.dcm);;All Files (*)",
            options=options)

        if file_name:
            try:
                if file_name.lower().endswith(".dcm"):
                    pixel_array, info = self.read_dicom(file_name)
                else:
                    pixel_array = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                    if pixel_array is None:
                        raise IOError("unsupported image format")
                    info = {}

                self.pixels = np.ascontiguousarray(pixel_array)
                self.laterality = info.get("laterality") or ai_service.guess_laterality(self.pixels)
                upper_name = os.path.basename(file_name).upper()
                self.view = info.get("view") or ("MLO" if "MLO" in upper_name else ("CC" if "CC" in upper_name else ""))
                self.case_id = os.path.basename(file_name)

                height, width = self.pixels.shape
                q_img = QImage(self.pixels.data, width, height, width, QImage.Format_Grayscale8).copy()
                self.image_viewer.set_image(QPixmap.fromImage(q_img))

                side = "Right" if self.laterality == "R" else "Left"
                self.lbl_patient_info.setText(
                    f"Patient: {info.get('name') or 'ANONYMOUS'}\n"
                    f"ID: {info.get('patient_id') or '—'}\n"
                    f"Study: Mammography\n"
                    f"View: {side} breast {self.view}\n"
                    f"File: {self.case_id}")

                self.image_loaded = True
                self.last_result = None
                self.btn_analyse.setEnabled(True)
                self.btn_heatmap.hide()
                self.lbl_patient_info.show()
                self.result_container.hide()
                self.lbl_status.hide()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image.\n\nDetails: {str(e)}")

    def read_dicom(self, file_name):
        dicom_data = pydicom.dcmread(file_name)
        pixel_array = dicom_data.pixel_array.astype(float)
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[..., 0] if pixel_array.shape[-1] in (3, 4) else pixel_array[0]
        if str(getattr(dicom_data, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array
        pixel_array -= pixel_array.min()
        if pixel_array.max() > 0:
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        laterality = str(getattr(dicom_data, "ImageLaterality", "") or getattr(dicom_data, "Laterality", "")).upper()
        view = str(getattr(dicom_data, "ViewPosition", "")).upper()
        info = {
            "name": str(getattr(dicom_data, "PatientName", "") or ""),
            "patient_id": str(getattr(dicom_data, "PatientID", "") or ""),
            "laterality": laterality[:1] if laterality[:1] in ("L", "R") else None,
            "view": "MLO" if "MLO" in view else ("CC" if "CC" in view else None),
        }
        return np.uint8(pixel_array), info

    def start_analysis(self):
        if not self.image_loaded or (self.worker is not None and self.worker.isRunning()):
            return

        self.btn_analyse.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_regenerate.setEnabled(False)
        self.result_container.hide()
        self.lbl_status.setStyleSheet("")
        self.lbl_status.setText("Analyzing your image...")
        self.lbl_status.show()

        self.worker = AnalysisThread(self.pixels, self.laterality, self.view, self.case_id, self)
        self.worker.finished_ok.connect(self.finish_analysis)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.start()
        self.timeout_timer.start(int(config.AI_TIMEOUT_S * 1000))

    def unlock_buttons(self):
        self.timeout_timer.stop()
        self.btn_analyse.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_regenerate.setEnabled(True)

    def analysis_failed(self, message):
        self.unlock_buttons()
        self.lbl_status.setStyleSheet("color: #ff5555;")
        self.lbl_status.setText(f"The AI could not analyse this image.\n{message}")
        self.lbl_status.show()

    def analysis_timeout(self):
        self.lbl_status.setStyleSheet("color: #ffb86c;")
        self.lbl_status.setText(f"No answer from the AI after {config.AI_TIMEOUT_S:.0f} s. "
                                "The result will appear if it arrives.")
        self.btn_upload.setEnabled(True)

    def finish_analysis(self, result):
        self.unlock_buttons()
        self.lbl_status.hide()
        self.last_result = result

        result_text = (f"<b>Urgency: {result.priority}</b><br><br>"
                       f"Malignancy score: {result.malignant:.3f}<br>"
                       f"Benign-finding score: {result.benign:.3f}<br>"
                       f"Why: {result.explanation}<br>"
                       f"Source: {result.model}<br>"
                       f"Explanation by: {result.explanation_source}")

        self.lbl_result.setText(result_text)

        if result.priority == "HIGH":
            self.lbl_result.setStyleSheet("color: #ff5555; border: 1px solid #ff5555; background-color: #2a1111;")
        elif result.priority == "MEDIUM":
            self.lbl_result.setStyleSheet("color: #ffb86c; border: 1px solid #ffb86c; background-color: #2a2011;")
        else:
            self.lbl_result.setStyleSheet("color: #50fa7b; border: 1px solid #50fa7b; background-color: #112a18;")

        self.image_viewer.set_overlay(result.heatmap, result.rois)
        self.image_viewer.set_overlay_visible(self.btn_heatmap.isChecked())
        self.btn_heatmap.show()
        self.result_container.show()

    def toggle_heatmap(self, visible):
        self.image_viewer.set_overlay_visible(visible)

    def submit_feedback(self, feedback_type):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(os.path.join(config.LOG_DIR, "feedback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "file": self.case_id,
                "feedback": feedback_type,
                "priority": self.last_result.priority if self.last_result else None,
                "malignant": self.last_result.malignant if self.last_result else None,
                "model": self.last_result.model if self.last_result else None,
            }) + "\n")
        QMessageBox.information(self, "Feedback Recorded",
                                f"Thank you for your {feedback_type} feedback. This helps improve the model.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PACSInterface()
    window.show()
    sys.exit(app.exec_())
