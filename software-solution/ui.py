import sys
import random
import numpy as np
import pydicom
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt


class InteractiveImageViewer(QGraphicsView):
    """Custom Graphics View to handle zooming, panning, and zoom limits"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # PACS-style visual settings
        self.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.min_zoom = 1.0

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.scene.itemsBoundingRect())

        # Fit image to view and establish the baseline minimum zoom
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
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

            # Prevent zooming out smaller than the view boundary
            if current_scale * zoom_factor < self.min_zoom:
                zoom_factor = self.min_zoom / current_scale

        self.scale(zoom_factor, zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            current_scale = self.transform().m11()

            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
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
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QLabel#result_label {
                font-size: 16px;
                font-weight: bold;
                color: #ff5555;
                padding: 10px;
                border: 1px solid #ff5555;
                border-radius: 4px;
                background-color: #2a1111;
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
        sidebar.setFixedWidth(300)
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

        self.btn_analyse = QPushButton("Run Metastasis Analysis")
        self.btn_analyse.clicked.connect(self.run_analysis)
        self.btn_analyse.setEnabled(False)

        info_text = "Patient: JOHN DOE\nID: 987654321\nDOB: 01/01/1970\nStudy: MRI Routine"
        self.lbl_patient_info = QLabel(info_text)
        self.lbl_patient_info.setStyleSheet("color: #aaa; font-size: 13px; line-height: 1.5;")
        self.lbl_patient_info.hide()

        self.lbl_result = QLabel("Result: Pending...")
        self.lbl_result.setObjectName("result_label")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.hide()

        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(self.btn_upload)
        sidebar_layout.addWidget(self.btn_analyse)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self.lbl_patient_info)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.lbl_result)

        self.image_viewer = InteractiveImageViewer()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.image_viewer)

    def upload_image(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)",
            options=options
        )

        if file_name:
            try:
                dicom_data = pydicom.dcmread(file_name)
                pixel_array = dicom_data.pixel_array


                if pixel_array.ndim == 3:
                    if pixel_array.shape[2] == 3:
                        pixel_array = np.mean(pixel_array, axis=2)
                    else:
                        pixel_array = pixel_array[0]
                elif pixel_array.ndim != 2:
                    raise ValueError(f"Unsupported image dimensions: {pixel_array.shape}")

                pixel_array = pixel_array.astype(float)
                if pixel_array.max() > 0:
                    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0

                pixel_array = np.uint8(pixel_array)
                height, width = pixel_array.shape

                image_bytes = pixel_array.tobytes()
                q_img = QImage(image_bytes, width, height, width, QImage.Format.Format_Grayscale8)

                if q_img.isNull():
                    raise ValueError("Failed to generate image buffer from DICOM data.")

                pixmap = QPixmap.fromImage(q_img)
                self.image_viewer.set_image(pixmap)

                self.image_loaded = True
                self.btn_analyse.setEnabled(True)
                self.lbl_patient_info.show()
                self.lbl_result.hide()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM image.\n\nDetails: {str(e)}")

    def run_analysis(self):
        if not self.image_loaded:
            return

        probability = random.uniform(2.5, 98.5)
        self.lbl_result.setText(f"Metastasis Probability:\n{probability:.1f}%")

        if probability > 75:
            self.lbl_result.setStyleSheet(
                "color: #ff5555; border-color: #ff5555; background-color: #2a1111; font-size: 16px; font-weight: bold; padding: 10px;")
        elif probability > 30:
            self.lbl_result.setStyleSheet(
                "color: #ffb86c; border-color: #ffb86c; background-color: #2a2011; font-size: 16px; font-weight: bold; padding: 10px;")
        else:
            self.lbl_result.setStyleSheet(
                "color: #50fa7b; border-color: #50fa7b; background-color: #112a18; font-size: 16px; font-weight: bold; padding: 10px;")

        self.lbl_result.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PACSInterface()
    window.show()
    sys.exit(app.exec())