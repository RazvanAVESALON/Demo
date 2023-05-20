
import os
import sys
import json
from predictie_pe_frame import DEMO_PredicitionClass
from predictie_pe_frame_segmentare import DEMO_PredicitionClass_seg
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *



def apply_transfer_function(frames, w, c):
    y_min = 0.0
    y_max = 255.0

    below = (frames <= c - 0.5 - (w-1) / 2.0)
    above = (frames > c - 0.5 + (w-1) / 2.0)
    between = (~below) & (~above)

    result = frames.copy()

    result[below] = y_min
    result[above] = y_max
    result[between] = ((frames[between] - (c - 0.5)) /
                       (w-1) + 0.5) * (y_max - y_min) + y_min

    return result


class DEMOWidow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi("demo.ui", self)
        self.show()
        self.data=0
        self.angio = None
        self.clipping_points = {}
        self.prediction_clipping_points={}

        self.current_case=0
        self.current_frame = 0
        self.last_opened_path = r""

    @pyqtSlot()
    def on_open_btn_clicked(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open study", self.last_opened_path)
        if path:
            self.last_opened_path = path
            self.cases = []
            for dir in os.listdir(path):
                dirpath = os.path.join(path, dir)
                if os.path.isdir(dirpath):
                    self.cases.append(dirpath)

            if len(self.cases) > 0:
                self.current_case_idx = 0
                self.load_case(self.cases[self.current_case_idx])
            else:
                print("No cases found. Make sure you select the study directory.")   
       
    @pyqtSlot()
    def on_previous_frame_btn_clicked(self):
        if self.angio is None:
            return

        self.current_frame -= 1
        self.current_frame %= self.angio.shape[0]
        self.update_image_view()

    @pyqtSlot()
    def on_next_frame_btn_clicked(self):
        if self.angio is None:
            return

        self.current_frame += 1
        self.current_frame %= self.angio.shape[0]
        self.update_image_view()

    @pyqtSlot()
    def on_previous_case_btn_clicked(self):

        self.current_case_idx -= 1
        self.current_case_idx %= len(self.cases)
        self.load_case(self.cases[self.current_case_idx])

    @pyqtSlot()
    def on_next_case_btn_clicked(self):

        self.current_case_idx += 1
        self.current_case_idx %= len(self.cases)
        self.load_case(self.cases[self.current_case_idx])

    def on_image_view_wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.on_previous_frame_btn_clicked()
        else:
            self.on_next_frame_btn_clicked()

    def load_case(self, case_path):
        self.current_case=case_path
        with open(os.path.join(case_path, "angio_loader_header.json"), "r") as f:
            self.metadata = json.load(f)
        self.Acq_case.setText(os.path.basename(os.path.normpath(case_path)))
        head, tail = os.path.split(case_path)
        self.patient_case.setText(os.path.basename(os.path.normpath(head)))

        self.annotation_path = os.path.join(case_path, "clipping_points.json")

        with open(self.annotation_path, "r") as f:
            self.clipping_points = json.load(f)
            
        with open(self.annotation_path, "r") as q:
            self.prediction_clipping_points=json.load(q)

        self.pred_path=os.path.join(case_path, "pred.csv")
        self.angio = np.load(os.path.join(
            case_path, "frame_extractor_frames.npz"))["arr_0"]
        self.angio = apply_transfer_function(
            self.angio, self.metadata['WindowWidth'], self.metadata['WindowCenter'])
        self.angio = self.angio.astype(np.uint8)

        self.current_frame = 0

        self.update_image_view()

    def update_image_view(self):
        if self.angio is None:
            return

        self.slider.setMaximum(self.angio.shape[0] - 1)
        self.slider.setValue(self.current_frame)

        qimage = QImage(self.angio[self.current_frame].data, self.metadata["ImageSize"][0],
                        self.metadata["ImageSize"][1], self.metadata["ImageSize"][0], QImage.Format_Grayscale8)
        self.angio_view.setFixedWidth(self.metadata["ImageSize"][1])
        self.angio_view.setFixedHeight(self.metadata["ImageSize"][0])
        self.angio_view.setPixmap(QPixmap(qimage))

        if self.find_btn.isChecked():
            if self.reg_check.isChecked():
                predictie, distance=DEMO_PredicitionClass(self.angio[self.current_frame],self.prediction_clipping_points[str(self.current_frame)],self.metadata).__predict__()
                
                self.dist.setText(str(distance))
                painter = QPainter(self.angio_view.pixmap())
                painter.setPen(QColor("yellow"))
                painter.setBrush(QColor("yellow"))

                painter.drawEllipse(QPoint(predictie[1], predictie[0]), 5, 5)

                painter.end()
            elif self.seg_check.isChecked():
                predictie , distance =DEMO_PredicitionClass_seg(self.angio[self.current_frame],self.prediction_clipping_points[str(self.current_frame)],self.metadata).__predict__()
                self.dist.setText(str(distance))
                
                for x, y in zip(predictie['x'], predictie['y']):
                    painter = QPainter(self.angio_view.pixmap())
                    painter.setPen(QColor("yellow"))
                    painter.setBrush(QColor("yellow"))

                    painter.drawEllipse(QPoint(x, y), 5, 5)

                    painter.end()
               
            if str(self.current_frame) in self.clipping_points:
                painter = QPainter(self.angio_view.pixmap())
                painter.setPen(QColor("red"))
                painter.setBrush(QColor("red"))

                painter.drawEllipse(QPoint(self.clipping_points[str(
                    self.current_frame)][1], self.clipping_points[str(self.current_frame)][0]), 5, 5)

                painter.end()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DEMOWidow()
    sys.exit(app.exec_())
