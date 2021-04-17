import sys
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(800,800)
    w.setStyleSheet("background-color: white;")
    w.setWindowTitle("eye examination App")

    label = QLabel(w)
    label.setStyleSheet("Color: black;")
    label.setText("0.1")
    label.setFont(QFont('Times', 100))
    label.move(340,50)
    label.show()

    pic = QLabel(w)
    pic.setPixmap(QPixmap("E.png"))
    pic.move(340, 400)
    pic.show() # You were missing this.

    w.show()
    sys.exit(app.exec_())