# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/andy/Dropbox/AAI_Eric/ui/mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(640, 480)
        MainWindow.setAcceptDrops(True)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("../../../Pictures/Screenshot from 2016-12-07 15-03-21.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralWidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.frame = QtGui.QFrame(self.centralWidget)
        self.frame.setFrameShape(QtGui.QFrame.Panel)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout_7.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.listContainer = QtGui.QHBoxLayout()
        self.listContainer.setObjectName(_fromUtf8("listContainer"))
        self.horizontalLayout.addLayout(self.listContainer)
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.pushButton = QtGui.QPushButton(self.frame)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.verticalLayout_5.addWidget(self.pushButton)
        self.pushButton_2 = QtGui.QPushButton(self.frame)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.verticalLayout_5.addWidget(self.pushButton_2)
        self.pushButton_4 = QtGui.QPushButton(self.frame)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.verticalLayout_5.addWidget(self.pushButton_4)
        self.pushButton_3 = QtGui.QPushButton(self.frame)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.verticalLayout_5.addWidget(self.pushButton_3)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 2)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.ImageGroupBox = QtGui.QGroupBox(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageGroupBox.sizePolicy().hasHeightForWidth())
        self.ImageGroupBox.setSizePolicy(sizePolicy)
        self.ImageGroupBox.setObjectName(_fromUtf8("ImageGroupBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.ImageGroupBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 1, 2, 1)
        self.imageLayout = QtGui.QVBoxLayout()
        self.imageLayout.setObjectName(_fromUtf8("imageLayout"))
        self.gridLayout_2.addLayout(self.imageLayout, 1, 0, 1, 1)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.gridLayout_2.addLayout(self.verticalLayout_4, 2, 0, 1, 1)
        self.widget = QtGui.QWidget(self.ImageGroupBox)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout_2.addWidget(self.widget, 0, 1, 1, 1)
        self.sumRowsLayout = QtGui.QHBoxLayout()
        self.sumRowsLayout.setObjectName(_fromUtf8("sumRowsLayout"))
        self.gridLayout_2.addLayout(self.sumRowsLayout, 0, 0, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setRowStretch(0, 2)
        self.gridLayout_2.setRowStretch(1, 2)
        self.gridLayout_2.setRowStretch(2, 1)
        self.horizontalLayout_3.addWidget(self.ImageGroupBox)
        self.groupBox_2 = QtGui.QGroupBox(self.frame)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_3 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.tableWidget = QtGui.QTableWidget(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setObjectName(_fromUtf8("tableWidget"))
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout_3.addWidget(self.tableWidget, 0, 1, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout_3.addLayout(self.verticalLayout, 1, 1, 1, 1)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setRowStretch(0, 1)
        self.gridLayout_3.setRowStretch(1, 1)
        self.horizontalLayout_3.addWidget(self.groupBox_2)
        self.horizontalLayout_3.setStretch(0, 3)
        self.horizontalLayout_3.setStretch(1, 2)
        self.verticalLayout_7.addLayout(self.horizontalLayout_3)
        self.verticalLayout_7.setStretch(0, 1)
        self.verticalLayout_7.setStretch(1, 2)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 640, 19))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        self.menuHere = QtGui.QMenu(self.menuBar)
        self.menuHere.setObjectName(_fromUtf8("menuHere"))
        MainWindow.setMenuBar(self.menuBar)
        self.actionHere = QtGui.QAction(MainWindow)
        self.actionHere.setObjectName(_fromUtf8("actionHere"))
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionNew_Project = QtGui.QAction(MainWindow)
        self.actionNew_Project.setObjectName(_fromUtf8("actionNew_Project"))
        self.actionChange_Project_Settings = QtGui.QAction(MainWindow)
        self.actionChange_Project_Settings.setObjectName(_fromUtf8("actionChange_Project_Settings"))
        self.menuHere.addAction(self.actionNew_Project)
        self.menuHere.addAction(self.actionQuit)
        self.menuHere.addAction(self.actionHere)
        self.menuHere.addAction(self.actionChange_Project_Settings)
        self.menuBar.addAction(self.menuHere.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Analyse and Integrate", None))
        self.pushButton.setText(_translate("MainWindow", "Reanalyse Selected", None))
        self.pushButton_2.setText(_translate("MainWindow", "Output PDF for Selected", None))
        self.pushButton_4.setText(_translate("MainWindow", "Ouptut CSV for selected", None))
        self.pushButton_3.setText(_translate("MainWindow", "CollateAll PDFs in the Output Folder", None))
        self.ImageGroupBox.setTitle(_translate("MainWindow", "Image", None))
        self.groupBox_2.setTitle(_translate("MainWindow", "Analysis", None))
        self.menuHere.setTitle(_translate("MainWindow", "Project", None))
        self.actionHere.setText(_translate("MainWindow", "Save Project", None))
        self.actionQuit.setText(_translate("MainWindow", "Open Existing Project", None))
        self.actionNew_Project.setText(_translate("MainWindow", "New Project", None))
        self.actionChange_Project_Settings.setText(_translate("MainWindow", "Change Project Settings", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

