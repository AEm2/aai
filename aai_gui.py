from PyQt4 import QtGui
#from PyQt4 import QtCore
from ui.mainwindow import MainWindow
#from ui.dialog_chooseproject import Dialog

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    ui.dialogChoose.show()
    sys.exit(app.exec_())
