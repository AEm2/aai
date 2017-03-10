# -*- coding: utf-8 -*-

"""
Module implementing Dialog.
"""

#from PyQt4.QtCore import pyqtSignature
from PyQt4.QtGui import QDialog
from PyQt4 import QtGui
from PyQt4 import QtCore

from .Ui_dialog_chooseproject import Ui_Dialog

import os

class myLineEdit(QtGui.QLineEdit):
    def __init__(self, parent=None):
            super(myLineEdit, self).__init__(parent)
            self.setText('Project Description')
            
    #reimplement the mousePressEvent
    def mousePressEvent (self, event):
        if 'Project Description' in str(self.text()):
            self.setText('')
        
class Dialog(QDialog, Ui_Dialog):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget (QWidget)
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.projectList = os.listdir(os.path.join(os.curdir, 'data'))
        #populate the ListView with available projects in the data directory
        for project in self.projectList:
            QtGui.QListWidgetItem(project, self.projectListWidget)
        self.lineEditWidget = myLineEdit()
        self.lineEditWidget.setGeometry(QtCore.QRect(10, 51, 371, 181))
        self.horizontalLayout.insertWidget(0,self.lineEditWidget)
        #self.horizontalLayout.addWidget(self.lineEditWidget)
    
