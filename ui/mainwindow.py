# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import QMainWindow
from PyQt4 import QtCore,  QtGui
from .Ui_mainwindow import Ui_MainWindow
from .dialog_chooseproject import Dialog
#sudo apt-get install pyqt4-dev-tools to compile form with pyuic4 in linux

from analyse import getLanesCwt,  getBackground,  getSumRows,  analyse,  binPeaks,  getResults
from curveFit import fitCurve
from dataOutput import outputPdf
from myClasses import Analysis

import pyqtgraph as pg
import os
import numpy as np
import readTif
import pickle
import h5py

def clearWidget(widget):
    for i in reversed(range(widget.count())): 
        widget.itemAt(i).widget().setParent(None)    
    return
    
class TestListView(QtGui.QListWidget):
    """Accepts file drops into the QListWidget"""
    def __init__(self, type, parent=None):
        super(TestListView, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.emit(QtCore.SIGNAL("dropped"), links)
        else:
            event.ignore()

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        #global setting for pyqtgraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'b')   
        pg.setConfigOptions(antialias=True)
        
        self.fnumpy = None
        self.fdict = None
        
        #set up sumCols widget
        self.scp = pg.PlotWidget()
        self.scp_itm = self.scp.getPlotItem()
        self.scp.hideAxis('bottom')
        self.scp.hideAxis('left')
        self.scp.setMouseEnabled(x=False,  y=False)
        self.scp.hideButtons()
        self.verticalLayout_4.addWidget(self.scp)

        #set up the imageView widget
        STEPS = np.array([0.0, 0.2, 0.6, 1.0])
        CLRS = ['k', 'r', 'y', 'w']        
        clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))
        self.lut = clrmp.getLookupTable()
        
        #self.imv = pg.ImageView()
        self.imv = pg.GraphicsView()
        self.imageLayout.addWidget(self.imv)
        self.img_itm = pg.ImageItem()
        self.imv.addItem(self.img_itm)
        
        self.grad = pg.HistogramLUTWidget()
        self.grad.gradient.tickPen = pg.mkPen('k')
        self.grad.setImageItem(self.img_itm)
        #self.grad.gradient.loadPreset('thermal')
        self.grad.gradient.restoreState({'ticks': [(0.0, (0, 255, 255, 255)), (1.0, (255, 255, 0, 255)), (0.5, (0, 0, 0, 255)), (0.25, (0, 0, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'})
        self.grad.gradient.restoreState({'ticks': [(0.0, (255, 255, 255, 255)), (1.0, (0, 0, 0, 255))], 'mode': 'rgb'})
        self.horizontalLayout_2.addWidget(self.grad)
        
        #add some plots to the sumRows layout
        self.plots = [] #create a list of mplWidgets
        for i in range(10):
            self.plots.append(pg.PlotWidget()) #using pyqtgraph
            self.sumRowsLayout.addWidget(self.plots[i])
            self.plots[i].showAxis('bottom', show=False)
            self.plots[i].hideAxis('left')
            self.plots[i].setMouseEnabled(x=False,  y=False)
            self.plots[i].hideButtons()
            
        #plot for the kobs/kinetics results
        self.kobs = pg.PlotWidget()
        #self.kobs.hideAxis('bottom') #don't hide the time axis for this one
        #self.kobs.hideAxis('left')
        self.kobs.setMouseEnabled(x=False,  y=False)
        self.kobs.hideButtons()
        self.verticalLayout.addWidget(self.kobs)
        #TODO: set y-range from 0 to 1 permanently
        
        #the QListWidget for accepting file drops
        self.view = TestListView(self)
        self.connect(self.view, QtCore.SIGNAL("dropped"), self.pictureDropped)
        self.listContainer.addWidget(self.view)
        self.view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        
        #what was the window's last position and size?
        self.settings = QtCore.QSettings("MyCompany", "MyApp")
        if self.settings.value("geometry") is not None:
            self.restoreGeometry(self.settings.value("geometry").toByteArray())
        if self.settings.value("windowState") is not None:
            self.restoreState(self.settings.value("windowState").toByteArray())
        
        #open a dialog so the user can open or create a project
        self.currentProject = None
        self.dialogChoose = Dialog()
        #dialogChoose SIGNALS AND SLOTS
        self.dialogChoose.pushButton.clicked.connect(self.dlg_pb1_clicked)
        self.dialogChoose.pushButton_2.clicked.connect(self.dlg_pb2_clicked)
        
        #MainWindow SIGNALS AND SLOTS
        self.pushButton.clicked.connect(self.pb1_clicked) #reanalyse a selection of images
        self.pushButton_2.clicked.connect(self.pb2_clicked) #output PDFs to ./pdf/ #TODO: make project specific
        self.pushButton_3.clicked.connect(self.pb3_clicked) #combine PDFs
        self.pushButton_4.clicked.connect(self.pb4_clicked) #output CSV
        self.view.clicked.connect(self.selectionClicked) #tif selection changed

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtGui.QMessageBox.question(self, 'Message', 
                         quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    
        if reply == QtGui.QMessageBox.Yes:
            #do what needs to be done before closing
            if self.fnumpy is not None:
                print('Closing hdf5 file')
                self.fnumpy.close()
            #this needs to be changed for project-based program
            if self.currentProject:
                print('Saving Analysis() data')
                dict_path = os.path.join(os.curdir,'data', self.currentProject,'aai.dict')
                f = open(dict_path,  'wb')
                pickle.dump(self.fdict, f)
                f.close()
            #save the size and position of the screen
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
            event.accept()
        else:
            event.ignore()
            
    @pyqtSlot()
    def pictureDropped(self, l):
        #TODO: Progress Dialog on this one
        for url in l:
            if os.path.exists(url):
                print(url)
                #needs to be a tif
                if '.tif' in url:
                    path,  filename = os.path.split(str(url))
                    file_noext = filename[:-4]
                    print path,  filename,  file_noext
                    #does the item already exist in the list?
                    items = []
                    for index in xrange(self.view.count()):
                        items.append(self.view.item(index).text())
                    #print(items)
                    #if it's not there already, add it
                    if file_noext not in items:
                        item = QtGui.QListWidgetItem(file_noext, self.view)    
                        item.setStatusTip(file_noext)

                    #save to hdf5 instead of directory
                    #TODO: test that this works properly
                    if file_noext not in self.fnumpy: 
                        #dataset = file_noext+'/img16'
                        #print('Try loading {0} img16 array from hdf5'.format(dataset))
                        image = readTif.tifFile(url)
                        img16 = image.image
                        print('Saving img16 array to hdf5')
                        #create the group called file_noext
                        self.fnumpy.create_group(file_noext)
                        #then load create the dataset 'img16'  
                        self.fnumpy[file_noext].create_dataset('img16',  data=img16)
                    else: #TODO: what if the user wants the data updated?
                        #the image exists in the hdf5 file, 
                        #does the user want to update the image?
                        #open a dialog to ask
                        pass
                    #now analyse the image    
                    self.analyseImg(file_noext)
                #find the last analysed item in the QListWidget
                items = self.view.findItems(file_noext, QtCore.Qt.MatchFlag(1)) #'1' = 'item contains'
                if len(items):
                    #if it was found, select it
                    self.view.setItemSelected(items[0], True)
                else:
                    #otherwise, select the first item in the list
                    self.view.setCurrentRow(0)
                self.selectionClicked() #update the plots
    
    @pyqtSlot()
    def on_selection_changed(self):
        #print('Selection Changed')
        #what is the name of the selected item?
        file_noext = str(self.view.item(self.view.currentRow()).text())
        #do the analysis to plot the necessary elements
        self.analyseImg(file_noext)
        
    @pyqtSlot()
    def pb1_clicked(self):
        """Reanalyse a selection of images"""
        file_noext = str(self.view.item(self.view.currentRow()).text())
        #maybe there are multiple selections?
        list = self.view.selectedItems()
        if len(list) > 1:
            dlg = QtGui.QProgressDialog("Analysing",  "Cancel Analysis",  0, len(list))
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setWindowTitle("")
            dlg.show()
            count = 0
            for item in list:
                dlg.setLabelText("Analysing {0}".format(str(item.text())))
                self.analyseImg(str(item.text()), reanalyse=True)
                count += 1
                dlg.setValue(count)
                if dlg.wasCanceled():
                    break
        else:
            self.analyseImg(file_noext, reanalyse=True)
        return
        
    @pyqtSlot()
    def pb2_clicked(self):
        """Reanalyse all selected items and output to PDF"""
        file_noext = str(self.view.item(self.view.currentRow()).text())
        #maybe there are multiple selections?
        list = self.view.selectedItems()
        if len(list) > 1:
            dlg = QtGui.QProgressDialog("Creating PDF",  "Cancel Analysis",  0, len(list))
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setWindowTitle("")
            dlg.show()
            count = 0
            for item in list:
                dlg.setLabelText("Analysing {0}".format(str(item.text())))
                self.analyseImg(str(item.text()), reanalyse=True,  pdf=True)
                count += 1
                dlg.setValue(count)
                if dlg.wasCanceled():
                    break
        else:
            self.analyseImg(file_noext, reanalyse=True,  pdf=True)
        return
    
    @pyqtSlot()
    def selectionClicked(self):
        #don't run analysis with control or shift pressed
        #wait for user to make multiple selections
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier or modifiers == QtCore.Qt.ShiftModifier:
            return
        else:
            list = self.view.selectedItems()
            self.analyseImg(str(list[0].text()))
        return
        
    def pb3_clicked(self):
        """Combine PDF button clicked"""
        # Loading the pyPdf Library
        from pyPdf import PdfFileWriter, PdfFileReader
        # Create a routine that appends files to the output file
        def append_pdf(input,output):
            [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]
        # Create an object where pdf pages are appended to
        output = PdfFileWriter()
        pdfs = os.listdir(os.path.join(os.curdir, 'pdf'))
        for apdf in pdfs:
            append_pdf(PdfFileReader(open(os.path.join(os.curdir, 'pdf',apdf),"rb")),output)
        # Write all the collected pages to a file
        output.write(open("{0}_combined.pdf".format(self.currentProject),"wb")) 
        return
        
    @pyqtSlot()
    def pb4_clicked(self):
        """Output CSV button clicked"""
        list = self.view.selectedItems()
        if len(list) > 1:
            dlg = QtGui.QProgressDialog("Output CSV",  "Cancel Output",  0, len(list))
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setWindowTitle("")
            dlg.show()
            count = 0
            f = open('results.csv',  'wb')
            f.write('Variant, k1 from XXX, k1 from XX, k2 from XX, k1 from X, k2 from X\n')
            for item in list: #item is the file_noext
                print(str(item.text()))
                kobs = self.fdict[str(item.text())].kobs
                print(kobs)
                print self.csv(kobs)
                alist = []
                f.write(str(item.text())+', ') #write the variant
                if len(kobs) > 1:
                    for i in kobs:
                        for j in i:
                            alist.append(j)
                    f.write(self.csv(alist))
                else:
                    print kobs[0].shape
                    print type(kobs[0])
                    print self.csv(kobs[0].tolist())
                    f.write(str(kobs[0].tolist()[0]))
                f.write('\n')
                count += 1
                dlg.setValue(count)
                if dlg.wasCanceled():
                    break
            f.close()
        return
        
    def csv(self, l,spacer = ', '):
        if len(l) > 1:
            a = ''
            a += str(l[0])
            for i in l[1:]:
                a += spacer
                a += str(i)
            return a
        return str(l)
        
    def showImage(self, img_fp):
        """Numpy Floating Point Array plotted to PyQtGraph Widget"""
        print("Show Image:")
        if type(img_fp) is not np.ndarray:
            raise Exception('showImage wants a numpy aray, received {0} instead'.format(type(img_fp)))
        img_fp /= 65535 #normalise, was a 16 bit unsigned integer
        self.img_itm.setImage(np.transpose(img_fp))
        print self.imv.geometry()
        print self.imageLayout.geometry()
        yscale = float(self.imageLayout.geometry().height())/self.img_itm.height()
        xscale = float(self.imageLayout.geometry().width())/self.img_itm.width()
        #resize the image to fit
        self.img_itm.resetTransform()
        self.img_itm.scale(xscale,  yscale)
        #TODO: maximise the dynamic range of the image by setting the HistogramLUT
        return
        
    def showSumCols(self, sumCols, lane_bnd=None):
        self.scp.clear()
        apen = pg.mkPen(width=1, color='k')
        self.scp.plot(sumCols, pen=apen)
        #plot the lane boundaries
        if lane_bnd is not None:
            for lb in lane_bnd: #TODOP: plot these as linear regions
                self.scp.addItem(pg.InfiniteLine(pos=lb[0]))
                self.scp.addItem(pg.InfiniteLine(pos=lb[1]))             
        self.scp.setXRange(0+40, sumCols.shape[0]-40) #pretty up the edges along the x-axis
        return
        
    def showSumRows(self,  sumRows, lanes=None):
        #set up the colors for marking out the linear region items
        alpha = 127
        red = QtGui.QColor(255, 0, 0, alpha)
        green = QtGui.QColor(0, 255, 0, alpha)
        blue = QtGui.QColor(0, 0, 255, alpha)
        color = [blue, red, green] #the matplotlib color order
        #clear the plots
        for ln in range(10):
            self.plots[ln].clear()
        apen = pg.mkPen(width=1, color='k')
        #iterate through each lane in sumRows
        for ln in range(sumRows.shape[1]):
            ##self.plots[ln].plot(np.arange(0, sumRows[:, ln].shape[0]), sumRows[:, ln],  pen=apen)
            ##self.plots[ln].setXRange(0+40, sumRows[:, ln].shape[0]-40 )
            #swap x and y axes
            self.plots[ln].plot(sumRows[:, ln], np.arange(0, sumRows[:, ln].shape[0]),  pen=apen)
            ##self.plots[ln].plot(sumRows[:, ln], np.arange(sumRows[:, ln].shape[0], 0,-1),  pen=apen)
            self.plots[ln].setYRange(sumRows[:, ln].shape[0]-40, 0+40)
            self.plots[ln].invertY()
            
            #put the LinearRegionItem on the y-axis (by oriientation = Horizontal)
            if lanes is not None:
                x = 0
                for peak in lanes[ln].peaks:
                    if peak.color is not None: #the peak is only colored if the associated results have been deleted.
                        self.plots[ln].addItem(pg.LinearRegionItem(values=[peak.bounds[0], peak.bounds[1]], orientation = pg.LinearRegionItem.Horizontal, movable=False,  brush=QtGui.QBrush(peak.color)))
                    else:
                        self.plots[ln].addItem(pg.LinearRegionItem(values=[peak.bounds[0], peak.bounds[1]], orientation = pg.LinearRegionItem.Horizontal, movable=False,  brush=QtGui.QBrush(color[x])))
                        x += 1
                        if x > 2:
                            x=0
        
        
            
        return
        
    def showResults(self, results):
        """Fill a QTableWidget with the results"""
        self.tableWidget.setRowCount(results.shape[1])
        self.tableWidget.setColumnCount(results.shape[0])
        for row in range(results.shape[0]):
            for col in range(results.shape[1]):
                self.tableWidget.setItem(col,  row,  QtGui.QTableWidgetItem('{0:2.2f}'.format(results[row][col])))            
        return
        
    def showKineticsCurves(self, curves):
        self.kobs.clear()
        self.kobs.plot(curves[0, :], curves[1, :],  pen='b')
        if curves.shape[0] > 2:
            self.kobs.plot(curves[0, :], curves[2, :],  pen='r')
        if curves.shape[0] > 3:
            self.kobs.plot(curves[0, :], curves[3, :],  pen='g')        
        return
        
    def saveThumbnail(self, img, file_path):
        """save a light background jpg"""
        import scipy.misc
        img_in = np.asarray(img, dtype = np.uint16)
        a = np.max(img_in) - img_in #invert the image
        #TODO: increase dynamic range of the image
        scipy.misc.imsave(file_path, a)
        return

            
    def showAnalysis(self, file_noext):
        """If the analysis has already been done, then just plot the results"""
        #this is now done at the end of analyseImg
        return
        
    def analyseImg(self, file_noext, reanalyse=False, pdf=False):
        """Analyse the image"""
        #the analysis will be stored in two files,
        #1) aai.hdf5, containing all the numpy arrays, this is self.fnumpy
        #2) aai.dict, a dict containing pickled Analysis() classes self.fdict
        
        #load npy array, should have been saved when file dropped
        #keep this as a backup for the hdf5 file
#        outFolder = os.path.join(self.data_dir, file_noext)
#        img_fn = os.path.join(outFolder, 'img.npy')
        
#        #does the dict of this analysis exist in self.dict?
        if file_noext in self.fdict:
            print('dict found for {0}'.format(file_noext))
            print('kobs = {0}'.format(self.fdict[file_noext].kobs))
        else: #no? create a new entry with file_noext for key and new Analysis class
            self.fdict[file_noext] = Analysis()
            
        if reanalyse:
            #what to do here?
            self.fdict[file_noext] = Analysis()
            
#        print('Loading NumPy array {0}'.format(img_fn))
#        img16 = np.load(img_fn)
        
        if file_noext in self.fnumpy:
            dataset = file_noext+'/img16'
            print('Loading {0} img16 array from hdf5... '.format(dataset)), 
            img16 = self.fnumpy[dataset]
            print('Success')
        else:
            #should have been loaded during drop into listWidget
            raise Exception('Image retrieval from hdf5 failed')
            
            #Only do this while transitioning from folder based data storage
            #print('Saving img16 array to hdf5')
            #create the group called file_noext
            #self.fnumpy.create_group(file_noext)
            #then load create the dataset 'img16'  
            #self.fnumpy[file_noext].create_dataset('img16',  data=img16)
        
        img = np.array(img16, dtype=np.float64)
        sumCols = np.sum(img, axis=0)
        np.save('output_sumCols.npy', sumCols)

        #do cwt to smooth data, find minima, these are the limits of the lanes
        self.fdict[file_noext].lane_bnd = getLanesCwt(sumCols)
        
        #find the global background, subtract it from the image
        self.fdict[file_noext].setBackground(getBackground(img))

        self.showImage(np.array(img,  dtype=np.float32)) #have to pass a copy of the image, otherwise, foobar
        img -= int(self.fdict[file_noext].background)
        
        sumRows = getSumRows(img, self.fdict[file_noext].lane_bnd)
        
        lanes = analyse(sumRows)
        
        bins = binPeaks(lanes)
            
        results, newlanes = getResults(sumRows,  lanes,  bins)
            
        #fix up the lanes object, sort peaks, remove origin spots
#        for alane in newlanes:
#            print alane.lane_number
#            for apeak in alane.peaks:
#                print apeak.bin, apeak.bounds,  apeak.max,  apeak.argmax

        #what is the standard deviation along axis 1 for results?
        delete = []
        if results.shape[0] > 1:
            stds = np.std(results, axis=1)
            print stds
            stds = stds[:-1] #don't care about the last band
            print stds
            #if the standard deviation is less than 1000 and it is not the last row, delete that row
            print np.max(stds)
            thresh = np.max(stds)/10
            print('Std Dev threshold to keep row needs to be above {0}'.format(thresh))
            row = 0
            for std in np.nditer(stds): #everything except the last row
                if std < thresh:
                    delete.append(row)
                row += 1
        
        #need to delete from the results, make the LinearRegionItem grey to show we have excluded data
        print results
        if len(delete):
            #the rows to delete?
            print('Deleting row {0}'.format(delete))
            delete.reverse()
            print delete
            #for row in delete:
            #    results = np.delete(results, (row), axis=0)
            print results
        
            for aLane in lanes:
                for aPeak in aLane.peaks:
                    if aPeak.bin in delete:
                        aPeak.setColor(QtGui.QColor(55, 55, 55, 127))

        #what are the kobs, get the curves at the same time
        if 'Mn' in file_noext:
            timepoints = np.array([1,2,3,4,5,10,15,20,30, 60],dtype=np.float)
        else: #it's either Mg or something else
            timepoints = np.array([1,3,5,10,20,30,60,90,150,210],dtype=np.float)
            
        #are the results and timepoints the same size?
        if results.shape[1] != timepoints.shape[0]:
            print('Results and Time Points Different Shapes')
            if results.shape[1] == 9:
                #probably missed the last lane, try to fit excluding the last time point
                timepoints = timepoints[:-1]
            elif results.shape[1] == 8:
                timepoints = timepoints[:-2]
            else:
                #something else has happened
                raise Exception('Not enough data points to time points')
                
        kobs, curves, r2 = fitCurve(results, timepoints)
        self.fdict[file_noext].kobs = kobs
        
        #do all the plotting here
        self.showSumCols(sumCols, self.fdict[file_noext].lane_bnd)
        self.showSumRows(sumRows, newlanes)
        self.showResults(results)
        self.showKineticsCurves(curves)
        
        print pdf
        if pdf:
            print('Preparing pdf for output')
            outputPdf(file_noext, img,  results,  timepoints, curves,  kobs,  r2)
        
        return
        
    #for the open project dialog
    def dlg_pb1_clicked(self):
        """This will be called when the initial dialog at program start
        has the choose existing project button clicked"""
        #has the user selected anything?
        if not self.dialogChoose.projectListWidget.currentItem():
            return #if nothing selected, return
        self.currentProject = str(self.dialogChoose.projectListWidget.currentItem().text())
        print("Opening Existing Project {0}".format(self.currentProject))
        #now that we have the current project directory,
        #load the hdf5 and dict files
        hdf5_path = os.path.join(os.curdir, 'data', self.currentProject,'aai.hdf5')
        print('Opening {0}'.format(hdf5_path))
        self.fnumpy = h5py.File(hdf5_path,  'a') #allows read/write, will create file if doesn't exist.
        keys = self.fnumpy.keys()
        for key in keys:
            QtGui.QListWidgetItem(key, self.view)        
        #open the dict file, containing all the Analysis() objects
        aai_path = os.path.join(os.curdir, 'data', self.currentProject,'aai.dict')
        print('Opening {0}'.format(aai_path))
        if os.path.exists(aai_path):
            f = open(aai_path,  'rb')
            self.fdict = pickle.load(f) #read the whole file into self.fdict
            f.close() 
        else:
            #raise Exception
            self.fdict = dict() #this will save on close
        #select the first item of the list if it exists 
        #TODO: save state and remember the last selected item
        print(self.view.count())
        if self.view.count():
            self.view.setCurrentRow(0)
            self.selectionClicked() #call the function to display the data
        #close the dialog
        self.dialogChoose.close()
        return
        
    def dlg_pb2_clicked(self):
        """User wants to create a new project"""
        print("New Project")
        #What is the name of the new project?
        name = str(self.dialogChoose.lineEditWidget.text())
        #try to create project directory
        project_path = os.path.join(os.curdir, 'data', name)
        if not os.path.exists(project_path):
            os.mkdir(project_path)
            self.currentProject = name
            item = QtGui.QListWidgetItem(name, self.dialogChoose.projectListWidget)
            self.dialogChoose.projectListWidget.setCurrentItem(item)
            self.dlg_pb1_clicked()
            #now call dlg_pb1_clicked
        else:
            #project folder exists, what to do?
            pass
        return
