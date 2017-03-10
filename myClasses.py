import numpy as np

#for analyse.py

class Peak(object):
    def __init__(self, integral, bounds, max, argmax, lane,  bin=None):
        #self.ydata = ydata
        self.bounds = bounds
        self.lane = lane
        self.max = max
        self.argmax = argmax #important for band binning
        self.integral = integral
        self.bin = bin
        self.color = None
        return
        
    def setColor(self, color):
        self.color = color
        
class Bin (object):
    def __init__(self, indx,  value):
        self.value_list = [value]
        self.peakList = None
        self.centre = value  
        self.indx = indx
        return
    
    def addValue(self, value):
        self.value_list.append(value)
        self.centre = np.average(self.value_list)

class Lane(object):
    def __init__(self,lane_number):
        self.lane_number = lane_number
        self.peaks = []
        self.bounds = None
        self.binwidth = 50
        
#for mainWindow.py

class Analysis(object):
    def __init__(self):
        #self.file_noext = None
        #self.img = None
        #self.sumCols = None
        self.lane_bnd = None #making a new 
        self.lanes = None
        self.results = None
        self.results_norm = None
        self.kobs = None
        #self.curves = None
        self.background = None
        self.timepoints = None
        return
        
    def setBackground(self,  bg):
        self.background = bg
        
    def setTimepoints(self,  tp):
        self.timepoints = tp
