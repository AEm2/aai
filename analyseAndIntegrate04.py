## -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:40:45 2016

@author: Andy1
"""

#version 0.0.9


##rev 04 -  save to json list of lane results, rather than separate files for
##          each lane

#import matplotlib
#matplotlib.use('Agg') #use a different backend

import matplotlib.pyplot as plt
import numpy as np
import readTif02 as readTif
import codecs, json
import os
import smooth

from numpyJSON import NumpyEncoder
from scipy import stats, signal
from scipy.signal import argrelmax, argrelmin
from numpyJSON import json_numpy_obj_hook



def getLanes(img,bands=16,plot=False):   
    
    #get the dimensions of the image
    shape = img.shape 
    
    sumCols = []
    x = range(shape[1])
    
    #sum all the columns to find the vertical lane profile
    for col in x:
        sumCols.append(np.sum(img[:,col:col+1]))
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, sumCols)

    #smooth the summed columns
    #smooth() needs sumCols as a numpy array
    sumCols = np.asarray(sumCols)
    sc_smooth = smooth.smooth(sumCols,60)
    sc_smooth = sc_smooth[30:]
    x = range(len(sc_smooth))
    
    if plot:
        ax.plot(x,sc_smooth)

    #find the minima and maxima of the smoothed sum of columns
    increasing = True
    last_y = img[0][0] #set last_y to the first y value of the array
    
    maxima_y = []
    maxima_x = []
    minima_y = []
    minima_x = []
    
    for i in x[50:-1]: #ignore the first 50 pixels near the edge
        #test the next y value in the smoothed summed column array
        if (last_y - sc_smooth[i])**2 > 16:
            if increasing:
                if sc_smooth[i] > sc_smooth[i+1]:
                    maxima_y.append(sc_smooth[i])
                    maxima_x.append(i)
                    increasing = False
                    last_y = sc_smooth[i]
            else:
                if sc_smooth[i] < sc_smooth[i+1]:
                    minima_y.append(sc_smooth[i])
                    minima_x.append(i)
                    increasing = True
                    last_y = sc_smooth[i]
        if len(minima_y) == bands: #
            break
        
    if plot:
        ax.plot(maxima_x,maxima_y,'ro')
        ax.plot(minima_x,minima_y,'go')

    #TODO: make halfLaneWidth dependent on maxima and minima
    #now the bands should be 90 microns wide, the scan was 100 um, so each pixel is 1 um?
    #get the length between minima

    count = range(len(minima_x))
    diffs = []
    for i in count[:-1]:
        diffs.append(minima_x[i+1] - minima_x[i])
    diffs = np.asarray(diffs)    
    halfLaneWidth = 50
    
    #make a list of tuples of lane boundaries
    lanes = []
    for i in maxima_x:
        lanes.append((i-halfLaneWidth, i+halfLaneWidth))
        
    #this script should give max, min, max, min etc.
    #give the lane boundaries by the minima. 
    #the first and last boundaries need to be determined with their maxima as well
    widths = np.arange(20,30,2)
    waves = signal.cwt(sc_smooth,signal.ricker,widths)
    plt.plot(waves[2])
    minima2 = argrelmin(waves[2], order=30, mode='clip')
    for m in minima2:
        plt.plot(m, waves[2][m],'co')
    lanes = []
    minima2= minima2[0]
    #print minima2
    #print len(minima2)
    for i in range(len(minima2)-1):
        lanes.append([minima2[i],minima2[i+1]])
    #print len(lanes)
        
#    for wave in waves:        
#        plt.plot(x, wave)  
        
    return lanes

class Peak(object):
    def __init__(self,ydata,bounds,maxima,lane):
        self.ydata = ydata
        self.bounds = bounds
        self.lane = lane
        self.maxima = maxima
        self.integral = np.trapz(self.ydata)
        self.bin = None
        return
        
    def __str__(self):
        string = '\nPeak in lane {0}\n'.format(self.lane)
        string += 'Maxima:\t{0}\n'.format(self.maxima)
        string += 'Integ.: \t{0}\n'.format(self.integral)
        string += 'Lane: \t{0}'.format(self.lane)
        return string
        
class Bin (object):
    def __init__(self,value):
        self.value_list = [value]
        self.peakList = None
        self.centre = value  
        self.index = None
        return
    
    def addValue(self, value):
        self.value_list.append(value)
        self.centre = np.average(self.value_list)

class Lane(object):
    def __init__(self,lane_number):
        self.lane_number = lane_number
        self.peaks = []
        
        
class TLC(object):
    def __init__(self, data):
        self.lanes = [] #a list of all Lane objects
        self.raw_data = data #a list of all the raw data coming from analyseTLC
        self.background = self.set_background() #find the global background
        self.number_lanes = len(self.raw_data)
    
    def set_background(self):
        ##concatenate all the data in the lists
        print stats.mode(self.raw_data)
        print stats.mode(self.raw_data, axis=None)
        self.background = stats.mode(self.raw_data)
        return
#        import itertools
#        concat = list(itertools.chain.from_iterable(self.raw_data))
#        ##make the array integers
#        concat = np.asarray(concat,dtype=np.int16)
#        ##find the mode of the flattened array
#        mode = stats.mode(concat)
#        return mode[0][0]

def analyse(filename, number_lanes):
    
    #plot = 1
    
    #save everything to the output folder
    file_noext = filename[:-4]
    #ext = filename[-4:]
    outFolder = os.path.join(os.curdir,file_noext)
    print('Output directory {0}'.format(outFolder))
    import site
    print(site.getsitepackages())
    import scipy
    print(str(scipy.__version__))
    #create the output folder if it doesn't exist
    if not os.path.exists(outFolder):
        print('Creating output directory {0}'.format(outFolder))
        
        os.mkdir(outFolder)
        
#    #read the tif file
#    if 'tif' in ext:
#        try:
#            image = readTif.tifFile(filename)
#            img = image.image
#            print img.shape, np.max(img), np.min(img), img.dtype
#            height, width = img.shape
#        except Exception as e:
#            print e
#            #maybe it's LZW tiff
#            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
#            print img.shape, np.max(img), np.min(img), img.dtype
#            if img.dtype != np.uint16:
#                raise AssertionError('Image needs to be read as 16-bit')

    #read the tif using readTif
    image = readTif.tifFile(filename)
    img = image.image
    print img.shape, np.max(img), np.min(img), img.dtype
    height, width = img.shape   
    
    #save image as jpg into subfolder
    import scipy.misc
    file_path = os.path.join(outFolder,"{0}2.jpg".format(file_noext)) #dark background
    img_in = np.asarray(img, dtype = np.uint16)
    a = np.max(img_in) - img_in #invert the image
    scipy.misc.imsave(file_path,img_in)
    file_path = os.path.join(outFolder,"{0}.jpg".format(file_noext))#light background
    scipy.misc.imsave(file_path, a)
    
    

    
#    #find the lane boundaries
#    x = np.linspace(0,img.shape[1],img.shape[1])
#    sumCols = np.sum(img,axis=0)
#    
#    if plot:
#        fig, ax = plt.subplots()
#        ax.plot(x, sumCols)
#
#    sc_smooth = smooth.smooth(sumCols,60) #smooth the summed columns
#    sc_smooth = sc_smooth[30:] #justify the smoothed plot
#    x = np.linspace(0, sc_smooth.shape[0], sc_smooth.shape[0])
#    
#    if plot:
#        ax.plot(x,sc_smooth)
#
#    from scipy.signal import argrelmax, argrelmin
#    maxima2 = argrelmax(sc_smooth, order=30, mode='clip')[0] #other mode is 'wrap'
#    minima2 = argrelmin(sc_smooth, order=30, mode='clip')
#    
#    #just take the top (lane_number) values
#    print maxima2
#    print maxima2.shape[0]
#    print np.argmin(maxima2)
#    print sc_smooth[maxima2]
#    lowest = np.argmin(sc_smooth[maxima2])
#    print np.argmin(sc_smooth[maxima2])
#    print maxima2[np.argmin(sc_smooth[maxima2])]
#    maxima2 = np.delete(maxima2,lowest)
#    print maxima2
#    plt.plot(maxima2,sc_smooth[maxima2], 'ro')
#    
#    diff = maxima2[1:] - maxima2[:-1]
#    print diff
#    print np.mean(diff)
#    print np.std(diff)
#    res = np.mean(diff) - diff
#    print res, np.power(res,2)
#    start = maxima2[0]
#    width = np.mean(diff)
#    peaks = np.linspace(0,number_lanes-1, number_lanes)
#    print peaks
#    print width
#    peaks *= width
#    print peaks
#    peaks += start
#    print peaks
#    for i in np.nditer(peaks):
#        plt.axvline(i)
#    
##    residuals = np.zeros((11,11))
##    width_pm = np.linspace(-5,5,11) #width along i
##    start_pm = np.linspace(-5,5,11) #start along j
##    print range(0,10)
##    print width_pm
##    print start_pm
##    for i in range(0,11):
##        for j in range(0,11):
##            peaks = np.array(range(number_lanes))
##            #print peaks
##            peaks *= (width + width_pm[i])
##            peaks += (start + start_pm[j])
##            res = maxima2 - peaks
##            res2 = np.power(res,2)
##            residuals[i][j] = np.sum(res2)
##    print residuals
##    print np.min(residuals), np.argmin(residuals)
##    print np.argmin(residuals)/11, np.argmin(residuals)%11
##    row = np.argmin(residuals)/11
##    col = np.argmin(residuals)%11
##    print residuals[row][col]
##    print width_pm[row], start_pm[col]
##    
##    peaks = np.array(range(number_lanes))
##    peaks *= (width + width_pm[row])
##    peaks += (start + start_pm[col])
##    for i in np.nditer(peaks):
##        plt.axvline(i,color='r')  
#    
#    halfwidth = (width)/2
#    lanes = []    
#    for p in peaks:
#        lanes.append([int(p-halfwidth), int(p+halfwidth)])
#    print lanes

    #get the lanes the old way, this works much better.

    lanes = getLanes(img, bands=number_lanes, plot=True)
    
    for l in lanes:
        plt.axvline(l[0])
        plt.axvline(l[1])
    plt.savefig(os.path.join(outFolder,file_noext + '_lanes_cols.jpg'),dpi=300)
    plt.clf()
    
    sumRows= np.zeros((img.shape[0],number_lanes)) #rows, cols
    
    for i in range(number_lanes):
        sumRows[:,i] = np.sum(img[:,lanes[i][0]:lanes[i][1]],axis=1)
        
    lane_bounds = lanes
    
    extent = [0,1,0,1]
    plt.imshow(sumRows, cmap = 'jet', interpolation='nearest', extent = extent, aspect = 1)
    plt.savefig(os.path.join(outFolder,file_noext + '_lanes_rows.jpg'),dpi=300)
    plt.clf()
    
    
    
    
    
    
    ##start of integrate

    file_noext = filename[:-4]
    outFolder = os.path.join(os.curdir,file_noext) 
    
    #verbose = False
    verbose = True
    
    ##load json from analyseTLC here
    #f = open(os.path.join(outFolder,file_noext+'_sumRows.json'),'rb')
    #sumRows = json.loads(f.read(), object_hook=json_numpy_obj_hook) #should get a sumRowsArray
    #plt.imshow(sumRows, cmap = 'jet', interpolation='nearest', extent = [0,1,0,1], aspect = 1)   
    
    #need a better way of determining background
    background_est = stats.mode(sumRows, axis=None)[0]
    background = 0
    print("Background = {0}".format(background))
    ##loop through each lane to determine its characteristics
    lanes = []
    smoothies = []
    for ln in range(sumRows.shape[1]):
        array = sumRows[:,ln]
        
        ##create plots for the upcoming analysis
        f, ax = plt.subplots(3,1,sharex=True)
        x = np.linspace(0,array.shape[0],array.shape[0])
        plt.sca(ax[0])
        ##plot the raw data
        plt.plot(x,array,'b-')
        plt.xlim(0,array.shape[0])
        
        ##convolve data with a smoothing window
        #window = signal.hann(10)
        window = signal.gaussian(10,2)
        window = window/np.sum(window)
        smoothed = signal.convolve(array,window)
        smoothed -= background #remove background from smoothed
        smoothies.append(smoothed)
        plt.plot(np.arange(0,smoothed.shape[0]),smoothed)
        ##perform a continous wave transform
        
        #widths = np.arange(1,100,1)
        widths = np.arange(20,30,2)
        waves = signal.cwt(array,signal.ricker,widths)
        plt.sca(ax[1])
        plt.imshow(waves)
        
        plt.sca(ax[2])
        
        waveno = 2
        #plt.plot(x, waves[50])
        #plt.plot(x, waves[30]) #TODO: waves 30 is an arbitrary choice, fix this
        for wave in waves:        
            plt.plot(x, wave)
        
        ##find relative extrema of the waves
        threshold = 40 #TODO: threshold 40 is an arbitrary choice, fix.
        threshold = np.max(array)/20 #threshold is one hundreth of the lane max
        #print "Threhsold = {0}".format(threshold)
        
        maxima = signal.argrelmax(waves[waveno],axis=0,order=5)
        maxima = maxima[0]
        peak_max = []
        for maxi in maxima:
            if smoothed[maxi] > threshold: #see if peak is above a certain value
                #print "Peak height: {0}".format(smoothed[maxi])
                plt.axvline(maxi,color='r')
                plt.text(maxi, waves[waveno][maxi], '{0}'.format(int(smoothed[maxi])),fontsize=8)
                peak_max.append(maxi)
            
        minima = signal.argrelmin(waves[waveno],axis=0,order=3)
        #print minima, minima[0]
        minima = minima[0]
        minima_list = []
        minima_list.append(0)
        for mini in minima:
            #print mini
            plt.axvline(mini)
            plt.text(mini, waves[waveno][mini], '{0}'.format(int(smoothed[mini])),fontsize=6)
            minima_list.append(mini)
        minima_list.append(len(smoothed))
        
        aLane = Lane(ln)
        
        ##for all minima in minima_list, find the two closest to the peak
        #minima_array = np.asarray(minima_list)
        #print minima_list
        parse = len(minima_list) - 1 
        peak_bounds = []
        for peak in peak_max:
            for i in range(parse):
                if peak > minima_list[i] and peak < minima_list[i+1]:
                    peak_bounds.append([peak,minima_list[i],minima_list[i+1]])
                    ##create a Peak object with ydata and bounds
                    aPeak = Peak(smoothed[minima_list[i]:minima_list[i+1]],\
                        (minima_list[i],minima_list[i+1]), peak, ln)
                    ##add peaks and their bounds to the lane object
                    aLane.peaks.append(aPeak)
        color = 0.1
        for item in peak_bounds:
            plt.axvspan(item[1],item[2],facecolor="0.1",alpha=0.2)
            color += 0.4

        ##output the results of peak finding to output folder
        output = os.path.join(os.curdir,outFolder,file_noext+'_intPeaks01_cwt_{0}.jpg'.format(ln))
        plt.savefig(output,dpi=300)
        plt.close()
        
        ##sort the peaks in reverse order so highest MW peak is first
        aLane.peaks.reverse()
        
#        for item in aLane.peaks:
#            print item.maxima
        
        ##add the Lane object to the myTLC object
        lanes.append(aLane)

    #bin the peaks
    bins = [] #this will contain a list of all the binned peaks
    
    #initialise with the first peak in the first lane
    #print lanes[0].peaks[0] ##the first peak in the first lane
    bins.append(Bin(lanes[0].peaks[0].maxima)) ##assign it to the first bin
    lanes[0].peaks[0].bin = 0 #make sure the peak knows which bin it is in
    binWidth = 50
    
    ##iterate through all the peaks in all the lanes and add them to bins
    for aLane in lanes:
        for aPeak in aLane.peaks:
            if not aPeak.bin:
                ##make a new bin unless an adequate bin is found
                newBin = True
                ##test the mean value of all the existing bins to see if a new bin is required
                for abin in bins:
                    ##does the peak we're testing lie inside the range of the current bin?
                    if (aPeak.maxima < (abin.centre + binWidth)) and \
                            (aPeak.maxima > (abin.centre - binWidth)):
                        abin.addValue(aPeak.maxima)
                        #yes! add value of current bin to the Peak.bin variable
                        aPeak.bin = bins.index(abin)
                        newBin = False #new bin not required
                        break # the bin is found so exit the loop and test the next peak
                if newBin: #have reached the end of the bins, and no adequate bin has been found
                    bins.append(Bin(aPeak.maxima)) #need to create a new bin
                    aPeak.bin = bins.index(abin)+1 #with an index 1 greater than the last bin
            
    if verbose:
        print("\nCurrent binning status:")
        for abin in bins:
            print abin.centre, abin.value_list, bins.index(abin)
    
    ## create a numpy array the size of:
    ## columns are the number lanes
    ## rows are the number of bins
    cols = sumRows.shape[1]
    rows = len(bins)
    #print cols, rows
    results = np.zeros((rows, cols),dtype=np.float) #create an empty array
    for aLane in lanes:
        for aPeak in aLane.peaks:
            results[aPeak.bin][aPeak.lane] = aPeak.integral
    
    #get the lane bounds
    bounds = np.full((rows,cols,2),np.NaN) #three dimensional array
    for aLane in lanes:
        for aPeak in aLane.peaks:
            bounds[aPeak.bin][aPeak.lane][0] = aPeak.bounds[0]
            bounds[aPeak.bin][aPeak.lane][1] = aPeak.bounds[1]
    #get the average bounds for each row of bands
    bounds_avg = np.nanmean(bounds,axis=1)

    #finally plot the image with the lane and band boundaries
    plt.clf() #empty the plot
    plt.imshow(img) #show the image
    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0],0)
    
    
    #mark out the average bounds
    for i in range(bounds_avg.shape[0]):
        plt.axhline(bounds_avg[i][0])
        plt.axhline(bounds_avg[i][1])
    #mark out the lanes
    for i in range(number_lanes):
        plt.axvline(lane_bounds[i][0])
        plt.axvline(lane_bounds[i][1]) 
    #plot the band boundaries for detected bands
    for i in range(number_lanes):
        bins = bounds.shape[0]
        for a in range(bins):
            if np.isnan(bounds[a][i][0]): #if there are no bounds, do nothing
                pass
            else:
                plt.hlines(bounds[a][i][0],lane_bounds[i][0],lane_bounds[i][1])
                plt.hlines(bounds[a][i][1],lane_bounds[i][0],lane_bounds[i][1])
    
    #fill the zero values of results with 'background'
    #this will be a good measure of background correction later
    print results
    print len(smoothies)
    for i in range(results.shape[0]): #each bin (band row)
        for j in range(results.shape[1]): #for every lane 
            if results[i][j] == 0:
                #use smoothed and bounds_avg to find the signal
                a = np.trapz(smoothies[j][bounds_avg[i][0]:bounds_avg[i][1]])
                #b = np.sum(smoothies[j][bounds_avg[i][0]:bounds_avg[i][1]])
                results[i][j] = a
    print results
    

    for i in range(results.shape[0]): #for every band row
        for j in range(results.shape[1]): #for every lane col
            #print i,j,bounds_avg[i][0],bounds_avg[i][1],lane_bounds[j][0],lane_bounds[j][1]
            y = np.average(bounds_avg[i])
            x = np.average(lane_bounds[j]) 
            text = '{0:2.0f}'.format(results[i][j])
            plt.text(x, y, text, color='w',  fontsize=7, horizontalalignment='center', verticalalignment='center')
    output = os.path.join(os.curdir,outFolder,file_noext+'_analyseAndIntegrate.jpg')
    plt.savefig(output,dpi=300)
    plt.show()
    #plt.close()

    
    dumped = json.dumps(results, cls=NumpyEncoder)
    f = open(os.path.join(outFolder,file_noext+'-array.json'),'wb')
    f.write(dumped)
    f.close()


if __name__ == '__main__': 
    #analyse(filename='20160812.coPARN-A383V(p1).Mn.Kobs.tif', number_lanes=10)
    analyse(filename='20161025.A1.Y91c(p2)-Mg.tif', number_lanes=10)
