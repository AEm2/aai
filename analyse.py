import numpy as np
from scipy import signal
from myClasses import Lane, Peak,  Bin

def getLanesCwt(sumCols):
    #do cwt to smooth data, find minima, these are the limits of the lanes
    import scipy
    wave = scipy.signal.cwt(sumCols,scipy.signal.ricker,[22, 24])
    minima2 = scipy.signal.argrelmin(wave[0], order=30, mode='clip')
    lanes = []
    minima2= minima2[0]
    for i in range(len(minima2)-1):
        lanes.append([minima2[i],minima2[i+1]])
    return lanes
    
def getBackground(img):
    #what is the background?, first estimate, get strip outside of lanes
    #what about just the 20 pixels around the border of the image?
    bg = []
    bg.append(np.mean(img[:, 0:20]))
    bg.append(np.mean(img[0:20, :]))
    bg.append(np.mean(img[:, img.shape[1]-20:img.shape[1]]))
    bg.append(np.mean(img[img.shape[0]-20:img.shape[0], :]))
    #just the top of the image for now
    return float(bg[1])
    
def getSumRows(img, lanes):
    number_lanes = len(lanes) #lanes is a list
    sumRows= np.zeros((img.shape[0],number_lanes)) #rows, cols
    for i in range(number_lanes):
        sumRows[:,i] = np.sum(img[:,lanes[i][0]:lanes[i][1]],axis=1)
    np.save('output_sumRows.npy', sumRows)
    return sumRows
 
def analyse(sumRows,  threshold_proportion = 15.):
    ##start of integrate
    ##loop through each lane to determine its characteristics
    lanes = []
    smoothies = []
    #print sumRows.shape[1]
    for ln in range(sumRows.shape[1]):
        array = sumRows[:,ln]
        window = signal.gaussian(10,2)
        window = window/np.sum(window)
        smoothed = signal.convolve(array,window)
        smoothies.append(smoothed)
        #perform a continous wave transform
        widths = np.arange(20,30,2)
        waves = signal.cwt(array,signal.ricker,widths)
        waveno = 2
        #find relative extrema of the waves
        threshold_proportion = 25
        threshold = np.max(array)/threshold_proportion
        print ln, threshold
        maxima = signal.argrelmax(waves[waveno],axis=0,order=5)
        maxima = maxima[0]
        peak_max = []
        for maxi in maxima:
            if smoothed[maxi] > threshold: #see if peak is above a certain value
                peak_max.append(maxi)

        minima = signal.argrelmin(waves[waveno],axis=0,order=3)
        minima = minima[0]
        minima_list = []
        minima_list.append(0)
        for mini in minima:
            minima_list.append(mini)
        minima_list.append(waves[waveno].shape[0])
        aLane = Lane(ln)
        
        #for all minima in minima_list, find the two bounding the peak
        parse = len(minima_list) - 1 
        peak_bounds = []
        for peak in peak_max:
            for i in range(parse):
                if peak > minima_list[i] and peak < minima_list[i+1]:
                    peak_bounds.append([peak,minima_list[i],minima_list[i+1]])
                    #create a Peak object with ydata and bounds
                    x1 = minima_list[i]
                    x2 = minima_list[i+1]
                    #integral = np.sum(smoothed[minima_list[i]:minima_list[i+1]])
                    #argmax = np.argmax(smoothed[minima_list[i]:minima_list[i+1]])
                    #integral = np.trapz(smoothed[minima_list[i]:minima_list[i+1]])
                    integral = np.sum(sumRows[x1:x2, ln])
                    max = np.max(sumRows[x1:x2, ln])
                    #argmax = np.argmax(sumRows[x1:x2, ln])+x1 #needs to be offset from start of data slice
                    argmax = peak #changed max to argmax, max is really the max now
                    aPeak = Peak(integral, [minima_list[i],minima_list[i+1]], max, argmax, ln)
                    #add peaks and their bounds to the lane object
                    aLane.peaks.append(aPeak)

        #sort the peaks in reverse order so highest MW peak is first
        #this doesn't matter so much, we will be sorting them again later.
        aLane.peaks.reverse()
        
        #add the Lane object to the myTLC object
        lanes.append(aLane)
        
    return lanes

def binPeaks(lanes,  binWidth = 50):
    #bin the peaks according to comigration along the TLC plate
    bins = [] #this will contain a list of Bin objects
    indx = 0

    #initialise with the first peak in the first lane
    bins.append(Bin(indx, lanes[0].peaks[0].argmax)) #assign it to the first bin
    lanes[0].peaks[0].bin = 0 #make sure the peak knows which bin it is in
    
    #iterate through all the peaks in all the lanes and add them to bins
    for aLane in lanes:
        for aPeak in aLane.peaks:
            if not aPeak.bin:
                #make a new bin unless an adequate bin is found
                newBin = True
                #test the mean value of all the existing bins to see if a new bin is required
                for abin in bins:
                    #does the peak we're testing lie inside the range of the current bin?
                    if (aPeak.argmax < (abin.centre + binWidth)) and \
                            (aPeak.argmax > (abin.centre - binWidth)):
                        abin.addValue(aPeak.argmax)
                        #yes! add value of current bin to the Peak.bin variable
                        aPeak.bin = abin.indx
                        newBin = False #new bin not required
                        break # the bin is found so exit the loop and test the next peak
                if newBin: #have reached the end of the bins, and no adequate bin has been found
                    indx += 1 #increment the indx, we are creating a new bin
                    bins.append(Bin(indx,  aPeak.argmax)) #need to create a new bin
                    aPeak.bin = indx #with an indx 1 greater than the last bin
        aLane.binwidth = binWidth

    #the highest bin centre should be the lowest (or highest, it's arbitrary) number.
    #but there should be an order from low to high
    
#    print('Old bins')
#    for abin in bins:
#        print abin.centre, abin.value_list, abin.indx
        
    map = {}
    map2 = {}
    bins = sorted(bins, reverse=True, key=lambda x:x.centre)
    #print('New bins')
    count = 0
    for abin in bins:
        #print abin.centre, abin.value_list, abin.indx
        map[abin.centre] = count #remap the bins according their abin.centre order
        map2[abin.indx] = count #remap peak.bin to new bin mapping
        count += 1
    #print map
    #print map2
    
    #print('Remapped bins')
    for abin in bins:
        #print abin.centre,  map[abin.centre]
        abin.indx = map[abin.centre] #will this work?
#    for abin in bins:
#        print abin.centre, abin.value_list, abin.indx
 
    #now to sort the peaks
    
    for aLane in lanes:
        for aPeak in aLane.peaks:
            #print aPeak.bin, 
            aPeak.bin = map2[aPeak.bin] #old bin to new bin
            #print aPeak.bin

    return bins
  
def getResults(sumRows,  lanes,  bins):
    cols = sumRows.shape[1]
    rows = len(bins)
        
    #number_lanes = sumRows.shape[1]
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
    
    #include bands along the same height in the image that have not been detected by peak detection
    for i in range(results.shape[0]): #each bin (band row)
        for j in range(results.shape[1]): #for every lane 
            if results[i][j] == 0:
                #use bounds_avg to find the signal
                x1 = int(bounds_avg[i][0])
                x2 = int(bounds_avg[i][1])
                a = np.sum(sumRows[x1:x2, j])
                #a = np.trapz(smoothies[j][bounds_avg[i][0]:bounds_avg[i][1]])
                results[i][j] = a
                bounds = [bounds_avg[i][0], bounds_avg[i][1]]
                #Peak(integral, bounds, max, argmax, lane)
                max = np.max(sumRows[x1:x2, j])
                argmax = np.argmax(sumRows[x1:x2, j])+x1
                #which bin does the new peak belong to?
                mid = x1+(x2-x1)/2 #middle of the range, can't use max, too unstable
                
                for abin in bins:
                    #print mid,  abin.centre+lanes[j].binwidth,  abin.centre-lanes[j].binwidth,  abin.indx
                    if mid < abin.centre+lanes[j].binwidth and\
                        mid > abin.centre-lanes[j].binwidth:
                            bin_indx = abin.indx
                            break
                    else:
                        bin_indx=None
                lanes[j].peaks.append(Peak(a, [x1, x2], max, argmax, j,  bin_indx))
                 
    results /= 1000
    
    #sort peaks in lanes
    for aLane in lanes:
        aLane.peaks = sorted(aLane.peaks, key=lambda x:x.bin) #sort the peaks by their bin
    
    return results, lanes

# the old getLanes function
#def getLanes(img,bands=16,plot=False):
#    
#    #get the dimensions of the image
#    shape = img.shape 
#    
#    sumCols = []
#    x = range(shape[1])
#    
#    #sum all the columns to find the vertical lane profile
#    for col in x:
#        sumCols.append(np.sum(img[:,col:col+1]))
#
#    #smooth the summed columns
#    #smooth() needs sumCols as a numpy array
#    sumCols = np.asarray(sumCols)
#    sc_smooth = smooth.smooth(sumCols,60)
#    sc_smooth = sc_smooth[30:]
#    x = range(len(sc_smooth))
#
#    #find the minima and maxima of the smoothed sum of columns
#    increasing = True
#    last_y = img[0][0] #set last_y to the first y value of the array
#    
#    maxima_y = []
#    maxima_x = []
#    minima_y = []
#    minima_x = []
#    
#    for i in x[50:-1]: #ignore the first 50 pixels near the edge
#        #test the next y value in the smoothed summed column array
#        if (last_y - sc_smooth[i])**2 > 16:
#            if increasing:
#                if sc_smooth[i] > sc_smooth[i+1]:
#                    maxima_y.append(sc_smooth[i])
#                    maxima_x.append(i)
#                    increasing = False
#                    last_y = sc_smooth[i]
#            else:
#                if sc_smooth[i] < sc_smooth[i+1]:
#                    minima_y.append(sc_smooth[i])
#                    minima_x.append(i)
#                    increasing = True
#                    last_y = sc_smooth[i]
#        if len(minima_y) == bands: #
#            break
#            
#    count = range(len(minima_x))
#    diffs = []
#    for i in count[:-1]:
#        diffs.append(minima_x[i+1] - minima_x[i])
#    diffs = np.asarray(diffs)    
#    halfLaneWidth = 50
#    
#    #make a list of tuples of lane boundaries
#    lanes = []
#    for i in maxima_x:
#        lanes.append((i-halfLaneWidth, i+halfLaneWidth))
#    return lanes
    
