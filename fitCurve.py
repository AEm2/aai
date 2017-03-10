# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:02:19 2016

@author: Andy1
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
#import codecs
#import smooth
from scipy.optimize import curve_fit
#from scipy import stats
#from curves import *
from numpyJSON import json_numpy_obj_hook
## http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
## http://stackoverflow.com/users/3768982/tlausch

#import cv2
#import readTif
#import scipy

#curves for kobs

def conc_A(t, k1):
    return np.exp(-k1*t)
    
def conc_B(t,k1,k2):
    A = (k1/(k2-k1))
    C = np.exp(-k1*t)
    D = np.exp(-k2*t) #overflow encountered here...
    B = (C-D)
    return A*B
    
def conc_C(t, k1, k2):
#    A = 1/(k1-k2)
#    C = k2*np.exp(-k1*t)
#    D = k1*np.exp(-k2*t)
#    B = (1-A*(C-D))
#    return B
    #return 1 - (1/(k2-k1))*(k2*np.exp(-k1*t)-k1*np.exp(-k2*t))
    return 1 - ((1/(k2-k1))*(k2*np.exp(-k1*t)-k1*np.exp(-k2*t)))
    
    
def sumOne (X, a, b, c):
    x,y,z = X
    return a*x + b*y + c*z

    
def printR2 (func, coeff, x, y):
    #print (x, coeff)
    #yf = func(x,*coeff)
    #print yf
    #print yf**2 
    #print (func(x, *coeff))**2
    #print yf - yf**2
    #print y, yf
    #print y - yf
    #print np.mean(y - yf), np.mean(y - yf)**2
    #print "Mean Squared Error: {0:.5f}".format(np.mean(yf-func(x, *coeff)**2))
    ss_res = np.dot((y - func(x, *coeff)),(y - func(x, *coeff)))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    #print ss_res, ss_tot
    #print "Mean R :",  1-ss_res/ss_tot
    #R2 = (1-ss_res/ss_tot)**2 #this might be wrong?
    R2 = (1-ss_res/ss_tot)**2
    print "R-squared: {0:.3f}".format(R2)
    return R2

def csv(l,spacer = ', '):
    if len(l) > 1:
        a = ''
        a += str(l[0])
        for i in l[1:]:
            a += spacer
            a += str(i)
        return a
    return str(l)
   
def fit(filename):
    ##vars
    #filename = "20160812.coPARN-A383V(p1).Mg.Kobs.tif"
    file_noext = filename[:-4]
    outFolder = os.path.join(os.curdir,file_noext)
    
    ##import the numpy array
    f = open(os.path.join(outFolder,file_noext+'-array.json'),'rb')
    results = json.loads(f.read(), object_hook=json_numpy_obj_hook)
    

    
    ##fit first bin
    ##x values are time for Mg
    

    if 'Mg' in filename:
        x = np.asarray([1,3,5,10,20,30,60,90,150,210],dtype=np.float)
    elif 'Mn' in filename:
        x = np.asarray([0.5,1,2,3,4,5,10,15,20,30],dtype=np.float)
    else:
        return
    x *= 60
    
    #create the axes
    fig, axes = plt.subplots(2,1)
    #resize to A4 paper
    fig.set_size_inches(8.27,11.69)
    #big title
    fig.suptitle('{0}'.format(file_noext), fontsize=14, fontweight='bold')
    
    #plot the TLC image
    ax = axes[0]
    import scipy.misc
    
    #print a.shape
    extent = [0,1,0,1]
    if 0:
        file_path = os.path.join(outFolder,"{0}.jpg".format(file_noext))
        a = scipy.misc.imread(file_path)
        ax.imshow(a, cmap='gray',interpolation='nearest', extent = extent, aspect=0.8)
    else:
        #for the inverse image
        file_path = os.path.join(outFolder,"{0}2.jpg".format(file_noext))
        a = scipy.misc.imread(file_path) 
        ax.imshow(a, cmap='Greys',interpolation='nearest', extent = extent, aspect=0.8) 
                   

    #turn off the ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #make the axis spines thinner
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.2)
    #maybe change the bounding box
    box = ax.get_position()

    #plot the graph
    ax = axes[1]
    
    results.flags['WRITEABLE'] = True
    
    #the sum across for each time point, this should equal one.
    print np.sum(results, axis=0) 
    print np.sum(results, axis=1) 
    
    #normalise the results
    sumCols = np.sum(results, axis=0)
    results = results/sumCols
    
    A = results[0]
    ax.plot(x, A, 'b.')

    B = results[1]
    ax.plot(x, B, 'r.')

    C = results[2]
    ax.plot(x, C, 'g.')
    
    X = np.linspace(1,x[-1:],500)

    print scipy.__version__

    popt_A, pcov_A = curve_fit(conc_A, x, A, p0=(0.001))
    k1 = popt_A[0]
    print k1
    popt_B, pcov_B = curve_fit(conc_B, x, B, p0=(k1,1))
    k1 = (popt_A[0] + popt_B[0]) / 2
    k2 = popt_B[1]
    print k1, k2
    popt_C, pcov_C = curve_fit(conc_C, x, C, p0=(k1,k2))
    
    ##plot the fitted curves
    X = np.linspace(0,x[-1:],x[-1:])
    #manually enter the k1 and k2 parameters
    if 0:
        popt_A = [0.015]
        popt_B = [0.015,0.05]
        Y_A = conc_A(X,*popt_A)
        ax.plot(X, Y_A,'b', label='AAA')
        Y_B = conc_B(X,*popt_B)
        ax.plot(X, Y_B,'r', label='AA')
    else:
        Y_A = conc_A(X,*popt_A)
        ax.plot(X, Y_A,'b', label='AAA')
        Y_B = conc_B(X,*popt_B)
        ax.plot(X, Y_B,'r', label='AA')
        ax.plot(X,conc_C(X,*popt_B),'m')
        Y_C = conc_C(X,*popt_C)
        ax.plot(X, Y_C,'g', label='A')

    #what are the fitted parameters for k1 and k2?        
    if 1:
        print popt_A 
        print popt_B 
        print popt_C
        
    #log of linear scale on x-axis
    if 0:
        ax.xscale('log',basex=2.718)
        #ax.xlim(1,210*60)
    else:
        #ax.xlim(0,210*60)
        pass
    

    #Micke wants the legend outside
    box = ax.get_position()
    #print box
    #shrink the plot
    ax.set_position([box.x0, box.y0 + 0.1, box.width * 0.8, box.height-0.1])
    #put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #write out the k1 and k2 parameters on the plot
    if 1:
        vspace = 0.15
        
        #plot the values from the AAA fit
        start1 = -0.4
        #ax.text(10000,0.5,'$k_1$ {0:.3f}'.format(k1)+' s$^{-1}$')
        x1 = 0
        ax.text(x1,start1,'$k_1$ {0:.8f}'.format(popt_A[0])+' s$^{-1}$')
        ax.text(x1,start1-vspace,'$R^2$ {0:.3f}'.format(printR2(conc_A, popt_A, x, A)))
        
        #plot the values from the AA fit
        if 'Mg' in filename:
            x2 = 5000
        else:
            x2 = 600
        start2 = -0.34
        ax.text(x2,start2,'$k_1$ {0:.8f}'.format(popt_B[0])+' s$^{-1}$')
        ax.text(x2,start2-vspace,'$k_2$ {0:.8f}'.format(popt_B[1])+' s$^{-1}$')
        ax.text(x2,start2-vspace*2,'$R^2$ {0:.3f}'.format(printR2(conc_B, popt_B, x, B)))
        
        #plot the values from the A fit
        if 'Mg' in filename:
            x2 = 11000
        else:
            x2 = 1300
        start2 = -0.34
        ax.text(x2,start2,'$k_1$ {0:.8f}'.format(popt_C[0])+' s$^{-1}$')
        ax.text(x2,start2-vspace,'$k_2$ {0:.8f}'.format(popt_C[1])+' s$^{-1}$')
        ax.text(x2,start2-vspace*2,'$R^2$ {0:.3f}'.format(printR2(conc_C, popt_C, x, C)))
    
    #set the x- and y-axis limits, label, etc.
    ax.set_xlim(0,x[-1:])
    ax.set_xlabel('Time (sec)')
    plt.ylim(0,1)
    plt.ylabel('Normalised Intensity')
       
    #save to pdf and close figure
    output = os.path.join(outFolder,file_noext+'_fitCurve_kobs_07.jpg')    
    plt.savefig(output, dpi=300)
    plt.close(fig)
        
        
        
        
    #output the fitted curves
    if 0:
        fig = plt.gcf()
        fig.set_figwidth(8)
        fig.set_figheight(5)
        plt.tight_layout()   
    
    #output raw parameters for curves A and B
    
    if 0:
        print 'popt_A: {0}'.format(popt_A)
        print 'popt_C: {0}'.format(csv(popt_C))
        #save this data to a file to use as p0s for subsequent curve fits
        
    
    if 0 :
        ##output the parameters for librecalc import
        print '\nk1s and 2s For LibreCalc Import:'
        print csv([popt_A[0], np.sqrt(np.diag(pcov_A))[0],\
            popt_B[0], np.sqrt(np.diag(pcov_B))[0],
            popt_B[1], np.sqrt(np.diag(pcov_B))[1]],'\t ')
            
        ##output results[1] for graphpad prism
        print '\nFor GraphPad Prism:'
        print 0.
        for i in results[1]:
            print i*10
    
    #print the input array data
    if 0:
        for result in results:
            print result
            
  
    plt.close()
        
    
    
if __name__ == '__main__':
    
    if 0:
    #replot all tif files in the directory    
        files = os.listdir(os.curdir)
        tifs = []
        for afile in files:
            if '.tif' in afile:
                tifs.append(afile)
        for atif in tifs:
            fit(atif)
    else:
        #plot just one tif
        #fit("20160628-coPARN.S87A(p1).Mg.tif")
        #fit('20160705-coPARN-N470A(p1)Mg2.tif')
        #fit('20160728-coPARN-A383V(p2)-Mg.fit')
        fit('20161025.A1.Y91c(p2)-Mg.tif')
