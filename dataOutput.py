import matplotlib.pyplot as plt
import os
import numpy as np

def outputPdf(file_noext, img,  results,  timepoints, curves,  kobs,  r2):

    #TODO: get the output folder using dialog
    outFolder = os.path.join(os.curdir, 'pdf')

    #create the axes
    fig, axes = plt.subplots(2,1)
    #resize to A4 paper
    fig.set_size_inches(8.27,11.69)
    #big title
    fig.suptitle('{0}'.format(file_noext), fontsize=14, fontweight='bold')
    
    #plot the TLC image
    ax = axes[0]
    extent = [0,1,0,1]
    #ax.imshow(img, cmap='gray',interpolation='nearest', extent = extent, aspect=0.8)
    ax.imshow(img, cmap='Greys',interpolation='nearest', extent = extent, aspect=0.8) 
    ax.get_xaxis().set_visible(False)#turn off the ticks
    ax.get_yaxis().set_visible(False)#turn off the ticks
    for axis in ['top','bottom','left','right']:#make the axis spines thinner
        ax.spines[axis].set_linewidth(0.2)
    box = ax.get_position()#maybe change the bounding box

    #plot the data points and curves on axes[1]
    ax = axes[1]
    sumCols = np.sum(results, axis=0)
    results = results/sumCols
    
    x = curves[0]
    A = curves[1]
    ax.plot(x, A, 'b',  label = 'XXX')
    ax.plot(timepoints, results[0], 'b.')
    if curves.shape[0] > 2:
        B = curves[2]
        ax.plot(x, B, 'r',  label = 'XX')
        ax.plot(timepoints, results[1], 'r.')
    if curves.shape[0] > 3:
        C = curves[3]
        ax.plot(x, C, 'g', label = 'X')
        ax.plot(timepoints, results[2], 'g.')

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
        
        #plot the values from the A fit
        start1 = -0.4
        #ax.text(10000,0.5,'$k_1$ {0:.3f}'.format(k1)+' s$^{-1}$')
        x1 = 0
        ax.text(x1,start1,'$k_1$ {0:.8f}'.format(kobs[0][0])+' s$^{-1}$')
        ax.text(x1,start1-vspace,'$R^2$ {0:.3f}'.format(r2[0]))
        
        if curves.shape[0] > 2:
            #plot the values from the B fit
            if 'Mg' in file_noext:
                x2 = 5000
            else:
                x2 = 600
            x2 = timepoints[-1]/2
            start2 = -0.34
            ax.text(x2,start2,'$k_1$ {0:.8f}'.format(kobs[1][0])+' s$^{-1}$')
            ax.text(x2,start2-vspace,'$k_2$ {0:.8f}'.format(kobs[1][1])+' s$^{-1}$')
            ax.text(x2,start2-vspace*2,'$R^2$ {0:.3f}'.format(r2[1]))
        
        if curves.shape[0] > 3:
            #plot the values from the C fit
            if 'Mg' in file_noext:
                x2 = 11000
            else:
                x2 = 1300
            #what about getting the last time point?
            x2 = timepoints[-1]
            start2 = -0.34
            ax.text(x2,start2,'$k_1$ {0:.8f}'.format(kobs[2][0])+' s$^{-1}$')
            ax.text(x2,start2-vspace,'$k_2$ {0:.8f}'.format(kobs[2][1])+' s$^{-1}$')
            ax.text(x2,start2-vspace*2,'$R^2$ {0:.3f}'.format(r2[2]))
    
    #set the x- and y-axis limits, label, etc.
    ax.set_xlim(0,x[-1:])
    ax.set_xlabel('Time (sec)')
    plt.ylim(0,1)
    plt.ylabel('Normalised Intensity')
       
    #save to pdf and close figure
    output = os.path.join(outFolder,file_noext+'_results_01.pdf')    
    plt.savefig(output, dpi=300)
    plt.close(fig)
  
    plt.close()
