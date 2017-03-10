import numpy as np
import scipy

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
    
def rsq (func, coeff, x, y):
    ss_res = np.dot((y - func(x, *coeff)),(y - func(x, *coeff)))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    R2 = (1-ss_res/ss_tot)**2
    return R2

def fitCurve(results, timepoints):
    """do a curve fit of the results, Mg time points as default"""
    #will return a list of popts
    kobs = []
    kobs_arrays = []
    r2 = []
            
    #make the minutes of the timepoints into seconds (required for k1 and k2 determination
    timepoints *= 60
    results.flags['WRITEABLE'] = True
    
    #normalise the results to the sum of each lane
    sumCols = np.sum(results, axis=0)
    results = results/sumCols

    A = results[0]
    if results.shape[0] < 2:
        B= None
    else:
        B = results[1]
    #enzyme might be really slow
    if results.shape[0] < 3:
        C = None
    else:
        C = results[2]
    #print scipy.__version__
    guess=0.0001
    k1=0.0001
    count = 0
    #TODO: try own curve fitting using http://scipy-cookbook.readthedocs.io/items/robust_regression.html
    while k1==guess:
        guess *= 1.1
        popt_A, pcov_A = scipy.optimize.curve_fit(conc_A, timepoints, A, p0=guess,  maxfev=5000)
        k1 = popt_A[0]
        count += 1
        #print count, k1, guess
        if count > 100:
            k1 = 0.001
            popt_A, pcov_A = scipy.optimize.curve_fit(conc_A, timepoints, A, p0=0.001,  maxfev=10000)
            break
    #print popt_A
    kobs.append(popt_A.tolist())
    kobs_arrays.append(popt_A)
    r2.append(rsq(conc_A, popt_A, timepoints, A))

    if B is not None:
        popt_B, pcov_B = scipy.optimize.curve_fit(conc_B, timepoints, B, p0=(k1,1))
        #print popt_B
        kobs.append(popt_B.tolist())
        kobs_arrays.append(popt_B)
        r2.append(rsq(conc_B, popt_B, timepoints, B))
        k1 = popt_B[0]
        k2 = popt_B[1]
        
    if C is not None:
        popt_C, pcov_C = scipy.optimize.curve_fit(conc_C, timepoints, C, p0=(k1,k2))
        #print popt_C
        kobs.append(popt_C.tolist())
        kobs_arrays.append(popt_C)
        r2.append(rsq(conc_C, popt_C, timepoints, C))
        
    X = np.linspace(1,timepoints[-1:],500)
    curves = np.zeros((results.shape[0]+1, 500)) #array with first row timepoints, the rest the fitted curves
    curves[0, :] = X #first row, the timepoints
    curves[1, :] = conc_A(X,*popt_A)
    if curves.shape[0] > 2:
        curves[2, :] = conc_B(X,*popt_B)
    if curves.shape[0] > 3:
        curves[3, :] = conc_C(X,*popt_C)
        
    return kobs_arrays, curves,  r2
    
