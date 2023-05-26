#-------------------------------------------------------------------------------
#
#  An exstension of scipy.stats.gaussian_kde 
#  to estimate the multivairate joint distrubite of N+1 variable (x1,x2,...,xN)
#    The KDE take the
#     - add a selective KDE factor b=(b1,...,b_n)
#     - compute the conditional distribution:  f(xN | (x1,...,x{n-1}))  
#
#  Written by: Hai Bui (hai.bui@uib.no)
#. This work is part of the Project ....
#  Please cite this paper .... if....
#.  
#  Date: 2022-08-09
#
#
#  Features: usage inheritted from scipy gaussian_kde, however with these additional function
#    - set_diag_cov(): use data diagonal maxtrix instead of full covariance matrix (this is what awkde use.)
#    - set_selective_factor: to set the selective band width factor, 
#    - calc_local_weights: calculate local bandwidth factor for adaptive bandwidh kde,
#                        must be called at least once before aw_evaluate
#    - aw_evaluate: similar to evaluate, but using adaptive bandwidh
#    - conditional_distribution:  compute  f(xN | (x1,...,x{n-1}))  
#
#-------------------------------------------------------------------------------
import pyximport
from scipy.stats import gaussian_kde 
import numpy as np
from scipy import linalg, special
from scipy.special import logsumexp
from scipy._lib._util import check_random_state

from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)


from backend import gaussian_aw_kernel_estimate
from sklearn.model_selection import LeaveOneOut  


class saw_gausian_kde(gaussian_kde):
    
    
    #
    # for example, b=np.array([1,0.1,1])
    #
    def set_selective_factor_old(self,h):
        if (len(h.shape)==1):
            h=h.reshape(1,-1)
        
        if (len(h.shape)!=2):
            raise ValueError("Number of dimensions of `b` should be 1 or 2")
        
        if (h.shape[1]!=self.d):
            raise ValueError(f"Error: Mistmach number of dimensions {h.shape[1]} vs. {self.d}")
            
        self.selective_factor=h  
        self.covariance=(self._data_covariance)*np.matmul(h.T,h)
        self.inv_cov = linalg.inv(self.covariance)   
        #print("KDE bandwidth matrix:\n",self.covariance)
       
    
    def set_selective_factor(self,h):
        if (len(h.shape)==1):
            h=h.reshape(1,-1)
        
        if (len(h.shape)!=2):
            raise ValueError("Number of dimensions of `b` should be 1 or 2")
        
        if (h.shape[1]!=self.d):
            raise ValueError(f"Error: Mistmach number of dimensions {h.shape[1]} vs. {self.d}")
                  
        self.selective_factor=h  
        
        L,Q = linalg.eigh(self._data_covariance)
        L1  = np.diag(L)*np.matmul(h.T,h)
        self.covariance=np.matmul(np.matmul(Q,L1),linalg.inv(Q))              
        self.inv_cov = linalg.inv(self.covariance)   
        
        #print("KDE bandwidth matrix:\n",self.covariance)
        
    
    def set_diag_cov(self):
        #print(self.covariance)
        self.covariance=np.diag(np.diag(self.covariance))
        #print(self.covariance)
        self.inv_cov = linalg.inv(self.covariance)
        
    
    #
    # 
    #
    def calc_local_weights(self,alpha=0.5):
        kde_values=self.evaluate(self.dataset)
        g = np.exp(np.sum(np.log(kde_values)) / self.n)
        self.inv_gamma = (kde_values / g)**alpha
        #print("inv_gamma:",self.inv_gamma) # checked
        
    
    #
    # Calculate the 1-D conditional distribution f(x1!(x2,x3,...,x_n))
    # x_in(n_data, nx): n_data row x nx columns, id n_data points of the prior information
    # ymin,ymax,dy is the output cooridniate (resolutions) 
    #
    def conditional_distribution(self,x_in,adaptive=False,ymin=-15,ymax=15,dy=0.05):
        ny=round((ymax-ymin)/dy)
        if (len(x_in.shape)==1): #That mean only one data point to be calcualte
            x_in=x_in.reshape(1,-1)
        
        
        n_data,nx=x_in.shape[0],x_in.shape[1]
        if (nx!=self.d-1):
            raise ValueError(f"Number of inputs is {nx}, should be {self.d-1}")
        #Preparing inputs
        xcoord = np.broadcast_to(x_in,(ny,n_data,nx)).reshape(ny*n_data,nx)        
        ycoord = np.linspace(ymin,ymax,ny)
        dy = ycoord[1]-ycoord[0]    
        yycoord=np.broadcast_to(ycoord.reshape(-1,1),(ny,n_data)).reshape(ny*n_data,1)
        inputs = np.hstack((xcoord,yycoord)).T   
        
        #print(f"Calcualte joint PDF at {n_data} data points...")    
        if (adaptive):
            ff=self.aw_evaluate(inputs).reshape(-1,n_data) #dims=(ny,n_data)
        else:
            ff=self.evaluate(inputs).reshape(-1,n_data) #dims=(ny,n_data)
        
        denominator = np.sum(ff*dy,axis=0)
        d = np.broadcast_to(denominator.reshape(1,-1), ff.shape)
        pdfs=ff/d
        ymean=np.sum( ycoord.reshape(-1,1)*pdfs*dy,axis=0 )
    
        #print(f"Calcualte CDF...")    
        cdfs = np.cumsum(pdfs,axis=0)*dy
        cdfs=cdfs/cdfs[-1,:]  #to make sure the last values are always 1

        return ycoord,ymean,pdfs,cdfs  
    
     #
    
    def aw_evaluate(self, points):
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        output_dtype = np.common_type(self.covariance, points)
        itemsize = np.dtype(output_dtype).itemsize
        if itemsize == 4:
            spec = 'float'
        elif itemsize == 8:
            spec = 'double'
        elif itemsize in (12, 16):
            spec = 'long double'
        else:
            raise TypeError('%s has unexpected item size %d' %
                            (output_dtype, itemsize))
        result = gaussian_aw_kernel_estimate[spec](self.dataset.T, self.weights[:, None],
                                                points.T, self.inv_cov,self.inv_gamma, output_dtype)
        return result[:, 0]

                

def lscv(h,xyvalues,adaptive=False):
    from math import sqrt
    loo = LeaveOneOut()
    n_features,n_samples= xyvalues.shape[0],xyvalues.shape[1]
    Selective=True
    if (np.isscalar(h)):
        Selective=False
        
    if (Selective):
        kde2=multivariate_gausian_kde(xyvalues)
        kde2.set_selective_factor(sqrt(2)*h)
        if  (adaptive):
            kde_full = multivariate_gausian_kde(xyvalues)
            kde_full.set_selective_factor(h)
    else:
        kde2=multivariate_gausian_kde(xyvalues,bw_method=sqrt(2)*h)
        if (adaptive):
             kde_full = multivariate_gausian_kde(xyvalues,bw_method=h)
    
    if (adaptive):
        kde2.calc_local_weights()
        f2_hat = kde2.aw_evaluate(xyvalues)
        kde_full.calc_local_weights()
        inv_gamma=kde_full.inv_gamma  #Use local bandwith for leave one out instead of recaculate every time
        
    else:
        f2_hat = kde2.evaluate(xyvalues)
            
    Ef2 = np.average(f2_hat)
    
    f_loo = np.zeros(n_samples,"float")
    for i,(train_idx, test_idx) in enumerate(loo.split(xyvalues.T)):
        train_data=xyvalues.T[train_idx].T
        test_data=xyvalues.T[test_idx].T
        if (Selective):
            kde_loo=multivariate_gausian_kde(train_data)
            kde_loo.set_selective_factor(h)  
        else:
            kde_loo=multivariate_gausian_kde(train_data,bw_method=h)
            
        if (adaptive):
            kde_loo.inv_gamma=inv_gamma[train_idx] #This approximate will take a shorter time
            #kde_loo.calc_local_weights()
            f_loo[i]=kde_loo.aw_evaluate(test_data)
        else:
            f_loo[i]=kde_loo.evaluate(test_data)

    return Ef2 - 2.*np.mean(f_loo)




#Note Dimension of sklean is: n_samples, n_features,
#But of scipy is opposite:    n_features,n_samples
from sklearn.model_selection import LeaveOneOut  
loo = LeaveOneOut()
#Calculate mean conditional squared error with leave-one-out rule
#If h take a single value, then use Scalar Factor
#else: use adaptive Factor 
def mcse(h,xyvalues,adaptive,y0=-10,y1=10,dy=0.1):
    n_features,n_samples = xyvalues.shape[0],xyvalues.shape[1]
    y_loo=np.zeros(n_samples,"float")
    Selective=True
    if (np.isscalar(h)):
        Selective=False

    #Use local bandwith for leave one out instead of recaculate every time
    if (adaptive):
        if (Selective):
            kde_full=multivariate_gausian_kde(xyvalues)
            kde_full.set_selective_factor(h)
        else:
            kde_full = multivariate_gausian_kde(xyvalues,bw_method=h)
        kde_full.calc_local_weights()
        inv_gamma=kde_full.inv_gamma  
        
        
    for i,(train_idx, test_idx) in enumerate(loo.split(xyvalues.T)):
        train_data=xyvalues.T[train_idx].T
        test_data=xyvalues.T[test_idx].T
        
        if (Selective):
            kde_loo=multivariate_gausian_kde(train_data)
            kde_loo.set_selective_factor(h)
        else:
            kde_loo=multivariate_gausian_kde(train_data,bw_method=h)
            
        if (adaptive):
            kde_loo.inv_gamma=inv_gamma[train_idx] #This approximate will take a shorter time
       
        xin = test_data[0:n_features-1,:].T
        ycoord,y_loo[i],pdfs,cdfs = kde_loo.conditional_distribution(xin,adaptive,y0,y1,dy)

    mse = np.mean((y_loo-xyvalues[n_features-1,:])**2)
    return mse

        
def cdf2quantile(yr,cdfs,q):
    from scipy.interpolate import interp1d
    dims=cdfs.shape
    ny,nx=dims[0],dims[1]    
    Qt = np.zeros(nx,"float")
    for i in range(nx):
        Qt[i]=interp1d(cdfs[:,i],yr)(q)
    return Qt


