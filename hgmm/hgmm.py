# -*- coding: utf-8 -*-
"""
Implementation of Frank Havlak's hybrid Gaussian mixture model, from his 2013 
journal paper: Discrete and Continuous, Probabilistic Anticipation
for Autonomous Robots in Urban Environments (https://arxiv.org/pdf/1309.0766.pdf)
 
Created on Sun Apr 24 07:27:13 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
import scipy.linalg as la
import gaussian_splitting


def get_sigma_points(mu, cov):
    ''' Find sigma points for a gaussian with mean mu, covariance cov 
    and a parameter lambda
    
    Returns
    --------
        pts 
            shape (n,2n+1) np array of sigma points
        weights
            shape (2n+1,) array of point weights
            '''
     
    n = cov.shape[0]  
    mu = mu.flatten()
    
    kappa = 3 - n # n + kappa = 3 is a good heuristic for gaussian dist data
    
    S_squared = (n+kappa)*cov # matrix we want the square root of
    
    S = la.cholesky(S_squared)
    
    n_pts = 1 + 2*n
    pts = np.zeros((n, n_pts))    
    weights = np.zeros(n_pts)    
    
    for j in range(n_pts):
        if j == 0:
            pts[:,j] = mu
            weights[j] = kappa/(n+kappa)
        elif j > 0 and j <= n:
            pts[:,j] = mu + S[j-1,:]
            weights[j] = 0.5/(n+kappa)
            
            pts[:,j+n] = mu - S[j-1,:]
            weights[j+n] = 0.5/(n+kappa)
            
    
    assert np.allclose(weights.sum(), 1.0)
    
    # expensive check to see if 1st and 2nd moments of sigma points match gaussian
    #assert compare_sigma_pts(mu, cov, pts, weights)
    
    return pts, weights
        
def get_sigma_points_mean_cov(pts, weights):
    '''Find mean and covariance of a set of weighted sigma points pts
    
    Returns
    -------
        mean, cov
            np arrays (n,1) and (n,n) respectively
    '''

    n = pts.shape[0]
    
    n_pts = pts.shape[1]
    
    mean = np.sum(pts*weights, axis=1)[:,np.newaxis]
    cov = np.dot(weights*(pts-mean), (pts-mean).T)
    
    # Sometimes if kappa < 0, cov may become non positive semi-definite. If so, 
    # approximate 'covariance' matrix according to UKF paper Julier 1997
    try: # check positive semi-definiteness
        la.cholesky(cov)
    except la.LinAlgError:
        print 'Covariance matrix is not positive semi-definite, aproximating...'
        # take 'covariance' about propagated 0th sigma point instead of new mean
        X0 = pts[:,0,np.newaxis] # first sigma point
        cov = np.dot(weights*(pts-X0), (pts-X0).T) 
        # Check positive semi-definiteness again (should always be)
        la.cholesky(cov)# will throw LinAlgError if not
        
    return mean, cov
    
def get_state_sigma_points(pts, n_x, n_v):
    ''' Return the sigma points that correspond to the state as an array of size
    (n_x)*(2*n_x+1) from an array *pts* containing sigma points for the augmented 
    state/noise sigma point array (see propagate for details on how that is 
    calcualted).'''
    idx = np.concatenate((range(n_x+1), range(n_x+n_v+1, 2*n_x+n_v+1)))
    assert idx.shape[0] == 1 + 2*n_x, str(idx)
    return pts[:n_x, idx]

def compare_sigma_pts(mu, cov, pts, weights):
    ''' return true if the generated pts have a mean and cov that matches the input 
    to get_sigma_points (hint: it always should)'''
    mu = mu.flatten()
    mu_check, cov_check = get_sigma_points_mean_cov(pts, weights)
    try:        
        assert np.allclose(cov, cov_check), 'cov values are shite'
    except:
        print 'mu check:', mu_check.flatten()
        print 'mu', mu
        print cov
        print cov_check
        assert False, 'covariance is bung'
    return True


def propagate_discrete(mixture, func, k):
    ''' Propagate discrete components of the state for mixands in a hybrid gaussian 
    mixture. Mixand will split if multiple discrete states are possible
    
    Parameters
    ----------
    mixture : gmm.GaussianMixture object
        gaussian mixture to be propagated

    func : function
        function of the form X_d_new, prob = func(m, k)
        
        - m is the gaussian mixand at timestep k
        - k is the timestep, often not really needed
        
        - X_d_new is a list of possible new discrete states (at time k+1)
        - prob is a list of probabilities of transition to each x_d in X_d_new
    
    k : integer
        timestep
    '''
    
    for m in mixture.mixands.copy():
        
        X_d_new, probs = func(m, k)
        
        if len(X_d_new) > 1:
            print 'Splitting mixands during discrete propagation......',
            new_mixands = []
            
            for x_d_new, prob in zip(X_d_new, probs):
                new_m = m.copy()
                new_m.x_d = x_d_new
                new_m.w *= prob
                new_mixands.append(new_m)
                
            mixture.mixands.remove(m)
            mixture.mixands.update(new_mixands)
            print 'Done'
            
        else:
            m.x_d = X_d_new[0]
        


def propagate_continuous(mixture, func, k, cov_v, e_res_max=0.5, max_mixands=30):
    ''' Propagate continuous components of the state for mixands in a hybrid gaussian 
    mixture. Mixand will split if linearity residual criteria *e_res_max* is exceeded, 
    using a precomputed split solution. Mixture will reduce if number of mixands 
    exceeds *max_mixands*. 
    
    Parameters
    ----------
    mixture : gmm.GaussianMixture object
        gaussian mixture to be propagated

    func : function
        function of the form X_new = fc(X, V, m, k) where X , X_new are (n_x)x(p) and V is (n_v)x(p).
        
        - X and V are an array of p state vectors, and m noise vectors, respectively, where n_x is the (continuous) state dimension and n_v is the noise dimension. 
        - m is the mixand, included to provide access to the x_d, but also the ability to modify some other mixand attributes to help with other calculations (the same for all p points propagated through).
        - k is the timestep, often not really needed
    
    e_res_max :
        maximum residual error before splitting
    
    max_mixands : 
        after each propagation the mixture will be reduced to this number of mixands
        using moment preserving merges that give the lowest upper-bound kld convergence.
    '''
    for m in mixture.mixands:
        break # get a single mixand from the set of mixands
    n_x = m.mu.shape[0] # number of states
    n_v = cov_v.shape[0] # number of noise components
    
    for m in mixture.mixands.copy():
        # Create augmented mean vector and covariance matrix
        cov_aug = la.block_diag(m.P, cov_v)
        
        assert m.mu.shape == (n_x,1), str(m.mu.shape)
        mu_aug = np.concatenate((m.mu, np.zeros((n_v,1))))
        
        # Find sigma points
        pts, weights = get_sigma_points(mu_aug, cov_aug)
        X = pts[:n_x,:] # values of the state for each sigma point
        Y = pts[n_x:,:] # vaues of noise for each sigma point
        
        # Propagate sigma points and get new mean, cov, and sigma points
        X_new = func(X, Y, m, k)        
        
        # Calcs to help find splitting axis
        Xbar = get_state_sigma_points(X, n_x, n_v)
        Xbar_augmented = np.concatenate( ( Xbar, np.ones((1,2*n_x+1)) )  )

        # LQ factorisation
        Q, R = np.linalg.qr(Xbar_augmented.T, mode='complete') 
        Q = Q.T
        #L = R.T
        
        # Variable names should match those in the paper (if this is unclear read it)
        Xbar_new = get_state_sigma_points(X_new, n_x, n_v) # sigma points relating to state
        Xhat_all_new = np.dot(Xbar_new, Q.T)
        Xhat_res_new = Xhat_all_new[:,n_x+1:] # X_hat_all partition not explained by linear model
            
        # Calculate linearisation error residual e_res (mixand splitting criteria)
        zeromat = np.zeros((n_x,n_x+1))
        e_res = np.linalg.norm( np.concatenate((zeromat, Xhat_res_new), axis=1))
        
        print 'e_res =', e_res,
        if e_res > e_res_max:
            print ' splitting....',
            # Residual associated with each sigma point
            E_new = np.dot( np.concatenate( (zeromat, Xhat_res_new), axis=1 ),
                            Q )
            error_norms = np.linalg.norm(E_new, axis=0)
            # 2nd mom of weighted sigma pts before propagation
            second_moments = np.cov(Xbar, aweights=error_norms, bias=True) 
            # first eigvector (one with largest eigval) is the optimal split axis
            w, V = np.linalg.eig(second_moments) # unsorted eigvals and eigvects
            idx = w.argsort()[::-1] # sort by largest eigval
            w = w[idx]
            V = V[:,idx]
                        
            e_split = V[:,0] # split axis
            
            ### Now split gaussian
            new_mixands = gaussian_splitting.split_gaussian(m, e_split)
            
            for new_m in new_mixands:
                # Create augmented mean vector and covariance matrix
                cov_aug = la.block_diag(new_m.P, cov_v)
                
                mu_aug = np.concatenate((new_m.mu, np.zeros((n_v,1))))
                
                # Find sigma points
                pts, weights = get_sigma_points(mu_aug, cov_aug)
                X = pts[:n_x,:] # values of the state for each sigma point
                Y = pts[n_x:,:] # vaues of noise for each sigma point
                
                # Propagate sigma points and get new mean, cov, and sigma points
                X_new = func(X, Y, new_m, k)  
                new_m.mu, new_m.P = get_sigma_points_mean_cov(X_new, weights)
            
            # Remove current mixand and add split results
            mixture.mixands.remove(m)
            mixture.mixands.update(new_mixands)
            print 'done'
        else:
            print ''
            # No splitting case
            m.mu, m.P = get_sigma_points_mean_cov(X_new, weights)
        
    ### Now reduce number of mixands
    mixture.reduce_N(max_mixands)
  




''' Make up some propagation functions for testing '''

def f_d(m, k):
    if k == 30:
        return [{'x':True}, {'x':False}], [0.75, 0.25]
    else:
        return [m.x_d], [1.]      
        
def f_c(x, v, m, k=0):
    ''' Generic nonlinear discrete-time dynamics function for testing purposes'''
    
    if m.x_d['x']:    
        r = 10 #- k/49
    else:
        r = 5
    vel = np.pi*r/200 + v*np.ones_like(x[0,:])
    dt = 1
    
    x = np.reshape(x, (3,-1))
    x_new = np.zeros_like(x)

    for j in range(x_new.shape[1]):
        x_new[0,j] = x[0,j] + r * (np.sin(x[2,j] + vel[0,j]*dt/r) - np.sin(x[2,j]))
        x_new[1,j] = x[1,j] + r * (np.cos(x[2,j] + vel[0,j]*dt/r) - np.cos(x[2,j]))
        x_new[2,j] = x[2,j] + vel[0,j]*dt/r

    return x_new
        
        
        
        
        
if __name__ == "__main__":
    ''' Run simple test case '''    
    
    
    import gmm
    import matplotlib.pyplot as plt
    
    plt.close('all')
    
    n = 3 # number of states
    
    mu = np.zeros((n,1), dtype=np.float64) # initial mixand mean
    
    P = np.eye(n)*0.005 # covariance matrix
    P[1,1] = 1e-3
    P[2,2] = 1e-3
    
    # rotate cov matrix to make it more interesting
    R = gaussian_splitting.get_rot_mat_from_vects(
                   np.array([[1,0,0]]).T, np.array([[1,1,0]]).T  )
    P = np.dot(R, np.dot(P, R.T))

    cov_v = np.array([[1e-3]], dtype=np.float64)  # noise covariance matrix
               
               
    #mixands = [gmm.Mixand(1/3, mu, P),gmm.Mixand(1/3, mu+0.1, P),gmm.Mixand(1/3, mu-0.1, P)]
    mixands = [gmm.Mixand(1, mu, P, x_d={'x':True})]
               
    mixture = gmm.GaussianMixture(mixands)
    print mixture
    
    
    plt.figure()
    mixture.plot()
    plt.xlim(-11,11)
    plt.ylim(-21,1)
    plt.pause(0.1)

    # precompute optimal gaussian splits    
    gaussian_splitting.precompute(N=3, cov_reduction=0.3, n=n)
    
    for k in range(200):
        print '\n timestep:', k
        print 'N =', mixture.N   
        
        propagate_discrete(mixture, f_d, k)
        propagate_continuous(mixture, f_c, k, cov_v, e_res_max=0.0005, max_mixands=10)

        mixture.plot(nsteps=200)
        plt.pause(0.01)
