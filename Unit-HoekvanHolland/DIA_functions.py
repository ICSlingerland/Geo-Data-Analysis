# Shared functions to generate the model and PDF of DIA
 
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from scipy.linalg import null_space
import matplotlib.pyplot as plt

def compute_DIA_matrices(A, B, Q_yy, Q_tt, c_vectors):

    # Input
    # The design matrix                                    A
    # The B matrix such that B.T @ A = 0                   B
    # The vc-matrix of the measurement vector              Q_yy                                       
    # The vc-matrix of the misclosure vector               Q_tt
    # The matrix containing 1D vectors                     c_vectors 

    # Output 
    # The BLUE-inverse of B.T @ ci                         Btci_plus
    # The projection matrices on the range of cti          Pcti

    # Do checks on the c_vectors
    if c_vectors.shape[0] != Q_yy.shape[0]:
        raise Exception("The dimensions of the c_vectors are not correct")
    elif (not np.any(c_vectors)):
        raise Exception("The c_vectors matrix is filled with zeros")

    # Compute the BLUE-inverse of A
    invQ_yy = linalg.inv(Q_yy)
    A_plus = linalg.inv(A.T @ invQ_yy @ A) @ A.T @ invQ_yy

    # Get the number of alternative hypotheses (k), number of measurements (m), the redundancy (r), and the dimension of the unknown vector (n)
    k = c_vectors.shape[1]
    m = Q_yy.shape[0]
    r = B.shape[1]
    n = A_plus.shape[0]
   
    # Compute the inverse of the vc-matrix Q_tt
    invQ_tt = linalg.inv(Q_tt)

    # Compute the "DIA" matrices
    Btci_plus = np.zeros((k, 1, r))
    Pcti = np.zeros((k, r, r))
    Li = np.zeros((k, n, r))
    for idx in range(0, k):
        # Extract the one-dimensional model error signature vector
        ci = c_vectors[:,idx].reshape(m,1)   
        
        # Comput the (B.T @ ci)^(-1) term
        Btci_plus[idx] = linalg.inv(ci.T @ B @ invQ_tt @ B.T @ ci) @ ci.T @ B @ invQ_tt   

        # Compute the projection matrices on the ranges of cti
        cti = B.T @ ci  
        Pcti[idx] = cti @ linalg.inv(cti.T @ invQ_tt @ cti) @ cti.T @ invQ_tt

        # Compute the Li terms 
        Li[idx] = A_plus @ ci @ Btci_plus[idx]

    return Pcti, Li

def plot_partitionied_2D_misclosure_space(Pcti, Q_tt, Q_yy, alpha):

    # Input
    # The projection matrices of the cti vectors           Pcti
    # The vc-matrix of the misclosure vector               Q_tt
    # The vc-matrix of the measurement vector              Q_yy
    # The level of significance of the OMT                 alpha

    # Output 
    # Plot of the partitioned misclosure space

    # Do a check on the dimension of the misclosure space
    if (Q_tt.shape[0] != 2):
        raise Exception("The redundancy is higher than 2 is not supported!")     

    # Set the limits of the 2D space
    t1_min, t1_max, t2_min, t2_max = -2, 2, -2, 2

    # Set the number of points to generate
    nb_samples_t_2D = 100000

    # Generate random t1 and t2 coordinates
    t1_coords = np.random.uniform(t1_min, t1_max, nb_samples_t_2D)
    t2_coords = np.random.uniform(t2_min, t2_max, nb_samples_t_2D)

    # Combine the t1 and t2 coordinates into a 2D array
    t_2D = np.vstack((t1_coords, t2_coords))

    # Get the number of alternative hypotheses (k), number of measurements (m) and the redundancy (r)
    k = Pcti.shape[0]
    m = Q_yy.shape[0]
    r = Q_tt.shape[0]    

    # Compute the threshold for the Overall Model Test (OMT)
    threshold_squared = chi2.ppf(1-alpha, r)

    # Compute the OMT statistics
    invQ_tt = linalg.inv(Q_tt)
    omt_test = np.sum((t_2D.T @ invQ_tt) * t_2D.T, axis=1)

    # Compute the indicator functions for which the samples are outside the acceptance region P0
    i_outside_P0_bool = omt_test > threshold_squared
    i_outside_P0 = i_outside_P0_bool.astype(int)
    indices_i_P0 =  np.nonzero(i_outside_P0)

    # Compute the squared w-tests
    samples_t_2D_outside_P0 = t_2D[:,i_outside_P0_bool].T
    nb_samples_outside_P0 = np.sum(i_outside_P0)

    computed_squared_wtest = np.zeros((nb_samples_outside_P0, k))
    for idx in range(0, k):
        Pcti_t_2D = Pcti[idx] @ samples_t_2D_outside_P0.T
        computed_squared_wtest[:,idx] = np.sum((Pcti_t_2D.T @ invQ_tt) * Pcti_t_2D.T, axis=1)

    # Create the array with the indicator functions for each partition
    max_indices_squared_wtest = computed_squared_wtest.argmax(axis=1)
    temp_arr = np.zeros_like(computed_squared_wtest)
    temp_arr[np.arange(nb_samples_outside_P0), max_indices_squared_wtest] = 1

    # Place the indices into a larger array
    i_Pi_arr = np.zeros((nb_samples_t_2D, k))
    for idx in range(0,nb_samples_outside_P0):
        i_Pi_arr[indices_i_P0[0][idx],:] = temp_arr[idx,:]

    # Transform the array of indicator functions in an array of booleans
    i_Pi_arr_bool = i_Pi_arr.astype(bool)

    # Plot the partitioned misclosure space
    fig, ax = plt.subplots()
    ax.set_xlim([t1_min, t1_max])
    ax.set_ylim([t2_min, t2_max])

    for idx in range(0,k):
        ax.plot(t_2D[0,i_Pi_arr_bool[:,idx]], t_2D[1,i_Pi_arr_bool[:,idx]],'o', label = "P" + str(idx+1))
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlabel('t1')
    ax.set_ylabel('t2')
    ax.set_title("The partitioned misclosure space")

def compute_indicator_function(Pcti, t_samples, Q_tt, alpha):

    # Input
    # The projection matrices of the cti vectors           Pcti
    # Samples of t                                         t_samples
    # The vc-matrix of the misclosure vector               Q_tt
    # The level of significance of the OMT                 alpha

    # Output 
    # Indicators for the samples of t


    # Get the number of alternative hypotheses (k), number of measurements (m) and the redundancy (r)
    k = Pcti.shape[0]
    n_t, r = t_samples.shape 

    # Compute the threshold for the Overall Model Test (OMT)
    threshold_squared = chi2.ppf(1-alpha, r)

    # Compute the OMT statistics
    invQ_tt = linalg.inv(Q_tt)
    omt_test = np.sum((t_samples @ invQ_tt) * t_samples, axis=1)

    # Compute the indicator functions for which the samples are outside the acceptance region P0
    i_outside_P0_bool = omt_test > threshold_squared
    i_outside_P0 = i_outside_P0_bool.astype(int)
    indices_i_P0 =  np.nonzero(i_outside_P0)

    # Compute the squared w-tests
    samples_t_2D_outside_P0 = t_samples[i_outside_P0_bool,:]
    nb_samples_outside_P0 = np.sum(i_outside_P0)

    computed_squared_wtest = np.zeros((nb_samples_outside_P0, k))
    for idx in range(0, k):
        Pcti_t_2D = Pcti[idx] @ samples_t_2D_outside_P0.T
        computed_squared_wtest[:,idx] = np.sum((Pcti_t_2D.T @ invQ_tt) * Pcti_t_2D.T, axis=1)

    # Create the array with the indicator functions for each partition
    max_indices_squared_wtest = computed_squared_wtest.argmax(axis=1)
    temp_arr = np.zeros_like(computed_squared_wtest)
    temp_arr[np.arange(nb_samples_outside_P0), max_indices_squared_wtest] = 1

    # Place the indices into a larger array
    i_Pi_arr = np.zeros((n_t, k))
    for idx in range(0,nb_samples_outside_P0):
        i_Pi_arr[indices_i_P0[0][idx],:] = temp_arr[idx,:]

    return i_Pi_arr