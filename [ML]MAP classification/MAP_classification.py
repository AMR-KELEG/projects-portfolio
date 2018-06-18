# CSE 465: Assignment 1
# Amr Mohamed
# Maximum likelihood / Maximum a-posteriori Classifiers
# Python 3

import numpy as np
import matplotlib.pyplot as plt

def mahalanobis_dis(x,mean,cov_mat):
  #TODO: Input Checking
  #1) Dimensions
  #2) Types
  #3) Invertable covariance matrix
  v=(x-mean)
  dis=np.matmul(np.matmul(v.T,np.linalg.inv(cov_mat)),v)
  return np.sqrt(dis)

def disc_func(x,mean,cov_mat,p_wi,d=3):
   det_cov=np.linalg.det(cov_mat)
   mah_dis=mahalanobis_dis(x,mean,cov_mat)
   return (-0.5*mah_dis*mah_dis)-(d*np.log(np.pi/2)/2)-(0.5*np.log(det_cov))+np.log(p_wi)

def classify(point,prior_p,cov_matrices,mean_vectors):
  prob_class=[]
  for i in range(len(cov_matrices)):
    prob_class.append(disc_func(point,mean_vectors[i],cov_matrices[i],prior_p[i]))
  return np.argmax(prob_class)

def solve(testing_patterns,prior_p,cov_matrices,mean_vectors,figure_description):
  for p in testing_patterns:
    print('Vector:',p,'belongs to class',1+classify(p,prior_p,cov_matrices,mean_vectors))
  x = y = np.arange(-10.0, 10.0, 0.1)
  X, Y = np.meshgrid(x, y)
  image_size = X.shape
  xy=np.array([X,Y])
  print('Progress |',' ' * 16 ,'|',sep='')
  fig = plt.figure(figure_description)
  for indx in range(0,16):
    z=np.array([classify(np.array([x,y,indx]),prior_p,[cov1,cov2,cov3], [c1_mean,c2_mean,c3_mean])
     for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    plt.subplot(4, 4, indx+1)
    plt.pcolor(X, Y, Z, cmap='winter', vmin=0, vmax=2)
    plt.title('X3='+str(indx))
    print('Progress |','#'*(indx+1), ' ' * (15-indx) ,'|',sep='')

if __name__=='__main__':
  x1 = [[-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.51,-2.25,5.56,1.03],
  [-8.12,-3.48,-5.52,-3.78,0.63,3.29,2.09,-2.13,2.86,-3.33],
  [-3.68,-3.54,1.66,-4.11,7.39,2.08,-2.59,-6.94,-2.26,4.33]];
  x2 = [[-0.91,1.30,-7.75,-5.47,6.14,3.60,5.37,7.18,-7.39,-7.50],
  [-0.18,-2.06,-4.54,0.50,5.72,1.26,-4.63,1.46,1.17,-6.32],
  [-0.05,-3.53,-0.95,3.92,-4.85,4.36,-3.65,-6.66,6.30,-0.31]];
  x3 = [[5.35,5.12,-1.34,4.48,7.11,7.17,5.75,0.77,0.90,3.52],
  [2.26,3.22,-5.31,3.42,2.39,4.33,3.97,0.27,-0.43,-0.36],
  [8.13,-2.66,-9.87,5.19,9.21,-0.98,6.65,2.41,-8.71,6.43]]
  c1=np.array(x1).transpose()
  c2=np.array(x2).transpose()
  c3=np.array(x3).transpose()
  c1_mean=np.mean(c1,axis=0)
  c2_mean=np.mean(c2,axis=0)
  c3_mean=np.mean(c3,axis=0)
  cov1=np.cov(c1.transpose())
  cov2=np.cov(c2.transpose())
  cov3=np.cov(c3.transpose())
  points=[np.array([1,2,1]) , np.array([5,3,1]) , np.array([0,0,0]) , np.array([1,0,0])]

  print('MAP Classifier:')
  solve(points,np.array([0.8,0.1,0.1]),[cov1,cov2,cov3],[c1_mean,c2_mean,c3_mean],'MAP Classifier')
  print('-'*20)
  print('ML Classifier:')
  # Prior Probabilities are set to one to cancel their effect on the disc. function
  solve(points,np.array([1.,1.,1.]),[cov1,cov2,cov3],[c1_mean,c2_mean,c3_mean],'ML Classifier')
  plt.show()
