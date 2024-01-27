import math
import numpy as np
from sklearn.metrics import accuracy_score

class MKNearestNeighbors:
    """Implements a k-Nearest Neighbor variant proposed for gene expression cancer classification.
    
    Please check the following paper to get further details regarding the method employed to implement this code:

    S. M. Ayyad, A. I. Saleh, and L. M. Labib, “Gene expression cancer classification using modified K-Nearest Neighbors technique,” BioSystems, vol. 176, pp. 41–51, 2019.
    """

    def __init__(self,mode='smknn'):
        """Constructor to initialize the class properties.

        Args:
            mode (str, optional): Mode to get the nearest neighbors. Defaults to 'smknn'.

        Raises:
            Exception: Thrown if the mode is different from 'smknn' or 'lmknn'
        """
        if (mode!='smknn' and mode!='lmknn'):
            raise Exception('Mode parameter must be smknn or lmknn')
            
        self.X_train = None
        self.y_train = None
        self.centers = None
        self.classes = None
        self.k = None
        self.nearest_neighbors = None
        self.mode = mode
    
    @property
    def mode(self) -> str:
        """Mode of the Modified k-NN.
        """
        return self.__mode

    @mode.setter
    def mode(self,mode : str) -> None:
        if (mode!='smknn' and mode!='lmknn'):
            raise Exception('Mode parameter must be smknn or lmknn')

        self.__mode = mode
    
    def __get_distances(self, X, Y, check_same_idx=True):
        """Computes the distance matrix from the samples in X and Y.

        Args:
            X (array): A MxN array.
            Y (array): A KxZ array.
            check_same_idx (bool, optional): If True, the diagonal of the distance matrix is assigned zero. Defaults to True.

        Returns:
            array: A MxK array with the distance between each element from X to all elements of Y.
        """
        
        distances = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(X.shape[0]):
            p = X[i]
            for j in range(Y.shape[0]):
                if (check_same_idx and i == j):
                    continue
                
                ed = np.linalg.norm(Y[j]-p)                
                distances[i][j] = ed
        
        return distances
            
    def fit(self, X, y):
        """Computes the center of the classes and the weights of the samples.

        Args:
            X (array): A MxN dimensional array with the samples of the training set.
            y (array): A Mx1 dimensional array with the labels of each sample in X.
        
        Returns:
            None
        """
        
        self.X_train = np.copy(X)
        
        # Adding to more columns to the X_train array to store the labels (y) and the weights of each training sample to the center of the class, respectively
        self.X_train = np.append(self.X_train,np.zeros((X.shape[0],2)),axis=1)
        self.X_train[:,-2] = np.copy(y)
        
        # Auxiliary variables to store the classes and their respective centers
        classes = np.unique(y)
        centers = []
        
        # Auxiliary variable to store the weigths of the samples (with respect to the class centers)
        w = []
        
        for c in classes:
            indices = np.where(y==c)[0]
            X_ = X[indices,:]
            
            # Getting the center of the class
            center = np.mean(X_,axis=0)
            centers.append(center)
            
            # Getting the weights of each sample
            w = []
            for s in X_:
                w.append(1 / (np.linalg.norm(s-center)+0.0001))
            
            # Adding the class weights to the last column of the X_train array
            self.X_train[indices,-1] = np.array(w)
        
        self.centers = np.vstack(centers)
        self.classes = classes
    
    def predict(self, X):
        """Predicts the labels of each sample in test set X.

        Args:
            X (array): A MxN array with the samples of the test set.

        Returns:
            array: A Mx1 array with the predicted labels.
        """

        y_pred = []
        
        for t in X:
            t = np.expand_dims(t,axis=0)
            
            # Distance of the test sample to all centers of class
            distances_center = self.__get_distances(t,self.centers,False).flatten()
            
            # Distance of the test sample to all instances of the training set
            distances = self.__get_distances(t,self.X_train[:,:-2],False).flatten()
            
            # Getting the distance of the test sample to the class centers
            if (self.mode=='smknn'):
                ed = distances_center[np.argmin(distances_center)]
            else:
                ed = distances_center[np.argmax(distances_center)]
            
            # Getting the nearest neighbors, i.e., all training instances whose distances are less than the distances
            idx_min = np.where(distances <= ed)
            nearest_neighbors = self.X_train[idx_min]
            
            # Concatenating the distances of the test sample to the nearest neighbors of the training set
            nearest_neighbors = np.concatenate((nearest_neighbors,distances[idx_min].reshape(-1,1)),axis=1)
                        
            # Determining the final class based on the nearest neighbors
            if (len(nearest_neighbors) == 0):
                final_class = self.classes[np.argmin(distances_center)]
            elif (len(np.unique(nearest_neighbors[:,-3])) == 1):
                final_class = np.unique(nearest_neighbors[:,-3]).astype(int)
            else:
                # Finding the classes of the nearest neighbors                
                cl = np.unique(nearest_neighbors[:,-3]).astype(int)
                classes = dict.fromkeys(cl,0)
                
                # Weighted sum considering the considering the distances and weights of the training instances
                for n in nearest_neighbors:
                    c = int(n[-3])
                    classes[c]+=(1/n[-1]) * n[-2]
                
                final_class = max(classes,key=lambda x : classes[x])
            
            y_pred.append(final_class)
        
        y_pred = np.vstack(y_pred)
        return y_pred