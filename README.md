# PL-kNN
This repository retains the source code of PL-kNN, a model proposed to avoid initializing the value of k of the standard k-NN classifier. The proposed model relies on the SMKNN presented by Ayyad et al., but with improvements on the class center calculation and the neighbors' determination considering the circle enclosing the test sample.

The model seeks the nearest neighbors using a semicircle computed as the distance between the test sample under analysis and the nearest class center of the training samples. This approach is efficient when there is a mixing of instances of different classes.

# Citation
