# PL-kNN
Thank you for being so interested in our research. This repository contains the source code of PL-kNN, a model proposed to bypass the choice of the k parameter of the standard k-NN classifier. The proposed model is based on the Smallest Modified k-Nearest Neighbors (SMKNN) presented by Ayyad et al. in the following paper:

**Ayyad, S. M., Saleh, A. I., & Labib, L. M. (2019). Gene expression cancer classification using modified K-Nearest Neighbors technique. *Biosystems*, 176, 41-51.**

Compared to the original SMKNN model, the proposed approach improves the class center calculation and the nearest neighbors' choice inside the circle enclosing the test sample. The model seeks the nearest neighbors using a semicircle computed as the distance between the test sample under analysis and the nearest class center of the training samples. This approach is efficient when there is a mixing of instances of different classes.

# Usage

Please, check the **example.py** file to see a simple example of how to use our method.

If you want to use it in your source code, you have to set the PYTHONPATH variable to point to the folder where you cloned this repository. If you clone to **/home/user/PL-kNN**, you have to set the PYTHONPATH as follows:

```sh
export PYTHONPATH=/home/user/PL-kNN
```

# Citation request

By using this repo, you are accepting to cite the following paper in all publications that use this source code:

**Jodas, D.S., Passos, L.A., Papa, J.P. (2022, June 01–03). PL-kNN: A parameterless nearest neighbors classifier. [Paper presentation]. *IWSSIP 2022 - International Conference on Systems, Signals and Image Processing*, Sofia, Bulgaria. [http://iwssip.bg](http://iwssip.bg)**

# Additional info

Please, check the websites of the Recogna Laboratory to find more information about the work in progress in several domains of Machine Learning:

Recogna Laboratory: [https://www.recogna.tech](https://www.recogna.tech) <br>

# Contact

If you have any further questions, please do not hesitate to contact us.

| Author                    | E-mail                        |
| ----------------------    | ----------------------        |
| Danilo Samuel Jodas       | danilojodas@gmail.com         | <br>
| João Paulo Papa           | joao.papa@unesp.br            | <br>
| Leandro Aparecido Passos  | l.passosjunior@wlv.ac.uk      | <br>

