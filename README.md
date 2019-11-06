# Foraminifera-Identification
A Keras implementation for foraminifera identification. Data set and detailed project description are available on website: https://research.ece.ncsu.edu/aros/foram-identification/

## Steps to run the code
Step1: please download and extract the forams images 'NCSU-CUB Foram Images 01' to the same folder as the scripts.

Step2: run keras_extract_features.py to extract features from pre-trained models.

Step3: run keras_train_new_layers.py to train the new fully-connected layers for forams classification. 

## Related publications
B. Zhong, Q. Ge, B. Kanakiya, R. Mitra, T. Marchitto, E. Lobaton, “A Comparative Study of Image Classification Algorithms for Foraminifera Identification,” IEEE Symp. Series on Computational Intelligence (SSCI), 2017. [Makes use of NCSU-CUB Foram Images 01 Dataset]

Q. Ge, B. Zhong, B. Kanakiya, R. Mitra, T. Marchitto, E. Lobaton, “Coarse-to-Fine Foraminifera Image Segmentation through 3D and Deep Features,” IEEE Symp. Series on Computational Intelligence (SSCI), 2017. [Makes use of NCSU-CUB Foram Labels 01 Dataset]