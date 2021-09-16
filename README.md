# 1. Autoencoder

execution instuctions: $ python3 autoencoder.py -d <dataset_path>
  
  The program can't be executed without the path of the dataset.

---------------
  
The program trains an autoencoder to autoencode images. 
  
---------------  

Best results with hyperparameters:
  
    layer_num = 5
    filter_size = 64
    filter_num = 3
    epochs = 80
    batch_size = 256
  
![image](https://user-images.githubusercontent.com/62807134/133614581-a32ab11c-ce50-4bbd-91c0-72a05d7d5acd.png)
  
---------------
# 2. Classification
  
execution instuctions: $ python3 classification.py -d <dataset_path> -dl
<dataset_labels> -t <trainset_path> -tl <trainset_labels> -model <saved_model_name>
  
  The program can't be executed without these instructions in this specific order.

----------------
  
The program classifies the same images by using a new model which contains the autoencoder from above.
  
----------------
  
The dataset used for this project is the MNIST dataset of hand written digits.  
