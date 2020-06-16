## If you like my work then please visit 
www.sdak.epizy.com

# Crowd-Counting-Using-Pytorch
Pytorch based Crowd-Counting model using SCNN_CSRNet.

## What is Crowd-Counting?
  Crowd counting is a technique to count the number of people present in the image. Counting the number of people in sparse crowd is way simpler than counting the count in dense areas where the amount of people is huge like sports stadium or any tomorrowland festival. So by this technique we can do it in few seconds.

## Why we need Crowd-Counting?
  By looking in the below image can we calculate the crowd count well yess but this is time consuming task right? This is where your deep learning skills kick in we just automate the process with the help of something known as Convolutional Neural Network or CNN. We will talk in lil depth about how we implement our model, trained and tested it.
  
  <img src="https://static.timesofisrael.com/www/uploads/2012/04/crowd-in-berlin.jpg" height = "350" width="900"/>
  
### Link for the Shanghai Dataset:
 https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0
 
### Installation:
  - Pytorch > 1.X
  - Cuda > 10.X
  - Scipy/Numpy/h5py
  - Python 3.x
### Steps to train the model:
  - First install all the required libraries 
  - Download the Shanghai Dataset and move the python files to the Dataset folder.
  - Generate the ground truth and density map with GenerateGroundnTruth.py file
    - Change the path in python script where you have downloaded your dataset.
    - Shanghai dataset has two parts Part_A and Part_B.
      - Shanghai Part_A dataset contains images that are overcrowded or more crowded
      - Shanghai Part_B dataset contains images in which the crowd is scattered or sparsed.
   - After generating the ground truth and density map you can see the following output of the images with the sum count.
   - Now the train.py file to train the model on Part_A and you will get the output weights file with .tar extension save it at your current directory.
   - Note Part_A will perform good on crowded images so for scattered crowd you have to train the Part_B dataset.
   - Now predict any random or new image on the trained model.
 <img src="https://github.com/surajdakua/Crowd-Counting-Using-Pytorch/blob/master/Density_map1.png" height = "350" width="900"/>
  
## Density Map of the following image.
 <img src="https://github.com/surajdakua/Crowd-Counting-Using-Pytorch/blob/master/Desnity_Map50.png" height = "350" width="900"/>

## Papers to make you understand the concept more clear.
    - https://arxiv.org/pdf/1907.02724v1.pdf
    - https://arxiv.org/pdf/1708.00199v2.pdf
    - https://arxiv.org/pdf/1707.09605v2.pdf
    - https://arxiv.org/pdf/1803.03095v1.pdf
## Reference and Motivation.
https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
 

 

