# Image Classification of Surgical Instruments
This project is a practical one that compares the differences between the various deep learning methods. through out the base model (ResNet50,VGG-16) to add attention and fine-grained classification model, this project illustrates comparison of the efficacy of the different models. 
## Surgical Instruments Dataset
- Dataset from : https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools
- In total it contains 3009 images and the respective labels classifying the objects as Scalpel, Straight Dissection Clamp, Straight Mayo Scissor or Curved Mayo Scissor in this dataset contain labeled data which use for train and predict bounding box for Image Detection
- But in this work I just use Alone image in each class for classification only , It's mean just use only 2010 image for train/val/test for classification by ratio 0.7/0.1/0.2<br>

![ImageClass](https://github.com/tanutb/onborad/blob/main/img/class_image.png)
### Obstacle 
These two classes have a similar shape, with the only difference being the end of the scissor. 
The Curved Mayo Scissor has a curved end while the Straight Mayo Scissor has a straight end. 
According to blue box in image above.
To classify objects that look similar, the task would fall under the domain of fine-grained 
classification.

![ImageClass](https://github.com/tanutb/onborad/blob/main/img/obstacle.png)

## Image Classification 
### Base state of the art model
- using ResNet50
- using VGG16
### Attention model
- SEResNet50 that is ResNet50 with Squeeze-and-Excitation Networks 
### Fine-Grained Image Classification
- Bilinear Pooling to complexier
### Fine-Grained Image Classification with Attention model
- WSDAN(Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification) <br>
In Abstract said “WS-DAN improves the classification accuracy in two folds. In the first 
stage, images can be seen better since more discriminative parts’ features will be extracted. In the 
second stage, attention regions provide accurate location of object, which ensures our model to look 
at the object closer and further improve the performance.”
- model from https://github.com/GuYuc/WS-DAN.PyTorch
## Result
WSDAN Get the best performance over the other model in F1score 
![F1score](https://github.com/tanutb/onborad/blob/main/img/Result_F1score.png)
## Visualize WSDAN
The models can localize the object. And can generate attention maps to represent the 
object’s discriminative parts.
![Visual1](https://github.com/tanutb/onborad/blob/main/img/visualize1.png)
Some image -models cant localize object well due to it’s have reflection of light and 
something but model still can classify this class correctly maybe it’s because of dataset have small 
number of data classes
![Visual2](https://github.com/tanutb/onborad/blob/main/img/visualize2.png)
