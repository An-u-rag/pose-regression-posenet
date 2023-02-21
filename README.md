# pose-regression-posenet
Pytorch implementation of PoseNet with pretrained weights. 

Regression of pose i.e. 3D position and orientation of camera given multiple RGB images representing the a similar scene. 
PoseNet is a convolutional neural network that utilizes pretrained weights from InceptionV1 blocks and Googlenet architecture in combination with 3 loss headers to regress 3 positional coordinates x, y and z of the camera and 4 quaternion values w, p, q and r. 
The pretrained wights of InceptionV1 blocks are on the Places dataset and this project tries to make the model predict on the King's College dataset. 


