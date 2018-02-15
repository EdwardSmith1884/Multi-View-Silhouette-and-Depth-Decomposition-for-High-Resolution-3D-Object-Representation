# 3D Object Super-Resolution
This is a repository to reproduce the method from the paper "3D Object Super-Resolution". The code here allows one to train models to accuractly and efficently upsample voxelized objects to high resolutions, and to reconstruct voxelized objects from single RGB images at high resolutions.


<p align="center">
  <img  src="images/SR.png" width="512" >
</p>

Example super-resolution results, when increasing from 32^3 resolution to 256^3 resolution. The first row is the low resolution inputs and the second row is the corresponing high resolution predictions. 

## Super-Resolution
The first element of this repo is the 3D super-resolution method. For this method 6 primary orthographic depth maps(ODM) are extracted from the low resolution object, and passed through our image super resolution method to produce a prediction for the corresponding high resolution object's 6 primary OMDs. The predicted ODM's are then used to carve away at the low resolution object(upsampled ot the higher resolution using nearest nighbor interpolation), to produce an estimate for the high resolution object. 

![Diagram](images/SRMethod.png?raw=true "Title")
Intuitive Diagram to understand the 3D super-resolution method. 

The image super-resolution technique used to predict high resolution odms, makes use to two deep convolutional neural networks. The first network, outlined in 'depth.py', estimates only the new depths into the known surface of the low resolution object  within a predefined range. The output of this network is added the corresponing low resolution odms(upsampled to the higher resolution), to produce a complete estimate for the depths to the new objects surface. The second network, outlined in 'occupancy.py", predicts an occupancy map for the high resolution odm, basically predicting silhouettes for the predicted object. The outputs of these two networks are combined to produce a rough estimate of the high reoslution ODM, and then smoothed to produce a final prediction. 

![Diagram](images/DepthPipeLine.png?raw=true "Title")
Intuitive Diagram to understand the 3D super-resolution method. 

## Reference:
please cite my paper: ,if you use this repo for research 
