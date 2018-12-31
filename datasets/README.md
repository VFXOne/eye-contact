# UnityEyes

UnityEyes is a synthetic dataset used to train the ELG CNN for eye region landmarks localization.

The published code and software can be found at https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/

Please download the software and generate images such that there exists the folder `imgs/` when you are done.

The more images you generate, the more robust the trained neural network would be.

# Columbia Gaze
This is the dataset used to evaluate the algorithm in this project. It features over 5,880 images of 56 people in laboratory conditions. There are 5 different horizontal gaze angles and 3 different vertical angles.

Even if there is not much different gaze directions, it is sufficient for evaluation purposes.

You can download the dataset [here](http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)

# MPIIGaze
This dataset can be used to evaluate the gaze estimation. Unfortunately feeding some images into the algorithm directly won't work because it requires an image containing a fully visible face. 

The dataset is listed here because it is one of the biggest and the images are taken in unconstrained settings (i.e. "in the wild").

This dataset can be found [here](https://www.mpi-inf.mpg.de/de/abteilungen/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/).
