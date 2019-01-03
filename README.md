# Eye contact detection
This is my bachelor semester project at the VITA laboratory (EPFL).

## Description
The project is based on the gaze estimation algorithm from [https://github.com/swook/GazeML](https://github.com/swook/GazeML). It converts the output of the gaze estimation to classify whether the person is looking at the camera or not.

## Dependencies
This project has been tested using Tensorflow 1.11 and Python 3.6.6 in an Anaconda 4.5.4 virtual environment.
You'll have to install ```coloredlogs```, ```opencv```, ```numpy```, ```cmake```, ```dlib```, ```matplotlib``` (if you want to [evaluate the algorithm](https://github.com/VFXOne/eye-contact/wiki/Evaluating-the-algorithm)), ```scipy``` and ```ujson```.

## Install
Navigate to the directory and run this command 
```
python setup.py install
```
I would recommend installing it in a virtual environment such as [Anaconda](https://www.anaconda.com).

Tensorflow is assumed to be installed, you can follow the official guide [here](https://www.tensorflow.org/install).

If you don't have any of the dependencies listed above you can run this command to install them all at once
```
pip install coloredlogs numpy scipy cmake dlib matplotlib ujson opencv-python
```

## Demo
You can run the algorithm from a webcam stream:
```
cd src
python eye-contact.py
```
Or from a video, in this case you have to provide the path of the video. For example:
```
cd src
python eye-contact.py --from_video my-video.mp4
```
To list the available commands, type
```
python eye-contact.py --help
```

## Evaluation
This algorithm has been tested on the [Columbia Gaze dataset](http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/).
The gaze is described with two angles: a vertical (theta) and a horizontal (phi) angle. Here are the results on a random sample of 1,000 images from the dataset (click on the images to expand):

Phi errors histogram | Theta errors histogram | Absolute Phi errors | Absolute Theta errors
:---:|:---:|:---:|:---:
![](https://github.com/VFXOne/eye-contact/blob/master/results/histogram_phi.png?raw=true) | ![](https://github.com/VFXOne/eye-contact/blob/master/results/histogram_theta.png?raw=true) | ![](https://github.com/VFXOne/eye-contact/blob/master/results/Phi_errors.png?raw=true) | ![](https://github.com/VFXOne/eye-contact/blob/master/results/Theta_errors.png?raw=true)

You can evaluate the algorithm yourself with different images by following the tutorial on the [wiki](https://github.com/VFXOne/eye-contact/wiki/Evaluating-the-algorithm)
