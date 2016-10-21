# Feature-Detection
Feature Detection using OpenCV-python and scikit-learn. Feature detection is implemented using both [Haar-cascades](https://github.com/nikv96/Feature-Detection/tree/master/Haar) and [HOG descriptor](https://github.com/nikv96/Feature-Detection/tree/master/HOG) methods.

# How to Run
1. Install all dependencies by running install.sh
2. Training
  1. Haar-cascades:
    1. Add positive and negative images to info/ and neg/ respectively
    2. Create descriptions file by commenting out respective function calls in HaarCascades.py and running ```python HaarCascades.py```
    3. Run ```opencv_createsamples -info positive.txt -num 500 -w 40 -h 40 -vec positive.vec``` to create the vector file
    4. Run ```opencv_traincascade -data data -vec positive.vec -bg negative.txt -numPos 500 -numNeg 500 -numStages 10 -w 40 -h 40``` to train
  2. HOG:
    1. Add positive and negative images to data/positive, data/negative and hard negative tests to data/negative_hard respectively.
    2. Run Trainer with ```python Trainer.py```
3. Testing
  1. Haar-cascades:
    1. Run ```python Haar-cascades.py```
  2. HOG:
    2. Run ```python Tester.py```
    
# Contributors
1. [Nikhil Venkatesh](https://github.com/nikv96)
2. [Nicholas Adrian](https://github.com/nicholasadr)

# References
1. [Creating your own Haar Cascade OpenCV Python Tutorial](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/)
2. [Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
3. [(Faster) Non-Maximum Suppression in Python](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
4. [Sliding Windows for Object Detection with Python and OpenCV](http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
