#cv imports
import cv2
import numpy as np

#scikit imports
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from skimage.transform import pyramid_gaussian

#python imports
import os
import glob
import warnings

#global variables
pos_features = "features/pos"
neg_features = "features/neg"
min_wdw_size = (40, 40)
step_size = [10, 10]
orientations = 9
pixels_per_cell = [8,8]
cells_per_block = [4,4]
visualize = False
normalize = True
model_path = "models/svm.model"
threshold = 0
downscale = 1.25

image_size = (40,40)
def sliding_window(image, window_size, step_size):
    for a in xrange(0, image.shape[0], step_size[1]):
        for b in xrange(0, image.shape[1], step_size[0]):
            yield (b, a, image[a:a + window_size[1], b:b + window_size[0]])

def fix_size():
	pos_im_path = "data/positive"
	neg_im_path = "data/negative"

	i=1	
	for im_path in glob.glob(os.path.join(pos_im_path, "*")):
		im = cv2.imread(im_path, 0)
		res = cv2.resize(im, image_size, interpolation = cv2.INTER_CUBIC)
		cv2.imwrite("data/pos_corrected/"+str(i)+".png",res)
		i+=1
	print "Positive features saved"
	i=1

	for im_path in glob.glob(os.path.join(neg_im_path, "*")):
		im = cv2.imread(im_path, 0)
		res = cv2.resize(im, image_size, interpolation = cv2.INTER_CUBIC)
		cv2.imwrite("data/neg_corrected/"+str(i)+".png",res)
		i+=1
	print "Negative features saved"

def feature_extractor():
	pos_im_path = "data/pos_corrected"
	neg_im_path = "data/neg_corrected"
	
	for im_path in glob.glob(os.path.join(pos_im_path, "*")):
		print im_path
		im = imread(im_path, as_grey=True)
		#HOG Descriptor
		f = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
		f_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
		f_path = os.path.join(pos_features, f_name)
		joblib.dump(f, f_path)
	print "Positive features saved"

	for im_path in glob.glob(os.path.join(neg_im_path, "*")):
		print im_path
		im = imread(im_path, as_grey=True)
		#HOG Descriptor
		f = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
		f_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
		f_path = os.path.join(neg_features, f_name)
		joblib.dump(f, f_path)
	print "Negative features saved"

def classifier_trainer(hard_neg=False):
	fds = []
	labels = []
	
	for feat_path in glob.glob(os.path.join(pos_features,"*.feat")):
		print feat_path
		fd = joblib.load(feat_path)
		fds.append(fd)
		labels.append(1)

	for feat_path in glob.glob(os.path.join(neg_features,"*.feat")):
		print feat_path
		fd = joblib.load(feat_path)
		fds.append(fd)
		labels.append(0)

	if hard_neg:
		for feat_path in glob.glob(os.path.join("features/hardneg","*.feat")):
			print feat_path
			fd = joblib.load(feat_path)
			fds.append(fd)
			labels.append(0)


	clf = LinearSVC()
	clf.fit(fds, labels)

	if not os.path.isdir(os.path.split(model_path)[0]):
		os.makedirs(os.path.split(model_path)[0])
	joblib.dump(clf, model_path)
	print "Classifier saved"

def hard_negative_mining():
	neg_im_path = "data/negative_hard"
	clf = joblib.load(model_path)
	i=260
	
	for im_path in glob.glob(os.path.join(neg_im_path, "*")):
		detections = []
		fds = {}
		im = imread(im_path, as_grey=True)
		scale = 0
		for im_scaled in pyramid_gaussian(im, downscale=downscale):
			if im_scaled.shape[0] < min_wdw_size[1] or im_scaled.shape[1] < min_wdw_size[0]:
				break
			for (x, y, im_window) in sliding_window(im_scaled, min_wdw_size, step_size):
				if im_window.shape[0] != min_wdw_size[1] or im_window.shape[1] != min_wdw_size[0]:
					continue
				fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
				pred = clf.predict(fd)
				if pred == 1:
					det = (x, y, clf.decision_function(fd), int(min_wdw_size[0]*(downscale**scale)), int(min_wdw_size[1]*(downscale**scale)))
					detections.append(det)
					fds[(det[0],det[1],det[3],det[4])] = fd
			scale+=1
		detections = nms(detections, threshold)
		for det in detections:
			f = fds[(det[0],det[1],det[3],det[4])] 
			f_name = str(i) + ".feat"
			f_path = os.path.join("features/hardneg", f_name)
			print f_path
			joblib.dump(f, f_path)
			i = i+1
		
	classifier_trainer(True)
	print "Done with Hard Negative Mining"

def warn(*args, **kwargs):
	pass
warnings.warn = warn
	

if __name__ == "__main__":
	fix_size()
	feature_extractor()
	classifier_trainer()
	hard_negative_mining()
