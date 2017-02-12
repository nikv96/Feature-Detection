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

def overlapping_area(detection_1, detection_2):
	x1_tl = detection_1[0]
	x2_tl = detection_2[0]
	x1_br = detection_1[0] + detection_1[3]
	x2_br = detection_2[0] + detection_2[3]
	y1_tl = detection_1[1]
	y2_tl = detection_2[1]
	y1_br = detection_1[1] + detection_1[4]
	y2_br = detection_2[1] + detection_2[4]
	x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
	y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
	overlap_area = x_overlap * y_overlap
	area_1 = detection_1[3] * detection_2[4]
	area_2 = detection_2[3] * detection_2[4]
	total_area = area_1 + area_2 - overlap_area
	return overlap_area / float(total_area)

def nms(detections, threshold=.5):
	if len(detections) == 0:
		return []
	detections = sorted(detections, key=lambda detections: detections[2],reverse=True)
	new_detections=[]
	new_detections.append(detections[0])
	del detections[0]
	for index, detection in enumerate(detections):
		for new_detection in new_detections:
			if overlapping_area(detection, new_detection) > threshold:
				del detections[index]
				break
		else:
			new_detections.append(detection)
			del detections[index]

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
