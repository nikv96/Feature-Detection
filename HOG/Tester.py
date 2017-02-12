from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import warnings
import sys

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
	return new_detections

def warn(*args, **kwargs):
	pass
warnings.warn = warn

def sliding_window(image, window_size, step_size):
    for a in xrange(0, image.shape[0], step_size[1]):
        for b in xrange(0, image.shape[1], step_size[0]):
            yield (b, a, image[a:a + window_size[1], b:b + window_size[0]])

def test():

	im = imread("4.jpg", as_grey=True)
	min_wdw_sz = [40, 40]
	step_size = [10, 10]
	orientations = 9
	pixels_per_cell = [8,8]
	cells_per_block = [4,4]
	visualize = False
	normalize = True
	model_path = "models/svm.model"
	threshold = 0
	downscale = 1.25

	clf = joblib.load(model_path)

	detections = []
	scale = 0
	for im_scaled in pyramid_gaussian(im, downscale=downscale):
		cd = []
		if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
			break
		for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
			if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
				continue
			fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
			pred = clf.predict(fd)
			if pred == 1:
				detections.append((x, y, clf.decision_function(fd),
				    int(min_wdw_sz[0]*(downscale**scale)),
				    int(min_wdw_sz[1]*(downscale**scale))))
				cd.append(detections[-1])
		scale+=1

	clone = im.copy()
	detections = nms(detections, threshold)

	for (x_tl, y_tl, _, w, h) in detections:
		cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
	cv2.imshow("RESULT", clone)
	cv2.waitKey()

if __name__=="__main__":
	test()
