'''
This program allows you to train and test a HAAR based classifier.

Call the function using

	python FeatureDetection.py <<method>> <<argument>>

List of <<method>>:
1. storeRawImages
	This method can be used to retrieve raw images from a list of URLs. The list of URLs have to be stored in a txt file called img_links.txt in the root directory of the repository. Provide argument as "pos" or "neg" for imageType.
2. convertAndSave
	This method is used to resize the images and save them for uniformity. Provide argument as "pos" or "neg" for imageType. 
3. createDataFiles
	This method is used to create a list of all the positive and negative images in info.dat and bg.txt files.
4. getTargetPosition
	This method is used to test the code. Provide argument as "video" or "image" to test.

Train HAAR classifier:
1. Collect positive and negative images using the storeRawImages() function.
2. Create the bg.txt and info.dat file using function createDataFiles().
3. Create vector file using ```opencv_createsamples -info info/info.lst -num 1950 -w 20 -h 20 -vec positives.vec```
4. Run classifier trainer using ```opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20```
'''

import urllib
import cv2
import numpy as np
import os

def storeRawImages(image_type):
    f = open("img_links.txt",'r')
    image_urls = f.read()
    pic_num = 1
    
    if not os.path.exists(image_type):
        os.makedirs(image_type)
        
    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, image_type+"/"+str(pic_num)+".jpg")
            img = cv2.imread(image_type+"/"+str(pic_num)+".jpg")
            resized_image = cv2.resize(img, (320, 240))
            cv2.imwrite(image_type+"/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))
    f.close()

def convertAndSave(image_type):
    pic_num=1
    for img in os.listdir(image_type):
        img = cv2.imread(image_type+"/"+str(pic_num)+".jpg")
        resized_image = cv2.resize(img, (320, 240))
        cv2.imwrite(image_type+"/"+str(pic_num)+".jpg",resized_image)
        pic_num += 1

def createDataFiles():
    for file_type in ['pos','neg']:
        for img in os.listdir(file_type):
            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 320 240\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

def getTargetPosition(option):
	'''
	This function searches the video stream for the target and returns the position

	input: 
	output:
	'''

	target_cascade = cv2.CascadeClassifier('data/cascade.xml')
	if option == "image":
		img = cv2.imread("sample.jpg")

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		target = target_cascade.detectMultiScale(gray, 1.3, 5, minSize = (200,100))

		print target

		# add this
		for (x,y,w,h) in target:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
		cv2.imshow('RESULT',img)
		k = cv2.waitKey() & 0xff
		while not k == ord('q'):
			pass
	else:
		cap = cv2.VideoCapture(1)

		cv2.namedWindow("RESULT",cv2.WINDOW_NORMAL)

		while True:
			ret, img = cap.read()
			if ret == False:
				print("Stream has ended")
				break
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#Modify the cars array every 25 frames
			target = target_cascade.detectMultiScale(gray,1.01, 5)
		
			for (x,y,w,h) in target:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		
			cv2.imshow('RESULT',img)
			k = cv2.waitKey(1) & 0xff
			if k == ord('q'):
				break

		cap.release()
	
	cv2.destroyAllWindows()

if __name__=='__main__':
	import sys
	print(__doc__)
	if len(sys.argv) > 1:
		opt = sys.argv[1]
		if opt == "storeRawImages":
			if len(sys.argv) > 2 :
				image_type = sys.argv[2] 
			else:
				image_type = "pos"
			storeRawImages(image_type)
		elif opt == "convertAndSave":
			if len(sys.argv) > 2:
				image_type = sys.argv[2] 
			else:
				image_type = "pos"
			convertAndSave(image_type)
		elif opt == "createDataFiles":
			createDataFiles()
		elif opt == "getTargetPosition":
			getTargetPosition(sys.argv[2])
		else:
			print("Invalid Option")
	else:
		print("Argument not provided. Exiting program now.")

