'''
Steps to train HaarCascades classifier:

1. Collect positive and negative images using the store_raw_images() function
2. Have to create the bg.txt and info.dat file using function create_pos_n_neg().
3. Create positive samples using 
	```opencv_createsamples -img watch5050.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950```
4. Create vector file using 
	```opencv_createsamples -info info/info.lst -num 1950 -w 20 -h 20 -vec positives.vec```
5. Run classifier trainer using 
	```opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20```
6. Copy the cascade file from data/ to /
'''
import urllib
import cv2
import numpy as np
import os

def store_raw_images():
    #http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513
    #http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152
    #http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513
    f = open("img_links.txt",'r')
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04606574'  
    neg_image_urls = f.read()
    pic_num = 1
    
    if not os.path.exists('pos'):
        os.makedirs('pos')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "pos/"+str(pic_num)+".jpg")
            img = cv2.imread("pos/"+str(pic_num)+".jpg")
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("pos/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))
    f.close()

def create_pos_n_neg():
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

def get_target_pos():
	'''
	This function searches the video stream for the target and returns the position

	input: 
	output:
	'''

	target_cascade = cv2.CascadeClassifier('cascade.xml')
	#Uncomment to test with image
	'''
	img = cv2.imread("sample.jpg")

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	target = target_cascade.detectMultiScale(gray, 1.01, 3)

	print target

	# add this
	for (x,y,w,h) in target:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
	cv2.imshow('RESULT',img)
	k = cv2.waitKey() & 0xff
	while not k == ord('q'):
		pass
	'''

	#Comment when using video sample
	cap = cv2.VideoCapture(1)

	#Uncomment to see video output
	#cv2.namedWindow("RESULT",cv2.WINDOW_NORMAL)

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
		
		#Uncomment to see video output in gui
		cv2.imshow('RESULT',img)
		k = cv2.waitKey(1) & 0xff
		if k == ord('q'):
			break

	cap.release()
	
	cv2.destroyAllWindows()


get_target_pos()
#store_raw_images()
#create_pos_n_neg()
