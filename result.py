# using SVM and KNN to classifier a image
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())
print "loading models..."
model = pickle.loads(open("model_svc.cpickle", "rb").read())

print "extracting image..."
img = cv2.imread(args["image"])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])

print "predicting..."
hist = hist.reshape(1, -1)
label = model.predict(hist)[0]

label = "{}".format(label)
if label == "1":
    label = "Dog"
else:
    label = "Cat"

output = imutils.resize(img, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

