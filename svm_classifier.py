from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	return hist.flatten()

print("extracting images...")
imagePaths = list(paths.list_images("kaggle_dogs_vs_cats"))

features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = extract_color_histogram(image)
	features.append(hist)
	labels.append(label)
	if i > 0 and i % 1000 == 0:
		print("{}/{}".format(i, len(imagePaths)))

features = np.array(features)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

print("histogram feature accurracy")
# model = LinearSVC()
model = SVC()
model.fit(trainFeat, trainLabels)

acc = model.score(testFeat, testLabels)
print("{:.2f}%".format(acc * 100))
# predictions = model.predict(testFeat)
# print(classification_report(testLabels, predictions, target_names=le.classes_))


