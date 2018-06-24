from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

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

rawImages=[]
features=[]
labels=[]

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	if i > 0 and i % 1000 == 0:
		print("{}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

model=KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(trainRI, trainRL)
acc=model.score(testRI, testRL)
print("vector feature accuracy: {:.2f}%".format(acc * 100))

model=KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(trainFeat, trainLabels)

print("histogram feature accuracy: ")
predictions=model.predict(testFeat)
print(classification_report(testLabels, predictions, target_names=le.classes_))

