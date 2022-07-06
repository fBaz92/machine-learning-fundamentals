import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from pathlib import Path
from tqdm import tqdm
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class BagOfWords():

    def __init__(self, A_, y_train, target_w=150, channels=4, k = 50, c_params = [1, 10, 100], gamma_params = [0.01, 0.1, 1, 10]):
        self.A_ = A_
        self.y_train = y_train
        self.target_w = target_w
        self.channels = channels
        self.n_images_paths, self.n_features = self.A_.shape
        self.k = k
        self.sift = cv2.SIFT_create()
        self.des_list, self.kp_list, self.descriptors_float = self.get_descriptors(A_=self.A_)
        self.voc, self.variance = kmeans(self.des_list, self.k, 20)
        self.features = self.vector_quantization()
        self.c_params = c_params
        self.gamma_params = gamma_params
        self.model = self.train_model()


    def get_descriptors(self, A_):
        des_list = []
        kp_list = []
        images, features = A_.shape
        for i in tqdm(range(images)):
            im = A_[i,:]
            im = im.reshape([-1, self.target_w, self.channels])
            im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            kp = self.sift.detect(im,None)
            keypoints,descriptor= self.sift.compute(im, kp)
            des_list.append(descriptor)
            kp_list.append(keypoints)

        descriptors = des_list[0]
        for descriptor in des_list[1:]:
            descriptors = np.concatenate((descriptors, descriptor))
        descriptors_float = descriptors.astype(float)
        return des_list, kp_list, descriptors_float

    def draw_keypoints(self, index=0, color=(255, 255, 255)):
        keypoints = self.kp_list[index]
        im = self.A_[index,:]
        for kp in tqdm(keypoints):
            x, y = kp.pt
            plt.imshow(cv2.circle(im, (int(x), int(y)), 2, color))

    def vector_quantization(self):
        features = np.zeros((self.n_images_paths, self.k), "float32")
        for i in tqdm(range(self.n_images_paths)):
            try:
                # Use the codebook to assign each observation to a cluster via vector quantization
                # labels, distance = vq(dataset, codebook)
                image_words, distance = vq(self.des_list[i], self.voc)
                for w in image_words:
                    features[i][w] += 1
            except:
                # if the image has no image_words, continue (the row will be all zeros)
                continue
        return features

    def train_model(self):

        from sklearn.svm import SVC
        features = self.features - np.mean(self.features, axis=0)
        clf = SVC(kernel='rbf', class_weight='balanced')
        param_grid = dict(gamma=self.gamma_params, C=self.c_params)
        grid = GridSearchCV(clf, param_grid)
        grid.fit(features, self.y_train)
        return grid.best_estimator_


    def predict(self, A_test, y_test):
        des_list, _, _ = self.get_descriptors()
        n_images_path, _ = A_test.shape
        im_test_features = self.vector_quantization(n_images_paths=n_images_path, descriptors_list=des_list, k=self.k)

        yfit = self.model.predict(im_test_features)
        print(classification_report(y_test, yfit))
        return yfit
