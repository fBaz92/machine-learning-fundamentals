import scipy.stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM

class GM1D():

    def __init__(self, data: np.array, autotune=True, k=2, maximum_modes=2, max_iter = 1000, verbose=False):
        self.data = data.reshape(-1,1)
        self.maximum_modes = maximum_modes
        self.max_iter = max_iter
        self.verbose = verbose
        if autotune:
            self.k = self.get_number_modes()
        else:
            self.k = k
        self.gmm = GMM(n_components=self.k, random_state=0).fit(self.data)
        self.mean = self.gmm.means_
        self.covariances = self.gmm.covariances_
        self.weights = self.gmm.weights_


    def get_number_modes(self):
        # first of all, let's confirm the optimal number of components
        bics = []
        min_bic = 0
        counter=1
        for i in range(self.maximum_modes): # test the AIC/BIC metric between 1 and 10 components
            gmm = GMM(n_components = counter, max_iter=self.max_iter, random_state=0, covariance_type = 'full')
            labels = gmm.fit(self.data).predict(self.data)
            bic = gmm.bic(self.data)
            bics.append(bic)
            if bic < min_bic or min_bic == 0:
                min_bic = bic
                opt_bic = counter
            counter = counter + 1

        if self.verbose:
            # plot the evolution of BIC/AIC with the number of components
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1,2,1)
            # Plot 1
            plt.plot(np.arange(1,self.maximum_modes+1), bics, 'o-', lw=3, c='black', label='BIC')
            plt.legend(frameon=False, fontsize=1)
            plt.xlabel('Number of components', fontsize=20)
            plt.ylabel('Information criterion', fontsize=20)
            plt.xticks(np.arange(0,11,2))
            plt.title('Opt. components = '+str(opt_bic), fontsize=20)

        return opt_bic

    def pdf(self, x):
        tmp = 0
        for index in range(self.k):
            tmp += norm.pdf(x, self.mean[index,0], np.sqrt(self.covariances[index,0]))*self.weights[index]
        return np.log(tmp[0])

    def plot_pdf_modeled(self, n_bins=25):
        x_data = np.linspace(np.min(self.data),np.max(self.data), n_bins+1)
        data_ = [np.exp(self.pdf(d)) for d in x_data]

        plt.hist(self.data,density=True,bins=n_bins)
        plt.plot(x_data, np.array(data_))


class BayesianClassifier():
    def __init__(self, data: np.ndarray, labels: list, normalize = False, verbose=False):
        if normalize:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(data)
        else:
            self.data = data
        self.labels = np.array(labels)
        self.n_classes = len(set(labels))
        self.unique_labels = {code: label for code, label in zip(range(self.n_classes), set(labels))}
        self.n_features = len(data[0])
        self.prior = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
        self.std_ = np.zeros((self.n_classes, self.n_features))
        #self.likelihood = {class_: [] for class_ in self.unique_labels}
        self.mixture_model = [GM1D(data=self.data[:,i], verbose=verbose) for i in range(self.n_features)]
        self.train()

    def train(self):
        for i in range(self.n_classes):
            self.prior[i] = np.log(np.sum(self.labels == self.unique_labels[i]) / self.labels.shape[0])
            for j in range(self.n_features):
                # I create the likelihood model for each feature
                self.means[i][j] = np.mean(self.data[self.labels == self.unique_labels[i], j])
                self.std_[i][j] = np.std(self.data[self.labels == self.unique_labels[i], j])
                #each entry is a pdf object with the mean and std of the feature
                #self.likelihood[class_].append(lambda x: np.log(scipy.stats.norm(self.means[i][j], self.std_[i][j]).pdf(x)))

    def predict(self, data:np.ndarray, weights=None, y_true = None, use_mixture_model = True):
        predictions = []
        prob_dataframe = pd.DataFrame(columns=[code for code in self.unique_labels.keys()])
        for i in tqdm(range(data.shape[0])):
            observation = data[i]
            if use_mixture_model:
                prediction, probabilities = self.predict_single(data=observation, use_mixture_model=True, weights=weights)
            else:
                prediction, probabilities = self.predict_single(data=observation)
            prob_dataframe = pd.concat([prob_dataframe, pd.DataFrame(data=probabilities.reshape(1,-1))], ignore_index=True)
            predictions.append(prediction)
        prob_dataframe = prob_dataframe.rename(columns=self.unique_labels)
        if y_true is not None:
            score = accuracy_score(y_true=y_true,y_pred=predictions)
            print("The score of the classifier is: {}".format(score))
        return (predictions, prob_dataframe)

    def predict_single(self, data:np.ndarray, use_mixture_model=True, weights=None):
        probabilities = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            #add the prior at the estimate
            probabilities[i] = self.prior[i]
            for j in range(self.n_features):
                #calcluate the likehood for this class and this feature
                if use_mixture_model:
                    calculations = self.mixture_model[j].pdf(data[j])
                else:
                    calculations = np.log(
                        scipy.stats.norm(self.means[i][j], self.std_[i][j]).pdf(data[j])
                    )
                if weights is not None:
                    #rescale the calculations: if a feature has an higher eigenvalue the associated
                    #contribute will be smaller and, in global, it will have an higher impact in the
                    #classification.
                    calculations = calculations/weights[j]

                probabilities[i] += calculations
        index = np.argmax(probabilities)
        return (self.unique_labels[index], probabilities)