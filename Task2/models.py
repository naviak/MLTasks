import numpy as np
import matplotlib.pyplot as plt


class MNISTClassifier():
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.model.fit(X_train, y_train)
        self.predictions = self.model.predict(X_test)
        self.modelScore = self.model.score(X_test, y_test)
        self.wrongArgs = np.where(self.predictions != y_test)[0]
        
        self.y_test = y_test
        self.X_test = X_test

    
    def showWrongOnes(self, num):
        _, ax = plt.subplots(2, 5)
        ax = ax.flatten()
        for i in range(10):
            im_idx = np.where((self.y_test == i) & (self.y_test != self.predictions))[0][num]
            plottable_image = np.reshape(self.X_test[im_idx], (28, 28))
            ax[i].imshow(plottable_image, cmap='gray_r')
            ax[i].title.set_text(f"pred {self.predictions[im_idx]}")
            
            
def showMnistExamples(X, y, num):
    _, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        im_idx = np.where(y == i)[0][num]
        plottable_image = np.reshape(X[im_idx], (28, 28))
        ax[i].imshow(plottable_image, cmap='gray_r')