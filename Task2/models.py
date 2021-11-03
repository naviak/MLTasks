import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class MNISTClassifier():
    def __init__(self, model, X, y, test_size=0.25, random_state=2, fitting = True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model = model
        
        if fitting:
            self.model.fit(X_train, y_train)
            self.predictions = self.model.predict(X_test)
            self.modelScore = self.model.score(X_test, y_test)
        
        self.y_test = y_test
        self.X_test = X_test
        self.X = X
        self.y = y
        self.fitted = fitting

    
    def showWrongOnes(self, num):
        if self.fitted:
            _, ax = plt.subplots(2, 5)
            ax = ax.flatten()
            for i in range(10):
                im_idx = np.where((self.y_test != self.predictions))[0][i]
                plottable_image = np.reshape(self.X_test[im_idx], (28, 28))
                ax[i].imshow(plottable_image, cmap='gray_r')
                ax[i].title.set_text(f"pred {self.predictions[im_idx]}")
            
            
    def getCrossValidateScore(self):
        return cross_val_score(self.model, self.X, self.y, cv=5)
            

def showMnistExamples(X, y, num, ximg = 28, yimg = 28):
    _, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        im_idx = np.where(y == i)[0][num]
        plottable_image = np.reshape(X[im_idx], (ximg,yimg))
        ax[i].imshow(plottable_image, cmap='gray_r')
        
def showDistributions(X, y):
    _, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        im_idx = np.where(y == i)[0]
        X_mod = np.copy(X[im_idx])
        X_mod[X_mod > 0] = 1
        s = np.sum(X_mod, axis = 0)
        s = np.reshape(s, (28,28))
        plottable_image = np.reshape(s, (28, 28))
        ax[i].imshow(plottable_image, cmap='jet')