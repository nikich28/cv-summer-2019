import numpy as np
from sklearn.svm import LinearSVC
from scipy.ndimage import convolve1d
from skimage.transform import resize
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def ang(x):
    binCount = 9
    if x == np.pi:
        return binCount - 1
    return int(binCount * x / np.pi)

def extract_hog(img):
    img = resize(img, (64, 64))
    br = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    i_x = convolve1d(br, np.array([1, 0, -1]), axis=1, mode='wrap')
    i_y = convolve1d(br, np.array([1, 0, -1]), axis=0, mode='wrap')
    grad = (i_x ** 2 + i_y ** 2) ** (1 / 2)
    th = abs(np.arctan2(i_y, i_x))
    
    #histogram
    h, w = grad.shape
    cellRows = 8
    cellCols = 8
    binCount = 9
    hist = np.zeros((h // cellRows, w // cellCols, binCount))
    for y in range(h // cellRows):
        for x in range(w // cellCols):
            for i in range(y*cellRows, (y+1)*cellRows):
                for j in range(x*cellCols, (x+1)*cellCols):
                    hi = ang(th[i][j])
                    hist[y][x][hi] += grad[i][j]
                    
    #blocks
    eps = 1e-10
    blockRowCells = 2
    blockColCells = 2
    h, w, c = hist.shape
    h = h - blockRowCells + 1
    w = w - blockColCells + 1
    conc = []
    for y in range(h):
        for x in range(w):
            vect = []
            for i in range(y, y + blockRowCells):
                for j in range(x, x + blockColCells):
                    vect = np.append(vect, hist[i][j])
            conc = np.append(conc, vect / ((np.dot(vect, vect) + eps) ** (1 / 2)))
    return conc

def fit_and_classify(train_features, train_labels, test_features):
    clf = LinearSVC()
    for i in range(5):
        input_train, input_test, ans_train, ans_test = train_test_split(train_features, train_labels, train_size = 0.80)
        clf.fit(input_train, ans_train)
    #clf.fit(train_features, train_labels)
    #cross_val_score(clf, train_features, train_labels, cv=5)
    return clf.predict(test_features)