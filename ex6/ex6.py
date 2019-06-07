import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from nltk.stem import PorterStemmer


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.square(np.linalg.norm(x1 - x2)) / (2 * np.square(sigma)))


def plot_boundary(X, model, linear=False):
    if linear:
        w1, w2 = model.coef_.ravel()
        b = model.intercept_[0]
        return -(w1 * x + b) / (w2)
    else:
        X1, X2 = np.meshgrid(X[0], X[1])
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = model.predict(this_X)
        return vals


def process_email(email):
    email = re.sub(r'<[^<>]+>', ' ', email)
    email = re.sub(r'[\d]+', 'number', email)
    email = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email)
    email = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub(r'[\$]+', 'dollar', email)
    email = re.split(r'[ @\$\/#\.\-\:\&\*\+=\[\]\?!\(\)\{\},\'\">_<;%]', email)
    email = [re.sub('[^a-zA-Z0-9]', '', s) for s in email]
    email = [ps.stem(w.strip()) for w in email]
    return email


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Linear Kernel ====================
    data = loadmat('ex6data1.mat')
    X, y = (data['X'], data['y'].ravel())
    model = svm.SVC(C=1, kernel='linear', tol=1e-3)
    model.fit(X, y)

    x = np.arange(min(X[:, 0]), max(X[:, 1]))
    plt.plot(X[np.argwhere(y == 0), 0], X[np.argwhere(y == 0), 1], 'yo')
    plt.plot(X[np.argwhere(y == 1), 0], X[np.argwhere(y == 1), 1], 'k+')
    plt.plot(x, plot_boundary(x, model, linear=True), 'b-')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()

    # ==================== Part 1.2 Gaussian Kernel ====================
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    assert np.round(gaussian_kernel(x1, x2, sigma), decimals=6) == 0.324652

    data = loadmat('ex6data2.mat')
    X, y = (data['X'], data['y'].ravel())
    C = 1
    sigma = 0.1
    gamma = (1 / (2 * np.square(sigma)))
    model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(X, y)

    plt.plot(X[np.argwhere(y == 0), 0], X[np.argwhere(y == 0), 1], 'yo')
    plt.plot(X[np.argwhere(y == 1), 0], X[np.argwhere(y == 1), 1], 'k+')
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    vals = plot_boundary([x1, x2], model)
    plt.contour(x1, x2, vals, [0.5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()

    data = loadmat('ex6data3.mat')
    X, y = (data['X'], data['y'].ravel())
    X_val, y_val = (data['Xval'], data['yval'].ravel())
    params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    vals = np.zeros((len(params), len(params)))
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            gamma = (1 / (2 * np.square(p2)))
            model = svm.SVC(C=p1, kernel='rbf', gamma=gamma)
            model.fit(X, y)
            vals[i, j] = np.mean(np.where(model.predict(X_val) == y_val, 0, 1))
    c = np.argmin(vals) // vals.shape[1]
    s = np.argmin(vals) % vals.shape[1]
    C = params[c]
    sigma = params[s]
    gamma = (1 / (2 * np.square(sigma)))
    model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(X, y)

    plt.plot(X[np.argwhere(y == 0), 0], X[np.argwhere(y == 0), 1], 'yo')
    plt.plot(X[np.argwhere(y == 1), 0], X[np.argwhere(y == 1), 1], 'k+')
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    vals = plot_boundary([x1, x2], model)
    plt.contour(x1, x2, vals, [0.5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()

    # ==================== Part 2.1 Preprocessing ====================
    vocab = {}
    ps = PorterStemmer()
    with open('vocab.txt', 'r') as f:
        vocab = {l.rstrip().split('\t')[1]: int(
            l.rstrip().split('\t')[0]) for l in f}
        n = len(vocab)
    vocab_i = {v: k for k, v in vocab.items()}

    with open('emailSample1.txt', 'r') as f:
        email = f.read().replace('\n', ' ').lower()
    email = process_email(email)

    words = np.array([vocab[w] for w in email if w in vocab and len(w) > 1])
    data = loadmat('word_indices.mat')
    word_incides = data['word_indices'].ravel()
    assert np.array_equal(words, word_incides)

    # ==================== Part 2.2 Extracting Features ====================
    features = np.zeros(n)
    features[words] = 1
    assert int(sum(features)) == 45

    # ==================== Part 2.3 Training an SVM ====================
    data = loadmat('spamTrain.mat')
    X, y = (data['X'], data['y'].ravel())
    data = loadmat('spamTest.mat')
    X_test, y_test = (data['Xtest'], data['ytest'].ravel())
    model = svm.SVC(C=0.1, kernel='linear')
    model.fit(X, y)
    p = model.predict(X)
    assert np.mean(np.where(p == y, 1, 0)) == 0.99825
    p_test = model.predict(X_test)
    assert np.mean(np.where(p_test == y_test, 1, 0)) == 0.989

    # ==================== Part 2.4 Top Predictors ====================
    w = model.coef_.ravel()
    w_s = np.sort(model.coef_.ravel())
    print('Top 15 predictors of spam:')
    for i in range(1, 16):
        print('{0:10s}\t({1:0.6f})'.format(
            vocab_i[np.argwhere(w == w_s[-i]).ravel()[0] + 1], w_s[-i]))
    print('')

    # ==================== Part 2.5 Sample Classification ====================
    samples = ['emailSample1.txt', 'emailSample2.txt',
               'spamSample1.txt', 'spamSample2.txt']
    print('Sample classifications:')
    for s in samples:
        with open(s, 'r') as f:
            email = f.read().replace('\n', ' ').lower()
        email = process_email(email)
        words = np.array([vocab[w]
                          for w in email if w in vocab and len(w) > 1])
        features = np.zeros(n)
        features[words] = 1
        p = model.predict(features.reshape(1, -1))[0]
        print('{0:20s} {1}'.format(s, p))
