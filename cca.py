# -*- coding: utf-8 -*-
import numpy
import scipy.linalg
import sklearn.cross_decomposition
import sklearn.metrics


class LinearCCA(object):
    def __init__(self, n_components):
        self._n_components = n_components
        self._wx = None
        self._wy = None

    def fit(self, X, Y):
        """ fit the model

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        assert X.shape[0] == Y.shape[0],\
            "number of samples of X and Y should be the same"

        # calculate covariance matrices
        Cxx = numpy.dot(X.T, X)
        Cxy = numpy.dot(X.T, Y)
        Cyy = numpy.dot(Y.T, Y)
        Cyy_inv = numpy.linalg.inv(Cyy)

        # solve generalized eigenvalue problem
        A = Cxy.dot(Cyy_inv).dot(Cxy.T)
        eig, wx = scipy.linalg.eigh(A, Cxx)
        eig = numpy.real(eig)
        wx = wx[:, eig > 0]
        eig = eig[eig > 0]

        idx = numpy.argsort(eig)[::-1]

        eig = eig[:self._n_components]
        self._wx = wx[:, idx[:self._n_components]]
        self._wy = Cyy_inv.dot(Cxy.T).dot(self._wx)
        self._wy /= numpy.sqrt(eig)

        return

    def fit_transform(self, X, Y):
        self.fit(X, Y)

        return self._wx.T.dot(X.T).T, self._wy.T.dot(Y.T).T


class KernelCCA(object):

    def __init__(self, n_components, kernel='linear', kernel_params=[],
                 nystrom_approximation=False, reg_param=0.1):
        self._n_components = n_components
        self._reg_param = reg_param
        self._alpha = None
        self._beta = None
        self._X = None
        self._Y = None

        self._nystrom_approximation = nystrom_approximation

        if kernel == 'linear':
            self._kernel = linear_kernel
        elif kernel == 'rbf':
            self._kernel = lambda x, y: rbf(x, y, *kernel_params)
        elif callable(kernel):
            self._kernel = lambda x, y: kernel(x, y, *kernel_params)

    def fit(self, X, Y):
        num_samples = X.shape[0]
        self._X = X
        self._Y = Y

        Kx = sklearn.metrics.pairwise_distances(X, metric=self._kernel)
        Ky = sklearn.metrics.pairwise_distances(Y, metric=self._kernel)

        Z = numpy.zeros(shape=(num_samples, num_samples))
        A = numpy.block([[Z, Kx.dot(Ky)], [Ky.dot(Kx), Z]])
        B = numpy.block([
            [Kx.dot(Kx) + self._reg_param * Kx, Z],
            [Z, Ky.dot(Ky) + self._reg_param* Ky]])

        # # solve generalize eigenvalue problem
        # A = Ky.dot(numpy.linalg.inv(
        #     Ky + self._reg_param * numpy.eye(Ky.shape[0]))).dot(Kx)
        # B = (Kx + self._reg_param * numpy.eye(Kx.shape[0]))
        eig, coef = scipy.linalg.eig(A, B)
        eig = numpy.real(eig)
        coef = coef[:, eig > 0]
        eig = eig[eig > 0]

        idx = numpy.argsort(eig)[::-1]
        eig = eig[:self._n_components]
        self._alpha = coef[:num_samples, idx[:self._n_components]]
        self._beta = coef[num_samples:, idx[:self._n_components]]

        return self._alpha.T.dot(Kx), self._beta.T.dot(Ky)

    def predict(self, X, Y):
        Kx = sklearn.metrics.pairwise_distances(self._X, X, metric=self._kenel)
        Ky = sklearn.metrics.pairwise_distances(self._Y, Y, metric=self._kernel)

        return self._alpha.T.dot(Kx), self._beta.T.dot(Ky)


def linear_kernel(x, y):
    return x.dot(y)


def rbf(x, y, sigma):
    return numpy.exp(-((x - y)**2).sum() / (2 * sigma**2))


if __name__ == '__main__':
    num_samples = 400
    x_dim = 2
    y_dim = 2

    noise1 = numpy.random.normal(size=num_samples)
    noise2 = numpy.random.normal(size=num_samples)
    u = numpy.arange(num_samples)
    u = (u % 80) / 80
    # u = numpy.repeat(numpy.array([0, 1, 2, 1, 0, 1, 2, 1]), 50)

    X = numpy.zeros(shape=(num_samples, x_dim))
    X[:, 0] = noise1 + u * 0.1
    X[:, 1] = -noise1 + u * 0.1
    Y = numpy.zeros(shape=(num_samples, y_dim))
    Y[:, 0] = noise2 + u * 0.1
    Y[:, 1] = -noise2 + u * 0.1

    model = KernelCCA(n_components=2, kernel='rbf', kernel_params=[0.1, ])
    X2, Y2 = model.fit(X, Y)
