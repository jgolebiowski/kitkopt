import numpy as np
import unittest
from blackop.kernels import poly_kernel, rbf


class KernelTest(unittest.TestCase):
    mat1 = np.array([[0.533426, 0.59575161, 0.50389644],
                     [0.62640336, 0.49767812, 0.31952826]])
    mat2 = np.array([[0.96380476, 0.83472723, 0.13314283],
                     [0.99353882, 0.07326699, 0.91934838],
                     [0.77286147, 0.97525652, 0.57766557],
                     [0.30939199, 0.12165684, 0.75722879]])

    def test_polynomial(self):
        deg = 1
        gamma = 2
        coef0 = 3

        result11 = poly_kernel(self.mat1, self.mat1, deg, gamma, coef0)
        result22 = poly_kernel(self.mat2, self.mat2, deg, gamma, coef0)
        result12 = poly_kernel(self.mat1, self.mat2, deg, gamma, coef0)

        target11 = np.array([[4.7867498, 4.58328307],
                             [4.58328307, 4.48432598]])
        target22 = np.array([[6.28683235, 5.28228008, 6.27174553, 4.00112667],
                             [5.28228008, 6.67537776, 5.74079578, 5.02492689],
                             [6.27174553, 5.74079578, 6.76427528, 4.59037755],
                             [4.00112667, 5.02492689, 4.59037755, 4.36783846]])
        target12 = np.array([[5.15699761, 5.07376948, 5.56871734, 4.23815976],
                             [5.12339783, 4.90515444, 5.30813465, 3.99261225]])

        np.testing.assert_almost_equal(result11, target11, decimal=5)
        np.testing.assert_almost_equal(result22, target22, decimal=5)
        np.testing.assert_almost_equal(result12, target12, decimal=5)

    def test_rbf(self):
        length = 2
        sigma_f = 1

        result11 = rbf(self.mat1, self.mat1, length, sigma_f)
        result22 = rbf(self.mat2, self.mat2, length, sigma_f)
        result12 = rbf(self.mat1, self.mat2, length, sigma_f)

        target11 = np.array([[1., 0.98702],
                             [0.98702, 1.]])
        target22 = np.array([[1., 0.74104, 0.93852, 0.71781],
                             [0.74104, 1., 0.78289, 0.88323],
                             [0.93852, 0.78289, 1., 0.78355],
                             [0.71781, 0.88323, 0.78355, 1.]])
        target12 = np.array([[0.90942, 0.84847, 0.94961, 0.91871],
                             [0.93656, 0.84478, 0.924, 0.8973]])

        np.testing.assert_almost_equal(result11, target11, decimal=5)
        np.testing.assert_almost_equal(result22, target22, decimal=5)
        np.testing.assert_almost_equal(result12, target12, decimal=5)
