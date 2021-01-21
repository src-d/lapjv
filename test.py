import unittest

from numpy import array, dstack, float32, float64, linspace, meshgrid, random, sqrt
from scipy.spatial.distance import cdist

from lapjv import lapjv


class LapjvTests(unittest.TestCase):
    def _test_random_100(self, dtype):
        random.seed(777)
        size = 100
        dots = random.random((size, 2))
        grid = dstack(meshgrid(linspace(0, 1, int(sqrt(size))),
                               linspace(0, 1, int(sqrt(size))))).reshape(-1, 2)
        cost = cdist(dots, grid, "sqeuclidean").astype(dtype)
        cost *= 100000 / cost.max()
        row_ind_lapjv, col_ind_lapjv, _ = lapjv(cost, verbose=True, force_doubles=True)
        # Obtained from pyLAPJV on Python 2.7
        row_ind_original = array([
            32, 51, 99, 77, 62, 1, 35, 69, 57, 42, 13, 24, 96, 26, 82, 52, 65,
            6, 95, 7, 63, 47, 28, 45, 74,
            61, 34, 14, 94, 31, 25, 3, 71, 49, 58, 83, 91, 93, 23, 98, 36, 40,
            4, 97, 21, 92, 89, 90, 29, 46,
            79, 2, 76, 84, 72, 64, 33, 37, 41, 15, 59, 85, 70, 78, 81, 20, 18,
            30, 8, 66, 38, 87, 44, 67, 68,
            39, 86, 54, 11, 50, 16, 17, 56, 0, 5, 80, 10, 48, 60, 73, 53, 75,
            55, 19, 22, 12, 9, 88, 43, 27])
        col_ind_original = array([
            83, 5, 51, 31, 42, 84, 17, 19, 68, 96, 86, 78, 95, 10, 27, 59, 80,
            81, 66, 93, 65, 44, 94, 38, 11,
            30, 13, 99, 22, 48, 67, 29, 0, 56, 26, 6, 40, 57, 70, 75, 41, 58, 9,
            98, 72, 23, 49, 21, 87, 33,
            79, 1, 15, 90, 77, 92, 82, 8, 34, 60, 88, 25, 4, 20, 55, 16, 69, 73,
            74, 7, 62, 32, 54, 89, 24,
            91, 52, 3, 63, 50, 85, 64, 14, 35, 53, 61, 76, 71, 97, 46, 47, 36,
            45, 37, 28, 18, 12, 43, 39, 2])
        self.assertTrue((row_ind_lapjv == row_ind_original).all())
        self.assertTrue((col_ind_lapjv == col_ind_original).all())

    def test_random_100_float64(self):
        self._test_random_100(float64)

    def test_random_100_float32(self):
        self._test_random_100(float32)

    def test_1024(self):
        random.seed(777)
        size = 1024
        dots = random.random((size, 2))
        grid = dstack(meshgrid(linspace(0, 1, int(sqrt(size))),
                               linspace(0, 1, int(sqrt(size))))).reshape(-1, 2)
        cost = cdist(dots, grid, "sqeuclidean")
        cost *= 100000 / cost.max()
        row_ind_lapjv32, col_ind_lapjv32, _ = lapjv(cost, verbose=True)
        self.assertEqual(len(set(col_ind_lapjv32)), dots.shape[0])
        self.assertEqual(len(set(row_ind_lapjv32)), dots.shape[0])
        row_ind_lapjv64, col_ind_lapjv64, _ = lapjv(cost, verbose=True, force_doubles=True)
        self.assertTrue((row_ind_lapjv32 == row_ind_lapjv64).all())
        self.assertTrue((col_ind_lapjv32 == col_ind_lapjv64).all())


if __name__ == "__main__":
    unittest.main()
