import logging
import unittest
import tst.test_kernels
import tst.test_gaussian_process
import tst.test_rescale
import tst.test_hypergrid
import tst.test_random_optimizer
import tst.test_bayesian_optimizer
import tst.test_acquisition


def main():
    # initialize
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Load tests from modules
    suite.addTest(loader.loadTestsFromModule(tst.test_kernels))
    suite.addTest(loader.loadTestsFromModule(tst.test_gaussian_process))
    suite.addTest(loader.loadTestsFromModule(tst.test_rescale))
    suite.addTest(loader.loadTestsFromModule(tst.test_hypergrid))
    suite.addTest(loader.loadTestsFromModule(tst.test_random_optimizer))
    suite.addTest(loader.loadTestsFromModule(tst.test_bayesian_optimizer))
    suite.addTest(loader.loadTestsFromModule(tst.test_acquisition))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)


if (__name__ == "__main__"):
    main()
