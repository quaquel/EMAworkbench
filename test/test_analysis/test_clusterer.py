import matplotlib.pyplot as plt
import numpy as np
import unittest

from ema_workbench.analysis import clusterer
from test import utilities

class ClusterTestCase(unittest.TestCase):

    def test_cluster(self):
        n = 10
        experiments, outcomes = utilities.load_flu_data()
        data = outcomes['infected fraction R1'][0:n, :]

        distances = clusterer.calculate_cid(data)
        self.assertEqual(distances.shape, (n,n))
        clusterer.plot_dendrogram(distances)
        plt.draw()

        assignment = clusterer.apply_agglomerative_clustering(distances, 2)
        self.assertEqual(assignment.shape, (10,))

        distances = clusterer.calculate_cid(data, condensed_form=True)
        self.assertEqual(distances.shape, sum(np.arange(0, n)))
        clusterer.plot_dendrogram(distances)
        plt.draw()

        plt.close('all')



if __name__ == '__main__':
    unittest.main()