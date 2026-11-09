import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ema_workbench.analysis import clusterer
from test import utilities


class ClusterTestCase(unittest.TestCase):
    def test_cluster(self):
        n = 10
        experiments, outcomes = utilities.load_flu_data()
        data = outcomes["infected_fraction_R1"][0:n, :]

        distances = clusterer.calculate_cid(data)
        self.assertEqual(distances.shape, (n, n))
        clusterer.plot_dendrogram(distances)
        plt.draw()

        assignment = clusterer.apply_agglomerative_clustering(distances, 2)
        self.assertEqual(assignment.shape, (10,))

        distances = clusterer.calculate_cid(data, condensed_form=True)
        self.assertEqual(distances.shape, sum(np.arange(0, n)))
        clusterer.plot_dendrogram(distances)
        plt.draw()

        plt.close("all")

    def test_calculate_cid_matches_pairwise_definition(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(25, 30)).cumsum(axis=1)

        ce = np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=1))
        expected = np.zeros((25, 25))
        for i in range(25):
            for j in range(25):
                if i != j:
                    expected[i, j] = clusterer.CID(data[i], data[j], ce[i], ce[j])

        distances = clusterer.calculate_cid(data, chunk_size=7)  # several chunks
        np.testing.assert_allclose(distances, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(distances, distances.T)
        np.testing.assert_array_equal(np.diag(distances), 0)

        condensed = clusterer.calculate_cid(data, condensed_form=True)
        np.testing.assert_allclose(
            condensed, sp.spatial.distance.squareform(expected, checks=False)
        )

    def test_calculate_cid_flat_series(self):
        rng = np.random.default_rng(0)
        data = np.vstack(
            [np.ones(20), np.full(20, 2.0), rng.normal(size=(3, 20)).cumsum(axis=1)]
        )
        distances = clusterer.calculate_cid(data)
        self.assertTrue(np.all(np.isfinite(distances)))
        # two flat series: complexity factor is 1, so CID equals Euclidean distance
        self.assertAlmostEqual(distances[0, 1], np.linalg.norm(data[0] - data[1]))
        # flat vs non-flat is far larger than non-flat vs non-flat
        self.assertGreater(distances[0, 2:].min(), distances[2:, 2:].max())

    def test_normalize_distances(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(300, 40)).cumsum(axis=1)
        distances = clusterer.calculate_cid(data)
        median = np.median(distances[np.triu_indices_from(distances, 1)])

        normalized, scale = clusterer.normalize_distances(distances)
        self.assertAlmostEqual(scale, median)
        np.testing.assert_allclose(normalized, distances / median)

        _, scale = clusterer.normalize_distances(distances, method="sample", seed=3)
        self.assertAlmostEqual(scale / median, 1, delta=0.02)

        condensed = clusterer.calculate_cid(data, condensed_form=True)
        normalized, scale = clusterer.normalize_distances(condensed)
        self.assertEqual(normalized.shape, condensed.shape)
        self.assertAlmostEqual(scale, median)

        with self.assertRaises(ValueError):
            clusterer.normalize_distances(distances, method="unknown")
        with self.assertRaises(ValueError):
            clusterer.normalize_distances(np.zeros((5, 5)))


if __name__ == "__main__":
    unittest.main()
