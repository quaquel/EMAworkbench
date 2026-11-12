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

    def test_calculate_cid_memory_options(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(23, 30)).cumsum(axis=1)
        reference = clusterer.calculate_cid(data)

        # chunking does not change the result, and the result is exactly symmetric
        for chunk_size in (1, 5, 23, 100):
            distances = clusterer.calculate_cid(data, chunk_size=chunk_size)
            np.testing.assert_allclose(distances, reference, rtol=1e-12)
            np.testing.assert_array_equal(distances, distances.T)
            np.testing.assert_array_equal(np.diag(distances), 0)

        # condensed form is built directly and matches the square form; exactly
        # for the same chunks, up to round-off of the matrix product otherwise
        for chunk_size in (1, 5, 100):
            square = clusterer.calculate_cid(data, chunk_size=chunk_size)
            condensed = clusterer.calculate_cid(
                data, condensed_form=True, chunk_size=chunk_size
            )
            np.testing.assert_array_equal(
                condensed, sp.spatial.distance.squareform(square, checks=False)
            )
            np.testing.assert_allclose(
                condensed,
                sp.spatial.distance.squareform(reference, checks=False),
                rtol=1e-12,
            )

        # float32 results
        distances = clusterer.calculate_cid(data, dtype=np.float32, chunk_size=5)
        self.assertEqual(distances.dtype, np.float32)
        np.testing.assert_allclose(distances, reference, rtol=1e-6)
        np.testing.assert_array_equal(distances, distances.T)
        condensed = clusterer.calculate_cid(data, condensed_form=True, dtype=np.float32)
        self.assertEqual(condensed.dtype, np.float32)

        with self.assertRaises(ValueError):
            clusterer.calculate_cid(data, dtype=np.int64)

    def test_normalize_distances_memory_options(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(50, 20)).cumsum(axis=1)
        distances = clusterer.calculate_cid(data, dtype=np.float32)

        normalized, scale = clusterer.normalize_distances(distances)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertIsNot(normalized, distances)

        original = distances.copy()
        normalized, scale_in_place = clusterer.normalize_distances(
            distances, copy=False
        )
        self.assertIs(normalized, distances)
        self.assertAlmostEqual(scale_in_place, scale)
        np.testing.assert_allclose(distances, original / scale, rtol=1e-6)

        # integer input is converted, never divided in place
        integers = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]])
        normalized, scale = clusterer.normalize_distances(integers, copy=False)
        self.assertEqual(scale, 4)
        np.testing.assert_array_equal(integers[0], [0, 2, 4])
        np.testing.assert_allclose(normalized[0], [0, 0.5, 1])


if __name__ == "__main__":
    unittest.main()
