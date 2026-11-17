"""Automated tests for visionline.py. Run: python -m unittest -v"""
import unittest
import numpy as np

import visionline


class VisionLineTests(unittest.TestCase):
    def test_sobel_expected_edge_map_and_border(self):
        img = [
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 200, 200, 200],
            [10, 10, 10, 200, 200, 200],
            [10, 10, 10, 200, 200, 200],
            [10, 10, 10, 200, 200, 200],
        ]
        mag, edge = visionline.sobel_edges(img, 100)
        expected = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(edge, expected)
        self.assertAlmostEqual(mag[1, 2], 600.832756, places=5)
        self.assertTrue(np.all(mag[[0, -1], :] == 0))
        self.assertTrue(np.all(mag[:, [0, -1]] == 0))

    def test_sobel_small_and_uniform_images(self):
        for image in ([[5]], [[5, 5], [5, 5]], np.full((5, 6), 42)):
            mag, edge = visionline.sobel_edges(image, 10)
            self.assertFalse(np.any(mag))
            self.assertFalse(np.any(edge))

    def test_label_blobs_expected_labels_and_count(self):
        mask = [
            [1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
        ]
        labels, count = visionline.label_blobs(mask)
        expected = np.array([
            [1, 1, 1, 0, 0, 0, 0, 2],
            [1, 1, 1, 0, 3, 3, 0, 2],
            [1, 1, 1, 0, 3, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 5, 5, 5, 0],
            [0, 4, 4, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        self.assertEqual(count, 5)
        np.testing.assert_array_equal(labels, expected)

    def test_label_blobs_empty_and_diagonal_connectivity(self):
        labels, count = visionline.label_blobs(np.zeros((3, 3), dtype=np.uint8))
        self.assertEqual(count, 0)
        self.assertFalse(np.any(labels))
        labels, count = visionline.label_blobs([[1, 0], [0, 1]])
        self.assertEqual(count, 2)  # diagonal pixels are not 4-connected

    def test_all_shape_features(self):
        labels = np.array([
            [1, 1, 1, 0, 0, 0, 0, 2],
            [1, 1, 1, 0, 3, 3, 0, 2],
            [1, 1, 1, 0, 3, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 5, 5, 5, 0],
            [0, 4, 4, 0, 0, 0, 0, 0],
        ])
        f = visionline.all_shape_features(labels)
        self.assertEqual(f[1], {"area": 9, "bbox": (0, 0, 2, 2),
                                "perimeter": 8, "extent": 1.0, "kind": "RECT"})
        self.assertEqual(f[2]["perimeter"], 2)
        self.assertEqual(f[3]["area"], 4)
        self.assertEqual(f[4]["extent"], 0.75)
        self.assertEqual(f[4]["kind"], "NON-RECT")
        self.assertEqual(f[5]["bbox"], (4, 4, 4, 6))

    def test_shape_features_single_pixel_and_image_border(self):
        self.assertEqual(
            visionline.shape_features([[7]], 7),
            {"area": 1, "bbox": (0, 0, 0, 0), "perimeter": 1,
             "extent": 1.0, "kind": "RECT"}
        )
        with self.assertRaises(ValueError):
            visionline.shape_features([[0, 0]], 1)

    def test_match_template_exact_match_and_last_valid_position(self):
        scene = [
            [3, 1, 4, 1, 5, 9],
            [2, 6, 5, 3, 5, 8],
            [9, 7, 9, 3, 2, 3],
            [8, 4, 6, 2, 6, 4],
            [3, 3, 8, 3, 2, 7],
        ]
        tpl = [[6, 4], [2, 7]]
        self.assertEqual(visionline.match_template(scene, tpl), (3, 4, 0.0))

    def test_match_template_whole_image_and_oversized(self):
        image = [[1, 2], [3, 4]]
        self.assertEqual(visionline.match_template(image, image), (0, 0, 0.0))
        self.assertIsNone(visionline.match_template([[1]], [[1, 2]]))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            visionline.sobel_edges([[]], 1)
        with self.assertRaises(ValueError):
            visionline.label_blobs([1, 2, 3])
        with self.assertRaises(ValueError):
            visionline.sobel_edges([[1, float("nan")], [2, 3]], 1)

    @unittest.skipIf(visionline.cv2 is None, "OpenCV is not installed")
    def test_opencv_matcher_agrees(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        template = np.array([[5, 6], [8, 9]], dtype=np.uint8)
        self.assertEqual(visionline.match_template_opencv(image, template),
                         visionline.match_template(image, template))


if __name__ == "__main__":
    unittest.main(verbosity=2)
