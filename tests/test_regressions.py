"""Hardware-free regression checks for numerical and image-processing fixes."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegressionTests(unittest.TestCase):
    def test_signed_sobel(self):
        m = load("filtering_edge_detection/Exercise5_3.py")
        image = np.array([[0, 0, 255, 255, 0, 0]] * 5, np.uint8)
        fm, fn, magnitude, _ = m.sobel_filter(image)
        np.testing.assert_array_equal(fm, ndimage.sobel(image.astype(float), axis=1))
        self.assertLess(fm.min(), 0)
        np.testing.assert_allclose(magnitude, np.hypot(fm, fn))

    def test_gaussian_derivatives(self):
        m = load("filtering_edge_detection/Exercise5_4.py")
        image = np.array([[0, 0, 255, 255, 0, 0]] * 5, np.uint8)
        fm, fn = m.gauss_filter(image, 1)
        np.testing.assert_allclose(fm, ndimage.gaussian_filter(image.astype(float), 1, order=(0, 1)))
        self.assertLess(fm.min(), 0)

    def test_black_foreground_and_transparency(self):
        for path in ("Exercise9_5.py", "Exercise9_7.py"):
            m = load("haar_grabcut_background_replacement/" + path)
            image = np.zeros((4, 4, 3), np.uint8)
            mask = np.zeros((4, 4), np.uint8)
            mask[1:3, 1:3] = 1
            fg = m.add_alpha_channel(image, mask)
            for composite_path in ("Exercise9_6.py", "Exercise9_7.py"):
                compositor = load("haar_grabcut_background_replacement/" + composite_path)
                bg = np.full_like(image, 200)
                result = compositor.paste_image(fg, bg, -1, -1, 6, 6)
                np.testing.assert_array_equal(result[1:3, 1:3], 0)
                np.testing.assert_array_equal(result[0], 200)

    def test_grabcut_mask_contract(self):
        image = np.full((64, 64, 3), 255, np.uint8)
        image[20:44, 20:44] = 0
        for path in ("Exercise9_4.py", "Exercise9_7.py"):
            m = load("haar_grabcut_background_replacement/" + path)
            cutout, mask = m.grab_cut(image, 10, 10, 44, 44, return_mask=True)
            self.assertEqual(mask[30, 30], 1)
            self.assertEqual(mask[0, 0], 0)
            np.testing.assert_array_equal(cutout, image * mask[:, :, None])

    def test_panorama_color_and_translation(self):
        m = load("sift_panorama_stitching/Experiment7.py")
        image = np.random.default_rng(42).integers(0, 256, (240, 400, 3), dtype=np.uint8)
        left, right = image[:, :300].copy(), image[:, 100:].copy()
        _, _, matches, panorama = m.create_panorama(left, right)
        self.assertEqual(matches.shape, (240, 600, 3))
        self.assertTrue(395 <= panorama.shape[1] <= 405)
        np.testing.assert_array_equal(panorama[:240, :300], left)

    def test_panorama_blank_input(self):
        m = load("sift_panorama_stitching/Experiment7.py")
        with self.assertRaisesRegex(ValueError, "No usable features"):
            m.create_panorama(np.zeros((80, 80), np.uint8), np.zeros((80, 80), np.uint8))

    def test_panorama_failed_homography(self):
        m = load("sift_panorama_stitching/Experiment7.py")
        image = np.random.default_rng(2).integers(0, 256, (200, 200), dtype=np.uint8)
        with patch.object(m.cv2, "findHomography", return_value=(None, None)):
            with self.assertRaisesRegex(ValueError, "Homography"):
                m.create_panorama(image, image)

    def test_cancelled_capture_cleans_up_thread(self):
        m = load("sift_panorama_stitching/Experiment7.py")
        with patch.object(m, "interface", return_value=None):
            m.main()

    def test_hough_uses_pristine_input(self):
        m = load("raspberry_pi_camera_pen_detection/Exercise8_5.py")
        source = np.zeros((240, 240, 3), np.uint8)
        output = source.copy()
        with patch.object(m.cv2, "HoughLines", return_value=np.array([[[100., 0.]]], np.float32)), patch.object(m.cv2, "putText") as label:
            m.hough_transform(20, 20, 180, 180, source, output)
            self.assertEqual(label.call_args.args[1], "90.00")
        self.assertFalse(source.any())
        self.assertTrue(output.any())

    def test_video_open_failure_releases_handles(self):
        m = load("haar_grabcut_background_replacement/Exercise9_7.py")
        with patch.object(m.cv2, "CascadeClassifier") as classifier, patch.object(m.cv2, "VideoCapture") as capture:
            classifier.return_value.empty.return_value = False
            capture.return_value.isOpened.return_value = False
            with self.assertRaises(OSError):
                m.background_replacement("missing", "missing", "classifier")
            self.assertEqual(capture.return_value.release.call_count, 2)


if __name__ == "__main__":
    unittest.main()
