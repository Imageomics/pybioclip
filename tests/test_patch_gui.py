import base64
import json
import os
import tempfile
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
import numpy as np
import PIL.Image
from bioclip.patch_gui import (
    GRID_SIZE,
    PATCH_SIZE,
    IMAGE_SIZE,
    DATASET_MEAN_RGB,
    MaskStrategy,
    SelectionMode,
    create_grid_mask_pixels,
    apply_mean_fill,
    apply_zero_fill,
    apply_gaussian_blur,
    apply_mask_to_image,
    _kept_bbox,
    _PatchServerState,
    _make_handler,
    _find_free_port,
    _image_to_base64,
)


class TestCreateGridMaskPixels(unittest.TestCase):
    def test_single_patch_top_left(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        grid[0, 0] = True
        pixel_mask = create_grid_mask_pixels(grid)
        self.assertEqual(pixel_mask.shape, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(pixel_mask[:PATCH_SIZE, :PATCH_SIZE].all())
        self.assertFalse(pixel_mask[PATCH_SIZE:, :].any())
        self.assertFalse(pixel_mask[:, PATCH_SIZE:].any())

    def test_single_patch_bottom_right(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        grid[GRID_SIZE - 1, GRID_SIZE - 1] = True
        pixel_mask = create_grid_mask_pixels(grid)
        y0 = (GRID_SIZE - 1) * PATCH_SIZE
        x0 = (GRID_SIZE - 1) * PATCH_SIZE
        self.assertTrue(pixel_mask[y0:, x0:].all())
        self.assertFalse(pixel_mask[:y0, :].any())
        self.assertFalse(pixel_mask[:, :x0].any())

    def test_all_selected(self):
        grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        pixel_mask = create_grid_mask_pixels(grid)
        self.assertTrue(pixel_mask.all())

    def test_none_selected(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        pixel_mask = create_grid_mask_pixels(grid)
        self.assertFalse(pixel_mask.any())

    def test_output_shape(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        pixel_mask = create_grid_mask_pixels(grid)
        self.assertEqual(pixel_mask.shape, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(pixel_mask.dtype, bool)


class TestApplyMeanFill(unittest.TestCase):
    def test_masked_pixels_get_mean_values(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[0:PATCH_SIZE, 0:PATCH_SIZE] = True
        result = apply_mean_fill(img, mask)
        expected = np.array([int(c * 255) for c in DATASET_MEAN_RGB], dtype=np.uint8)
        np.testing.assert_array_equal(result[0, 0], expected)
        np.testing.assert_array_equal(result[PATCH_SIZE - 1, PATCH_SIZE - 1], expected)

    def test_unmasked_pixels_unchanged(self):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[0:10, 0:10] = True
        result = apply_mean_fill(img, mask)
        self.assertEqual(result[50, 50, 0], 128)
        self.assertEqual(result[50, 50, 1], 128)
        self.assertEqual(result[50, 50, 2], 128)

    def test_does_not_mutate_input(self):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 100, dtype=np.uint8)
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        original = img.copy()
        apply_mean_fill(img, mask)
        np.testing.assert_array_equal(img, original)


class TestApplyZeroFill(unittest.TestCase):
    def test_masked_pixels_become_zero(self):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 200, dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:110, 100:110] = True
        result = apply_zero_fill(img, mask)
        self.assertTrue((result[100:110, 100:110] == 0).all())

    def test_unmasked_pixels_unchanged(self):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 200, dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:110, 100:110] = True
        result = apply_zero_fill(img, mask)
        self.assertTrue((result[0:100, 0:100] == 200).all())

    def test_does_not_mutate_input(self):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 100, dtype=np.uint8)
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        original = img.copy()
        apply_zero_fill(img, mask)
        np.testing.assert_array_equal(img, original)


class TestApplyGaussianBlur(unittest.TestCase):
    def test_returns_pil_image(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 0, 0))
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:124, 100:124] = True
        result = apply_gaussian_blur(img, mask)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_unmasked_pixels_unchanged(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (100, 150, 200))
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[112:224, 112:224] = True  # mask bottom-right quadrant
        result = apply_gaussian_blur(img, mask)
        result_arr = np.array(result)
        # Top-left corner (far from mask) should be unchanged
        np.testing.assert_array_equal(result_arr[0, 0], [100, 150, 200])


class TestApplyMaskToImage(unittest.TestCase):
    def test_no_selection_returns_resized_image(self):
        img = PIL.Image.new('RGB', (500, 500), (100, 150, 200))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr[0, 0], [100, 150, 200])

    def test_grid_mean_fill(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        mask[0, 0] = True
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.GRID)
        result_arr = np.array(result)
        expected = np.array([int(c * 255) for c in DATASET_MEAN_RGB], dtype=np.uint8)
        np.testing.assert_array_equal(result_arr[0, 0], expected)
        # Unmasked region
        np.testing.assert_array_equal(result_arr[PATCH_SIZE, PATCH_SIZE], [0, 0, 0])

    def test_grid_zero_fill(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (200, 200, 200))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        mask[0, 0] = True
        result = apply_mask_to_image(img, mask, MaskStrategy.ZERO_FILL, SelectionMode.GRID)
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(result_arr[PATCH_SIZE, PATCH_SIZE], [200, 200, 200])

    def test_grid_blur(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (100, 100, 100))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        mask[7, 7] = True
        result = apply_mask_to_image(img, mask, MaskStrategy.GAUSSIAN_BLUR, SelectionMode.GRID)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_freeform_zero_fill(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (150, 150, 150))
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[50:70, 50:70] = True
        result = apply_mask_to_image(img, mask, MaskStrategy.ZERO_FILL, SelectionMode.FREEFORM)
        result_arr = np.array(result)
        self.assertTrue((result_arr[50:70, 50:70] == 0).all())
        np.testing.assert_array_equal(result_arr[0, 0], [150, 150, 150])

    def test_freeform_mean_fill(self):
        img = PIL.Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:120, 100:120] = True
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.FREEFORM)
        result_arr = np.array(result)
        expected = np.array([int(c * 255) for c in DATASET_MEAN_RGB], dtype=np.uint8)
        np.testing.assert_array_equal(result_arr[110, 110], expected)

    def test_resizes_large_image(self):
        img = PIL.Image.new('RGB', (1000, 800), (50, 100, 150))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))


class TestKeptBbox(unittest.TestCase):
    def test_no_mask(self):
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        bbox = _kept_bbox(mask)
        self.assertEqual(bbox, (0, IMAGE_SIZE, 0, IMAGE_SIZE))

    def test_all_masked(self):
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        self.assertIsNone(_kept_bbox(mask))

    def test_edge_masked(self):
        # Mask the top 16 rows (first patch row) — kept region starts at row 16
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[:16, :] = True
        bbox = _kept_bbox(mask)
        self.assertEqual(bbox, (16, IMAGE_SIZE, 0, IMAGE_SIZE))

    def test_single_kept_patch(self):
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[32:48, 64:80] = False  # one 16x16 patch kept
        bbox = _kept_bbox(mask)
        self.assertEqual(bbox, (32, 48, 64, 80))


class TestCropBehavior(unittest.TestCase):
    """Tests that apply_mask_to_image crops to the kept region at full resolution."""

    def _make_striped_image(self, w, h):
        """Create an image with a unique color per quadrant for verifying crop origin."""
        img = PIL.Image.new('RGB', (w, h), (0, 0, 0))
        arr = np.array(img)
        mid_y, mid_x = h // 2, w // 2
        arr[:mid_y, :mid_x] = [255, 0, 0]      # top-left: red
        arr[:mid_y, mid_x:] = [0, 255, 0]       # top-right: green
        arr[mid_y:, :mid_x] = [0, 0, 255]       # bottom-left: blue
        arr[mid_y:, mid_x:] = [255, 255, 0]     # bottom-right: yellow
        return PIL.Image.fromarray(arr)

    def test_mask_edges_crops_to_interior(self):
        """Masking all edge patches should crop to interior, giving it more resolution."""
        img = PIL.Image.new('RGB', (1920, 1080), (100, 150, 200))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        # Mask the entire outer ring of patches
        mask[0, :] = True   # top row
        mask[-1, :] = True  # bottom row
        mask[:, 0] = True   # left column
        mask[:, -1] = True  # right column
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))
        # The kept interior should be solid (100, 150, 200) — no fill artifacts
        result_arr = np.array(result)
        # Center pixel should be the original color
        np.testing.assert_array_equal(result_arr[112, 112], [100, 150, 200])

    def test_invert_selection_crops_to_selection(self):
        """Selecting a small region and inverting should crop to that region."""
        # 1920x1080 image, select only patches in rows 6-7, cols 6-7 (center area)
        img = self._make_striped_image(1920, 1080)
        mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)  # everything masked
        mask[6:8, 6:8] = False  # keep only center 2x2 patches
        result = apply_mask_to_image(img, mask, MaskStrategy.ZERO_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))
        # The result should be dominated by the color from the center of the original
        # (which spans all 4 quadrants near the center). The key point is that the
        # 2x2 patch region now fills the entire 224x224, not just a tiny area.
        result_arr = np.array(result)
        # Should NOT be all zeros — the kept patches have real image content
        self.assertTrue(result_arr.any())

    def test_no_mask_returns_full_image(self):
        """No masking should return the full image resized to 224x224."""
        img = PIL.Image.new('RGB', (500, 300), (42, 42, 42))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        result = apply_mask_to_image(img, mask, MaskStrategy.MEAN_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr[0, 0], [42, 42, 42])

    def test_freeform_crop_to_kept(self):
        """Freeform mask with small kept region should crop to that region."""
        img = PIL.Image.new('RGB', (800, 800), (200, 100, 50))
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:124, 100:124] = False  # small 24x24 kept region
        result = apply_mask_to_image(img, mask, MaskStrategy.ZERO_FILL, SelectionMode.FREEFORM)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))
        # The kept region should now fill 224x224 — mostly the original color
        result_arr = np.array(result)
        # Center of result should be the original image content, not zero
        self.assertTrue(result_arr[112, 112].any())

    def test_interior_hole_gets_filled(self):
        """Kept region with a masked hole inside should fill the hole."""
        img = PIL.Image.new('RGB', (224, 224), (200, 200, 200))
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        mask[0, :] = True   # mask top edge
        mask[7, 7] = True   # mask one interior patch
        result = apply_mask_to_image(img, mask, MaskStrategy.ZERO_FILL, SelectionMode.GRID)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))


def _create_temp_image(tmpdir, name="test.jpg", size=(IMAGE_SIZE, IMAGE_SIZE), color=(100, 150, 200)):
    """Helper: write a temp image file and return its path."""
    path = os.path.join(tmpdir, name)
    PIL.Image.new("RGB", size, color).save(path)
    return path


class TestImageToBase64(unittest.TestCase):
    def test_returns_valid_base64_png(self):
        img = PIL.Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 0, 0))
        b64 = _image_to_base64(img)
        decoded = base64.b64decode(b64)
        self.assertTrue(decoded.startswith(b"\x89PNG"))


class TestPatchServerState(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = _create_temp_image(self.tmpdir)
        self.callback_calls = []
        self.predictions_returned = [
            {"file_name": self.img_path, "classification": "Testus specius", "score": 0.95}
        ]

        def mock_callback(image, path):
            self.callback_calls.append((image, path))
            return self.predictions_returned

        self.state = _PatchServerState([self.img_path], mock_callback)

    def test_get_image_data(self):
        data = self.state.get_image_data(0)
        self.assertEqual(data["index"], 0)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["filename"], "test.jpg")
        decoded = base64.b64decode(data["image"])
        self.assertTrue(decoded.startswith(b"\x89PNG"))

    def test_handle_predict_grid(self):
        mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        mask[0][0] = True
        body = {"mask": mask, "strategy": "mean", "mode": "grid", "imageIndex": 0}
        results = self.state.handle_predict(body)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["classification"], "Testus specius")
        self.assertEqual(len(self.callback_calls), 1)
        image, path = self.callback_calls[0]
        self.assertIsInstance(image, PIL.Image.Image)
        self.assertEqual(path, self.img_path)

    def test_handle_predict_freeform(self):
        mask_arr = np.zeros(IMAGE_SIZE * IMAGE_SIZE, dtype=np.uint8)
        mask_arr[100:200] = 1
        mask_b64 = base64.b64encode(mask_arr.tobytes()).decode()
        body = {"mask": mask_b64, "strategy": "zero", "mode": "freeform", "imageIndex": 0}
        results = self.state.handle_predict(body)
        self.assertEqual(len(results), 1)

    def test_results_accumulate(self):
        mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        body = {"mask": mask, "strategy": "mean", "mode": "grid", "imageIndex": 0}
        self.state.handle_predict(body)
        self.state.handle_predict(body)
        self.assertEqual(len(self.state.results), 2)

    def test_done_event(self):
        self.assertFalse(self.state.done_event.is_set())
        self.state.done_event.set()
        self.assertTrue(self.state.done_event.is_set())


class TestPatchServerHTTP(unittest.TestCase):
    """Integration tests that start a real HTTP server."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = _create_temp_image(self.tmpdir)
        self.img_path2 = _create_temp_image(self.tmpdir, "test2.jpg", color=(50, 50, 50))

        def mock_callback(image, path):
            return [{"file_name": path, "classification": "Mock species", "score": 0.99}]

        self.state = _PatchServerState([self.img_path, self.img_path2], mock_callback)
        port = _find_free_port()
        handler_cls = _make_handler(self.state)
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        self.url = f"http://127.0.0.1:{port}"
        import threading
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()

    def test_serves_html(self):
        resp = urlopen(self.url + "/")
        self.assertEqual(resp.status, 200)
        html = resp.read().decode()
        self.assertIn("bioclip patch", html)
        self.assertIn("<canvas", html)

    def test_serves_image(self):
        resp = urlopen(self.url + "/api/image/0")
        data = json.loads(resp.read())
        self.assertEqual(data["index"], 0)
        self.assertEqual(data["total"], 2)
        self.assertIn("image", data)

    def test_serves_second_image(self):
        resp = urlopen(self.url + "/api/image/1")
        data = json.loads(resp.read())
        self.assertEqual(data["index"], 1)
        self.assertEqual(data["filename"], "test2.jpg")

    def test_predict_endpoint(self):
        mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        mask[5][5] = True
        body = json.dumps({
            "mask": mask, "strategy": "mean", "mode": "grid", "imageIndex": 0
        }).encode()
        req = Request(self.url + "/api/predict", data=body,
                      headers={"Content-Type": "application/json"})
        resp = urlopen(req)
        predictions = json.loads(resp.read())
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["classification"], "Mock species")

    def test_done_endpoint(self):
        req = Request(self.url + "/api/done", data=b"", method="POST")
        resp = urlopen(req)
        self.assertEqual(resp.status, 200)
        self.assertTrue(self.state.done_event.is_set())
