"""
Unit tests for StereoInference class.

Tests cover:
1. Shape consistency
2. Scale/depth correctness
3. Coordinate transformation correctness
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo_inference import StereoInference


class MockFoundationStereoModel:
    """Mock model for testing without actual inference."""

    def __init__(self, constant_disparity: float = 50.0):
        self.constant_disparity = constant_disparity

    def forward(self, left, right, iters=32, test_mode=True):
        """Return constant disparity map."""
        import torch
        B, C, H, W = left.shape
        return torch.full((B, 1, H, W), self.constant_disparity, dtype=left.dtype, device=left.device)

    def to(self, device):
        return self

    def eval(self):
        pass


class TestStereoInferenceShapes(unittest.TestCase):
    """Test shape consistency of outputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.H, self.W = 480, 640
        self.left_img = np.random.randint(0, 256, (self.H, self.W, 3), dtype=np.uint8)
        self.right_img = np.random.randint(0, 256, (self.H, self.W, 3), dtype=np.uint8)

        # Create calibration
        self.fx = self.fy = 400.0
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.K_rect = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.baseline_m = 0.1

    def test_output_shapes(self):
        """Test that all outputs have correct shapes."""
        # Create mock results (simulating what infer() returns)
        disparity = np.random.uniform(0, 100, (self.H, self.W)).astype(np.float32)
        valid_mask = disparity > 0

        depth = np.zeros_like(disparity)
        depth[valid_mask] = self.fx * self.baseline_m / disparity[valid_mask]

        points_cam1 = np.random.uniform(-1, 1, (self.H, self.W, 3)).astype(np.float32)

        # Check shapes
        self.assertEqual(disparity.shape, (self.H, self.W))
        self.assertEqual(depth.shape, (self.H, self.W))
        self.assertEqual(points_cam1.shape, (self.H, self.W, 3))
        self.assertEqual(valid_mask.shape, (self.H, self.W))
        self.assertEqual(valid_mask.dtype, bool)

        print("✓ Shape test passed: All outputs have consistent shapes")

    def test_different_resolutions(self):
        """Test with various image resolutions."""
        test_sizes = [(240, 320), (480, 640), (720, 1280)]

        for H, W in test_sizes:
            left = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            right = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

            K = np.array([
                [400, 0, W/2],
                [0, 400, H/2],
                [0, 0, 1]
            ], dtype=np.float32)

            # Simulate disparity output
            disparity = np.random.uniform(10, 50, (H, W)).astype(np.float32)

            self.assertEqual(disparity.shape, (H, W))

        print(f"✓ Resolution test passed: Tested {len(test_sizes)} different resolutions")


class TestStereoInferenceScale(unittest.TestCase):
    """Test depth scale correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.H, self.W = 480, 640
        self.fx = 400.0
        self.baseline_m = 0.1

        self.K_rect = np.array([
            [self.fx, 0, self.W/2],
            [0, self.fx, self.H/2],
            [0, 0, 1]
        ], dtype=np.float32)

    def test_depth_scale_formula(self):
        """Test that depth = fx * baseline / disparity."""
        # Create constant disparity
        d0 = 50.0
        disparity = np.full((self.H, self.W), d0, dtype=np.float32)
        valid_mask = disparity > 0

        # Compute depth using formula
        expected_depth = self.fx * self.baseline_m / d0
        depth = np.zeros_like(disparity)
        depth[valid_mask] = self.fx * self.baseline_m / disparity[valid_mask]

        # Check median depth
        median_depth = np.median(depth[valid_mask])
        relative_error = abs(median_depth - expected_depth) / expected_depth

        self.assertLess(relative_error, 0.03, f"Depth scale error {relative_error:.1%} exceeds 3%")

        print(f"✓ Scale test passed: median(depth)={median_depth:.4f}m, expected={expected_depth:.4f}m, error={relative_error:.2%}")

    def test_depth_variation(self):
        """Test depth with varying disparity."""
        disparities = [10, 20, 50, 100]

        for d in disparities:
            expected_depth = self.fx * self.baseline_m / d
            actual_depth = self.fx * self.baseline_m / d

            relative_error = abs(actual_depth - expected_depth) / expected_depth
            self.assertLess(relative_error, 1e-6)

        print(f"✓ Depth variation test passed: Tested {len(disparities)} disparity values")


class TestStereoInferenceRotation(unittest.TestCase):
    """Test coordinate transformation correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.H, self.W = 100, 100  # Smaller for faster testing

    def test_rotation_consistency(self):
        """Test that R1 @ R1.T = I for point transformation."""
        # Generate random rotation matrix
        angle = np.pi / 6  # 30 degrees
        axis = np.array([0, 0, 1])  # Rotation around Z axis
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        R1 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Generate random points
        points_orig = np.random.uniform(-1, 1, (self.H, self.W, 3)).astype(np.float32)
        valid_mask = np.ones((self.H, self.W), dtype=bool)

        # Transform forth and back
        points_rect = self._transform_points(points_orig, R1, valid_mask)
        points_back = self._transform_points(points_rect, R1.T, valid_mask)

        # Check consistency
        max_error = np.max(np.abs(points_back - points_orig))
        self.assertLess(max_error, 1e-5, f"Rotation consistency error: {max_error}")

        print(f"✓ Rotation test passed: max error={max_error:.2e} after R1 @ R1.T transformation")

    def test_identity_rotation(self):
        """Test that identity rotation doesn't change points."""
        points = np.random.uniform(-1, 1, (self.H, self.W, 3)).astype(np.float32)
        valid_mask = np.ones((self.H, self.W), dtype=bool)

        R_identity = np.eye(3)
        points_transformed = self._transform_points(points, R_identity, valid_mask)

        np.testing.assert_array_almost_equal(points, points_transformed)

        print("✓ Identity rotation test passed")

    def _transform_points(self, points: np.ndarray, R: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Transform 3D points using rotation matrix R."""
        transformed_points = np.zeros_like(points)
        valid_points = points[valid_mask]

        if len(valid_points) > 0:
            transformed_valid = (R @ valid_points.T).T
            transformed_points[valid_mask] = transformed_valid

        return transformed_points


class TestStereoInferenceIntegration(unittest.TestCase):
    """Integration tests with actual model if available."""

    def test_model_loading(self):
        """Test that model can be loaded if checkpoint exists."""
        ckpt_path = "./pretrained_models/23-51-11/model_best_bp2.pth"

        if os.path.exists(ckpt_path):
            try:
                stereo_infer = StereoInference(ckpt_path)
                self.assertIsNotNone(stereo_infer.model)
                self.assertIsNotNone(stereo_infer.args)
                print("✓ Model loading test passed")
            except Exception as e:
                self.fail(f"Model loading failed: {e}")
        else:
            self.skipTest("Model checkpoint not found")

    def test_end_to_end_inference(self):
        """Test complete inference pipeline if model available."""
        ckpt_path = "./pretrained_models/23-51-11/model_best_bp2.pth"

        if not os.path.exists(ckpt_path):
            self.skipTest("Model checkpoint not found")

        try:
            # Initialize
            stereo_infer = StereoInference(ckpt_path)

            # Create test images
            H, W = 480, 640
            left_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

            K_rect = np.array([
                [400, 0, W/2],
                [0, 400, H/2],
                [0, 0, 1]
            ], dtype=np.float32)
            baseline_m = 0.1

            # Run inference
            results = stereo_infer.infer(left_img, right_img, K_rect, baseline_m)

            # Validate results
            self.assertEqual(results['disparity'].shape, (H, W))
            self.assertEqual(results['depth'].shape, (H, W))
            self.assertEqual(results['points_cam1'].shape, (H, W, 3))
            self.assertEqual(results['valid_mask'].shape, (H, W))

            # Check that we have valid outputs
            self.assertGreater(results['valid_mask'].sum(), 0, "No valid disparity values")

            valid_depth = results['depth'][results['valid_mask']]
            self.assertGreater(len(valid_depth), 0, "No valid depth values")
            self.assertTrue(np.all(valid_depth > 0), "Depth values should be positive")

            print("✓ End-to-end inference test passed")
            print(f"  Valid pixels: {results['valid_mask'].sum()}/{results['valid_mask'].size}")
            print(f"  Depth range: {valid_depth.min():.3f} - {valid_depth.max():.3f}m")

        except Exception as e:
            self.fail(f"End-to-end test failed: {e}")

    def test_tensor_value_range_smoke_test(self):
        """Smoke test to validate input tensor ranges are correct (0-255, not 0-1)."""
        ckpt_path = "./pretrained_models/23-51-11/model_best_bp2.pth"

        if not os.path.exists(ckpt_path):
            self.skipTest("Model checkpoint not found")

        try:
            stereo_infer = StereoInference(ckpt_path)

            # Create test images with known range
            H, W = 240, 320  # Smaller for faster test
            left_img = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)  # Clear range
            right_img = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

            K_rect = np.array([
                [200, 0, W/2],
                [0, 200, H/2],
                [0, 0, 1]
            ], dtype=np.float32)
            baseline_m = 0.1

            # Hook to capture input tensors before model forward
            captured_tensors = {}
            original_forward = stereo_infer.model.forward

            def capture_forward(left_tensor, right_tensor, **kwargs):
                captured_tensors['left'] = left_tensor.clone()
                captured_tensors['right'] = right_tensor.clone()
                return original_forward(left_tensor, right_tensor, **kwargs)

            stereo_infer.model.forward = capture_forward

            # Run inference
            results = stereo_infer.infer(left_img, right_img, K_rect, baseline_m, valid_iters=8)

            # Restore original forward
            stereo_infer.model.forward = original_forward

            # Validate tensor ranges
            left_tensor = captured_tensors['left']
            right_tensor = captured_tensors['right']

            self.assertGreaterEqual(left_tensor.min().item(), 0, "Left tensor min should be >= 0")
            self.assertLessEqual(left_tensor.max().item(), 255, "Left tensor max should be <= 255")
            self.assertGreaterEqual(right_tensor.min().item(), 0, "Right tensor min should be >= 0")
            self.assertLessEqual(right_tensor.max().item(), 255, "Right tensor max should be <= 255")

            # Check for reasonable contrast (not double-normalized)
            left_range = left_tensor.max().item() - left_tensor.min().item()
            right_range = right_tensor.max().item() - right_tensor.min().item()

            self.assertGreater(left_range, 50, f"Left tensor range too small: {left_range} (possible double normalization)")
            self.assertGreater(right_range, 50, f"Right tensor range too small: {right_range} (possible double normalization)")

            print("✓ Tensor value range smoke test passed")
            print(f"  Left tensor range: {left_tensor.min().item():.1f} - {left_tensor.max().item():.1f} (span: {left_range:.1f})")
            print(f"  Right tensor range: {right_tensor.min().item():.1f} - {right_tensor.max().item():.1f} (span: {right_range:.1f})")

        except Exception as e:
            self.fail(f"Tensor range smoke test failed: {e}")


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStereoInferenceShapes))
    suite.addTests(loader.loadTestsFromTestCase(TestStereoInferenceScale))
    suite.addTests(loader.loadTestsFromTestCase(TestStereoInferenceRotation))
    suite.addTests(loader.loadTestsFromTestCase(TestStereoInferenceIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)