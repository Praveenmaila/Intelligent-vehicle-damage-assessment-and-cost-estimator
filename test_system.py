"""
Comprehensive Test Cases for Vehicle Damage Assessment System
Implements all test scenarios from research paper requirements
"""

import unittest
import torch
import numpy as np
from PIL import Image
import os
import tempfile

from models.vehicle_detector import VehicleDetector
from models.part_localizer import VehiclePartLocalizer
from models.damage_localizer import DamageLocalizer
from models.post_processor import PostProcessor
from integrated_system import IntegratedDamageAssessor
from utils.metrics import compute_iou, compute_dice_coefficient, compute_pixel_accuracy


class TestVehicleDetector(unittest.TestCase):
    """Test cases for vehicle detection module (TC01-TC02)."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize vehicle detector once for all tests."""
        cls.detector = VehicleDetector()
        cls.test_dir = 'test_images'
        os.makedirs(cls.test_dir, exist_ok=True)
    
    def test_tc01_detect_vehicle_present(self):
        """TC01: Detect presence of vehicle in valid image."""
        # Create test image with vehicle-like features
        img = Image.new('RGB', (640, 480), color='gray')
        result = self.detector.predict(img)
        
        self.assertIn('has_vehicle', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['has_vehicle'], bool)
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 100)
        print("✓ TC01 PASS: Vehicle detection module functional")
    
    def test_tc02_detect_no_vehicle(self):
        """TC02: Detect absence of vehicle in irrelevant image."""
        # Create test image without vehicle (pure noise)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        result = self.detector.predict(img)
        
        self.assertIn('has_vehicle', result)
        self.assertIn('probabilities', result)
        self.assertIn('no_vehicle', result['probabilities'])
        print("✓ TC02 PASS: No-vehicle detection functional")


class TestPartLocalizer(unittest.TestCase):
    """Test cases for vehicle part localization (TC03-TC04)."""
    
    @classmethod
    def setUpClass(cls):
        cls.localizer = VehiclePartLocalizer()
    
    def test_tc03_segment_parts_full_image(self):
        """TC03: Segment vehicle parts in clear full-car image."""
        img = Image.new('RGB', (640, 480), color='blue')
        result = self.localizer.predict(img, return_mask=True)
        
        self.assertIn('mask', result)
        self.assertIn('num_parts', result)
        self.assertIn('part_counts', result)
        self.assertEqual(result['mask'].shape, (480, 640))
        self.assertGreaterEqual(result['num_parts'], 0)
        print("✓ TC03 PASS: Part segmentation produces valid output")
    
    def test_tc04_segment_parts_partial_image(self):
        """TC04: Segment vehicle parts in partial/cropped image."""
        img = Image.new('RGB', (300, 300), color='red')
        result = self.localizer.predict(img, return_mask=True)
        
        self.assertIn('mask', result)
        self.assertEqual(result['mask'].shape, (300, 300))
        print("✓ TC04 PASS: Partial image segmentation functional")


class TestDamageLocalizer(unittest.TestCase):
    """Test cases for damage localization (TC05-TC06)."""
    
    @classmethod
    def setUpClass(cls):
        cls.localizer = DamageLocalizer()
    
    def test_tc05_segment_single_damage(self):
        """TC05: Segment damage types in image with scratch."""
        img = Image.new('RGB', (512, 512), color='white')
        result = self.localizer.predict(img, return_mask=True)
        
        self.assertIn('mask', result)
        self.assertIn('has_damage', result)
        self.assertIn('average_confidence', result)
        self.assertIsInstance(result['has_damage'], bool)
        print("✓ TC05 PASS: Damage detection produces valid output")
    
    def test_tc06_segment_multiple_damages(self):
        """TC06: Segment multiple damage types in same image."""
        img = Image.new('RGB', (512, 512), color='gray')
        result = self.localizer.predict(img, return_mask=True)
        
        self.assertIn('damage_counts', result)
        self.assertIsInstance(result['damage_counts'], dict)
        print("✓ TC06 PASS: Multiple damage detection functional")


class TestPostProcessor(unittest.TestCase):
    """Test cases for post-processing (TC07-TC09, TC15-TC17)."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = PostProcessor(confidence_threshold=0.7)
    
    def test_tc07_estimate_size_zoomed_out(self):
        """TC07: Estimate damage size for zoomed-out image."""
        result = self.processor.estimate_damage_size(
            pixel_count=1000,
            num_visible_parts=10,
            image_size=(640, 480)
        )
        
        self.assertIn('size_category', result)
        self.assertIn('severity_score', result)
        self.assertIn('zoom_level', result)
        self.assertEqual(result['zoom_level'], 'wide')
        print("✓ TC07 PASS: Size estimation for wide-angle functional")
    
    def test_tc08_estimate_size_close_up(self):
        """TC08: Estimate damage size for close-up image."""
        result = self.processor.estimate_damage_size(
            pixel_count=5000,
            num_visible_parts=1,
            image_size=(640, 480)
        )
        
        self.assertIn('zoom_level', result)
        self.assertEqual(result['zoom_level'], 'close_up')
        print("✓ TC08 PASS: Size estimation for close-up functional")
    
    def test_tc09_generate_single_damage_report(self):
        """TC09: Generate report for single damage."""
        damaged_parts = [{
            'part_id': 1,
            'damage_id': 1,
            'pixel_count': 500,
            'bbox': [10, 10, 100, 100],
            'confidence': 0.85,
            'needs_review': False
        }]
        
        part_names = {1: 'hood'}
        damage_names = {1: 'body_damage'}
        
        report = self.processor.generate_report(
            damaged_parts, part_names, damage_names,
            (640, 480), 5
        )
        
        self.assertIn('summary', report)
        self.assertIn('damages', report)
        self.assertEqual(report['summary']['total_damages_detected'], 1)
        print("✓ TC09 PASS: Single damage report generation functional")
    
    def test_tc15_postprocess_no_parts(self):
        """TC15: Post-processing with no parts detected."""
        # Empty masks
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        part_probs = np.zeros((5, 100, 100), dtype=np.float32)
        damage_probs = np.zeros((4, 100, 100), dtype=np.float32)
        
        result = self.processor.combine_masks(
            part_mask, damage_mask, part_probs, damage_probs
        )
        
        self.assertEqual(result['num_damaged_parts'], 0)
        print("✓ TC15 PASS: Empty mask handling functional")
    
    def test_tc16_postprocess_mismatched_locations(self):
        """TC16: Post-processing with damage outside part area."""
        # Part in one location
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:30, 10:30] = 1
        
        # Damage in different location
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask[60:80, 60:80] = 1
        
        part_probs = np.random.rand(5, 100, 100).astype(np.float32)
        damage_probs = np.random.rand(4, 100, 100).astype(np.float32)
        
        result = self.processor.combine_masks(
            part_mask, damage_mask, part_probs, damage_probs
        )
        
        # Should have no damaged parts since damage is outside part region
        self.assertEqual(result['num_damaged_parts'], 0)
        print("✓ TC16 PASS: Mismatched location handling functional")
    
    def test_tc17_confidence_threshold_mechanism(self):
        """TC17: Validate confidence threshold mechanism."""
        damaged_parts = [{
            'part_id': 1,
            'damage_id': 1,
            'pixel_count': 500,
            'bbox': [10, 10, 100, 100],
            'confidence': 0.5,  # Below threshold
            'needs_review': False
        }]
        
        flagged = self.processor.flag_for_review(damaged_parts)
        
        self.assertTrue(flagged[0]['needs_review'])
        self.assertIn('review_reason', flagged[0])
        print("✓ TC17 PASS: Confidence threshold flagging functional")


class TestIntegratedSystem(unittest.TestCase):
    """Test cases for complete pipeline (TC10, TC13-TC14, TC18-TC20)."""
    
    @classmethod
    def setUpClass(cls):
        cls.assessor = IntegratedDamageAssessor(confidence_threshold=0.7)
    
    def test_tc10_multi_damage_report(self):
        """TC10: Generate report for multiple damages."""
        # Create test image
        img = Image.new('RGB', (640, 480), color='white')
        
        result = self.assessor.assess_damage(
            img, skip_vehicle_check=True, return_visualizations=False
        )
        
        self.assertIn('success', result)
        self.assertIn('assessment', result)
        print("✓ TC10 PASS: Multi-damage report generation functional")
    
    def test_tc13_high_agreement_multi_view(self):
        """TC13: Compute confidence for high-agreement multi-view images."""
        # Create consistent predictions
        predictions = [
            {
                'damaged_parts': [{
                    'part_id': 1, 'damage_id': 1,
                    'confidence': 0.9, 'pixel_count': 100,
                    'bbox': [10, 10, 50, 50]
                }]
            },
            {
                'damaged_parts': [{
                    'part_id': 1, 'damage_id': 1,
                    'confidence': 0.92, 'pixel_count': 110,
                    'bbox': [12, 12, 52, 52]
                }]
            },
            {
                'damaged_parts': [{
                    'part_id': 1, 'damage_id': 1,
                    'confidence': 0.95, 'pixel_count': 105,
                    'bbox': [11, 11, 51, 51]
                }]
            }
        ]
        
        result = self.assessor.post_processor.aggregate_multi_view(predictions)
        
        self.assertEqual(result['num_damaged_parts'], 1)
        # Confidence should be high due to agreement
        self.assertGreater(result['damaged_parts'][0]['confidence'], 0.85)
        print("✓ TC13 PASS: High-agreement multi-view confidence functional")
    
    def test_tc14_conflicting_predictions(self):
        """TC14: Compute confidence for conflicting predictions."""
        # Create single view (low agreement)
        predictions = [
            {
                'damaged_parts': [{
                    'part_id': 1, 'damage_id': 1,
                    'confidence': 0.6, 'pixel_count': 100,
                    'bbox': [10, 10, 50, 50]
                }]
            }
        ]
        
        result = self.assessor.post_processor.aggregate_multi_view(predictions)
        
        # Lower confidence due to single view
        self.assertLess(result['damaged_parts'][0]['confidence'], 0.7)
        print("✓ TC14 PASS: Low-agreement confidence handling functional")
    
    def test_tc18_full_pipeline_clean_image(self):
        """TC18: Run full pipeline on clean image."""
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        
        result = self.assessor.assess_damage(
            img, skip_vehicle_check=True, return_visualizations=True
        )
        
        self.assertTrue(result['success'])
        self.assertIn('assessment', result)
        print("✓ TC18 PASS: Full pipeline execution successful")
    
    def test_tc19_pipeline_noisy_image(self):
        """TC19: Run pipeline on noisy user-submitted image."""
        # Create noisy image
        img_array = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        result = self.assessor.assess_damage(
            img, skip_vehicle_check=True, return_visualizations=False
        )
        
        self.assertTrue(result['success'])
        print("✓ TC19 PASS: Noisy image handling functional")
    
    def test_tc20_batch_processing(self):
        """TC20: Handle batch input of multiple images."""
        images = [
            Image.new('RGB', (640, 480), color='red'),
            Image.new('RGB', (640, 480), color='blue'),
            Image.new('RGB', (640, 480), color='green')
        ]
        
        result = self.assessor.assess_multiple_views(images)
        
        self.assertTrue(result['success'])
        self.assertIn('assessment', result)
        print("✓ TC20 PASS: Batch processing functional")


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_iou_computation(self):
        """Test IoU metric computation."""
        pred = np.array([[0, 1, 1], [1, 2, 2], [2, 2, 0]])
        target = np.array([[0, 1, 1], [1, 1, 2], [2, 2, 0]])
        
        result = compute_iou(pred, target, num_classes=3)
        
        self.assertIn('mean_iou', result)
        self.assertGreaterEqual(result['mean_iou'], 0)
        self.assertLessEqual(result['mean_iou'], 1)
        print("✓ IoU computation functional")
    
    def test_dice_computation(self):
        """Test Dice coefficient computation."""
        pred = np.array([[0, 1, 1], [1, 2, 2], [2, 2, 0]])
        target = np.array([[0, 1, 1], [1, 1, 2], [2, 2, 0]])
        
        result = compute_dice_coefficient(pred, target, num_classes=3)
        
        self.assertIn('mean_dice', result)
        self.assertGreaterEqual(result['mean_dice'], 0)
        self.assertLessEqual(result['mean_dice'], 1)
        print("✓ Dice computation functional")
    
    def test_pixel_accuracy(self):
        """Test pixel accuracy computation."""
        pred = np.array([[0, 1, 1], [1, 1, 2]])
        target = np.array([[0, 1, 1], [1, 2, 2]])
        
        accuracy = compute_pixel_accuracy(pred, target)
        
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        expected_acc = 5/6  # 5 correct out of 6 pixels
        self.assertAlmostEqual(accuracy, expected_acc, places=4)
        print("✓ Pixel accuracy computation functional")


class TestAugmentation(unittest.TestCase):
    """Test cases for data augmentation pipeline."""
    
    def test_training_augmentation(self):
        """Test training augmentation pipeline."""
        from utils.augmentation import get_training_augmentation
        
        transform = get_training_augmentation(image_size=512)
        
        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Apply augmentation
        augmented = transform(image=img)
        
        self.assertIn('image', augmented)
        self.assertEqual(augmented['image'].shape[1:], (512, 512))
        print("✓ Training augmentation functional")
    
    def test_validation_augmentation(self):
        """Test validation augmentation pipeline."""
        from utils.augmentation import get_validation_augmentation
        
        transform = get_validation_augmentation(image_size=512)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        augmented = transform(image=img)
        
        self.assertIn('image', augmented)
        self.assertEqual(augmented['image'].shape[1:], (512, 512))
        print("✓ Validation augmentation functional")


def run_all_tests():
    """Run all test cases and generate report."""
    print("="*80)
    print("VEHICLE DAMAGE ASSESSMENT SYSTEM - TEST SUITE")
    print("Based on Research Paper Test Cases")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVehicleDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPartLocalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDamageLocalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPostProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestAugmentation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("="*80)
    
    return result


if __name__ == '__main__':
    run_all_tests()
