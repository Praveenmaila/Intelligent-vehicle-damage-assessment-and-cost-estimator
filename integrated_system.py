"""
Integrated Vehicle Damage Assessment System
Complete end-to-end pipeline implementing research paper methodology
"""

import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from models.vehicle_detector import VehicleDetector
from models.part_localizer import VehiclePartLocalizer, VEHICLE_PART_CLASSES
from models.damage_localizer import DamageLocalizer, DAMAGE_TYPE_CLASSES
from models.post_processor import PostProcessor


class IntegratedDamageAssessor:
    """
    End-to-end vehicle damage assessment system.
    
    Pipeline:
    1. Vehicle Detection - Filter out non-vehicle images
    2. Part Localization - Segment vehicle parts (13 classes)
    3. Damage Localization - Segment damage types (3 categories)
    4. Post-Processing - Combine masks, estimate severity, calculate confidence
    5. Report Generation - Create structured assessment report
    
    Implements complete methodology from research paper.
    """
    
    def __init__(self, 
                 vehicle_detector_path: Optional[str] = None,
                 part_localizer_path: Optional[str] = None,
                 damage_localizer_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 device: Optional[str] = None):
        """
        Initialize integrated system.
        
        Args:
            vehicle_detector_path: Path to vehicle detector weights
            part_localizer_path: Path to part localizer weights
            damage_localizer_path: Path to damage localizer weights
            confidence_threshold: Threshold for human review flagging
            device: Device for inference
        """
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print("="*80)
        print("Initializing Intelligent Vehicle Damage Assessment System")
        print("="*80)
        
        # Initialize modules
        print("\n📦 Loading Module 1: Vehicle Detection...")
        self.vehicle_detector = VehicleDetector(vehicle_detector_path, self.device)
        
        print("📦 Loading Module 2: Part Localization...")
        self.part_localizer = VehiclePartLocalizer(part_localizer_path, self.device)
        
        print("📦 Loading Module 3: Damage Localization...")
        self.damage_localizer = DamageLocalizer(damage_localizer_path, self.device)
        
        print("📦 Loading Module 4: Post-Processing...")
        self.post_processor = PostProcessor(confidence_threshold)
        
        # Class mappings
        self.part_names = VEHICLE_PART_CLASSES
        self.damage_names = DAMAGE_TYPE_CLASSES
        
        print("\n✓ System initialized successfully")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print("="*80)
    
    def assess_damage(self, image: Image.Image, 
                     skip_vehicle_check: bool = False,
                     return_visualizations: bool = True) -> Dict:
        """
        Complete damage assessment pipeline.
        
        Args:
            image: PIL Image
            skip_vehicle_check: Skip vehicle presence check (for debugging)
            return_visualizations: Return annotated images
            
        Returns:
            Comprehensive assessment dictionary
        """
        result = {
            'success': False,
            'error': None,
            'vehicle_detected': False,
            'assessment': None
        }
        
        # Step 1: Vehicle Detection
        if not skip_vehicle_check:
            print("\n🔍 Step 1: Vehicle Detection...")
            vehicle_result = self.vehicle_detector.predict(image)
            result['vehicle_detection'] = vehicle_result
            
            if not vehicle_result['has_vehicle']:
                result['error'] = 'No vehicle detected in image'
                print(f"   ❌ No vehicle detected (confidence: {vehicle_result['confidence']}%)")
                return result
            
            result['vehicle_detected'] = True
            print(f"   ✓ Vehicle detected (confidence: {vehicle_result['confidence']}%)")
        else:
            result['vehicle_detected'] = True
        
        # Step 2: Part Localization
        print("\n🔍 Step 2: Vehicle Part Localization...")
        part_result = self.part_localizer.predict(
            image, 
            return_mask=True, 
            return_probabilities=True
        )
        
        num_parts = part_result['num_parts']
        print(f"   ✓ Detected {num_parts} vehicle parts")
        print(f"   Parts: {list(part_result['part_counts'].keys())}")
        
        # Step 3: Damage Localization
        print("\n🔍 Step 3: Damage Localization...")
        damage_result = self.damage_localizer.predict(
            image,
            return_mask=True,
            return_probabilities=True
        )
        
        has_damage = damage_result['has_damage']
        if not has_damage:
            result['success'] = True
            result['assessment'] = {
                'has_damage': False,
                'message': 'No damage detected',
                'num_parts_visible': num_parts
            }
            print("   ✓ No damage detected")
            return result
        
        print(f"   ✓ Damage detected (confidence: {damage_result['average_confidence']}%)")
        print(f"   Damage types: {list(damage_result['damage_counts'].keys())}")
        
        # Step 4: Post-Processing
        print("\n🔍 Step 4: Post-Processing & Analysis...")
        
        # Combine masks
        combined = self.post_processor.combine_masks(
            part_result['mask'],
            damage_result['mask'],
            part_result['probabilities'],
            damage_result['probabilities']
        )
        
        if combined['num_damaged_parts'] == 0:
            result['success'] = True
            result['assessment'] = {
                'has_damage': False,
                'message': 'Damage detected but not associated with vehicle parts',
                'num_parts_visible': num_parts
            }
            print("   ⚠️  Damage detected but not on vehicle parts")
            return result
        
        print(f"   ✓ Identified {combined['num_damaged_parts']} damaged part instances")
        
        # Flag for review
        damaged_parts_flagged = self.post_processor.flag_for_review(
            combined['damaged_parts']
        )
        
        # Generate report
        report = self.post_processor.generate_report(
            damaged_parts_flagged,
            self.part_names,
            self.damage_names,
            image.size,
            num_parts
        )
        
        print(f"   ✓ Assessment complete")
        print(f"   Total damages: {report['summary']['total_damages_detected']}")
        print(f"   Average confidence: {report['summary']['average_confidence']}%")
        print(f"   Human review needed: {report['summary']['needs_human_review']}")
        
        # Step 5: Visualization
        if return_visualizations:
            print("\n🔍 Step 5: Generating Visualizations...")
            
            # Annotated image with bounding boxes
            annotated_image = self.post_processor.visualize_results(
                image,
                damaged_parts_flagged,
                self.part_names,
                self.damage_names
            )
            
            # Colored segmentation masks
            part_colored = self.part_localizer.get_colored_mask(part_result['mask'])
            damage_colored = self.damage_localizer.get_colored_mask(damage_result['mask'])
            
            report['visualizations'] = {
                'annotated_image': annotated_image,
                'part_segmentation': Image.fromarray(part_colored),
                'damage_segmentation': Image.fromarray(damage_colored)
            }
            
            print("   ✓ Visualizations ready")
        
        result['success'] = True
        result['assessment'] = report
        
        return result
    
    def assess_multiple_views(self, images: List[Image.Image]) -> Dict:
        """
        Assess damage from multiple images (multi-view aggregation).
        
        Implements cross-view consistency checking from research paper.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Aggregated assessment
        """
        print(f"\n🔍 Multi-View Assessment ({len(images)} images)")
        print("="*80)
        
        predictions = []
        
        for i, image in enumerate(images, 1):
            print(f"\n📸 Processing Image {i}/{len(images)}...")
            result = self.assess_damage(image, return_visualizations=False)
            
            if result['success'] and result['assessment'].get('has_damage'):
                predictions.append(result['assessment'])
        
        if not predictions:
            return {
                'success': True,
                'assessment': {
                    'has_damage': False,
                    'message': 'No damage detected across all views',
                    'num_images': len(images)
                }
            }
        
        # Aggregate across views
        print("\n🔍 Aggregating Across Views...")
        aggregated_parts = []
        
        for pred in predictions:
            aggregated_parts.extend(pred.get('damages', []))
        
        # Use post-processor to aggregate
        aggregated_result = self.post_processor.aggregate_multi_view(predictions)
        
        # Generate final report
        if aggregated_result['damaged_parts']:
            # Re-flag with updated confidences
            flagged = self.post_processor.flag_for_review(
                aggregated_result['damaged_parts']
            )
            
            # Get representative image size and parts from first prediction
            first_pred = predictions[0]
            num_parts = first_pred['summary']['num_visible_parts']
            image_size = images[0].size
            
            final_report = self.post_processor.generate_report(
                flagged,
                self.part_names,
                self.damage_names,
                image_size,
                num_parts
            )
            
            final_report['summary']['num_images'] = len(images)
            final_report['summary']['aggregation_method'] = 'multi_view_consensus'
            
            print(f"\n✓ Multi-View Assessment Complete")
            print(f"   Images processed: {len(images)}")
            print(f"   Total unique damages: {final_report['summary']['total_damages_detected']}")
            print(f"   Average confidence: {final_report['summary']['average_confidence']}%")
            
            return {
                'success': True,
                'assessment': final_report
            }
        
        return {
            'success': True,
            'assessment': {
                'has_damage': False,
                'message': 'No consistent damage across views',
                'num_images': len(images)
            }
        }
    
    def get_system_info(self) -> Dict:
        """Get information about the assessment system."""
        return {
            'vehicle_detector': self.vehicle_detector.get_model_info(),
            'part_localizer': self.part_localizer.get_model_info(),
            'damage_localizer': self.damage_localizer.get_model_info(),
            'device': str(self.device),
            'confidence_threshold': self.post_processor.confidence_threshold,
            'num_part_classes': len(self.part_names),
            'num_damage_classes': len(self.damage_names)
        }


def quick_assess(image_path: str, save_output: bool = True, 
                output_dir: str = 'assessment_results') -> Dict:
    """
    Quick assessment function for standalone use.
    
    Args:
        image_path: Path to image
        save_output: Whether to save results
        output_dir: Directory for outputs
        
    Returns:
        Assessment results
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Initialize system
    assessor = IntegratedDamageAssessor()
    
    # Run assessment
    result = assessor.assess_damage(image, return_visualizations=True)
    
    if save_output and result['success']:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save visualizations
        if 'visualizations' in result['assessment']:
            vis = result['assessment']['visualizations']
            
            vis['annotated_image'].save(
                os.path.join(output_dir, f'{base_name}_annotated.jpg')
            )
            vis['part_segmentation'].save(
                os.path.join(output_dir, f'{base_name}_parts.jpg')
            )
            vis['damage_segmentation'].save(
                os.path.join(output_dir, f'{base_name}_damage.jpg')
            )
            
            print(f"\n✓ Results saved to {output_dir}/")
    
    return result


if __name__ == '__main__':
    print("Integrated Damage Assessment System")
    print("="*80)
    
    # Initialize system
    assessor = IntegratedDamageAssessor()
    
    # Print system info
    info = assessor.get_system_info()
    print("\n📊 System Information:")
    print(f"   Vehicle Detector: {info['vehicle_detector']['architecture']}")
    print(f"   Part Localizer: {info['part_localizer']['architecture']}")
    print(f"   Damage Localizer: {info['damage_localizer']['architecture']}")
    print(f"   Device: {info['device']}")
    print(f"   Part Classes: {info['num_part_classes']}")
    print(f"   Damage Classes: {info['num_damage_classes']}")
    
    print("\n✓ System ready for assessment")
