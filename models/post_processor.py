"""
Post-Processing Module
Combines part and damage predictions, estimates severity, computes confidence
Based on research paper methodology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage


class PostProcessor:
    """
    Post-processing for vehicle damage assessment.
    
    Key functions:
    1. Combine part and damage masks to identify damaged parts
    2. Estimate damage size based on number of visible parts (camera distance proxy)
    3. Calculate confidence scores from probabilities
    4. Aggregate predictions across multiple views
    5. Flag low-confidence predictions for human review
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize post-processor.
        
        Args:
            confidence_threshold: Threshold for flagging human review (0-1)
        """
        self.confidence_threshold = confidence_threshold
        
        # Damage size thresholds based on number of visible parts
        # More parts visible = wider angle = object appears smaller
        self.size_thresholds = {
            'close_up': (1, 2),      # 1-2 parts visible
            'medium': (3, 5),        # 3-5 parts visible
            'wide': (6, 100)         # 6+ parts visible
        }
    
    def combine_masks(self, part_mask: np.ndarray, damage_mask: np.ndarray,
                     part_probs: np.ndarray, damage_probs: np.ndarray) -> Dict:
        """
        Combine part and damage segmentation masks.
        
        Args:
            part_mask: Part segmentation (H x W) with part class IDs
            damage_mask: Damage segmentation (H x W) with damage class IDs
            part_probs: Part probabilities (C_part x H x W)
            damage_probs: Damage probabilities (C_damage x H x W)
            
        Returns:
            Dictionary with damaged part instances and their properties
        """
        # Find damaged regions (exclude no_damage class 0)
        damaged_pixels = damage_mask > 0
        
        if not np.any(damaged_pixels):
            return {
                'damaged_parts': [],
                'num_damaged_parts': 0,
                'total_damage_pixels': 0
            }
        
        # Get damaged parts by intersection
        damaged_parts_mask = part_mask * damaged_pixels
        
        # Find unique damaged parts (exclude background 0)
        unique_damaged_parts = np.unique(damaged_parts_mask)
        unique_damaged_parts = unique_damaged_parts[unique_damaged_parts > 0]
        
        damaged_part_instances = []
        
        for part_id in unique_damaged_parts:
            # Get all pixels of this damaged part
            part_damaged_pixels = (part_mask == part_id) & damaged_pixels
            
            if not np.any(part_damaged_pixels):
                continue
            
            # Get damage types present on this part
            damage_types_on_part = damage_mask[part_damaged_pixels]
            unique_damages = np.unique(damage_types_on_part)
            unique_damages = unique_damages[unique_damages > 0]
            
            for damage_id in unique_damages:
                # Get pixels with this specific part-damage combination
                instance_pixels = (part_mask == part_id) & (damage_mask == damage_id)
                pixel_count = np.sum(instance_pixels)
                
                if pixel_count == 0:
                    continue
                
                # Label connected components to separate individual damage instances
                labeled, num_instances = ndimage.label(instance_pixels)
                
                for instance_id in range(1, num_instances + 1):
                    instance_mask = (labeled == instance_id)
                    instance_pixel_count = np.sum(instance_mask)
                    
                    if instance_pixel_count < 10:  # Filter very small regions (noise)
                        continue
                    
                    # Get bounding box
                    coords = np.argwhere(instance_mask)
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                    
                    # Calculate average confidence for this instance
                    # Use probabilities from both part and damage predictions
                    part_conf = part_probs[part_id][instance_mask].mean()
                    damage_conf = damage_probs[damage_id][instance_mask].mean()
                    avg_confidence = (part_conf + damage_conf) / 2.0
                    
                    damaged_part_instances.append({
                        'part_id': int(part_id),
                        'damage_id': int(damage_id),
                        'pixel_count': int(instance_pixel_count),
                        'bbox': bbox,
                        'confidence': float(avg_confidence),
                        'mask': instance_mask
                    })
        
        return {
            'damaged_parts': damaged_part_instances,
            'num_damaged_parts': len(damaged_part_instances),
            'total_damage_pixels': int(np.sum(damaged_pixels))
        }
    
    def estimate_damage_size(self, pixel_count: int, num_visible_parts: int,
                            image_size: Tuple[int, int]) -> Dict:
        """
        Estimate damage size/severity based on pixel count and camera distance.
        
        Uses number of visible vehicle parts as proxy for camera distance:
        - More parts visible = wider angle = damage appears smaller
        - Fewer parts visible = zoomed in = damage appears larger
        
        Args:
            pixel_count: Number of damaged pixels
            num_visible_parts: Number of vehicle parts visible in image
            image_size: (width, height) of image
            
        Returns:
            Dictionary with size category and normalized area
        """
        total_pixels = image_size[0] * image_size[1]
        damage_ratio = pixel_count / total_pixels
        
        # Adjust damage ratio based on zoom level
        if num_visible_parts <= 2:
            zoom_level = 'close_up'
            # Close-up: damage looks large, but actual size may be small
            adjusted_ratio = damage_ratio * 0.5
        elif num_visible_parts <= 5:
            zoom_level = 'medium'
            adjusted_ratio = damage_ratio * 0.75
        else:
            zoom_level = 'wide'
            # Wide angle: damage looks small but may be large
            adjusted_ratio = damage_ratio * 1.5
        
        # Categorize size
        if adjusted_ratio < 0.01:
            size_category = 'minor'
            severity_score = 1
        elif adjusted_ratio < 0.05:
            size_category = 'moderate'
            severity_score = 2
        elif adjusted_ratio < 0.15:
            size_category = 'major'
            severity_score = 3
        else:
            size_category = 'severe'
            severity_score = 4
        
        return {
            'size_category': size_category,
            'severity_score': severity_score,
            'damage_ratio': round(damage_ratio * 100, 2),
            'adjusted_ratio': round(adjusted_ratio * 100, 2),
            'zoom_level': zoom_level,
            'num_visible_parts': num_visible_parts
        }
    
    def aggregate_multi_view(self, predictions: List[Dict]) -> Dict:
        """
        Aggregate predictions from multiple images of same vehicle.
        
        Implements cross-view consistency checking:
        - If same part + damage appears in multiple views -> increase confidence
        - If predictions disagree -> decrease confidence
        
        Args:
            predictions: List of prediction dictionaries from multiple images
            
        Returns:
            Aggregated assessment with confidence adjustment
        """
        if len(predictions) == 1:
            return predictions[0]
        
        # Build consensus for each part-damage combination
        part_damage_votes = {}  # (part_id, damage_id) -> [confidences]
        
        for pred in predictions:
            for damaged_part in pred.get('damaged_parts', []):
                key = (damaged_part['part_id'], damaged_part['damage_id'])
                confidence = damaged_part['confidence']
                
                if key not in part_damage_votes:
                    part_damage_votes[key] = []
                part_damage_votes[key].append(confidence)
        
        # Calculate aggregated results
        aggregated_parts = []
        
        for (part_id, damage_id), confidences in part_damage_votes.items():
            num_views = len(confidences)
            avg_confidence = np.mean(confidences)
            
            # Agreement factor: more views with same detection = higher confidence
            agreement_factor = min(num_views / len(predictions), 1.0)
            adjusted_confidence = avg_confidence * (0.7 + 0.3 * agreement_factor)
            
            # Find representative instance (highest confidence)
            best_pred_idx = 0
            best_conf = 0
            for i, pred in enumerate(predictions):
                for dp in pred.get('damaged_parts', []):
                    if dp['part_id'] == part_id and dp['damage_id'] == damage_id:
                        if dp['confidence'] > best_conf:
                            best_conf = dp['confidence']
                            best_pred_idx = i
            
            # Get representative damaged part
            for dp in predictions[best_pred_idx].get('damaged_parts', []):
                if dp['part_id'] == part_id and dp['damage_id'] == damage_id:
                    aggregated_part = dp.copy()
                    aggregated_part['confidence'] = float(adjusted_confidence)
                    aggregated_part['num_views'] = num_views
                    aggregated_part['agreement_factor'] = float(agreement_factor)
                    aggregated_parts.append(aggregated_part)
                    break
        
        return {
            'damaged_parts': aggregated_parts,
            'num_damaged_parts': len(aggregated_parts),
            'num_images': len(predictions),
            'aggregation_method': 'multi_view_consensus'
        }
    
    def flag_for_review(self, damaged_parts: List[Dict]) -> List[Dict]:
        """
        Flag low-confidence predictions for human review.
        
        Args:
            damaged_parts: List of damaged part instances
            
        Returns:
            List with 'needs_review' flag added to each instance
        """
        for part in damaged_parts:
            confidence = part['confidence']
            part['needs_review'] = confidence < self.confidence_threshold
            
            if part['needs_review']:
                part['review_reason'] = f"Low confidence ({confidence:.2%})"
        
        return damaged_parts
    
    def generate_report(self, damaged_parts: List[Dict], part_names: Dict,
                       damage_names: Dict, image_size: Tuple[int, int],
                       num_visible_parts: int) -> Dict:
        """
        Generate comprehensive damage assessment report.
        
        Args:
            damaged_parts: List of damaged part instances
            part_names: Mapping of part IDs to names
            damage_names: Mapping of damage IDs to names
            image_size: (width, height)
            num_visible_parts: Number of vehicle parts detected
            
        Returns:
            Structured damage report
        """
        report_items = []
        total_confidence = 0
        needs_review_count = 0
        
        for dp in damaged_parts:
            part_name = part_names.get(dp['part_id'], f"Unknown Part {dp['part_id']}")
            damage_name = damage_names.get(dp['damage_id'], f"Unknown Damage {dp['damage_id']}")
            
            # Estimate size
            size_info = self.estimate_damage_size(
                dp['pixel_count'],
                num_visible_parts,
                image_size
            )
            
            confidence = dp['confidence']
            total_confidence += confidence
            
            if dp.get('needs_review', False):
                needs_review_count += 1
            
            report_item = {
                'part': part_name,
                'damage_type': damage_name,
                'severity': size_info['size_category'],
                'severity_score': size_info['severity_score'],
                'confidence': round(confidence * 100, 2),
                'bbox': dp['bbox'],
                'needs_review': dp.get('needs_review', False),
                'details': {
                    'pixel_count': dp['pixel_count'],
                    'damage_ratio': size_info['damage_ratio'],
                    'zoom_level': size_info['zoom_level']
                }
            }
            
            report_items.append(report_item)
        
        # Calculate overall metrics
        num_damages = len(report_items)
        avg_confidence = (total_confidence / num_damages * 100) if num_damages > 0 else 0
        
        # Sort by severity (highest first)
        report_items.sort(key=lambda x: x['severity_score'], reverse=True)
        
        return {
            'summary': {
                'total_damages_detected': num_damages,
                'average_confidence': round(avg_confidence, 2),
                'needs_human_review': needs_review_count > 0,
                'items_flagged': needs_review_count,
                'num_visible_parts': num_visible_parts
            },
            'damages': report_items
        }
    
    def visualize_results(self, image: Image.Image, damaged_parts: List[Dict],
                         part_names: Dict, damage_names: Dict) -> Image.Image:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Original PIL Image
            damaged_parts: List of damaged part instances
            part_names: Part ID to name mapping
            damage_names: Damage ID to name mapping
            
        Returns:
            Annotated PIL Image
        """
        # Create copy
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        # Color map for damage types
        colors = {
            1: (255, 0, 0),      # body_damage - blue
            2: (0, 255, 0),      # surface_damage - green
            3: (0, 0, 255)       # deformity - red
        }
        
        for dp in damaged_parts:
            bbox = dp['bbox']
            damage_id = dp['damage_id']
            confidence = dp['confidence']
            
            part_name = part_names.get(dp['part_id'], f"Part{dp['part_id']}")
            damage_name = damage_names.get(damage_id, f"Damage{damage_id}")
            
            color = colors.get(damage_id, (128, 128, 128))
            
            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label
            label = f"{part_name}: {damage_name} ({confidence*100:.1f}%)"
            
            # Label background
            bbox_text = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
            draw.rectangle([bbox_text[0]-2, bbox_text[1]-2, 
                          bbox_text[2]+2, bbox_text[3]+2], fill=color)
            
            # Label text
            draw.text((bbox[0], bbox[1] - 25), label, fill=(255, 255, 255), font=font)
            
            # Flag icon for review items
            if dp.get('needs_review', False):
                flag_pos = (bbox[2] - 20, bbox[1] + 5)
                draw.text(flag_pos, "⚠", fill=(255, 255, 0), font=font)
        
        return img_draw


if __name__ == '__main__':
    # Example usage
    processor = PostProcessor(confidence_threshold=0.7)
    print("Post-Processor initialized successfully")
    print(f"Confidence threshold for review: {processor.confidence_threshold}")
