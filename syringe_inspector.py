import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import time
import os
from datetime import datetime

@dataclass
class InspectionResult:
    is_pass: bool
    no_small_bubbles: bool
    no_large_bubbles: bool
    confidence: float

class SyringeInspector:
    def __init__(self):
        # ROI parameters
        self.roi_x_percent = 0.25
        self.roi_y_percent = 0.2
        self.roi_width_percent = 0.5
        self.roi_height_percent = 0.6
        
        # Small bubble detection parameters
        self.small_bubble_min_radius = 2
        self.small_bubble_max_radius = 10
        
        # Large oval bubble parameters
        self.large_bubble_min_area = 200  # Minimum area for large bubbles
        self.large_bubble_max_area = 5000  # Maximum area for large bubbles
        self.min_oval_ratio = 1.5  # Minimum width/height ratio for ovals
        
        # Create output directory for failed inspections
        self.output_dir = "failed_inspections"
        os.makedirs(self.output_dir, exist_ok=True)

    def define_roi(self, frame) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Define and draw the Region of Interest for syringe placement."""
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates
        roi_x = int(width * self.roi_x_percent)
        roi_width = int(width * self.roi_width_percent)
        roi_y = int(height * self.roi_y_percent)
        roi_height = int(height * self.roi_height_percent)
        
        # Draw guide rectangle
        cv.rectangle(frame, (roi_x, roi_y), 
                    (roi_x + roi_width, roi_y + roi_height),
                    (0, 255, 0), 2)
        
        # Draw centerline for alignment
        center_x = roi_x + roi_width // 2
        cv.line(frame, (center_x, roi_y), 
                (center_x, roi_y + roi_height),
                (0, 255, 0), 1)
        
        return frame, (roi_x, roi_y, roi_width, roi_height)

    def detect_bubbles(self, gray: np.ndarray) -> Tuple[bool, bool, list, list]:
        """Detect both small circular and large oval bubbles."""
        # Enhance contrast
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(enhanced, (3,3), 0)
        
        # Edge detection
        edges = cv.Canny(blurred, 30, 150)
        
        # Dilate edges slightly to ensure closed contours
        kernel = np.ones((2,2), np.uint8)
        dilated = cv.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        small_bubbles = []
        large_bubbles = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Get rotated rectangle for oval detection
            rect = cv.minAreaRect(contour)
            (x, y), (width, height), angle = rect
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # Check intensity
            mask = np.zeros_like(gray)
            cv.drawContours(mask, [contour], 0, 255, -1)
            mean_intensity = cv.mean(enhanced, mask=mask)[0]
            
            # Check surrounding intensity
            surrounding_mask = cv.dilate(mask, kernel, iterations=3) - mask
            surrounding_intensity = cv.mean(enhanced, mask=surrounding_mask)[0]
            intensity_diff = abs(mean_intensity - surrounding_intensity)
            
            if area < self.large_bubble_min_area:
                # Small bubble detection
                (x, y), radius = cv.minEnclosingCircle(contour)
                radius = int(radius)
                
                if (self.small_bubble_min_radius <= radius <= self.small_bubble_max_radius and
                    circularity > 0.7 and intensity_diff > 20):
                    small_bubbles.append((int(x), int(y), radius))
                    
            else:
                # Large oval bubble detection
                if (self.large_bubble_min_area <= area <= self.large_bubble_max_area and
                    aspect_ratio >= self.min_oval_ratio and
                    intensity_diff > 20):
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    large_bubbles.append((box, (int(x), int(y)), (int(width), int(height)), angle))
        
        return (len(small_bubbles) == 0, len(large_bubbles) == 0,
                small_bubbles, large_bubbles)

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, InspectionResult]:
        """Analyze a single frame and return inspection results."""
        # Get ROI
        frame, (x, y, w, h) = self.define_roi(frame)
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # Perform bubble detection
        no_small_bubbles, no_large_bubbles, small_bubbles, large_bubbles = self.detect_bubbles(gray)
        
        # Draw small bubbles
        for bx, by, br in small_bubbles:
            cv.circle(roi, (bx, by), br, (0, 0, 255), 1)
            cv.circle(roi, (bx, by), 1, (0, 0, 255), 1)
            cv.putText(roi, f"{br*2}µm", (bx + br + 2, by),
                      cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Draw large oval bubbles
        for box, center, (width, height), angle in large_bubbles:
            cv.drawContours(roi, [box], 0, (255, 0, 0), 2)
            cx, cy = center
            cv.putText(roi, f"{int(max(width, height))}µm", (cx + 5, cy),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Calculate confidence and overall result
        confidence = 1.0 if (no_small_bubbles and no_large_bubbles) else 0.5
        is_pass = no_small_bubbles and no_large_bubbles
        
        # Draw results
        self._draw_results(frame, is_pass, no_small_bubbles, no_large_bubbles,
                          len(small_bubbles), len(large_bubbles))
        
        return frame, InspectionResult(
            is_pass=is_pass,
            no_small_bubbles=no_small_bubbles,
            no_large_bubbles=no_large_bubbles,
            confidence=confidence
        )

    def _draw_results(self, frame: np.ndarray, is_pass: bool,
                     no_small_bubbles: bool, no_large_bubbles: bool,
                     small_count: int, large_count: int):
        """Draw inspection results on the frame."""
        # Overall result
        status = "PASS" if is_pass else "FAIL"
        color = (0, 255, 0) if is_pass else (0, 0, 255)
        cv.putText(frame, f"Status: {status}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Bubble counts
        y_pos = 60
        for label, count, is_ok in [
            ("Small Bubbles", small_count, no_small_bubbles),
            ("Large Bubbles", large_count, no_large_bubbles)
        ]:
            check_color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv.putText(frame, f"{label}: {count}", 
                      (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, check_color, 2)
            y_pos += 25

    def save_failed_inspection(self, frame: np.ndarray):
        """Save failed inspection images for review."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"failed_{timestamp}.jpg")
        cv.imwrite(filename, frame)

    def run(self):
        """Run the inspection system."""
        cap = cv.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting syringe inspection system...")
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
                
            # Analyze frame
            frame, result = self.analyze_frame(frame)
            
            # Save failed inspections
            if not result.is_pass:
                self.save_failed_inspection(frame)
            
            # Display the frame
            cv.imshow('Syringe Inspection', frame)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv.imwrite(f"syringe_inspection_{timestamp}.jpg", frame)
                print(f"Frame saved as syringe_inspection_{timestamp}.jpg")
        
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    inspector = SyringeInspector()
    inspector.run() 