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
    no_bubbles: bool
    confidence: float

class SyringeInspector:
    def __init__(self):
        # ROI parameters (percentages of frame size)
        self.roi_x_percent = 0.25
        self.roi_y_percent = 0.2
        self.roi_width_percent = 0.5
        self.roi_height_percent = 0.6
        
        # Bubble detection parameters - Increased sensitivity
        self.bubble_min_radius = 2  # Smaller minimum radius to detect tinier bubbles
        self.bubble_max_radius = 20  # Increased maximum radius
        self.bubble_sensitivity = 12  # Lower value for more sensitive detection
        self.bubble_intensity_min = 130  # Lower brightness threshold
        self.bubble_intensity_diff = 20  # Lower required brightness difference
        
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

    def detect_bubbles(self, gray: np.ndarray) -> Tuple[bool, int]:
        """Detect very small air bubbles while ignoring printed markings."""
        # Step 1: Enhanced image preprocessing for better sensitivity
        # Apply stronger contrast enhancement
        clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # Apply sharpening to make bubbles more distinct
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv.filter2D(enhanced, -1, kernel)
        
        # Bilateral filter to reduce noise while preserving bubble edges
        denoised = cv.bilateralFilter(sharpened, 5, 50, 50)
        
        # Step 2: Create a more sensitive mask for potential bubble regions
        local_mean = cv.GaussianBlur(denoised, (15, 15), 0)
        diff = cv.subtract(denoised, local_mean)
        _, bright_mask = cv.threshold(diff, self.bubble_intensity_diff, 255, cv.THRESH_BINARY)
        
        # Apply morphological operations to enhance bubble regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        bright_mask = cv.morphologyEx(bright_mask, cv.MORPH_OPEN, kernel)
        
        # Step 3: Multi-scale circle detection for better sensitivity
        all_circles = []
        
        # Detect circles at multiple sensitivity levels
        for sensitivity in [self.bubble_sensitivity, self.bubble_sensitivity + 2]:
            circles = cv.HoughCircles(
                denoised,
                cv.HOUGH_GRADIENT,
                dp=1,
                minDist=8,  # Reduced minimum distance between circles
                param1=40,  # Reduced edge threshold
                param2=sensitivity,
                minRadius=self.bubble_min_radius,
                maxRadius=self.bubble_max_radius
            )
            
            if circles is not None:
                all_circles.extend(circles[0])
        
        if not all_circles:
            return True, 0
        
        # Convert to numpy array for processing
        all_circles = np.array(all_circles)
        
        # Step 4: Validate detected circles with more lenient criteria
        valid_bubbles = []
        for circle in all_circles:
            x, y, r = np.uint16(np.around(circle))
            
            # Check if circle is within bounds
            if not (r < x < gray.shape[1]-r and r < y < gray.shape[0]-r):
                continue
            
            # Create masks for analysis
            bubble_mask = np.zeros_like(gray)
            cv.circle(bubble_mask, (x, y), r, 255, -1)
            
            # Analyze intensity profile with more lenient thresholds
            roi = denoised[y-r:y+r, x-r:x+r]
            mask_roi = bubble_mask[y-r:y+r, x-r:x+r]
            
            mean_intensity = cv.mean(roi, mask=mask_roi)[0]
            
            # Create a ring around the bubble
            outer_mask = np.zeros_like(gray)
            cv.circle(outer_mask, (x, y), r+2, 255, 2)
            surrounding_intensity = cv.mean(denoised, mask=outer_mask)[0]
            
            # More lenient criteria for bubble detection
            if (mean_intensity > self.bubble_intensity_min or  # Bright enough
                mean_intensity > surrounding_intensity + self.bubble_intensity_diff):  # OR significantly brighter than surroundings
                
                # Check if this circle overlaps with any previously detected ones
                is_unique = True
                for (px, py, pr) in valid_bubbles:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < (r + pr):
                        is_unique = False
                        break
                
                if is_unique:
                    valid_bubbles.append((x, y, r))
                    # Draw the detected bubble (for visualization)
                    cv.circle(gray, (x, y), r, (0, 255, 0), 1)  # Circle
                    cv.circle(gray, (x, y), 1, (0, 0, 255), 1)  # Center point
        
        return len(valid_bubbles) == 0, len(valid_bubbles)

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, InspectionResult]:
        """Analyze a single frame and return inspection results."""
        # Get ROI
        frame, (x, y, w, h) = self.define_roi(frame)
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale and normalize
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
        
        # Perform bubble detection
        no_bubbles, bubble_count = self.detect_bubbles(gray)
        
        # Calculate confidence based on bubble detection
        confidence = 1.0 if no_bubbles else 0.5
        
        # Determine overall result
        is_pass = no_bubbles
        
        # Draw results on frame
        self._draw_results(frame, is_pass, no_bubbles)
        
        # Draw bubble count
        cv.putText(frame, f"Bubbles: {bubble_count}", 
                  (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, InspectionResult(
            is_pass=is_pass,
            no_bubbles=no_bubbles,
            confidence=confidence
        )

    def _draw_results(self, frame: np.ndarray, is_pass: bool, no_bubbles: bool):
        """Draw inspection results on the frame."""
        # Overall result
        status = "PASS" if is_pass else "FAIL"
        color = (0, 255, 0) if is_pass else (0, 0, 255)
        cv.putText(frame, f"Status: {status}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Bubble check
        check_color = (0, 255, 0) if no_bubbles else (0, 0, 255)
        cv.putText(frame, f"No Bubbles: {'✓' if no_bubbles else '✗'}", 
                  (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, check_color, 2)

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