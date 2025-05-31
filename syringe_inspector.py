import cv2 as cv
import numpy as np
from datetime import datetime
import os
import glob

def preprocess_frame(frame):
    """Preprocess frame for particle and bubble detection."""
    # Convert to HSV for better color analysis
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Get grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv.Canny(blur, 50, 150)
    
    # Dilate edges to connect broken contours
    kernel = np.ones((3,3), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    
    return edges, gray, hsv

def is_scale_marking(contour, min_aspect_ratio=3.0, max_angle_deviation=20):
    """Check if a contour is likely to be a scale marking."""
    # Get the minimum area rectangle
    rect = cv.minAreaRect(contour)
    (_, _), (width, height), angle = rect
    
    # Normalize angle to 0-90 degrees
    angle = abs(angle) % 90
    
    # Calculate aspect ratio
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
    
    # Check if the contour is elongated and roughly horizontal
    is_horizontal = (angle < max_angle_deviation or angle > 90 - max_angle_deviation)
    is_elongated = aspect_ratio >= min_aspect_ratio
    
    return is_horizontal and is_elongated

def detect_particles(hsv, gray):
    """Detect colored particles in the transparent liquid."""
    # Create a mask for colored regions (excluding black and white/transparent)
    # Define saturation and value thresholds to exclude black and transparent
    lower_thresh = np.array([0, 40, 40])  # Increased minimum saturation and value
    upper_thresh = np.array([180, 255, 255])
    color_mask = cv.inRange(hsv, lower_thresh, upper_thresh)
    
    # Remove very dark regions (scale markings)
    black_thresh = 40  # Increased threshold for black
    not_black_mask = cv.threshold(gray, black_thresh, 255, cv.THRESH_BINARY)[1]
    
    # Combine masks to get only colored regions that aren't black
    particle_mask = cv.bitwise_and(color_mask, not_black_mask)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    particle_mask = cv.morphologyEx(particle_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours of potential particles
    contours, _ = cv.findContours(particle_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    particles = []
    for contour in contours:
        area = cv.contourArea(contour)
        # Filter by size (adjust these thresholds as needed)
        if 15 < area < 400:  # Increased minimum area to reduce false positives
            # Get bounding circle
            (x, y), radius = cv.minEnclosingCircle(contour)
            # Calculate circularity
            perimeter = cv.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Only add if the shape is reasonably circular
            if circularity > 0.6:  # Added circularity check
                # Calculate average color in HSV
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv.mean(hsv, mask=mask)[:3]
                particles.append({
                    'position': (int(x), int(y)),
                    'radius': int(radius),
                    'area': area,
                    'color': mean_color,
                    'circularity': circularity
                })
    
    return particles

def detect_bubbles(gray):
    """Detect small circular bubbles in the grayscale image."""
    # Enhance contrast for better bubble detection
    _, binary = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)  # Increased threshold
    
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    scale_markings = []  # For debugging
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 15:  # Increased minimum area
            continue
            
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Skip if the contour looks like a scale marking
        if is_scale_marking(contour):
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int32(box)
            scale_markings.append(box)
            continue
            
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Get bounding circle
        (x, y), radius = cv.minEnclosingCircle(contour)
        
        # More strict criteria for bubbles
        if 3 <= radius <= 15 and circularity > 0.8:  # Increased circularity threshold
            # Additional check: verify intensity
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv.drawContours(mask, [contour], -1, 255, -1)
            mean_intensity = cv.mean(gray, mask=mask)[0]
            
            # Bubbles should be bright
            if mean_intensity > 180:  # Added intensity threshold
                bubbles.append((int(x), int(y), int(radius)))
    
    return bubbles, scale_markings

def analyze_predictions(frame_results, confidence_threshold=0.7):
    """
    Analyze results across multiple frames of the same syringe.
    Returns final verdict with confidence percentage.
    """
    total_frames = len(frame_results)
    if total_frames == 0:
        return "Unknown", 0.0

    # Count defects and calculate weighted scores
    defect_score = 0
    for result in frame_results:
        # Weight the score based on number and size of defects
        bubble_weight = sum(1.0 for _ in range(result['bubble_count']))
        particle_weight = sum(1.0 for _ in range(result['particle_count']))
        frame_score = (bubble_weight + particle_weight) / 2.0
        defect_score += frame_score
    
    # Normalize the score
    max_possible_score = total_frames  # Assuming max 1 defect per frame
    defect_percentage = (defect_score / max_possible_score) * 100
    pass_percentage = 100 - defect_percentage
    
    # Make final decision based on confidence threshold
    if pass_percentage >= (confidence_threshold * 100):
        return "PASS", pass_percentage
    elif defect_percentage >= (confidence_threshold * 100):
        return "FAIL", defect_percentage
    else:
        return "INCONCLUSIVE", max(pass_percentage, defect_percentage)

def draw_results(frame, bubbles, particles, scale_markings=None):
    """Draw detection results on the frame."""
    # Draw scale markings in green (for debugging)
    if scale_markings is not None:
        for box in scale_markings:
            cv.drawContours(frame, [box], 0, (0, 255, 0), 1)  # Green for scale markings
    
    # Draw bubbles in red
    for x, y, r in bubbles:
        cv.circle(frame, (x, y), r, (0, 0, 255), 2)  # Red for bubbles
        cv.circle(frame, (x, y), 1, (0, 0, 255), 2)
        cv.putText(frame, f"{r*2}px", (x + r + 2, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw particles in yellow
    for particle in particles:
        x, y = particle['position']
        r = particle['radius']
        cv.circle(frame, (x, y), r, (0, 255, 255), 2)  # Yellow for particles
        cv.circle(frame, (x, y), 1, (0, 255, 255), 2)
        cv.putText(frame, f"{int(particle['area'])}px²", (x + r + 2, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Return inspection results for this frame
    is_pass = len(bubbles) == 0 and len(particles) == 0
    return {
        'is_pass': is_pass,
        'bubble_count': len(bubbles),
        'particle_count': len(particles)
    }

def load_and_predict_rois(clf, roi_dir="detected_syringes", size=(64, 64)):
    """
    Load ROIs from directory and predict their class using the trained classifier.
    
    Args:
        clf: Trained classifier
        roi_dir: Directory containing ROI images
        size: Size to resize images to
    
    Returns:
        Dictionary mapping filenames to predictions
    """
    # Create directory if it doesn't exist
    os.makedirs(roi_dir, exist_ok=True)
    
    # Get all image files in the directory
    roi_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        roi_files.extend(glob.glob(os.path.join(roi_dir, ext)))
    
    if not roi_files:
        print(f"\nNo ROIs were processed. Make sure there are images in the '{roi_dir}' directory.")
        return {}
    
    print(f"Found {len(roi_files)} images to process")
    
    predictions = {}
    for img_path in roi_files:
        # Load and preprocess image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Load directly as grayscale
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        # Resize to match training data
        resized = cv.resize(img, size)
        
        # Flatten and reshape for prediction
        features = resized.flatten().reshape(1, -1)
        
        # Get prediction
        prediction = clf.predict(features)[0]
        
        # Store result
        filename = os.path.basename(img_path)
        predictions[filename] = prediction
        print(f"File: {filename} - Prediction: {prediction}")
    
    return predictions

def run_detection():
    """Run the syringe inspection system."""
    print("Starting syringe inspection system...")
    
    # Create output directory for failed inspections
    output_dir = "failed_inspections"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv.VideoCapture(2)  # Iriun camera index
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        print("Please check:")
        print("1. Is Iriun app running on your phone?")
        print("2. Is Iriun client running on your PC?")
        print("3. Are both devices on the same WiFi network?")
        return

    print("Camera connected successfully")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset analysis")

    try:
        # Define capture zone (center rectangle)
        ret, frame = cap.read()
        if not ret:
            print("Failed to get initial frame")
            return
            
        height, width = frame.shape[:2]
        zone_width = int(width * 0.4)   # 40% of frame width
        zone_height = int(height * 0.6)  # 60% of frame height
        x = (width - zone_width) // 2
        y = (height - zone_height) // 2
        capture_zone = (x, y, zone_width, zone_height)

        # Store results for confidence calculation
        frame_results = []
        analysis_active = True
        frames_to_analyze = 30  # Analyze 30 frames before making final decision

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continue

            try:
                # Extract ROI
                x, y, w, h = capture_zone
                roi = frame[y:y+h, x:x+w].copy()
                
                # Process ROI
                edges, gray, hsv = preprocess_frame(roi)
                bubbles, scale_markings = detect_bubbles(gray)
                particles = detect_particles(hsv, gray)
                
                # Draw capture zone
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv.putText(frame, "Place syringe here", (x, y-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Get results and draw on ROI
                frame_result = draw_results(roi, bubbles, particles, scale_markings)
                
                # If analysis is active, collect results
                if analysis_active and len(frame_results) < frames_to_analyze:
                    frame_results.append(frame_result)
                
                # Show current status
                status_color = (0, 255, 0) if frame_result['is_pass'] else (0, 0, 255)
                cv.putText(frame, f"Bubbles: {frame_result['bubble_count']}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv.putText(frame, f"Particles: {frame_result['particle_count']}", (10, 60),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Show analysis progress
                if analysis_active:
                    progress = len(frame_results) / frames_to_analyze * 100
                    cv.putText(frame, f"Analysis Progress: {progress:.1f}%", (10, 90),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # If we have enough frames, show final verdict
                if len(frame_results) >= frames_to_analyze:
                    verdict, confidence = analyze_predictions(frame_results)
                    color = (0, 255, 0) if verdict == "PASS" else (0, 0, 255)
                    cv.putText(frame, f"Final Verdict: {verdict}", (10, 120),
                              cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv.putText(frame, f"Confidence: {confidence:.1f}%", (10, 150),
                              cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    analysis_active = False
                
                # Show edge detection for debugging
                edges_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                
                # Update the ROI in the main frame
                frame[y:y+h, x:x+w] = roi
                
                # Show results
                cv.imshow('Syringe Inspection', frame)
                cv.imshow('Edge Detection', edges_color)

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"syringe_capture_{timestamp}.jpg"
                cv.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
                
                # Save failed inspections
                if not frame_result['is_pass']:
                    fail_filename = os.path.join(output_dir, f"failed_{timestamp}.jpg")
                    cv.imwrite(fail_filename, frame)
                    print(f"Saved failed inspection as {fail_filename}")
            elif key == ord('r'):
                # Reset analysis
                frame_results = []
                analysis_active = True
                print("Analysis reset. Starting new inspection.")

    except Exception as e:
        print(f"Camera error: {str(e)}")
        
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    run_detection()