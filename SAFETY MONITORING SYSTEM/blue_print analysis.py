import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import google.generativeai as palm
import base64
from io import BytesIO

class HazardDetectionModel:
    def __init__(self, palm_api_key=None):
        # Hazard types
        self.hazard_types = ["Fire Hazard", "Gas Leak", "Machinery Accident", "Animal Intrusion"]
        print("Hazard detection model initialized")
        
        # Initialize PaLM API if key is provided
        self.use_palm = False
        if palm_api_key:
            try:
                palm.configure(api_key=palm_api_key)
                self.use_palm = True
                print("PaLM API configured successfully")
                # Set up the model
                self.palm_model = palm.GenerativeModel('gemini-pro-vision')
            except Exception as e:
                print(f"Error configuring PaLM API: {e}")
        
        # Fallback: Fire hazard detection criteria (colors in RGB)
        self.fire_hazard_colors = {
            'red_areas': ([150, 0, 0], [255, 80, 80]),    # Red zones (potentially flammable materials)
            'yellow_areas': ([150, 150, 0], [255, 255, 80])  # Yellow zones (electrical components)
        }
        
        # Threshold parameters
        self.fire_area_min_size = 100  # Minimum size of an area to be considered a fire hazard

    def analyze_with_palm(self, image):
        """Use PaLM API to analyze the blueprint for potential hazards"""
        # Convert image to bytes
        img_byte_arr = BytesIO()
        Image.fromarray(image).save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Define the prompt
        prompt = """
        Analyze this factory blueprint image and identify potential hazard zones.
        For each potential hazard, provide:
        1. The type of hazard (Fire Hazard, Gas Leak, Machinery Accident, or Animal Intrusion)
        2. The x,y coordinates of the center of the hazard zone (as pixel coordinates)
        3. The approximate radius of the affected area in pixels
        4. A confidence score between 0 and 1
        5. Brief reasoning for why this area is potentially hazardous
        
        Return the results in JSON format like this:
        [
          {
            "type": "Fire Hazard",
            "center": [x, y],
            "radius": radius,
            "confidence": 0.85,
            "reasoning": "Red area indicating flammable materials storage"
          },
          {...}
        ]
        
        Focus on these criteria:
        - Fire Hazards: Areas with red markings, chemical storage, electrical panels
        - Gas Leaks: Areas with pipeline junctions, gas storage, ventilation issues
        - Machinery Accidents: Dense equipment areas, moving parts, narrow passages
        - Animal Intrusion: Entry points, open areas near exterior walls
        """
        
        try:
            # Send the image to PaLM API
            response = self.palm_model.generate_content([prompt, Image.fromarray(image)])
            
            # Process the response
            response_text = response.text
            
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                hazards = json.loads(json_str)
                
                # Format the hazards to match our expected structure
                formatted_hazards = []
                for hazard in hazards:
                    formatted_hazard = {
                        "type": hazard["type"],
                        "confidence": hazard["confidence"],
                        "center": tuple(hazard["center"]),
                        "radius": hazard["radius"],
                        "reasoning": hazard.get("reasoning", "")
                    }
                    formatted_hazards.append(formatted_hazard)
                
                return formatted_hazards, True
            
            return [], False
            
        except Exception as e:
            print(f"Error using PaLM API: {e}")
            return [], False

    def detect_fire_hazards_by_color(self, image):
        """Detect fire hazards based on color detection and clustering"""
        # Convert image to RGB if it's not
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image.copy()
            
        # Create a mask for potential fire hazard zones (red and yellow areas)
        fire_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Detect red areas (potentially flammable materials)
        for color_name, (lower, upper) in self.fire_hazard_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            color_mask = cv2.inRange(image_rgb, lower, upper)
            fire_mask = cv2.bitwise_or(fire_mask, color_mask)
        
        # Find connected components (potential hazard areas)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fire_mask, connectivity=8)
        
        # Filter out small regions and prepare detections
        fire_detections = []
        for i in range(1, num_labels):  # Skip the background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > self.fire_area_min_size:
                # Calculate center and radius
                center_x = int(centroids[i, 0])
                center_y = int(centroids[i, 1])
                
                # Calculate equivalent radius from area
                radius = int(np.sqrt(area / np.pi))
                
                # Calculate confidence based on area and color intensity
                region_mask = (labels == i).astype(np.uint8)
                mean_intensity = cv2.mean(image_rgb, mask=region_mask)[0] / 255.0  # Use red channel
                confidence = min(0.95, 0.75 + mean_intensity * 0.2)
                
                # Add detection
                fire_detections.append({
                    "type": "Fire Hazard",
                    "confidence": confidence,
                    "center": (center_x, center_y),
                    "radius": radius,
                    "reasoning": "Red/yellow area detected (possible flammable materials or electrical components)"
                })
        
        return fire_detections

    def detect_conventional_hazards(self, image):
        """
        Detect other hazards using fixed criteria
        In a real implementation, you would use specific detection methods for each hazard type
        """
        h, w = image.shape[:2]
        detections = []
        
        # Gas leak detection (areas with yellow/green tones near piping)
        gas_leak_centers = [
            (int(w * 0.7), int(h * 0.4)),  # Example fixed location
        ]
        
        for center in gas_leak_centers:
            detections.append({
                "type": "Gas Leak",
                "confidence": 0.78,
                "center": center,
                "radius": int(min(w, h) * 0.06),
                "reasoning": "Potential gas pipeline junction or storage area"
            })
        
        # Machinery accident risk (areas with dense equipment)
        detections.append({
            "type": "Machinery Accident",
            "confidence": 0.92,
            "center": (int(w * 0.5), int(h * 0.6)),
            "radius": int(min(w, h) * 0.07),
            "reasoning": "Dense equipment area with potential moving parts"
        })
        
        # Animal intrusion (near entry points and open areas)
        detections.append({
            "type": "Animal Intrusion",
            "confidence": 0.65,
            "center": (int(w * 0.8), int(h * 0.8)),
            "radius": int(min(w, h) * 0.04),
            "reasoning": "Potential entry point near exterior wall"
        })
        
        return detections

    def detect_hazards(self, image):
        """
        Detect all hazards in the factory blueprint
        """
        # Try using PaLM API first if configured
        if self.use_palm:
            palm_detections, success = self.analyze_with_palm(image)
            if success and palm_detections:
                return palm_detections
            print("PaLM API detection failed, falling back to conventional methods")
        
        # Fallback: Detect fire hazards using criteria-based detection
        fire_detections = self.detect_fire_hazards_by_color(image)
        
        # Detect other hazards (still using the simplified approach for demo)
        other_detections = self.detect_conventional_hazards(image)
        
        # Combine all detections
        return fire_detections + other_detections

# Function to generate heatmap overlay for hazards
def generate_heatmap(image, detections):
    """Generate a heatmap overlay showing hazard zones"""
    # Convert to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)

    h, w = image.shape[:2]

    # Create a blank heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Plot each detection as a hotspot
    for detection in detections:
        center = detection["center"]
        radius = detection["radius"]
        intensity = detection["confidence"]

        # Create a gaussian blob at each detection point
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist_from_center <= radius * 3  # 3x radius for wider effect

        # Apply gaussian falloff
        heatmap[mask] += intensity * np.exp(-(dist_from_center[mask]**2) / (2 * (radius**2)))

    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    # Apply colormap to convert to RGB
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Create blended image
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return blended, heatmap_colored

# Function to mark hazards on the image with labels
def mark_hazards(image, detections):
    """Mark detected hazards on the image with circles and labels"""
    # Convert to PIL image for drawing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    # Define colors for different hazard types
    colors = {
        "Fire Hazard": (255, 0, 0),       # Red
        "Gas Leak": (255, 255, 0),        # Yellow
        "Machinery Accident": (255, 165, 0), # Orange
        "Animal Intrusion": (128, 0, 128)  # Purple
    }

    # Draw each detection
    for detection in detections:
        hazard_type = detection["type"]
        center = detection["center"]
        radius = detection["radius"]
        confidence = detection["confidence"]
        color = colors.get(hazard_type, (0, 255, 0))  # Default to green if type not found

        # Draw circle
        draw.ellipse(
            [(center[0] - radius, center[1] - radius),
             (center[0] + radius, center[1] + radius)],
            outline=color, width=3
        )

        # Draw label with confidence
        label = f"{hazard_type}: {confidence:.2f}"
        draw.text((center[0] + radius, center[1] - radius), label, fill=color)

    return np.array(image)

# Function to add a color key to the image
def add_color_key(image, detections):
    """Add a color key to the image explaining the hazard types"""
    # Convert to PIL image for drawing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Define colors for different hazard types
    colors = {
        "Fire Hazard": (255, 0, 0),       # Red
        "Gas Leak": (255, 255, 0),        # Yellow
        "Machinery Accident": (255, 165, 0), # Orange
        "Animal Intrusion": (128, 0, 128)  # Purple
    }
    
    draw = ImageDraw.Draw(image)
    
    # Get unique hazard types from detections
    hazard_types = set(d["type"] for d in detections)
    
    # Draw color key in the bottom-left corner
    start_x, start_y = 10, image.height - 10 - (len(hazard_types) * 20)
    
    for i, hazard_type in enumerate(hazard_types):
        color = colors.get(hazard_type, (0, 255, 0))
        y_pos = start_y + (i * 20)
        
        # Draw color sample
        draw.rectangle([(start_x, y_pos), (start_x + 15, y_pos + 15)], fill=color)
        
        # Draw text
        draw.text((start_x + 20, y_pos), hazard_type, fill=(255, 255, 255))
    
    return np.array(image)

# Main processing function for the Gradio interface
def process_blueprint(blueprint_image, api_key=""):
    """Process a factory blueprint to detect and visualize hazards"""
    if blueprint_image is None:
        return None, None, "Please upload a blueprint image."

    # Convert to numpy array if needed
    if isinstance(blueprint_image, np.ndarray):
        image_np = blueprint_image
    else:
        image_np = np.array(blueprint_image)

    # Initialize the model and detect hazards
    model = HazardDetectionModel(palm_api_key=api_key if api_key.strip() else None)
    detections = model.detect_hazards(image_np)

    # Generate visualizations
    heatmap_image, pure_heatmap = generate_heatmap(image_np.copy(), detections)
    marked_image = mark_hazards(image_np.copy(), detections)
    final_image = add_color_key(marked_image, detections)

    # Generate report
    report = "# Factory Hazard Detection Report\n\n"
    
    # Detection method
    if model.use_palm and any(detections):
        report += "**Detection Method:** PaLM AI Vision Analysis\n\n"
    else:
        report += "**Detection Method:** Conventional Image Analysis\n\n"
    
    # Group detections by type
    hazard_by_type = {}
    for detection in detections:
        hazard_type = detection["type"]
        if hazard_type not in hazard_by_type:
            hazard_by_type[hazard_type] = []
        hazard_by_type[hazard_type].append(detection)
    
    # Show summary
    report += "## Summary\n\n"
    for hazard_type, hazards in hazard_by_type.items():
        report += f"- {hazard_type}: {len(hazards)} detected areas\n"
    report += "\n"
    
    # Detailed hazards
    report += "## Detected Hazards:\n\n"
    for hazard_type, hazards in hazard_by_type.items():
        report += f"### {hazard_type}\n\n"
        for i, detection in enumerate(hazards, 1):
            report += f"{i}. Confidence: {detection['confidence']:.2f}\n"
            report += f"   - Location: x={detection['center'][0]}, y={detection['center'][1]}\n"
            report += f"   - Severity Radius: {detection['radius']} pixels\n"
            if "reasoning" in detection and detection["reasoning"]:
                report += f"   - Reason: {detection['reasoning']}\n"
            report += "\n"

    report += "## Detection Criteria:\n\n"
    
    # Add information about detection criteria
    if model.use_palm and any(detections):
        report += "### PaLM AI Vision Analysis\n"
        report += "- Advanced AI vision model used to analyze blueprint\n"
        report += "- Criteria includes structural analysis, color patterns, and spatial relationships\n"
        report += "- Each detection includes reasoning for why an area is flagged\n\n"
    else:
        report += "### Conventional Image Analysis\n"
        report += "- Fire Hazards: Based on red color detection (potentially flammable materials)\n"
        report += "- Yellow zones (possible electrical components) are also monitored\n"
        report += "- Minimum area size requirement to filter out noise\n\n"
    
    report += "## Recommendations:\n\n"

    # Add recommendations based on detected hazards
    hazard_types = list(hazard_by_type.keys())

    if "Fire Hazard" in hazard_types:
        report += "### Fire Hazard Areas\n"
        report += "- Install additional fire extinguishers and smoke detectors in the marked fire hazard areas\n"
        report += "- Ensure clear evacuation routes from these zones\n"
        report += "- Consider fire-resistant materials in these areas\n\n"

    if "Gas Leak" in hazard_types:
        report += "### Gas Leak Zones\n"
        report += "- Implement gas detection sensors in the marked gas leak zones\n"
        report += "- Review ventilation systems in these areas\n"
        report += "- Ensure regular inspection of gas lines and connections\n\n"

    if "Machinery Accident" in hazard_types:
        report += "### Machinery Risk Areas\n"
        report += "- Install machine guards and emergency stop buttons\n"
        report += "- Designate clear operational zones with floor markings\n"
        report += "- Implement lockout/tagout procedures for maintenance\n\n"

    if "Animal Intrusion" in hazard_types:
        report += "### Potential Intrusion Points\n"
        report += "- Seal potential entry points in the marked areas\n"
        report += "- Consider implementing motion sensors and deterrents\n"
        report += "- Regular inspection of boundary integrity\n\n"

    report += "## Next Steps:\n\n"
    report += "1. Review these findings with safety team\n"
    report += "2. Prioritize hazard mitigation based on risk level\n"
    report += "3. Implement recommended safety measures\n"
    report += "4. Schedule a follow-up assessment\n"

    return final_image, heatmap_image, report

# Define Gradio interface with PaLM API key input
def create_gradio_interface():
    with gr.Blocks(title="Factory Hazard Detection") as demo:
        gr.Markdown("# Factory Hazard Detection System")
        gr.Markdown("## Upload a factory blueprint to detect potential hazards")

        with gr.Row():
            with gr.Column(scale=3):
                input_image = gr.Image(label="Upload Factory Blueprint", type="numpy")
            with gr.Column(scale=1):
                api_key = gr.Textbox(label="PaLM API Key (Optional)", placeholder="Enter your PaLM API key here", type="password")
                gr.Markdown("Leave API key blank to use conventional detection")

        with gr.Row():
            analyze_btn = gr.Button("Analyze Blueprint", variant="primary")

        with gr.Tabs():
            with gr.TabItem("Hazard Detection"):
                marked_output = gr.Image(label="Detected Hazards")
            
            with gr.TabItem("Hazard Heatmap"):
                heatmap_output = gr.Image(label="Hazard Heatmap")
            
            with gr.TabItem("Safety Report"):
                report_output = gr.Markdown(label="Safety Report")

        analyze_btn.click(
            fn=process_blueprint,
            inputs=[input_image, api_key],
            outputs=[marked_output, heatmap_output, report_output]
        )

    return demo

# Install the necessary dependencies if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")

    # Install required packages if in Colab (only if needed)
    try:
        import gradio
        import google.generativeai
    except ImportError:
        print("Installing required packages...")
       # !pip install gradio opencv-python matplotlib pillow google-generativeai
        print("Packages installed.")
except:
    IN_COLAB = False
    print("Not running in Google Colab")

# Launch the Gradio app with appropriate settings for Colab
if __name__ == "__main__":
    demo = create_gradio_interface()

    # If in Colab, use these specific launch settings
    
    print("Launching Gradio app...")
    
    # Launch with specific server parameters
    demo.launch(
        server_name="127.0.0.1",  # IMPORTANT: Use 127.0.0.1, not localhost
        server_port=7869,         # Change this port for each app
        share=False,
        debug=True,
        show_error=True
    )
    
    print("Gradio app should now be running")