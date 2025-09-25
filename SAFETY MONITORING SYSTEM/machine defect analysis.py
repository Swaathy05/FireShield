import os
import tempfile
import time
import json
import numpy as np
import gradio as gr
import trimesh
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
import cv2
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class MachineDefectAnalyzer:
    """Class for analyzing machine images to detect defects, safety issues, and usage precautions"""
    
    def __init__(self, provider="google"):
        """Initialize the machine defect analyzer with a specific LLM provider
        
        Args:
            provider (str): The LLM provider to use ("google" or "groq")
        """
        self.provider = provider.lower()
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.provider == "google":
            # Configure the Gemini API
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Initialize LangChain integration with Google Gemini
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif self.provider == "groq":
            # Initialize LangChain integration with Groq
            self.llm = ChatGroq(
                temperature=0.1, 
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.1-70b-versatile"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'google' or 'groq'.")
    
    def extract_image_features(self, image):
        """Extract important features from the machine image using traditional CV techniques"""
        if image is None:
            return None
            
        # Convert image to numpy array if it's a PIL image
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
            
        # Convert to grayscale for processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
            
        # Run edge detection to find potential defects and features
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours to identify components
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Blur detection - detect if parts of the image are blurry (potentially indicating wear or damage)
        blur_map = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Create a simple structure analysis
        feature_stats = {
            "total_contours": len(contours),
            "avg_contour_size": np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,
            "edge_density": np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),
            "blur_level": float(blur_map),
            "img_width": img.shape[1],
            "img_height": img.shape[0],
        }
        
        # Create heatmap of potential defect regions (for visualization)
        # Highlight areas with high edge density or unusual patterns
        kernel = np.ones((5,5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        heatmap = cv2.applyColorMap(dilated_edges, cv2.COLORMAP_JET)
        
        # Overlay the heatmap on the original image
        if len(img.shape) == 3:
            overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        else:
            overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.7, heatmap, 0.3, 0)
            
        # Convert back to PIL for return
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
        return {
            "feature_stats": feature_stats,
            "visualization": overlay_pil,
            "raw_edges": edges
        }
    
    def get_machine_description(self, image):
        """Get a detailed description of the machine using the LLM"""
        if image is None:
            return None
            
        prompt = PromptTemplate.from_template(
            """
            You are an expert mechanical engineer and machine safety inspector.
            
            Analyze the machine in the image and provide a detailed description.
            Focus on:
            1. Machine type and purpose
            2. Key mechanical components visible
            3. Moving parts identification
            4. Control interfaces
            5. Safety features present or missing
            6. Signs of wear, damage, or misalignment
            7. Materials and construction quality
            
            Be specific about:
            - Precise mechanical terminology for all components
            - Proper technical terms for the machine type
            - Industry standards applicable to this machinery type
            - Engineering principles relevant to this equipment
            
            Provide only the technical analysis without any preamble.
            """
        )
        
        # Since we can't directly pass the image to the LLM through this interface,
        # we'll extract visual features to help inform our prompt
        features = self.extract_image_features(image)
        feature_info = json.dumps(features["feature_stats"]) if features else "No features extracted"
        
        chain = prompt | self.llm
        response = chain.invoke({"image_info": feature_info})
        return response.content
    
    def analyze_machine_defects(self, image, machine_type="industrial"):
        """Perform comprehensive defect analysis on a machine image"""
        if image is None:
            raise ValueError("Please upload a machine image")
            
        start_time = time.time()
        
        # Extract image features and get visualization
        features = self.extract_image_features(image)
        if not features:
            return None, {
                "error": "Failed to extract image features",
                "processing_time": 0
            }
            
        # Get machine description from LLM
        machine_description = self.get_machine_description(image)
        
        # Create a detailed prompt for comprehensive defect analysis
        defect_prompt = PromptTemplate.from_template(
            """
            # MACHINE ANALYSIS TASK
            
            You are a specialized AI system for industrial machinery defect analysis with expertise in mechanical engineering, safety standards, and preventive maintenance.
            
            ## INPUT DATA:
            - Image Features: {features}
            - Machine Description: {machine_description}
            - Machine Type: {machine_type}
            
            ## REQUIRED ANALYSIS:
            Perform a comprehensive machine defect analysis with these exact sections:
            
            ### DETAILED DEFECT ANALYSIS
            
            1. STRUCTURAL INTEGRITY ASSESSMENT:
               - Visible wear and fatigue points
               - Critical stress concentrations
               - Misalignment indicators
               - Fastener and connection issues
               - Non-standard or compromised components
               - Deformation evidence
               - Material degradation signs
               - Welding or joining flaws
               
            2. MECHANICAL COMPONENT ANALYSIS:
               - Bearing condition assessment
               - Gear and transmission problems
               - Belt/chain wear patterns
               - Lubrication status
               - Hydraulic/pneumatic system leaks
               - Shaft alignment issues
               - Vibration indicators
               - Thermal stress markers
               
            3. ELECTRICAL & CONTROL SYSTEM EVALUATION:
               - Wiring condition issues
               - Control interface defects
               - Sensor malfunctions
               - Power connection problems
               - Circuit protection concerns
               - Automation reliability issues
               - Software/firmware vulnerabilities
               - Emergency systems status
               
            4. SAFETY HAZARDS IDENTIFICATION:
               - Guarding and protection inadequacies
               - Access risk points
               - Pinch/crush/shear points
               - Fall or impact dangers
               - Chemical exposure risks
               - Noise or vibration hazards
               - Fire or explosion potential
               - Emergency stop accessibility
               
            5. COMPLIANCE & STANDARDS EVALUATION:
               - Regulatory non-compliance areas
               - Industry standard deviations
               - Documentation deficiencies
               - Certification status issues
               - Testing and inspection lapses
               - Modification tracking problems
               - Training requirement gaps
               - Maintenance record deficits
            
            6. OPERATIONAL PRECAUTIONS:
               - Specific safety procedures required
               - Personal protective equipment needs
               - Operational restrictions
               - Environmental considerations
               - Special training requirements
               - Monitoring recommendations
               - Maintenance intervals
               - Emergency response protocols
            
            7. REMEDIATION RECOMMENDATIONS:
               - Immediate action items (prioritized)
               - Short-term repairs needed
               - Long-term replacement strategy
               - Upgrade recommendations
               - Performance optimization opportunities
               - Safety enhancement suggestions
               - Reliability improvement measures
               - Maintenance procedure adjustments
            
            ## OUTPUT FORMAT:
            Return a valid JSON object with each section from the analysis above.
            Include a 'confidence_level' key in each section with a value from 1-10.
            Include a 'defect_severity' key in each section with a value from 1-10.
            Include an array of specific defects for each category.
            
            Do not include explanatory text outside the JSON structure.
            """
        )
        
        # Query the LLM for defect analysis
        chain = defect_prompt | self.llm
        response = chain.invoke({
            "features": json.dumps(features["feature_stats"]),
            "machine_description": machine_description,
            "machine_type": machine_type
        })
        
        # Extract the JSON from the response and parse it
        try:
            # Find JSON content - it might be wrapped in code blocks
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
                
            defect_analysis = json.loads(json_str)
            
            # Add processing time and summary
            processing_time = time.time() - start_time
            
            # Calculate overall severity and confidence
            severity_scores = [section.get("defect_severity", 0) for section in defect_analysis.values() 
                              if isinstance(section, dict) and "defect_severity" in section]
            
            confidence_scores = [section.get("confidence_level", 0) for section in defect_analysis.values() 
                                if isinstance(section, dict) and "confidence_level" in section]
            
            overall_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            summary = {
                "processing_time": processing_time,
                "overall_severity": round(overall_severity, 1),
                "overall_confidence": round(overall_confidence, 1),
                "machine_type": machine_type,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add summary to analysis
            defect_analysis["analysis_summary"] = summary
            
            # Save the analysis to a file
            timestamp = int(time.time())
            analysis_filename = f"machine_analysis_{timestamp}.json"
            analysis_path = os.path.join(self.output_dir, analysis_filename)
            
            with open(analysis_path, 'w') as f:
                json.dump(defect_analysis, f, indent=2)
                
            # Save the visualization image
            viz_filename = f"machine_analysis_{timestamp}_visualization.png"
            viz_path = os.path.join(self.output_dir, viz_filename)
            features["visualization"].save(viz_path)
            
            return features["visualization"], defect_analysis
            
        except Exception as e:
            traceback.print_exc()
            return features["visualization"] if features else None, {
                "error": str(e),
                "processing_time": time.time() - start_time
            }

def create_machine_analyzer_interface():
    """Create a Gradio interface for the machine defect analyzer"""
    # Initialize the analyzer
    analyzer = MachineDefectAnalyzer(provider="google")  # Default to Google

    # Define theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )

    # Create the interface
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("""
        # Industrial Machine Defect Analyzer
        
        Upload an image of industrial machinery for comprehensive defect analysis. The system will:
        1. Analyze the machine for structural, mechanical, and electrical defects
        2. Identify safety hazards and compliance issues
        3. Provide detailed operational precautions
        4. Recommend remediation steps
        
        The analysis includes visualization of potential defect areas.
        """)

        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_image = gr.Image(
                    label="Upload Machine Image", 
                    type="pil",
                    height=400
                )
                
                machine_type = gr.Dropdown(
                    choices=[
                        "industrial", "manufacturing", "cnc", "hydraulic", 
                        "pneumatic", "conveyor", "robotic", "electrical", 
                        "hvac", "power_tool", "agricultural", "construction"
                    ],
                    label="Machine Type",
                    value="industrial"
                )
                
                llm_provider = gr.Radio(
                    choices=["google", "groq"],
                    label="Analysis Engine",
                    value="google"
                )
                
                analyze_btn = gr.Button("Analyze Machine", variant="primary")
                
            with gr.Column(scale=1):
                # Output displays
                with gr.Tab("Visualization"):
                    output_visualization = gr.Image(
                        label="Defect Visualization",
                        height=400
                    )
                
                with gr.Tab("Detailed Analysis"):
                    with gr.Accordion("Structural Integrity", open=True):
                        structural_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Mechanical Components"):
                        mechanical_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Electrical & Control Systems"):
                        electrical_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Safety Hazards"):
                        safety_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Compliance & Standards"):
                        compliance_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Operational Precautions"):
                        precautions_analysis = gr.JSON(label="")
                    
                    with gr.Accordion("Remediation Recommendations"):
                        remediation_analysis = gr.JSON(label="")
                
                with gr.Tab("Summary"):
                    analysis_summary = gr.JSON(label="Analysis Summary")
                    
                    with gr.Row():
                        severity_indicator = gr.Number(label="Overall Severity (1-10)", precision=1)
                        confidence_indicator = gr.Number(label="Analysis Confidence (1-10)", precision=1)
                    
                    download_btn = gr.Button("Download Full Analysis Report")
                    download_path = gr.Textbox(visible=False)

        # Event handlers
        def update_analyzer_provider(provider):
            analyzer.provider = provider.lower()
            if analyzer.provider == "google":
                analyzer.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:  # groq
                analyzer.llm = ChatGroq(
                    temperature=0.1, 
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.1-70b-versatile"
                )
            return f"Updated analysis engine to {provider}"
        
        llm_provider.change(
            fn=update_analyzer_provider,
            inputs=[llm_provider],
            outputs=[gr.Textbox(visible=False)]
        )
        
        def analyze_machine(image, machine_type):
            if image is None:
                return [None, None, None, None, None, None, None, None, 0, 0, None]
            
            try:
                visualization, analysis = analyzer.analyze_machine_defects(image, machine_type)
                
                # Extract sections from the analysis
                structural = analysis.get("STRUCTURAL_INTEGRITY_ASSESSMENT", {})
                mechanical = analysis.get("MECHANICAL_COMPONENT_ANALYSIS", {})
                electrical = analysis.get("ELECTRICAL_&_CONTROL_SYSTEM_EVALUATION", {})
                safety = analysis.get("SAFETY_HAZARDS_IDENTIFICATION", {})
                compliance = analysis.get("COMPLIANCE_&_STANDARDS_EVALUATION", {})
                precautions = analysis.get("OPERATIONAL_PRECAUTIONS", {})
                remediation = analysis.get("REMEDIATION_RECOMMENDATIONS", {})
                summary = analysis.get("analysis_summary", {})
                
                # Save full analysis to a file for download
                timestamp = int(time.time())
                analysis_filename = f"machine_analysis_{timestamp}.json"
                analysis_path = os.path.join(analyzer.output_dir, analysis_filename)
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                # Return all components
                return [
                    visualization,
                    structural,
                    mechanical,
                    electrical,
                    safety,
                    compliance,
                    precautions,
                    remediation,
                    summary.get("overall_severity", 0),
                    summary.get("overall_confidence", 0),
                    summary,
                    analysis_path
                ]
            except Exception as e:
                traceback.print_exc()
                error = {
                    "error": str(e),
                    "message": "Analysis failed. Please try a clearer image or different machine type."
                }
                return [None, error, error, error, error, error, error, error, 0, 0, error, None]
        
        analyze_btn.click(
            fn=analyze_machine,
            inputs=[input_image, machine_type],
            outputs=[
                output_visualization,
                structural_analysis,
                mechanical_analysis,
                electrical_analysis,
                safety_analysis,
                compliance_analysis,
                precautions_analysis,
                remediation_analysis,
                severity_indicator,
                confidence_indicator,
                analysis_summary,
                download_path
            ]
        )
        
        # Download handler
        def prepare_download(file_path):
            if file_path and os.path.exists(file_path):
                return file_path
            return None
        
        download_btn.click(
            fn=prepare_download,
            inputs=[download_path],
            outputs=[gr.File(label="Download Analysis")]
        )
        
        # Add footer
        gr.Markdown("""
        ### Machine Defect Analysis Information
        The analysis uses AI and computer vision to evaluate potential issues with industrial machinery, including:
        - Structural integrity concerns (wear, stress points, misalignments)
        - Mechanical component problems (bearings, gears, transmission systems)
        - Electrical system issues (wiring, controls, sensors)
        - Safety hazards (guarding issues, pinch points, emergency systems)
        - Compliance with industrial standards
        
        The system provides a severity rating (1-10) and confidence score for each category.
        
        ⚠️ **Important**: This analysis is for informational purposes only. Always consult with qualified technicians and safety professionals before operating potentially defective machinery.
        """)
    
    return demo

# Create and launch the interface if run directly
if __name__ == "__main__":
    demo = create_machine_analyzer_interface()
    
    print("Launching Gradio app...")
    
    # Launch with specific server parameters
    demo.launch(
        server_name="127.0.0.1",  # IMPORTANT: Use 127.0.0.1, not localhost
        server_port=7870,         # Change this port for each app
        share=False,
        debug=True,
        show_error=True
    )
    
    print("Gradio app should now be running")