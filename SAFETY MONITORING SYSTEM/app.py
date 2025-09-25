import threading
import subprocess
import time
from flask import Flask, jsonify, render_template, request
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Function to check if Gradio is running
def get_gradio_url(port, max_wait=30):
    """Waits for Gradio to start and returns its local URL."""
    logger.info(f"Checking if Gradio is running on port {port}")
    
    for i in range(max_wait):
        try:
            logger.info(f"Attempt {i+1}/{max_wait} to connect to http://127.0.0.1:{port}")
            response = requests.get(f"http://127.0.0.1:{port}", timeout=1)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Gradio on port {port}")
                return f"http://127.0.0.1:{port}"
        except requests.ConnectionError:
            logger.info(f"Connection attempt {i+1} failed, waiting...")
            time.sleep(1)  # Wait 1 second before retrying
        except requests.Timeout:
            logger.info(f"Connection attempt {i+1} timed out, waiting...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error checking Gradio: {str(e)}")
            time.sleep(1)
            
    logger.error(f"Failed to connect to Gradio on port {port} after {max_wait} attempts")
    return None  # If Gradio never starts

# Function to start a script in a separate thread
def start_script(script_path, port):
    """Runs the given script and waits for Gradio to start."""
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return None
        
    try:
        logger.info(f"Starting script: {script_path} on port {port}")
        
        # Start the script with output capture
        process = subprocess.Popen(
            ["python", script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give some time for the process to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process has exited
            stdout, stderr = process.communicate()
            logger.error(f"Script exited immediately with code {process.returncode}")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
            
        # Check if Gradio is running
        gradio_url = get_gradio_url(port)
        if gradio_url:
            return gradio_url
        else:
            # Process is running but Gradio isn't responding
            # Try to get output without waiting for completion
            try:
                stdout_data, stderr_data = process.communicate(timeout=1)
                logger.error(f"Gradio failed to start. STDOUT: {stdout_data}")
                logger.error(f"STDERR: {stderr_data}")
            except subprocess.TimeoutExpired:
                logger.warning("Process still running but Gradio not responding")
                # Don't kill the process, it might still be starting up
            
            return None

    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return None

# Function to check if a port is in use
def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# Flask routes for each script
@app.route('/run_trisha')
def run_trisha():
    """Starts Trisha's script in a new thread and returns a message with the URL."""
    port = 7868
    gradio_url = f"http://127.0.0.1:{port}"
    
    # Check if Gradio is already running on this port
    if is_port_in_use(port):
        logger.info(f"Gradio already running on port {port}")
        return jsonify({
            "message": "3D Model Generator is already running", 
            "url": gradio_url
        })
    
    # Start the script in a new thread
    script_path = r"D:\trisha\sastradaks\sastradaksh\trishacode.py"
    logger.info(f"Starting new thread for {script_path}")
    
    thread = threading.Thread(
        target=lambda: start_script(script_path, port)
    )
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    
    # Wait briefly to see if it starts quickly
    time.sleep(2)
    
    # Check if Gradio is running
    if is_port_in_use(port):
        return jsonify({
            "message": "3D Model Generator is ready", 
            "url": gradio_url
        })
    else:
        return jsonify({
            "message": "3D Model Generator is starting (this may take up to 30 seconds)...", 
            "url": gradio_url
        })

@app.route('/run_rizwana')
def run_rizwana():
    """Starts Rizwana's script in a new thread and returns a message with the URL."""
    port = 7869
    gradio_url = f"http://127.0.0.1:{port}"
    
    # Check if Gradio is already running on this port
    if is_port_in_use(port):
        logger.info(f"Gradio already running on port {port}")
        return jsonify({
            "message": "Blueprint Analyzer is already running", 
            "url": gradio_url
        })
    
    # Start the script in a new thread
    script_path = r"D:\trisha\sastradaks\sastradaksh\rizwanacode.py"
    logger.info(f"Starting new thread for {script_path}")
    
    thread = threading.Thread(
        target=lambda: start_script(script_path, port)
    )
    thread.daemon = True
    thread.start()
    
    # Wait briefly to see if it starts quickly
    time.sleep(2)
    
    # Check if Gradio is running
    if is_port_in_use(port):
        return jsonify({
            "message": "Blueprint Analyzer is ready", 
            "url": gradio_url
        })
    else:
        return jsonify({
            "message": "Blueprint Analyzer is starting (this may take up to 30 seconds)...", 
            "url": gradio_url
        })

@app.route('/run_machine')
def run_machine():
    """Starts Machine Defect Analysis script in a new thread and returns a message with the URL."""
    port = 7870
    gradio_url = f"http://127.0.0.1:{port}"
    
    # Check if Gradio is already running on this port
    if is_port_in_use(port):
        logger.info(f"Gradio already running on port {port}")
        return jsonify({
            "message": "Machine Defect Analyzer is already running", 
            "url": gradio_url
        })
    
    # Start the script in a new thread
    script_path = r"D:\trisha\sastradaks\sastradaksh\machinedefect.py"
    logger.info(f"Starting new thread for {script_path}")
    
    thread = threading.Thread(
        target=lambda: start_script(script_path, port)
    )
    thread.daemon = True
    thread.start()
    
    # Wait briefly to see if it starts quickly
    time.sleep(2)
    
    # Check if Gradio is running
    if is_port_in_use(port):
        return jsonify({
            "message": "Machine Defect Analyzer is ready", 
            "url": gradio_url
        })
    else:
        return jsonify({
            "message": "Machine Defect Analyzer is starting (this may take up to 30 seconds)...", 
            "url": gradio_url
        })

@app.route('/check_status/<int:port>')
def check_status(port):
    """Endpoint to check if a Gradio app is running on a specific port."""
    is_running = is_port_in_use(port)
    return jsonify({
        "running": is_running,
        "url": f"http://127.0.0.1:{port}" if is_running else None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)