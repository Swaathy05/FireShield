document.addEventListener("DOMContentLoaded", function () {
    console.log("Script loaded successfully!");

    // Function to show loading state
    function setLoading(button, isLoading) {
        if (isLoading) {
            button.disabled = true;
            button.innerHTML = '<span class="spinner"></span> Loading...';
            button.classList.add('loading');
        } else {
            button.disabled = false;
            button.innerHTML = button.getAttribute('data-original-text');
            button.classList.remove('loading');
        }
    }

    // Function to check app status
    function checkAppStatus(port, maxAttempts = 30) {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            
            function checkPort() {
                attempts++;
                console.log(`Checking if app is available on port ${port} (attempt ${attempts}/${maxAttempts})`);
                
                fetch(`/check_status/${port}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.running) {
                            console.log(`App is running on port ${port}`);
                            resolve(data.url);
                        } else if (attempts < maxAttempts) {
                            console.log(`App not available yet on port ${port}, retrying...`);
                            setTimeout(checkPort, 1000);
                        } else {
                            reject(new Error(`App failed to start on port ${port} after ${maxAttempts} attempts`));
                        }
                    })
                    .catch(error => {
                        if (attempts < maxAttempts) {
                            console.log(`Error checking app status, retrying...`);
                            setTimeout(checkPort, 1000);
                        } else {
                            reject(error);
                        }
                    });
            }
            
            checkPort();
        });
    }

    // Function to fetch and handle responses
    function runScript(endpoint, button, port) {
        console.log(`Button clicked! Sending request to: ${endpoint}`);
        
        // Save original button text
        if (!button.getAttribute('data-original-text')) {
            button.setAttribute('data-original-text', button.innerHTML);
        }
        
        // Show loading state
        setLoading(button, true);

        // Show status message
        const statusElement = document.createElement('div');
        statusElement.className = 'status-message';
        statusElement.innerHTML = 'Starting application...';
        button.parentNode.appendChild(statusElement);

        fetch(endpoint)
            .then(response => response.json())
            .then(data => {
                console.log("Response received:", data);
                statusElement.innerHTML = data.message;
                
                // Periodically check if the app is running
                return checkAppStatus(port)
                    .then(url => {
                        console.log(`App is ready at ${url}`);
                        setLoading(button, false);
                        statusElement.innerHTML = 'Application is ready! Opening in new tab...';
                        
                        // Open the Gradio app in a new tab
                        window.open(url, "_blank");
                        
                        // Remove status message after a delay
                        setTimeout(() => {
                            statusElement.remove();
                        }, 3000);
                    })
                    .catch(error => {
                        console.error("Error starting application:", error);
                        setLoading(button, false);
                        statusElement.innerHTML = 'Failed to start application. Please try again.';
                        
                        // Remove status message after a delay
                        setTimeout(() => {
                            statusElement.remove();
                        }, 5000);
                    });
            })
            .catch(error => {
                console.error("Fetch error:", error);
                setLoading(button, false);
                statusElement.innerHTML = 'Error connecting to server. Please try again.';
                
                // Remove status message after a delay
                setTimeout(() => {
                    statusElement.remove();
                }, 5000);
            });
    }

    // Match button IDs from HTML
    let generateModelBtn = document.getElementById("generateModel");
    let analyzeMachineBtn = document.getElementById("analyzeMachine");
    let analyzeBlueprintsBtn = document.getElementById("analyzeBlueprints");

    if (generateModelBtn) {
        generateModelBtn.addEventListener("click", function () {
            runScript("/run_trisha", this, 7868);
        });
    } else {
        console.error("Button #generateModel not found in DOM!");
    }

    if (analyzeMachineBtn) {
        analyzeMachineBtn.addEventListener("click", function () {
            runScript("/run_machine", this, 7870);
        });
    } else {
        console.error("Button #analyzeMachine not found in DOM!");
    }

    if (analyzeBlueprintsBtn) {
        analyzeBlueprintsBtn.addEventListener("click", function () {
            runScript("/run_rizwana", this, 7869);
        });
    } else {
        console.error("Button #analyzeBlueprints not found in DOM!");
    }
});