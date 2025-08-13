#!/bin/bash

# Check if GITHUB_TOKEN is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is required"
    exit 1
fi

# Create a temporary directory for cloning
TEMP_DIR=$(mktemp -d)
SNR_DIR="$TEMP_DIR/signal-and-noise"

# Check if signal-and-noise is already installed by trying to import it
if python -c "import snr.constants" 2>/dev/null; then
    echo "signal-and-noise package already installed, skipping installation"
else
    echo "Installing signal-and-noise repository..."
    
    # Clone the repository using the GitHub token to temp directory
    git clone https://${GITHUB_TOKEN}@github.com/allenai/signal-and-noise.git "$SNR_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Successfully cloned signal-and-noise repository"
        
        # Install in editable mode
        cd "$SNR_DIR"
        pip install -e .
        
        if [ $? -eq 0 ]; then
            echo "Successfully installed signal-and-noise package"
        else
            echo "Error: Failed to install signal-and-noise package"
            exit 1
        fi
        
        # Return to the working directory
        cd /home
        
        # Clean up the temporary directory (optional, since container will be destroyed anyway)
        # rm -rf "$TEMP_DIR"
    else
        echo "Error: Failed to clone signal-and-noise repository"
        echo "Please check your GITHUB_TOKEN is valid and has access to the repository"
        exit 1
    fi
fi

# Start the gunicorn server
echo "Starting gunicorn server..."
exec gunicorn main:server -b 0.0.0.0:7860
