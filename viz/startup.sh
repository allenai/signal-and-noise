#!/bin/bash

# Check if GITHUB_TOKEN is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is required"
    exit 1
fi

# Create a temporary directory for cloning
TEMP_DIR=$(mktemp -d)
SNR_DIR="$TEMP_DIR/signal-and-noise"

# Set up data directory environment variable (directory created in Dockerfile)
DATA_DIR="/home/data"
export SNR_DATA_DIR="$DATA_DIR"

# Set up Hugging Face cache directory to avoid permission issues
export HF_HOME="$DATA_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$DATA_DIR/.cache/huggingface/hub"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# Ensure data directory has proper permissions
mkdir -p "$DATA_DIR"
chmod -R 777 "$DATA_DIR"

# Debug: Check directory permissions
echo "Data directory permissions:"
ls -la /home/data
echo "Current user: $(whoami)"
echo "Current user ID: $(id)"

# Check if signal-and-noise is already installed by trying to import it
if python -c "import snr.constants" 2>/dev/null; then
    echo "signal-and-noise package already installed, skipping installation"
else
    echo "Installing signal-and-noise repository..."
    
    # Clone the repository using the GitHub token to temp directory
    git clone https://${GITHUB_TOKEN}@github.com/allenai/signal-and-noise.git "$SNR_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Successfully cloned signal-and-noise repository"
        
        # Install in editable mode with proper permissions
        cd "$SNR_DIR"
        
        # Try installing with --break-system-packages flag for newer pip versions
        if pip install --break-system-packages -e . 2>/dev/null; then
            echo "Installed with --break-system-packages flag"
        elif pip install --user -e . 2>/dev/null; then
            echo "Installed with --user flag"
        else
            # Fallback: install without editable mode
            echo "Falling back to non-editable installation"
            pip install --break-system-packages . || pip install --user .
        fi
        
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
