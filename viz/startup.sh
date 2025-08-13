#!/bin/bash

# Check if GITHUB_TOKEN is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is required"
    exit 1
fi

# Check if signal-and-noise repo is already installed
if [ ! -d "/home/signal-and-noise" ]; then
    echo "Installing signal-and-noise repository..."
    
    # Clone the repository using the GitHub token
    git clone https://${GITHUB_TOKEN}@github.com/allenai/signal-and-noise.git /home/signal-and-noise
    
    if [ $? -eq 0 ]; then
        echo "Successfully cloned signal-and-noise repository"
        
        # Install in editable mode
        cd /home/signal-and-noise
        pip install -e .
        
        if [ $? -eq 0 ]; then
            echo "Successfully installed signal-and-noise package"
        else
            echo "Error: Failed to install signal-and-noise package"
            exit 1
        fi
        
        # Return to the working directory
        cd /home
    else
        echo "Error: Failed to clone signal-and-noise repository"
        echo "Please check your GITHUB_TOKEN is valid and has access to the repository"
        exit 1
    fi
else
    echo "signal-and-noise repository already exists, skipping installation"
fi

# Start the gunicorn server
echo "Starting gunicorn server..."
exec gunicorn main:server -b 0.0.0.0:7860
