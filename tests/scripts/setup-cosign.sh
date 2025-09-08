#!/bin/bash
set -e

echo "Setting up test keys for Cosign integration tests..."

COSIGN_PASSWORD=${COSIGN_PASSWORD:-"testpassword"}

# Create directories
mkdir -p tests/integration/keys

# Create cosign key pair if it doesn't exist
if [ ! -f tests/integration/keys/cosign.key ] || [ ! -f tests/integration/keys/cosign.pub ]; then
    echo "Generating cosign key pair..."
    cd tests/integration/keys
    
    echo $COSIGN_PASSWORD

    # Set environment variable and generate non-interactively
    export COSIGN_PASSWORD="${COSIGN_PASSWORD}"
    cosign generate-key-pair
    
    echo "Generated cosign key pair:"
    ls -la .
    cd ../../..
fi

# Generate a second key pair for testing wrong key scenarios
if [ ! -f tests/integration/keys/wrong.key ] || [ ! -f tests/integration/keys/wrong.pub ]; then
    echo "Generating wrong cosign key pair for testing..."
    cd tests/integration/keys
    
    # Set different password and generate
    export COSIGN_PASSWORD="wrongpassword"
    cosign generate-key-pair --output-key-prefix wrong
    
    echo "Generated wrong cosign key pair:"
    ls -la wrong*
    cd ../../..
fi

# Generate a bad key pair for testing invalid key scenarios
if [ ! -f tests/integration/keys/invalid.key ]; then
    echo "Generating wrong cosign key pair for testing..."
    cd tests/integration/keys
    
    # Set different password and generate
    echo "Invalid key" > invalid.key
    
    echo "Generated wrong cosign key pair:"
    ls -la wrong*
    cd ../../..
fi