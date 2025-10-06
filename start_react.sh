#!/bin/bash

echo "🚀 Installing Node.js and setting up React app..."

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Install and use Node.js
echo "📦 Installing Node.js..."
nvm install 18
nvm use 18

# Verify installation
echo "✅ Verifying Node.js installation..."
node --version
npm --version

# Navigate to frontend directory
cd /Users/mac/Downloads/fake-naira-detection/frontend

# Install dependencies
echo "📦 Installing React dependencies..."
npm install

# Start the development server
echo "🚀 Starting React development server..."
npm run dev
