#!/bin/bash

echo "🚀 Vercel Setup Script for NairaGuard"
echo "======================================"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null
then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
else
    echo "✅ Vercel CLI already installed"
fi

# Check if user is logged in
echo ""
echo "🔐 Checking Vercel authentication..."
if vercel whoami &> /dev/null; then
    echo "✅ Already logged in to Vercel"
else
    echo "🔑 Please login to Vercel:"
    vercel login
fi

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create .vercelignore file
echo ""
echo "📝 Creating .vercelignore..."
cat > .vercelignore << EOF
# Development files
node_modules/
.git/
.gitignore
README.md
*.md
docs/

# Large files
models/saved/*.h5
models/saved/*.pkl
models/saved/*.onnx
datasets/
*.db
*.sqlite

# Test files
tests/
htmlcov/
coverage/
.coverage

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
EOF

echo "✅ .vercelignore created"

# Link project to Vercel
echo ""
echo "🔗 Linking project to Vercel..."
vercel link

echo ""
echo "🎉 Vercel setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure environment variables in Vercel dashboard"
echo "2. Run 'vercel' to deploy to preview"
echo "3. Run 'vercel --prod' to deploy to production"
echo ""
echo "📖 For detailed instructions, see VERCEL_DEPLOYMENT_GUIDE.md"
