
# Create necessary directories
mkdir -p tests
mkdir -p examples
mkdir -p videos  # For test videos

# Create a .gitignore file
echo "📝 Creating .gitignore..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
robot_fights_output/
*.mp4
*.avi
*.mov
*.mkv
models/
EOF

echo "✅ Setup complete for ricklon!"
echo ""
echo "🚀 Quick start:"
echo "  cd $PROJECT_NAME"
echo "  robot-fight-detector test"
echo "  robot-fight-detector detect your_video.mp4"
echo ""
echo "💡 Tips:"
echo "  - Place test videos in the 'videos/' directory"
echo "  - Results will be saved to 'robot_fights_output/' by default"
echo "  - Use 'robot-fight-detector --help' for all options"
echo ""
echo "📖 See README.md for detailed usage instructions."
