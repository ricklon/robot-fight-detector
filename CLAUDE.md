# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robot Fight Detector is a CLI tool that uses the SmolVLM2 vision-language model to detect robot fights in videos and images. The tool analyzes video frames or images and identifies scenes where robots appear to be fighting or engaged in combat.

## Design Methodology

The project follows these design principles:

- **Object-Oriented Design**: Core functionality encapsulated in the `RobotFightDetector` class
- **Command Pattern**: Uses Click library to implement CLI commands and subcommands
- **Single Responsibility Principle**: Separate methods for frame analysis and video processing
- **Configuration Through Parameters**: Flexible configuration via CLI options and environment variables
- **Explicit Error Handling**: Clear error messages with proper exception handling

## Development Environment

### Setup and Installation

1. The project uses Python 3.12.3+ and uv for dependency management:

```sh
# Install dependencies with uv
uv pip install -e .
```

2. Install development dependencies:

```sh
uv pip install pytest black isort flake8
```

3. Environment Setup (IMPORTANT):

The project requires a Hugging Face API token to access the SmolVLM2 model. Create a `.env` file based on the `.env.example`:

```sh
# Copy example file
cp .env.example .env

# Edit the .env file to add your HuggingFace token
# HUGGINGFACE_TOKEN=your_token_here
```

You'll need to:
- Create a Hugging Face account at https://huggingface.co/
- Get your token from https://huggingface.co/settings/tokens
- Add your token to the `.env` file

## Commands

### Running the CLI

```sh
# Test if SmolVLM2 model can be loaded successfully
robot-fight-detector test

# Detect robot fights in a video file
robot-fight-detector detect VIDEO_PATH [--output-dir OUTPUT_DIR] [--interval INTERVAL] [--model MODEL]

# Analyze a single image for robot fights
robot-fight-detector analyze-image IMAGE_PATH [--model MODEL]

# Get help
robot-fight-detector --help
```

### Development Commands

```sh
# Format code
black .
isort .

# Lint code
flake8

# Run tests
pytest
```

## Architecture

The project has a simple architecture:

- `robot_fight_detector.py`: Contains the main `RobotFightDetector` class and CLI commands using Click
- Main components:
  - `RobotFightDetector`: Core class that loads the SmolVLM2 model and provides methods to analyze frames and process videos
  - CLI commands: `detect`, `analyze_image`, and `test` implemented with Click

The application works by:
1. Loading the SmolVLM2 vision-language model
2. For videos: Sampling frames at specified intervals
3. Processing each frame with SmolVLM2 to detect robot fights
4. Saving detected frames and generating a JSON report with timestamps and descriptions

## Output Formats

The robot-fight-detector supports multiple output formats and options:

1. **JSON**: The default format which provides detailed information about each detected frame
2. **WebVTT**: A subtitle format that groups detections into continuous fight segments for easy navigation
3. **Video Clips**: Extracted MP4 video segments for each detected fight

### WebVTT Output

The WebVTT format creates subtitle files that identify when robot fights occur and which robots are involved. This is particularly useful for:

- Processing long videos (hours of footage) without saving excessive frames
- Creating navigable timestamps in video players that support subtitles/captions
- Identifying specific robots and actions during fights
- Creating searchable fight databases with robot names and events

WebVTT output can be enabled with the `--format vtt` option:

```sh
# Generate WebVTT for a large video without saving frames
robot-fight-detector detect videos/tournament.mp4 --format vtt --no-save-frames --interval 5
```

### Video Clip Extraction

The clip extraction feature allows you to automatically cut out just the fight scenes from longer videos. This is especially useful for:

- Creating highlight reels from tournament videos
- Sharing specific fights without the entire video
- Analyzing fight sequences in isolation
- Building a library of fight clips organized by robot names

Clip extraction can be enabled with the `--extract-clips` option:

```sh
# Extract fight clips from a video
robot-fight-detector detect videos/tournament.mp4 --extract-clips --clip-padding 3 --clip-quality high

# Process a long tournament video, extract clips but don't save frames
robot-fight-detector detect videos/tournament.mp4 --extract-clips --no-save-frames --interval 10
```

Clip extraction requires FFmpeg to be installed on the system.

## Future Development Tasks

The following tasks are planned or could be implemented to enhance the project:

1. **Testing Framework**: Add comprehensive unit tests and integration tests
2. **Confidence Threshold**: Implement the confidence threshold parameter in the code
3. **Batch Processing**: Add support for processing multiple videos in batch mode
4. **Performance Optimization**: Optimize frame extraction and processing for faster analysis
5. **Web Interface**: Create a simple web UI for uploading and analyzing videos
6. **Advanced Detection**: Improve prompt engineering for more accurate robot fight detection
7. **Video Trimming**: Add option to create trimmed video clips of the detected fight scenes
8. **Report Visualization**: Generate visual HTML reports with thumbnails and timestamps
9. **Streaming Processing**: Add support for processing videos in chunks to handle large files

## Notes

- Video processing happens at specified intervals (default: 1 second)
- Output includes saved frames of detected fights and a JSON report
- The model runs on CUDA if available, otherwise on CPU
- SmolVLM2-2.2B-Instruct is used with the chat templating API to process images and identify robot fights
- The application uses Flash Attention 2 for faster processing on CUDA devices