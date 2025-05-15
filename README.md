# Robot Fight Detector

A CLI tool to detect robot fights in videos and images using SmolVLM2 vision language model.

## Features

- Analyze videos to detect moments where robots are fighting
- Process individual images to check for robot combat
- Save frames and timestamps of detected robot fights
- Generate detailed JSON reports of detections
- Create WebVTT subtitles for robot fight navigation
- Automatically extract video clips of fight segments

## Installation

Requirements:
- Python 3.12.3 or higher
- [ffmpeg](https://ffmpeg.org/download.html) (required for clip extraction only)

### Option 1: Install with pip

```bash
# Install directly from GitHub
pip install git+https://github.com/ricklon/robot-fight-detector.git

# Or install in development mode after cloning with GitHub CLI
gh repo clone ricklon/robot-fight-detector
cd robot-fight-detector
pip install -e .
```

### Option 2: Install with uv (recommended for modern Python projects)

```bash
# Clone the repository with GitHub CLI
gh repo clone ricklon/robot-fight-detector
cd robot-fight-detector

# Install in development mode
uv pip install -e .

# Alternative: install dependencies directly from pyproject.toml
uv sync

# Or add to an existing project
uv add git+https://github.com/ricklon/robot-fight-detector.git
```

### Environment Setup

This project requires a Hugging Face API token to access the SmolVLM2 model:

1. Create a Hugging Face account at https://huggingface.co/
2. Get your token from https://huggingface.co/settings/tokens
3. Create a `.env` file with your token:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file and add your token
# HUGGINGFACE_TOKEN=your_token_here
```

The `.env` file supports the following configurations:

```
HUGGINGFACE_TOKEN=your_huggingface_token_here    # Required
MODEL_NAME=HuggingFaceTB/SmolVLM2-1.7B-Instruct  # Optional: alternative model
OUTPUT_DIR=./robot_fights_output                 # Optional: default output directory
DEFAULT_INTERVAL=1.0                             # Optional: frame sampling interval
```

## Usage

### Testing the Setup

```bash
# Test if dependencies and model can be loaded
robot-fight-detector test

# Skip model loading (faster)
robot-fight-detector test --skip-model
```

### Analyzing a Single Image

```bash
# Analyze an image for robot fights
robot-fight-detector analyze-image PATH_TO_IMAGE

# Example
robot-fight-detector analyze-image examples/robot_sample.jpg
```

### Detecting Robot Fights in Videos

```bash
# Detect robot fights in a video
robot-fight-detector detect PATH_TO_VIDEO [OPTIONS]

# Example with custom output directory and interval
robot-fight-detector detect videos/robot_battle.mp4 --output-dir ./results --interval 0.5

# Generate WebVTT subtitles file for video fights
robot-fight-detector detect videos/robot_battle.mp4 --format vtt --interval 2

# Process a large video without saving frames (memory efficient)
robot-fight-detector detect videos/long_tournament.mp4 --no-save-frames --format vtt --interval 5

# Extract fight clips from a video
robot-fight-detector detect videos/robot_battle.mp4 --extract-clips --clip-quality high

# Extract clips from a long tournament video
robot-fight-detector detect videos/long_tournament.mp4 --no-save-frames --extract-clips --clip-padding 5 --interval 10

# Complete workflow for a 5-hour tournament video
robot-fight-detector detect videos/tournament_5hours.mp4 --format vtt --extract-clips --no-save-frames --interval 15 --clip-quality low
```

### Processing YouTube Livestreams

```bash
# Process a YouTube livestream in real-time
robot-fight-detector livestream YOUTUBE_URL [OPTIONS]

# Example with basic options
robot-fight-detector livestream https://www.youtube.com/watch?v=STREAM_ID

# Process livestream with WebVTT output and custom interval
robot-fight-detector livestream https://www.youtube.com/watch?v=STREAM_ID --format vtt --interval 2

# Limit processing duration to 30 minutes
robot-fight-detector livestream https://www.youtube.com/watch?v=STREAM_ID --duration 30

# Process stream without saving frames (reduces disk usage)
robot-fight-detector livestream https://www.youtube.com/watch?v=STREAM_ID --no-save-frames

# Customize stream URL refresh interval (in seconds, default: 300)
robot-fight-detector livestream https://www.youtube.com/watch?v=STREAM_ID --refresh-interval 600
```

Note: The livestream feature requires `yt-dlp` which is installed with the package. Press Ctrl+C at any time to stop livestream processing and save results.

### Options

#### Common Options (all commands)

- `--output-dir`, `-o`: Custom output directory (default: `./robot_fights_output`)
- `--interval`, `-i`: Analysis interval in seconds (default: `1.0`)
- `--model`, `-m`: Alternative model to use
- `--format`, `-f`: Output format (`json` or `vtt`, default: `json`)
- `--save-frames/--no-save-frames`: Whether to save detected frame images (default: `--save-frames`)
- `--event-name`: Name of robot combat event for better robot identification (e.g., "BattleBots Season 6")

#### Video Processing Options

- `--extract-clips/--no-extract-clips`: Extract video clips for each fight segment (default: `--no-extract-clips`)
- `--clip-padding`: Seconds to add before and after each fight clip (default: `2.0`)
- `--clip-quality`: Quality of extracted clips (`low`, `medium`, `high`, default: `medium`)

#### Livestream Processing Options

- `--duration`, `-d`: Duration in minutes to process the stream (default: continuous until Ctrl+C)
- `--refresh-interval`: How often to refresh the stream URL in seconds (default: `300`)
- All other video options apply to livestreams as well

## Output

For video analysis, the detector provides output in either JSON or WebVTT format:

### JSON Output (Default)

With JSON output, the detector will:
1. Create a directory with detected frames as JPG images (if `--save-frames` is used)
2. Generate a JSON report with timestamps and descriptions
3. Print a summary of detected robot fights

Example JSON output:
```json
{
  "video_file": "videos/robot_battle.mp4",
  "analysis_date": "2025-05-14T12:34:56.789012",
  "total_detections": 3,
  "detections": [
    {
      "timestamp": 14.5,
      "frame_number": 348,
      "description": "YES. [Gigabyte] vs [Free Shipping]: Spinner weapon hits opponent's front wedge.",
      "frame_file": "robot_fight_14.50s.jpg"
    },
    ...
  ],
  "settings": {
    "interval": 1.0,
    "model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  }
}
```

### WebVTT Output

With WebVTT output (`--format vtt`), the detector will:
1. Create a directory with detected frames as JPG images (if `--save-frames` is used)
2. Generate a JSON report with timestamps and descriptions
3. Create a WebVTT (.vtt) file with fight segments and descriptions
4. Print a summary of detected fight segments

Example WebVTT output:
```
WEBVTT

00:00:14:500 --> 00:00:32:250
[FIGHT 1] Gigabyte vs Free Shipping

00:00:17:750 --> 00:00:20:750
[FIGHT 1] Gigabyte vs Free Shipping: Spinner weapon hits opponent's front wedge

00:01:05:100 --> 00:01:18:200
[FIGHT 2] Tombstone vs Witch Doctor

00:01:10:100 --> 00:01:13:100
[FIGHT 2] Tombstone vs Witch Doctor: Horizontal blade causes major damage
```

The WebVTT format can be used with most video players that support subtitles, allowing you to see fight segments and descriptions while watching the video. It's especially useful for long videos where you want to quickly navigate to the fight scenes.

### Video Clips Output

With clip extraction (`--extract-clips`), the detector will:
1. Identify continuous segments of robot fights
2. Extract each segment as a separate MP4 video clip
3. Include a configurable padding before and after each fight
4. Save all clips to a `clips` subdirectory in the output directory

Clips are named using the format:
```
[video_name]_[robot1]_vs_[robot2]_[timestamp].mp4
```

Clip quality can be configured with the `--clip-quality` option:
- `high`: Slow encoding, best quality (CRF 18)
- `medium`: Balanced encoding (CRF 23)
- `low`: Fast encoding, smaller files (CRF 28)

For analyzing long tournament videos, combining WebVTT and clip extraction provides both navigation aids and shareable fight highlights.

## Development

```bash
# Install development dependencies
uv pip install pytest black isort flake8

# Format code
black .
isort .

# Lint code
flake8

# Run tests
pytest
```

## License

MIT

## Credits

- SmolVLM2 model by [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- Created by [ricklon](https://github.com/ricklon)