#!/usr/bin/env python3
"""
Robot Fight Detector - A CLI tool to detect robot fights in videos using SmolVLM2
"""

import click
import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re
from collections import defaultdict
import ffmpeg
import subprocess
import shutil

# Load environment variables from .env file
load_dotenv()

# Get environment variables
DEFAULT_MODEL = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./robot_fights_output")
DEFAULT_INTERVAL = float(os.getenv("DEFAULT_INTERVAL", "1.0"))

# Helper functions for WebVTT output
def format_timestamp(seconds):
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ":")

def extract_robot_names(response):
    """Extract robot names from model response"""
    if not response.upper().startswith("YES"):
        return None, None, ""
    
    # Try to extract [Robot 1] vs [Robot 2]: Description format
    match = re.search(r"YES\.\s*\[?([^][\n:]+?)\]?\s+(?:vs|VS|versus)\s+\[?([^][\n:]+?)\]?[:\.](.*)", response)
    if match:
        robot1 = match.group(1).strip()
        robot2 = match.group(2).strip()
        description = match.group(3).strip()
        return robot1, robot2, description
    
    # If the format doesn't match exactly, just use the whole response
    return "Unknown", "Unknown", response.replace("YES.", "").strip()

def group_segments(detections, interval, min_gap=5.0):
    """Group detections into continuous fight segments with start/end times"""
    if not detections:
        return []
    
    # Sort detections by timestamp
    sorted_detections = sorted(detections, key=lambda x: x["timestamp"])
    
    segments = []
    current_segment = {
        "start": sorted_detections[0]["timestamp"],
        "robots": set(),
        "descriptions": []
    }
    
    last_timestamp = sorted_detections[0]["timestamp"]
    
    for detection in sorted_detections[1:]:
        # Extract robot names
        robot1, robot2, desc = extract_robot_names(detection["description"])
        
        # If robots are detected, add them to the set
        if robot1 and robot1 != "Unknown":
            current_segment["robots"].add(robot1)
        if robot2 and robot2 != "Unknown":
            current_segment["robots"].add(robot2)
            
        # Add description
        if desc:
            current_segment["descriptions"].append({
                "time": detection["timestamp"],
                "text": desc
            })
        
        # Check if this is part of the same segment (gap less than min_gap)
        if detection["timestamp"] - last_timestamp > min_gap + interval:
            # Finalize current segment
            current_segment["end"] = last_timestamp
            current_segment["robots"] = list(current_segment["robots"]) if current_segment["robots"] else ["Unknown Robots"]
            segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                "start": detection["timestamp"],
                "robots": set(),
                "descriptions": []
            }
            if robot1 and robot1 != "Unknown":
                current_segment["robots"].add(robot1)
            if robot2 and robot2 != "Unknown":
                current_segment["robots"].add(robot2)
            if desc:
                current_segment["descriptions"].append({
                    "time": detection["timestamp"],
                    "text": desc
                })
        
        last_timestamp = detection["timestamp"]
    
    # Finalize the last segment
    current_segment["end"] = last_timestamp
    current_segment["robots"] = list(current_segment["robots"]) if current_segment["robots"] else ["Unknown Robots"]
    segments.append(current_segment)
    
    return segments

def create_webvtt(segments, output_path):
    """Create a WebVTT file from segments"""
    with open(output_path, 'w') as f:
        f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments):
            robots_str = " vs ".join(segment["robots"][:2])  # Limit to two robots for readability
            
            # Write segment header with start and end times
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"[FIGHT {i+1}] {robots_str}\n\n")
            
            # Write individual actions within segment if available
            for j, desc in enumerate(segment["descriptions"]):
                if j > 0:  # Skip the first one as we already included it in the header
                    f.write(f"{format_timestamp(desc['time'])} --> {format_timestamp(desc['time'] + 3.0)}\n")
                    f.write(f"[FIGHT {i+1}] {robots_str}: {desc['text']}\n\n")
    
    return output_path

def extract_fight_clips(video_path, segments, output_dir, padding=2.0, quality="medium"):
    """Extract video clips for each fight segment using ffmpeg
    
    Args:
        video_path (str): Path to the source video
        segments (list): List of fight segments with start and end times
        output_dir (Path): Directory to save clips to
        padding (float): Additional seconds to include before and after the fight (default: 2.0)
        quality (str): Video quality - low, medium, high (default: medium)
    
    Returns:
        list: List of paths to the generated clips
    """
    # Create clips directory
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    # Set encoding parameters based on quality
    if quality == "high":
        preset = "slow"
        crf = "18"
    elif quality == "medium":
        preset = "medium"
        crf = "23"
    else:  # low
        preset = "ultrafast"
        crf = "28"
    
    clips = []
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Add padding but ensure we don't go below 0
        start_time = max(0, segment["start"] - padding)
        # Get robots text for filename
        robots = "_vs_".join([r.replace(" ", "_") for r in segment["robots"][:2]])
        if not robots or robots == "Unknown_Robots":
            robots = f"fight_{i+1}"
            
        # Create output filename
        output_file = clips_dir / f"{Path(video_path).stem}_{robots}_{format_timestamp(start_time).replace(':', '_')}.mp4"
        
        # Calculate duration
        duration = segment["end"] - start_time + padding
        
        try:
            # Run ffmpeg command
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", crf,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_file)
            ]
            
            # Execute the command
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                click.echo(f"Warning: Failed to extract clip {i+1}: {result.stderr}", err=True)
            else:
                clips.append(output_file)
                click.echo(f"✓ Extracted clip {i+1}: {output_file.name}")
                
        except Exception as e:
            click.echo(f"Error extracting clip {i+1}: {str(e)}", err=True)
    
    if clips:
        click.echo(f"\nExtracted {len(clips)} fight clips to {clips_dir}")
    else:
        click.echo("No clips were extracted.")
        
    return clips


class RobotFightDetector:
    def __init__(self, model_name=None):
        """Initialize the robot fight detector with SmolVLM2 model."""
        self.model_name = model_name if model_name else DEFAULT_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        click.echo(f"Loading SmolVLM2 model on {self.device}...")
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            click.echo("Warning: HUGGINGFACE_TOKEN not found in .env file. Authentication may fail.", err=True)
        
        try:
            # Use the proper model loading technique for SmolVLM2
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=hf_token
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=hf_token
            ).to(self.device)
            click.echo("✓ Model loaded successfully!")
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            raise
    
    def analyze_frame(self, frame):
        """Analyze a single frame to detect robot fights."""
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Prepare prompt for robot fight detection with robot identification
        prompt = """Look at this image carefully. Are there any robots fighting or engaging in combat?
        
        If you see robots fighting, respond in this exact format:
        YES. [Robot 1 Name] vs [Robot 2 Name]: Brief description of what's happening in the fight.
        
        If there are no robot fights, just respond with:
        NO.
        
        Look for:
        - Robots in combat positions or arenas (especially BattleBots competitions)
        - Robot battles or fights with visible damage or attacks
        - Names or identifiers of the robots (often visible on the robots or arena)
        - Types of weapons the robots are using (spinners, flippers, hammers, etc.)
        - The specific action happening at this moment (attacking, defending, etc.)"""
        
        try:
            # Process with SmolVLM2 using the chat template API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},  # Pass PIL image directly
                        {"type": "text", "text": prompt},
                    ]
                },
            ]
            
            # Prepare inputs using the processor
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
                
                # Generate response
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True
                )
                
                # Decode response
                full_response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
                
                # Clean up response - strip out the prompt and headers
                response = full_response
                if "User:" in response and "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                    # If it starts with YES or NO, leave as is
                    # Otherwise, clean up any prefixes
                    if not response.upper().startswith("YES") and not response.upper().startswith("NO"):
                        if "." in response:
                            response = response.split(".", 1)[1].strip()
                
                # Check if robot fight is detected
                is_robot_fight = response.upper().startswith('YES')
                return is_robot_fight, response
                
        except Exception as e:
            click.echo(f"Error analyzing frame: {e}", err=True)
            return False, f"Error: {str(e)}"
    
    def process_video(self, video_path, output_dir, interval=1.0, confidence_threshold=0.7, 
                   output_format="json", save_frames=True, extract_clips=False, clip_padding=2.0, 
                   clip_quality="medium"):
        """Process video to find robot fights."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise click.ClickException(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_interval = int(fps * interval)
        
        click.echo(f"Processing video: {video_path.name}")
        click.echo(f"Total duration: {duration:.2f}s, FPS: {fps}, Analyzing every {interval}s")
        
        detections = []
        frame_count = 0
        processed_frames = 0
        
        with click.progressbar(length=total_frames//frame_interval) as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Analyze frame
                    is_fight, description = self.analyze_frame(frame)
                    
                    if is_fight:
                        # Save frame if requested
                        frame_filename = None
                        if save_frames:
                            frame_filename = f"robot_fight_{timestamp:.2f}s.jpg"
                            frame_path = output_dir / frame_filename
                            cv2.imwrite(str(frame_path), frame)
                        
                        detection = {
                            "timestamp": timestamp,
                            "frame_number": frame_count,
                            "description": description,
                            "frame_file": frame_filename
                        }
                        detections.append(detection)
                        click.echo(f"\n✓ Robot fight detected at {timestamp:.2f}s")
                        click.echo(f"  Description: {description}")
                    
                    processed_frames += 1
                    bar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        # Save results based on output format
        results = {
            "video_file": str(video_path),
            "analysis_date": datetime.now().isoformat(),
            "total_detections": len(detections),
            "detections": detections,
            "settings": {
                "interval": interval,
                "model": self.model_name
            }
        }
        
        # Save JSON results
        results_file = output_dir / f"{video_path.stem}_robot_fights.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Process based on output format
        if output_format.lower() == "vtt" or extract_clips:
            # Group detections into segments
            segments = group_segments(detections, interval)
            results["segments"] = segments
            
            # Create WebVTT file if requested
            if output_format.lower() == "vtt":
                vtt_file = output_dir / f"{video_path.stem}_robot_fights.vtt"
                create_webvtt(segments, vtt_file)
                results["vtt_file"] = str(vtt_file)
            
            # Extract video clips if requested
            if extract_clips and segments:
                click.echo("\nExtracting fight clips...")
                clips = extract_fight_clips(
                    video_path, segments, output_dir, 
                    padding=clip_padding, quality=clip_quality
                )
                results["clips"] = [str(clip) for clip in clips]
            
            return results, results_file
        
        return results, results_file


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Robot Fight Detector - Find robot fights in videos using SmolVLM2"""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), 
              default=DEFAULT_OUTPUT_DIR, help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
@click.option('--interval', '-i', type=float, default=DEFAULT_INTERVAL, 
              help=f'Analyze every N seconds (default: {DEFAULT_INTERVAL})')
@click.option('--model', '-m', default=DEFAULT_MODEL,
              help=f'SmolVLM2 model to use (default: {DEFAULT_MODEL})')
@click.option('--format', '-f', 'output_format', type=click.Choice(['json', 'vtt']), default='json',
              help='Output format (json or vtt)')
@click.option('--save-frames/--no-save-frames', default=True,
              help='Save frames of detected robot fights (default: True)')
@click.option('--extract-clips/--no-extract-clips', default=False,
              help='Extract video clips for each fight segment (default: False)')
@click.option('--clip-padding', type=float, default=2.0,
              help='Seconds to add before and after each fight clip (default: 2.0)')
@click.option('--clip-quality', type=click.Choice(['low', 'medium', 'high']), default='medium',
              help='Quality of extracted clips (default: medium)')
def detect(video_path, output_dir, interval, model, output_format, save_frames, 
           extract_clips, clip_padding, clip_quality):
    """Detect robot fights in a video file."""
    try:
        # Make sure ffmpeg is installed if extracting clips
        if extract_clips:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    click.echo("Error: ffmpeg is required for clip extraction but not found.\n"
                               "Please install ffmpeg: https://ffmpeg.org/download.html", err=True)
                    raise click.Abort()
            except FileNotFoundError:
                click.echo("Error: ffmpeg is required for clip extraction but not found.\n"
                           "Please install ffmpeg: https://ffmpeg.org/download.html", err=True)
                raise click.Abort()
        
        detector = RobotFightDetector(model_name=model)
        results, results_file = detector.process_video(
            video_path, output_dir, interval, 
            output_format=output_format, 
            save_frames=save_frames,
            extract_clips=extract_clips,
            clip_padding=clip_padding,
            clip_quality=clip_quality
        )
        
        click.echo(f"\n🤖 Analysis complete!")
        click.echo(f"Found {results['total_detections']} robot fight scenes")
        click.echo(f"Results saved to: {results_file}")
        
        if save_frames and results['total_detections'] > 0:
            click.echo(f"Frames saved to: {output_dir}")
        
        # If clips were extracted, display information about clips
        if extract_clips and 'clips' in results and results['clips']:
            click.echo(f"\nExtracted {len(results['clips'])} fight clips")
            clips_dir = Path(results['clips'][0]).parent
            click.echo(f"Clips saved to: {clips_dir}")
        
        # If WebVTT was generated or segments exist, display information about segments
        if 'segments' in results:
            click.echo(f"\nIdentified {len(results['segments'])} fight segments:")
            for i, segment in enumerate(results['segments']):
                robots = " vs ".join(segment['robots'][:2])
                click.echo(f"  • Fight {i+1}: {format_timestamp(segment['start'])} - {format_timestamp(segment['end'])} ({robots})")
        elif results['total_detections'] > 0:
            click.echo("\nDetected robot fights:")
            for detection in results['detections']:
                click.echo(f"  • {detection['timestamp']:.2f}s: {detection['description']}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model', '-m', default=DEFAULT_MODEL,
              help=f'SmolVLM2 model to use (default: {DEFAULT_MODEL})')
def analyze_image(image_path, model):
    """Analyze a single image for robot fights."""
    try:
        detector = RobotFightDetector(model_name=model)
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise click.ClickException(f"Could not load image: {image_path}")
        
        is_fight, description = detector.analyze_frame(frame)
        
        click.echo(f"\nImage: {image_path.name}")
        click.echo(f"Robot fight detected: {'YES' if is_fight else 'NO'}")
        click.echo(f"Description: {description}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--skip-model', is_flag=True, help="Skip model loading test")
def test(skip_model):
    """Test if dependencies and model can be loaded successfully."""
    try:
        from transformers import __version__ as transformers_version
        
        click.echo("Testing dependencies...")
        click.echo(f"✓ OpenCV version: {cv2.__version__}")
        click.echo(f"✓ PyTorch version: {torch.__version__}")
        click.echo(f"✓ Transformers version: {transformers_version}")
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        click.echo(f"✓ PyTorch device: {device}")
        
        # Check environment variables
        click.echo("\nEnvironment configuration:")
        click.echo(f"✓ MODEL_NAME: {DEFAULT_MODEL}")
        click.echo(f"✓ OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
        click.echo(f"✓ DEFAULT_INTERVAL: {DEFAULT_INTERVAL}")
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            click.echo("⚠ HUGGINGFACE_TOKEN is not set. Model loading may fail.", err=True)
            click.echo("  Please set your HuggingFace token in the .env file.")
        else:
            click.echo("✓ HUGGINGFACE_TOKEN is set")
        
        # Test model loading if not skipped
        if not skip_model:
            click.echo("\nTesting SmolVLM2 model loading...")
            detector = RobotFightDetector()
            click.echo("✓ Model test successful!")
            click.echo(f"Device: {detector.device}")
            click.echo(f"Model: {detector.model_name}")
        
        click.echo("\n✓ All tests completed successfully!")
    except Exception as e:
        click.echo(f"✗ Test failed: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()
