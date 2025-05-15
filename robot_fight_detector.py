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
import time
import yt_dlp
import threading
import signal
import sys

# Load environment variables from .env file
load_dotenv()

# Get environment variables
DEFAULT_MODEL = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./robot_fights_output")
DEFAULT_INTERVAL = float(os.getenv("DEFAULT_INTERVAL", "1.0"))
DEFAULT_STREAM_REFRESH_INTERVAL = int(os.getenv("STREAM_REFRESH_INTERVAL", "300"))  # 5 minutes
DEFAULT_EVENT_NAME = os.getenv("EVENT_NAME", "")  # Optional event name

# Helper functions for WebVTT output
def format_timestamp(seconds):
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ":")

def extract_robot_names(response):
    """Extract robot names, status, and match details from model response"""
    # If response indicates no robots or no fight, return None values
    if response.upper().startswith("NO ROBOTS") or response.upper().startswith("NO FIGHT"):
        return None, None, response
    
    if not response.upper().startswith("YES"):
        return None, None, ""
    
    # Initialize default values for all fields
    robot1 = "Unknown"
    robot2 = "Unknown"
    description = ""
    status = "Ongoing"  # Default status
    sponsors = ""
    damage = ""
    timer = ""
    score = ""
    
    # Try to extract [Robot 1] vs [Robot 2]: Description format from the first line
    lines = response.strip().split('\n')
    first_line = lines[0]
    
    match = re.search(r"YES\.\s*\[?([^][\n:]+?)\]?\s+(?:vs|VS|versus)\s+\[?([^][\n:]+?)\]?[:\.](.*)", first_line)
    if match:
        robot1 = match.group(1).strip()
        robot2 = match.group(2).strip()
        description = match.group(3).strip()
    
    # Extract additional information from remaining lines
    for line in lines[1:]:
        line = line.strip()
        
        # Extract match status
        if line.startswith("Status:"):
            status_value = line.replace("Status:", "").strip()
            status = status_value
        
        # Extract sponsors
        elif line.startswith("Sponsors:"):
            sponsors = line.replace("Sponsors:", "").strip()
        
        # Extract damage information
        elif line.startswith("Damage:"):
            damage = line.replace("Damage:", "").strip()
        
        # Extract timer information
        elif line.startswith("Timer:"):
            timer = line.replace("Timer:", "").strip()
        
        # Extract score information
        elif line.startswith("Score:"):
            score = line.replace("Score:", "").strip()
    
    # Sanitize robot names to avoid trademark issues
    trademark_replacements = {
        r"(?i)battlebots?": "combat robot",
        r"(?i)battlebots?\s+competition": "robot competition",
        r"(?i)battle\s*bots?": "combat robot" 
    }
    
    for pattern, replacement in trademark_replacements.items():
        robot1 = re.sub(pattern, replacement, robot1)
        robot2 = re.sub(pattern, replacement, robot2)
        description = re.sub(pattern, replacement, description)
        sponsors = re.sub(pattern, replacement, sponsors)
    
    # Create a structured response dictionary
    match_info = {
        "robot1": robot1,
        "robot2": robot2,
        "description": description,
        "status": status,
        "sponsors": sponsors,
        "damage": damage,
        "timer": timer,
        "score": score
    }
    
    return robot1, robot2, description, match_info

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
        "descriptions": [],
        "sponsors": set(),
        "match_status": [],
        "damage_reports": []
    }
    
    last_timestamp = sorted_detections[0]["timestamp"]
    
    # Process first detection
    first_detection = sorted_detections[0]
    try:
        robot1, robot2, desc, match_info = extract_robot_names(first_detection["description"])
        
        # Add match info to the first segment
        if robot1 and robot1 != "Unknown":
            current_segment["robots"].add(robot1)
        if robot2 and robot2 != "Unknown":
            current_segment["robots"].add(robot2)
            
        # Add description and other details
        if desc:
            current_segment["descriptions"].append({
                "time": first_detection["timestamp"],
                "text": desc
            })
        
        # Add match status
        if "status" in match_info and match_info["status"]:
            current_segment["match_status"].append({
                "time": first_detection["timestamp"],
                "status": match_info["status"]
            })
        
        # Add sponsors if available
        if "sponsors" in match_info and match_info["sponsors"]:
            for sponsor in match_info["sponsors"].split("|"):
                if sponsor.strip():
                    current_segment["sponsors"].add(sponsor.strip())
        
        # Add damage reports if available
        if "damage" in match_info and match_info["damage"]:
            current_segment["damage_reports"].append({
                "time": first_detection["timestamp"],
                "description": match_info["damage"]
            })
    except ValueError:
        # Handle case where extract_robot_names returns only 3 values (old format)
        robot1, robot2, desc = extract_robot_names(first_detection["description"])
        if robot1 and robot1 != "Unknown":
            current_segment["robots"].add(robot1)
        if robot2 and robot2 != "Unknown":
            current_segment["robots"].add(robot2)
        if desc:
            current_segment["descriptions"].append({
                "time": first_detection["timestamp"],
                "text": desc
            })
    
    # Process remaining detections
    for detection in sorted_detections[1:]:
        try:
            # Extract robot names and match info
            robot1, robot2, desc, match_info = extract_robot_names(detection["description"])
            
            # Check if this is part of the same segment based on time gap
            # Or if match status indicates a new match starting
            new_segment = False
            if detection["timestamp"] - last_timestamp > min_gap + interval:
                new_segment = True
            elif "status" in match_info and match_info["status"] and "Match start" in match_info["status"]:
                # Start a new segment if this is a match start and previous segment had content
                if current_segment["descriptions"]:
                    new_segment = True
            
            if new_segment:
                # Finalize current segment
                current_segment["end"] = last_timestamp
                current_segment["robots"] = list(current_segment["robots"]) if current_segment["robots"] else ["Unknown Robots"]
                current_segment["sponsors"] = list(current_segment["sponsors"]) if current_segment["sponsors"] else []
                segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    "start": detection["timestamp"],
                    "robots": set(),
                    "descriptions": [],
                    "sponsors": set(),
                    "match_status": [],
                    "damage_reports": []
                }
            
            # Add robot names to current segment
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
            
            # Add match status
            if "status" in match_info and match_info["status"]:
                current_segment["match_status"].append({
                    "time": detection["timestamp"],
                    "status": match_info["status"]
                })
            
            # Add sponsors if available
            if "sponsors" in match_info and match_info["sponsors"]:
                for sponsor in match_info["sponsors"].split("|"):
                    if sponsor.strip():
                        current_segment["sponsors"].add(sponsor.strip())
            
            # Add damage reports if available
            if "damage" in match_info and match_info["damage"]:
                current_segment["damage_reports"].append({
                    "time": detection["timestamp"],
                    "description": match_info["damage"]
                })
                
        except ValueError:
            # Handle case where extract_robot_names returns only 3 values (old format)
            robot1, robot2, desc = extract_robot_names(detection["description"])
            
            # Check if this is part of the same segment (gap less than min_gap)
            if detection["timestamp"] - last_timestamp > min_gap + interval:
                # Finalize current segment
                current_segment["end"] = last_timestamp
                current_segment["robots"] = list(current_segment["robots"]) if current_segment["robots"] else ["Unknown Robots"]
                current_segment["sponsors"] = list(current_segment["sponsors"]) if current_segment["sponsors"] else []
                segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    "start": detection["timestamp"],
                    "robots": set(),
                    "descriptions": [],
                    "sponsors": set(),
                    "match_status": [],
                    "damage_reports": []
                }
            
            # Add robot names to current segment
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
        
        last_timestamp = detection["timestamp"]
    
    # Finalize the last segment
    current_segment["end"] = last_timestamp
    current_segment["robots"] = list(current_segment["robots"]) if current_segment["robots"] else ["Unknown Robots"]
    current_segment["sponsors"] = list(current_segment["sponsors"]) if current_segment["sponsors"] else []
    segments.append(current_segment)
    
    return segments

def create_webvtt(segments, output_path):
    """Create a WebVTT file from segments with enhanced information"""
    with open(output_path, 'w') as f:
        f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments):
            robots_str = " vs ".join(segment["robots"][:2])  # Limit to two robots for readability
            
            # Add sponsors if available
            sponsors_str = ""
            if "sponsors" in segment and segment["sponsors"]:
                sponsors_list = [s for s in segment["sponsors"] if s]
                if sponsors_list:
                    sponsors_str = f" (Sponsors: {', '.join(sponsors_list[:3])})"  # Limit to 3 sponsors
            
            # Get match status from first entry if available
            match_status = ""
            if "match_status" in segment and segment["match_status"]:
                match_status = f" [{segment['match_status'][0]['status']}]" if segment["match_status"] else ""
            
            # Write segment header with start and end times
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"[FIGHT {i+1}] {robots_str}{match_status}{sponsors_str}\n\n")
            
            # Write individual actions within segment if available
            for j, desc in enumerate(segment["descriptions"]):
                if j > 0:  # Skip the first one as we already included it in the header
                    f.write(f"{format_timestamp(desc['time'])} --> {format_timestamp(desc['time'] + 3.0)}\n")
                    
                    # Find matching match status and damage report for this timestamp if available
                    status_at_time = ""
                    if "match_status" in segment:
                        for status_entry in segment["match_status"]:
                            if abs(status_entry["time"] - desc["time"]) < 1.0:  # Within 1 second
                                status_at_time = f" [{status_entry['status']}]"
                                break
                    
                    damage_at_time = ""
                    if "damage_reports" in segment and segment["damage_reports"]:
                        for damage_entry in segment["damage_reports"]:
                            if abs(damage_entry["time"] - desc["time"]) < 1.0:  # Within 1 second
                                damage_at_time = f" (Damage: {damage_entry['description']})"
                                break
                    
                    f.write(f"[FIGHT {i+1}] {robots_str}{status_at_time}: {desc['text']}{damage_at_time}\n\n")
            
            # Add specific match status changes
            if "match_status" in segment and len(segment["match_status"]) > 1:
                for status_entry in segment["match_status"][1:]:  # Skip the first one as we already included it
                    # Only include significant status changes: Match start, Victory, Match end, Entanglement pause
                    if any(key in status_entry["status"] for key in ["Match start", "Victory", "Match end", "Entanglement"]):
                        f.write(f"{format_timestamp(status_entry['time'])} --> {format_timestamp(status_entry['time'] + 3.0)}\n")
                        f.write(f"[FIGHT {i+1}] {robots_str}: STATUS CHANGE - {status_entry['status']}\n\n")
            
            # Add damage reports as separate entries if not already covered
            if "damage_reports" in segment and segment["damage_reports"]:
                for damage_entry in segment["damage_reports"]:
                    # Check if this damage report wasn't already included with a description
                    already_included = False
                    for desc in segment["descriptions"]:
                        if abs(desc["time"] - damage_entry["time"]) < 1.0:
                            already_included = True
                            break
                    
                    if not already_included:
                        f.write(f"{format_timestamp(damage_entry['time'])} --> {format_timestamp(damage_entry['time'] + 3.0)}\n")
                        f.write(f"[FIGHT {i+1}] {robots_str}: DAMAGE - {damage_entry['description']}\n\n")
    
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

def get_stream_url(url):
    """Extract actual video URL from YouTube link using yt-dlp
    
    Args:
        url (str): YouTube URL to extract stream from
        
    Returns:
        tuple: (stream_url, stream_title)
    """
    ydl_opts = {
        'format': 'best[height<=720]',  # Limit resolution to reduce processing load
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_title = info.get('title', 'Unknown Stream')
            return info['url'], stream_title
    except Exception as e:
        raise click.ClickException(f"Failed to extract stream URL: {str(e)}")


class RobotFightDetector:
    def __init__(self, model_name=None, event_name=None):
        """Initialize the robot fight detector with SmolVLM2 model."""
        self.model_name = model_name if model_name else DEFAULT_MODEL
        self.event_name = event_name if event_name is not None else DEFAULT_EVENT_NAME
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
            
            if self.event_name:
                click.echo(f"✓ Using event context: {self.event_name}")
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            raise
    
    def analyze_frame(self, frame):
        """Analyze a single frame to detect robot fights."""
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Prepare prompt for robot fight detection with robot identification
        event_context = f"\nThis is from the event: {self.event_name}.\n" if self.event_name else ""
        
        prompt = f"""Look at this image carefully and analyze if there are robot combat matches occurring.{event_context}

Answer in one of these precise formats:

If NO robots visible:
NO ROBOTS.

If robots visible but NOT fighting:
NO FIGHT. [Brief description: setup, display, introduction, etc]

If robots ARE fighting:
YES. [Robot 1 Name] vs [Robot 2 Name]: [Brief action description]
Status: [Match start/Ongoing/Entanglement pause/Match end/Victory]
Sponsors: [Robot 1: sponsor names] | [Robot 2: sponsor names]
Damage: [Description of visible damage to either robot]
Timer: [Match timer if visible on screen]
Score: [Points or score if displayed]

For accurate analysis:
- Typical matches last 2-5 minutes (usually with a 3-minute timer)
- Watch for "3, 2, 1, FIGHT!" countdowns indicating match start
- Note when referees pause matches for entanglements
- Watch for match end signals (buzzer/horn/lights)
- Victory can be by knockout (KO), technical knockout (TKO), or judges' decision
- Look for sponsor logos on robots, arena barriers, or team uniforms
- Note team members and their reactions for victory confirmation
- Check displays/graphics for official match information
- Observe damage indicators: sparks, detached parts, mobility issues
- Be extremely strict - only report fighting if you're highly confident"""
        
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
                
                # Also filter out "NO ROBOTS" responses
                if response.upper().startswith('NO ROBOTS'):
                    return False, response
                
                # Return false for "NO FIGHT" responses but keep the description
                if response.upper().startswith('NO FIGHT'):
                    return False, response
                
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
                        
                        # Try to extract detailed match information if available
                        try:
                            robot1, robot2, desc, match_info = extract_robot_names(description)
                            detection = {
                                "timestamp": timestamp,
                                "frame_number": frame_count,
                                "description": description,
                                "frame_file": frame_filename,
                                "robot1": robot1,
                                "robot2": robot2,
                                "match_info": match_info
                            }
                        except ValueError:
                            # Fall back to old format
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
        
    def process_livestream(self, stream_url, output_dir, interval=1.0, save_frames=True, 
                        output_format="json", extract_clips=False, clip_padding=2.0, 
                        clip_quality="medium", duration=None, refresh_interval=DEFAULT_STREAM_REFRESH_INTERVAL):
        """Process a YouTube livestream to find robot fights.
        
        Args:
            stream_url (str): YouTube livestream URL
            output_dir (Path): Directory to save results
            interval (float): Sample frame every N seconds
            save_frames (bool): Whether to save detected frames
            output_format (str): Output format ("json" or "vtt")
            extract_clips (bool): Whether to extract fight clips
            clip_padding (float): Seconds to add before/after clips
            clip_quality (str): Quality of extracted clips
            duration (float): Duration in minutes to process (None for continuous)
            refresh_interval (int): How often to refresh the stream URL in seconds
            
        Returns:
            tuple: (results, results_file)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract the actual stream URL from YouTube
        real_url, stream_title = get_stream_url(stream_url)
        
        click.echo(f"Processing livestream: {stream_title}")
        click.echo(f"Analyzing every {interval}s")
        
        # Set up results file name from YouTube title
        safe_title = re.sub(r'[\\/*?:"<>|]', "", stream_title).replace(" ", "_")
        results_file = output_dir / f"{safe_title}_robot_fights.json"
        
        # Initialize capture
        cap = cv2.VideoCapture(real_url)
        if not cap.isOpened():
            raise click.ClickException(f"Could not open stream URL")
        
        # Get FPS and calculate frame interval
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        # Set up duration limit if specified
        end_time = None
        if duration:
            end_time = time.time() + (duration * 60)
            click.echo(f"Will process for {duration} minutes")
        
        # Initialize variables
        detections = []
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        last_refresh_time = start_time
        elapsed_time = 0
        
        # Setup for handling interrupts
        stop_event = threading.Event()
        
        def signal_handler(sig, frame):
            click.echo("\nInterrupted! Finishing processing...")
            stop_event.set()
            
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            with click.progressbar(
                length=100, 
                label=f"Processing livestream",
                show_eta=False
            ) as bar:
                last_update = 0
                
                while not stop_event.is_set():
                    # Check if we need to refresh the stream URL
                    current_time = time.time()
                    if current_time - last_refresh_time > refresh_interval:
                        click.echo("\nRefreshing stream URL...")
                        cap.release()
                        real_url, _ = get_stream_url(stream_url)
                        cap = cv2.VideoCapture(real_url)
                        if not cap.isOpened():
                            click.echo("Error: Could not reopen stream. Trying again...")
                            time.sleep(5)
                            continue
                        last_refresh_time = current_time
                    
                    # Check if we've reached the duration limit
                    if end_time and current_time > end_time:
                        click.echo("\nReached specified duration. Stopping...")
                        break
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        click.echo("\nStream ended or frame could not be read. Trying to reconnect...")
                        cap.release()
                        time.sleep(5)  # Wait before reconnecting
                        real_url, _ = get_stream_url(stream_url)
                        cap = cv2.VideoCapture(real_url)
                        continue
                    
                    # Process frame at the specified interval
                    if frame_count % frame_interval == 0:
                        elapsed_time = current_time - start_time
                        timestamp = elapsed_time
                        
                        # Analyze frame
                        is_fight, description = self.analyze_frame(frame)
                        
                        if is_fight:
                            # Save frame if requested
                            frame_filename = None
                            if save_frames:
                                frame_filename = f"robot_fight_{timestamp:.2f}s.jpg"
                                frame_path = output_dir / frame_filename
                                cv2.imwrite(str(frame_path), frame)
                            
                            # Add detection with detailed match information if available
                            try:
                                robot1, robot2, desc, match_info = extract_robot_names(description)
                                detection = {
                                    "timestamp": timestamp,
                                    "frame_number": frame_count,
                                    "description": description,
                                    "frame_file": frame_filename,
                                    "time": datetime.now().isoformat(),
                                    "robot1": robot1,
                                    "robot2": robot2,
                                    "match_info": match_info
                                }
                            except ValueError:
                                # Fall back to old format
                                detection = {
                                    "timestamp": timestamp,
                                    "frame_number": frame_count,
                                    "description": description,
                                    "frame_file": frame_filename,
                                    "time": datetime.now().isoformat()
                                }
                            detections.append(detection)
                            
                            # Create temporary clip if requested
                            if extract_clips:
                                # Save a short clip of the detected fight
                                clip_filename = f"live_robot_fight_{timestamp:.2f}s.mp4"
                                clip_path = output_dir / "clips" / clip_filename
                                output_dir.joinpath("clips").mkdir(exist_ok=True)
                                
                                # Save current frame as a placeholder until we implement proper clip extraction
                                cv2.imwrite(str(output_dir / "clips" / f"frame_{timestamp:.2f}s.jpg"), frame)
                            
                            # Print detection
                            click.echo(f"\n✓ Robot fight detected at {timestamp:.2f}s")
                            click.echo(f"  Description: {description}")
                            
                            # Periodically save results to file
                            results = {
                                "stream_url": stream_url,
                                "stream_title": stream_title,
                                "analysis_date": datetime.now().isoformat(),
                                "total_detections": len(detections),
                                "detections": detections,
                                "settings": {
                                    "interval": interval,
                                    "model": self.model_name
                                }
                            }
                            
                            with open(results_file, 'w') as f:
                                json.dump(results, f, indent=2)
                        
                        processed_frames += 1
                        
                        # Update progress bar every 5 seconds
                        if int(elapsed_time) // 5 > last_update:
                            last_update = int(elapsed_time) // 5
                            bar.update(1)
                            if bar.length <= bar.pos:
                                bar.length = bar.pos + 100
                    
                    frame_count += 1
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.01)
        
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            
            # Release capture
            cap.release()
            
            # Final results
            results = {
                "stream_url": stream_url,
                "stream_title": stream_title,
                "analysis_date": datetime.now().isoformat(),
                "analysis_duration": time.time() - start_time,
                "total_detections": len(detections),
                "detections": detections,
                "settings": {
                    "interval": interval,
                    "model": self.model_name
                }
            }
            
            # Save final results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Process segments if needed
            if output_format.lower() == "vtt" and detections:
                segments = group_segments(detections, interval)
                results["segments"] = segments
                
                # Create WebVTT file
                vtt_file = output_dir / f"{safe_title}_robot_fights.vtt"
                create_webvtt(segments, vtt_file)
                results["vtt_file"] = str(vtt_file)
            
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
@click.option('--event-name', type=str, default=DEFAULT_EVENT_NAME,
              help='Name of robot combat event for better robot identification')
def detect(video_path, output_dir, interval, model, output_format, save_frames, 
           extract_clips, clip_padding, clip_quality, event_name):
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
        
        detector = RobotFightDetector(model_name=model, event_name=event_name)
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
                
                # Show sponsors if available
                sponsors_info = ""
                if "sponsors" in segment and segment["sponsors"]:
                    sponsors_list = [s for s in segment["sponsors"] if s]
                    if sponsors_list:
                        sponsors_info = f" - Sponsors: {', '.join(sponsors_list[:3])}"
                
                # Show match status info if available
                status_info = ""
                if "match_status" in segment and segment["match_status"]:
                    statuses = set(entry["status"] for entry in segment["match_status"] if entry["status"])
                    if statuses:
                        status_info = f" - Status: {', '.join(statuses)}"
                
                # Show damage info if available
                damage_info = ""
                if "damage_reports" in segment and segment["damage_reports"]:
                    damage_info = f" - Damage detected"
                
                duration = segment['end'] - segment['start']
                click.echo(f"  • Fight {i+1}: {format_timestamp(segment['start'])} - {format_timestamp(segment['end'])} ({duration:.1f}s) - {robots}{sponsors_info}{status_info}{damage_info}")
        elif results['total_detections'] > 0:
            click.echo("\nDetected robot fights:")
            for detection in results['detections']:
                # Try to extract enhanced information if available
                try:
                    robot1, robot2, desc, match_info = extract_robot_names(detection['description'])
                    
                    status_info = ""
                    if "status" in match_info and match_info["status"]:
                        status_info = f" [{match_info['status']}]"
                    
                    sponsors_info = ""
                    if "sponsors" in match_info and match_info["sponsors"]:
                        sponsors_info = f" - Sponsors: {match_info['sponsors']}"
                    
                    damage_info = ""
                    if "damage" in match_info and match_info["damage"]:
                        damage_info = f" - Damage: {match_info['damage']}"
                    
                    click.echo(f"  • {detection['timestamp']:.2f}s: {robot1} vs {robot2}{status_info}{sponsors_info}{damage_info}")
                
                except ValueError:
                    # Fall back to old format
                    click.echo(f"  • {detection['timestamp']:.2f}s: {detection['description']}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('stream_url')
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
@click.option('--duration', '-d', type=float, default=None,
              help='Duration in minutes to process the stream (default: continuous)')
@click.option('--refresh-interval', type=int, default=DEFAULT_STREAM_REFRESH_INTERVAL,
              help=f'How often to refresh the stream URL in seconds (default: {DEFAULT_STREAM_REFRESH_INTERVAL})')
@click.option('--event-name', type=str, default=DEFAULT_EVENT_NAME,
              help='Name of robot combat event for better robot identification')
def livestream(stream_url, output_dir, interval, model, output_format, save_frames, 
               extract_clips, clip_padding, clip_quality, duration, refresh_interval, event_name):
    """Process a YouTube livestream URL to detect robot fights in real-time.
    
    STREAM_URL should be a YouTube livestream URL like:
    https://www.youtube.com/watch?v=mt3hlsjx3dE
    
    The tool will continuously monitor the stream for robot fights.
    Press Ctrl+C to stop processing at any time.
    """
    try:
        # Check if URL is valid
        if not stream_url.startswith(('http://', 'https://')):
            raise click.ClickException("Invalid URL. Please provide a valid YouTube URL.")
        
        # Make sure yt-dlp is available
        try:
            yt_dlp_version = subprocess.run(
                ["yt-dlp", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ).stdout.strip()
            click.echo(f"Using yt-dlp version: {yt_dlp_version}")
        except FileNotFoundError:
            click.echo("Error: yt-dlp is required for livestream processing but not found.", err=True)
            raise click.Abort()
            
        # Initialize detector
        detector = RobotFightDetector(model_name=model, event_name=event_name)
        
        # Process livestream
        results, results_file = detector.process_livestream(
            stream_url, output_dir, interval,
            save_frames=save_frames,
            output_format=output_format,
            extract_clips=extract_clips,
            clip_padding=clip_padding,
            clip_quality=clip_quality,
            duration=duration,
            refresh_interval=refresh_interval
        )
        
        # Display summary
        click.echo(f"\n🤖 Livestream processing complete!")
        click.echo(f"Found {results['total_detections']} robot fight scenes")
        click.echo(f"Results saved to: {results_file}")
        
        if save_frames and results['total_detections'] > 0:
            click.echo(f"Frames saved to: {output_dir}")
        
        # If WebVTT was generated, display information about segments
        if 'segments' in results:
            click.echo(f"\nIdentified {len(results['segments'])} fight segments:")
            for i, segment in enumerate(results['segments']):
                robots = " vs ".join(segment['robots'][:2])
                
                # Show sponsors if available
                sponsors_info = ""
                if "sponsors" in segment and segment["sponsors"]:
                    sponsors_list = [s for s in segment["sponsors"] if s]
                    if sponsors_list:
                        sponsors_info = f" - Sponsors: {', '.join(sponsors_list[:3])}"
                
                # Show match status info if available
                status_info = ""
                if "match_status" in segment and segment["match_status"]:
                    statuses = set(entry["status"] for entry in segment["match_status"] if entry["status"])
                    if statuses:
                        status_info = f" - Status: {', '.join(statuses)}"
                
                # Show damage info if available
                damage_info = ""
                if "damage_reports" in segment and segment["damage_reports"]:
                    damage_info = f" - Damage detected"
                
                duration = segment['end'] - segment['start']
                click.echo(f"  • Fight {i+1}: {format_timestamp(segment['start'])} - {format_timestamp(segment['end'])} ({duration:.1f}s) - {robots}{sponsors_info}{status_info}{damage_info}")
        elif results['total_detections'] > 0:
            click.echo("\nDetected robot fights:")
            for detection in results['detections']:
                # Try to extract enhanced information if available
                try:
                    robot1, robot2, desc, match_info = extract_robot_names(detection['description'])
                    
                    status_info = ""
                    if "status" in match_info and match_info["status"]:
                        status_info = f" [{match_info['status']}]"
                    
                    sponsors_info = ""
                    if "sponsors" in match_info and match_info["sponsors"]:
                        sponsors_info = f" - Sponsors: {match_info['sponsors']}"
                    
                    damage_info = ""
                    if "damage" in match_info and match_info["damage"]:
                        damage_info = f" - Damage: {match_info['damage']}"
                    
                    click.echo(f"  • {detection['timestamp']:.2f}s: {robot1} vs {robot2}{status_info}{sponsors_info}{damage_info}")
                
                except ValueError:
                    # Fall back to old format
                    click.echo(f"  • {detection['timestamp']:.2f}s: {detection['description']}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model', '-m', default=DEFAULT_MODEL,
              help=f'SmolVLM2 model to use (default: {DEFAULT_MODEL})')
@click.option('--event-name', type=str, default=DEFAULT_EVENT_NAME,
              help='Name of robot combat event for better robot identification')
def analyze_image(image_path, model, event_name):
    """Analyze a single image for robot fights."""
    try:
        detector = RobotFightDetector(model_name=model, event_name=event_name)
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise click.ClickException(f"Could not load image: {image_path}")
        
        is_fight, description = detector.analyze_frame(frame)
        
        click.echo(f"\nImage: {image_path.name}")
        click.echo(f"Robot fight detected: {'YES' if is_fight else 'NO'}")
        
        if is_fight:
            try:
                # Try to extract enhanced information
                robot1, robot2, desc, match_info = extract_robot_names(description)
                
                click.echo(f"Robots: {robot1} vs {robot2}")
                
                if "status" in match_info and match_info["status"]:
                    click.echo(f"Match status: {match_info['status']}")
                
                if "sponsors" in match_info and match_info["sponsors"]:
                    click.echo(f"Sponsors: {match_info['sponsors']}")
                
                if "damage" in match_info and match_info["damage"]:
                    click.echo(f"Damage: {match_info['damage']}")
                
                if "timer" in match_info and match_info["timer"]:
                    click.echo(f"Timer: {match_info['timer']}")
                
                if "score" in match_info and match_info["score"]:
                    click.echo(f"Score: {match_info['score']}")
                
                if desc:
                    click.echo(f"Description: {desc}")
            
            except ValueError:
                # Fall back to old format
                click.echo(f"Description: {description}")
        else:
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
        
        # Check for yt-dlp
        try:
            yt_dlp_version = subprocess.run(
                ["yt-dlp", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ).stdout.strip()
            click.echo(f"✓ yt-dlp version: {yt_dlp_version}")
        except FileNotFoundError:
            click.echo("⚠ yt-dlp was not found in system PATH. Livestream processing will not work.", err=True)
        
        # Check environment variables
        click.echo("\nEnvironment configuration:")
        click.echo(f"✓ MODEL_NAME: {DEFAULT_MODEL}")
        click.echo(f"✓ OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
        click.echo(f"✓ DEFAULT_INTERVAL: {DEFAULT_INTERVAL}")
        click.echo(f"✓ STREAM_REFRESH_INTERVAL: {DEFAULT_STREAM_REFRESH_INTERVAL}")
        
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