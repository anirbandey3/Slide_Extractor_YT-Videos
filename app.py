import streamlit as st
import whisper
import torch
import os
import tempfile
import time
import logging
import yt_dlp
import subprocess
import re
import shutil
import zipfile
import cv2
import numpy as np
import imutils
from pathlib import Path
from PIL import Image
from urllib.parse import urlparse
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Video Transcriber & Slide Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .main {
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3rem;
        border-radius: 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF2E2E;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    .slide-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 1rem;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    .timestamp {
        color: #4CAF50;
        font-weight: bold;
    }
    .transcript-container {
        max-height: 400px;
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def is_valid_youtube_url(url):
    """Validate if the given URL is a valid YouTube URL."""
    try:
        parsed_url = urlparse(url)
        return 'youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc
    except:
        return False


def download_youtube_video(url):
    """Download YouTube video using yt-dlp and return both its path and title."""
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'video_{int(time.time())}.mp4')
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_path,
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')
            
        return temp_path, title
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise Exception(f"Failed to download video: {str(e)}")

def extract_audio_from_video(video_path, audio_output_path=None):
    """Extracts audio using moviepy (ffmpeg still used under-the-hood)"""
    if audio_output_path is None:
        audio_output_path = f"{os.path.splitext(video_path)[0]}_audio.wav"
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("No audio track found in the video file.")
        video.audio.write_audiofile(audio_output_path, codec="pcm_s16le")
        video.close()
        return audio_output_path
    except Exception as e:
        raise Exception(f"Error extracting audio: {str(e)}")

def transcribe_audio_with_whisper(audio_path, model_size="medium", use_cuda=True):
    """Transcribes the audio file using OpenAI's Whisper model with timestamps."""
    try:
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and not torch.cuda.is_available():
            st.warning("⚠️ CUDA is not available. Using CPU instead.")
        
        model = whisper.load_model(model_size).to(device)
        
        # Get transcription with timestamps
        result = model.transcribe(audio_path, word_timestamps=True)
        return result
    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")

def seconds_to_timecode(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def find_text_matches(full_transcription, segment_text):
    """Find and return indices where segment_text appears in full_transcription.
    Uses a more robust matching algorithm that handles variations better."""
    # Early return if segment is empty
    if not segment_text.strip():
        return []
        
    # Clean and normalize text to improve matching
    def normalize_text(text):
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    full_clean = normalize_text(full_transcription)
    segment_clean = normalize_text(segment_text)
    
    # Create a mapping from cleaned positions to original positions
    full_mapping = []
    clean_pos = 0
    for i, char in enumerate(full_transcription):
        if clean_pos < len(full_clean) and normalize_text(char) == full_clean[clean_pos]:
            full_mapping.append(i)
            clean_pos += 1
    
    # Find all occurrences with fuzzy matching
    matches = []
    
    # Split into words for more robust matching
    segment_words = segment_clean.split()
    if not segment_words:
        return []
    
    # Find longest sequence of words that appear in the full text
    start_word_idx = 0
    while start_word_idx < len(segment_words):
        best_match = None
        best_match_len = 0
        
        # Try different sequence lengths
        for end_word_idx in range(start_word_idx + 1, len(segment_words) + 1):
            phrase = ' '.join(segment_words[start_word_idx:end_word_idx])
            if len(phrase) < 5:  # Skip very short phrases
                continue
                
            # Look for this phrase in the full text
            pos = full_clean.find(phrase)
            if pos != -1:
                match_len = end_word_idx - start_word_idx
                if match_len > best_match_len:
                    # Map back to original text positions
                    try:
                        orig_start = full_mapping[pos]
                        orig_end = full_mapping[pos + len(phrase) - 1] + 1
                        best_match = (orig_start, orig_end)
                        best_match_len = match_len
                    except IndexError:
                        # This can happen due to mapping differences
                        continue
        
        if best_match:
            matches.append(best_match)
            start_word_idx += best_match_len
        else:
            # If no match found, skip this word
            start_word_idx += 1
    
    # Merge overlapping matches
    if matches:
        matches.sort(key=lambda x: x[0])
        merged = [matches[0]]
        
        for current in matches[1:]:
            previous = merged[-1]
            
            # If current overlaps with previous
            if current[0] <= previous[1]:
                # Merge them
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    return []

def highlight_transcription_ui(full_transcription, slides):
    """Create plain transcription text without any highlighting."""
    # Simply return the full transcription without any highlighting
    # Add line breaks for readability
    plain_html = full_transcription.replace('\n', '<br>')
    return plain_html

# VideoProcessor class - Using the original implementation from the second file
class VideoProcessor:
    """Class for processing videos to extract slides and transcriptions."""
    
    def __init__(self, output_dir="extracted_slides"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.temp_files = []
        
        # Constants for slide detection (from the new approach)
        self.FRAME_RATE = 3                   # No. of frames per second that needs to be processed
        self.WARMUP = self.FRAME_RATE         # Initial number of frames to be skipped
        self.FGBG_HISTORY = self.FRAME_RATE * 15  # No. of frames in background object
        self.VAR_THRESHOLD = 16               # Threshold for background subtraction model
        self.DETECT_SHADOWS = False           # Set False to improve speed
        self.MIN_PERCENT = 0.1                # Min % difference to detect if motion has stopped
        self.MAX_PERCENT = 3                  # Max % difference to detect if frame is still in motion
        
    def get_frames(self, video_path, progress_callback=None):
        """Extract frames from a video while skipping frames as defined in FRAME_RATE."""
        vs = cv2.VideoCapture(video_path)
        if not vs.isOpened():
            raise Exception(f'Unable to open file: {video_path}')

        total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration = vs.get(cv2.CAP_PROP_FRAME_COUNT) / vs.get(cv2.CAP_PROP_FPS)
        frame_time = 0
        frame_count = 0

        while frame_time < total_duration:
            vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)  # Move frame to a timestamp
            frame_time += 1 / self.FRAME_RATE

            (ret, frame) = vs.read()
            if not ret or frame is None:
                break

            frame_count += 1
            
            # Report progress if callback is provided
            if progress_callback:
                progress = min(99, (frame_time / total_duration) * 100)
                progress_callback(progress)
                
            yield frame_count, frame_time, frame

        vs.release()
        
        # Ensure 100% progress at the end
        if progress_callback:
            progress_callback(100)
            
    def get_slide_transcription(self, transcription_data, start_time, end_time):
        """Extract transcription for a specific time range from the full transcription."""
        segments = transcription_data.get("segments", [])
        slide_transcription = ""
        
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # If segment overlaps with slide timeframe
            if (segment_start <= end_time and segment_end >= start_time):
                slide_transcription += segment.get("text", "") + " "
                
        return slide_transcription.strip()
    
    def process_video(self, video_path, transcription_data, similarity_threshold=0.95, 
                      min_slide_duration=2, progress_callback=None):
        """
        Process video to extract slides with their corresponding transcriptions using 
        background subtraction method.
        
        Args:
            video_path: Path to the video file
            transcription_data: Whisper transcription output
            similarity_threshold: Not used in this implementation but kept for API compatibility
            min_slide_duration: Not used in this implementation but kept for API compatibility
            progress_callback: Callback function to report progress
            
        Returns:
            List of slides with image paths, timestamps, and transcriptions
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Initialize the background subtractor
            fgbg = cv2.createBackgroundSubtractorMOG2(
                history=self.FGBG_HISTORY, 
                varThreshold=self.VAR_THRESHOLD, 
                detectShadows=self.DETECT_SHADOWS
            )
            
            slides = []
            captured = False
            screenshot_count = 0
            W, H = None, None
            
            # Process frames
            for frame_count, frame_time, frame in self.get_frames(video_path, progress_callback):
                orig = frame.copy()
                
                # Resize for faster processing
                frame = imutils.resize(frame, width=600)
                
                # Apply background subtraction
                mask = fgbg.apply(frame)
                
                if W is None or H is None:
                    (H, W) = mask.shape[:2]
                
                # Calculate the percentage of difference
                p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100
                
                # If motion has stopped and we haven't captured this frame
                if p_diff < self.MIN_PERCENT and not captured and frame_count > self.WARMUP:
                    captured = True
                    
                    # Save the slide image
                    slide_image_path = os.path.join(self.output_dir, f"slide_{screenshot_count+1}.jpg")
                    cv2.imwrite(slide_image_path, orig)
                    
                    # Get the timestamp in seconds
                    timestamp_seconds = frame_time
                    
                    # For the first slide, the start time is 0
                    if screenshot_count == 0:
                        start_time = 0
                    else:
                        # Otherwise, use the end time of the previous slide
                        start_time = slides[-1]["end_seconds"]
                    
                    # Get transcription for this slide
                    slide_transcription = self.get_slide_transcription(
                        transcription_data, 
                        start_time, 
                        timestamp_seconds
                    )
                    
                    # Add slide to results
                    slides.append({
                        "image_path": slide_image_path,
                        "start_time": seconds_to_timecode(start_time),
                        "end_time": seconds_to_timecode(timestamp_seconds),
                        "start_seconds": start_time,
                        "end_seconds": timestamp_seconds,
                        "duration": round(timestamp_seconds - start_time, 2),
                        "transcription": slide_transcription
                    })
                    
                    screenshot_count += 1
                    
                # If we've captured a frame and significant motion is detected
                elif captured and p_diff >= self.MAX_PERCENT:
                    captured = False
            
            # If we have slides, update the duration of the last slide
            if slides:
                video = cv2.VideoCapture(video_path)
                total_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
                video.release()
                
                last_slide = slides[-1]
                last_slide["end_time"] = seconds_to_timecode(total_duration)
                last_slide["end_seconds"] = total_duration
                last_slide["duration"] = round(total_duration - last_slide["start_seconds"], 2)
                
                # Update the last slide's transcription to include to the end
                last_slide["transcription"] = self.get_slide_transcription(
                    transcription_data, 
                    last_slide["start_seconds"], 
                    total_duration
                )
            
            return slides
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
            
    def cleanup(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {str(e)}")

def create_results_package(output_dir, slides, full_transcription):
    """Create a structured directory with all results and a nice HTML report."""
    # Create directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Copy slide images to the images directory
    for i, slide in enumerate(slides, 1):
        src_path = slide["image_path"]
        dst_path = os.path.join(images_dir, f"slide_{i}.jpg")
        shutil.copy2(src_path, dst_path)
        # Store both original and relative paths
        slide["report_image_path"] = os.path.relpath(dst_path, output_dir)
        # Keep the original image_path for UI display
    
    # Save full transcription
    with open(os.path.join(output_dir, "full_transcription.txt"), "w", encoding="utf-8") as f:
        f.write(full_transcription)
    
    # Save individual slide transcriptions
    for i, slide in enumerate(slides, 1):
        with open(os.path.join(output_dir, f"slide_{i}_transcription.txt"), "w", encoding="utf-8") as f:
            f.write(slide["transcription"])
    
    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Analysis Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2a5298;
                border-bottom: 2px solid #2a5298;
                padding-bottom: 10px;
            }
            h2 {
                color: #1e3c72;
                margin-top: 30px;
            }
            .slide {
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
                border-left: 5px solid #1e3c72;
            }
            .slide-img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .metadata {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }
            .full-transcript {
                background-color: #f0f4f8;
                padding: 20px;
                border-radius: 8px;
                white-space: pre-wrap;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Analysis Results</h1>
            
            <h2>Extracted Slides</h2>
    """
    
    # Add slides to HTML
    for i, slide in enumerate(slides, 1):
        html_content += f"""
            <div class="slide">
                <h3>Slide {i}</h3>
                <div class="metadata">
                    <strong>Timestamp:</strong> {slide["start_time"]} - {slide["end_time"]} (Duration: {slide["duration"]} seconds)
                </div>
                <img src="{slide["report_image_path"]}" alt="Slide {i}" class="slide-img">
                <h4>Slide Transcription:</h4>
                <p>{slide["transcription"]}</p>
            </div>
        """
    
    # Add full transcription to HTML without highlights
    html_content += """
            <h2>Full Transcription</h2>
            <div class="full-transcript">
    """
    
    # Use plain transcription without highlights
    processed_transcription = full_transcription.replace('\n', '<br>')
    
    html_content += processed_transcription + """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

# Main app function
def main():
    st.title("🎬 Video Transcriber & Slide Extractor")
    st.markdown("Extract slides and transcriptions from videos")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("Transcription Settings")
        whisper_model = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium"],
            index=3
        )
        use_cuda = False
        if torch.cuda.is_available():
            use_cuda = st.checkbox("Use GPU (CUDA)", value=True)

        st.subheader("Slide Extraction Settings")
        min_percent = st.slider(
            "Minimum Change Percent",
            min_value=0.05,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        max_percent = st.slider(
            "Maximum Change Percent",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )

    # Removed FFmpeg check block

    try:
        import yt_dlp
    except ImportError:
        st.error("❌ yt-dlp is not installed. Please install it using: pip install yt-dlp")
        return

    tab1, tab2 = st.tabs(["Upload Video", "YouTube URL"])

    with tab1:
        uploaded_file = st.file_uploader("Upload a video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])
        process_upload = uploaded_file is not None
        video_path = None
        if process_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            st.success("✅ Video uploaded successfully!")

    with tab2:
        youtube_url = st.text_input("Enter YouTube URL")
        process_youtube = youtube_url and is_valid_youtube_url(youtube_url)
        if youtube_url and not process_youtube:
            st.error("❌ Please enter a valid YouTube URL")
        elif process_youtube:
            st.success("✅ Valid YouTube URL")

    if (process_upload or process_youtube) and st.button("🚀 Process Video"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            video_title = "Uploaded Video"
            if process_youtube:
                status_text.text("📥 Downloading YouTube video...")
                video_path, video_title = download_youtube_video(youtube_url)
                progress_bar.progress(10)

            timestamp = int(time.time())
            output_dir = os.path.join("processed_content", str(timestamp))
            os.makedirs(output_dir, exist_ok=True)

            status_text.text("🎵 Extracting audio from video...")
            audio_path = extract_audio_from_video(video_path)
            progress_bar.progress(20)

            status_text.text(f"🔍 Transcribing audio using AI ({whisper_model} model)...")
            transcription_result = transcribe_audio_with_whisper(audio_path, whisper_model, use_cuda)
            full_transcription = transcription_result["text"]
            progress_bar.progress(50)

            processor = VideoProcessor(output_dir=os.path.join(output_dir, "slides"))
            status_text.text("🖼️ Extracting slides from video...")

            processor.MIN_PERCENT = min_percent
            processor.MAX_PERCENT = max_percent

            slides = processor.process_video(
                video_path, 
                transcription_result,
                progress_callback=lambda p: progress_bar.progress(50 + int(p * 0.4))
            )

            status_text.text("📦 Creating results package...")
            create_results_package(output_dir, slides, full_transcription)
            progress_bar.progress(100)

            zip_path = f"{output_dir}.zip"
            shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', os.path.dirname(output_dir), os.path.basename(output_dir))
            status_text.text("✅ Processing complete!")

            if not slides:
                st.warning("No slides were detected in the video.")
            else:
                st.success(f"Successfully processed video and extracted {len(slides)} slides!")
                results_tab1, results_tab2 = st.tabs(["Slides", "Full Transcription"])

                with results_tab1:
                    for i, slide in enumerate(slides, 1):
                        with st.container():
                            st.markdown(f"""<div class="slide-card"><h4>Slide {i}</h4></div>""", unsafe_allow_html=True)
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                image = Image.open(slide['image_path'])
                                st.image(image, use_container_width=True)
                            with col2:
                                st.markdown(f"**Time:** {slide['start_time']} - {slide['end_time']}")
                                st.markdown(f"**Duration:** {slide['duration']} seconds")
                                with st.expander("Show Transcription"):
                                    st.write(slide['transcription'])

                with results_tab2:
                    plain_html = full_transcription.replace('\n', '<br>')
                    st.markdown("<h3>Full Transcription</h3>", unsafe_allow_html=True)
                    st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
                    st.markdown(plain_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with open(zip_path, 'rb') as f:
                file_name = f"{video_title.replace(' ', '_')}_analysis_{timestamp}.zip"
                st.download_button(
                    label="📦 Download All Results",
                    data=f,
                    file_name=file_name,
                    mime="application/zip"
                )

            if os.path.exists(audio_path):
                os.remove(audio_path)
            if video_path and os.path.exists(video_path) and process_youtube:
                os.remove(video_path)
            processor.cleanup()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
