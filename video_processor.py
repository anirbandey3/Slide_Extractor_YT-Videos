import cv2
import numpy as np
from datetime import timedelta
import os
import whisper
from moviepy.editor import VideoFileClip
import torch
import tempfile
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, output_dir="extracted_content"):
        """Initialize the VideoProcessor with output directory"""
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio_segments")
        self.slides_dir = os.path.join(output_dir, "slides")
        self.create_directories()
        
        # Initialize Whisper model
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
    def create_directories(self):
        """Create necessary directories for output"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.slides_dir, exist_ok=True)

    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))

    def is_slide_frame(self, frame):
        """Detect if a frame is likely to be a slide"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # Edge Detection
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

            h_lines = 0
            v_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    if angle < 5 or angle > 175:
                        h_lines += 1
                    elif 85 < angle < 95:
                        v_lines += 1

            # Text Detection
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
            text_regions = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            text_pixels = np.count_nonzero(text_regions)

            # Contour Analysis
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_rectangles = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > width * 0.2 and h > height * 0.1:
                    large_rectangles += 1

            return (h_lines >= 2 and v_lines >= 2 and 
                    text_pixels > width * height * 0.01 and 
                    large_rectangles >= 1)

        except Exception as e:
            logger.error(f"Error in slide detection: {e}")
            return False

    def extract_audio_segment(self, video_path, start_time, end_time, output_path):
        """Extract audio segment from video"""
        try:
            video = VideoFileClip(video_path)
            audio_segment = video.audio.subclip(start_time, end_time)
            audio_segment.write_audiofile(output_path, fps=16000, verbose=False, logger=None)
            video.close()
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""

    def process_video(self, video_path, similarity_threshold=0.95, min_slide_duration=2, progress_callback=None):
        """Process video to extract slides and transcriptions"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            slides = []
            current_slide = None
            current_slide_start = 0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_count / fps

                if frame_count % int(fps) == 0:  # Check every second
                    if current_slide is None:
                        if self.is_slide_frame(frame):
                            current_slide = frame
                            current_slide_start = current_time
                    else:
                        if not self.is_slide_frame(frame) or \
                           cv2.matchTemplate(cv2.cvtColor(current_slide, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          cv2.TM_CCOEFF_NORMED)[0][0] < similarity_threshold:
                            
                            duration = current_time - current_slide_start
                            if duration >= min_slide_duration:
                                slide_count = len(slides) + 1
                                
                                # Save slide image
                                image_path = os.path.join(self.slides_dir, f"slide_{slide_count}.jpg")
                                cv2.imwrite(image_path, current_slide)

                                # Extract and transcribe audio
                                audio_path = os.path.join(self.audio_dir, f"slide_{slide_count}.wav")
                                self.extract_audio_segment(video_path, current_slide_start, current_time, audio_path)
                                transcription = self.transcribe_audio(audio_path)

                                slides.append({
                                    'index': slide_count,
                                    'start_time': self.format_timestamp(current_slide_start),
                                    'end_time': self.format_timestamp(current_time),
                                    'duration': int(duration),
                                    'image_path': image_path,
                                    'audio_path': audio_path,
                                    'transcription': transcription
                                })

                            current_slide = frame if self.is_slide_frame(frame) else None
                            current_slide_start = current_time

                frame_count += 1
                if progress_callback and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)

            cap.release()
            return slides

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.audio_dir, ignore_errors=True)
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")