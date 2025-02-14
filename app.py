import streamlit as st
import os
import tempfile
import time
import logging
from pathlib import Path
from PIL import Image
import shutil
from video_processor import VideoProcessor
import streamlit as st
import os
import tempfile
import time
import logging
from pathlib import Path
from PIL import Image
import shutil
from pytube import YouTube

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="YouTube Slide Extractor",
    page_icon="🎥",
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
</style>
""", unsafe_allow_html=True)

def download_youtube_video(url):
    """Download YouTube video using pytube"""
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'video.mp4')
        
        # Download using pytube
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        video.download(filename=temp_path)
        
        return temp_path, yt.title
        
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None, None

def main():
    st.title("🎥 YouTube Slide Extractor")
    st.markdown("Extract slides and transcriptions from YouTube videos")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        similarity_threshold = st.slider(
            "Slide Similarity Threshold",
            min_value=0.10,
            max_value=0.99,
            value=0.95,
            help="Higher values mean more sensitive slide detection"
        )
        min_slide_duration = st.slider(
            "Minimum Slide Duration (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum time a slide should be visible"
        )

    # Main content
    url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("🚀 Extract Slides"):
        if not url:
            st.warning("Please enter a YouTube URL")
            return

        try:
            # Download video
            with st.spinner("📥 Downloading video..."):
                video_path, video_title = download_youtube_video(url)
                if not video_path:
                    return

            # Process video
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize processor
                output_dir = os.path.join("extracted_content", str(int(time.time())))
                processor = VideoProcessor(output_dir=output_dir)
                
                # Process video
                status_text.text("🔍 Processing video...")
                slides = processor.process_video(
                    video_path,
                    similarity_threshold=similarity_threshold,
                    min_slide_duration=min_slide_duration,
                    progress_callback=lambda p: progress_bar.progress(int(p))
                )

                if not slides:
                    st.warning("No slides were detected in the video.")
                    return

                # Display results
                st.success(f"✅ Successfully extracted {len(slides)} slides!")
                
                # Display slides
                for i, slide in enumerate(slides, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="slide-card">
                            <h4>Slide {i}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            image = Image.open(slide['image_path'])
                            st.image(image, use_column_width=True)
                        
                        with col2:
                            st.markdown(f"**Time:** {slide['start_time']} - {slide['end_time']}")
                            st.markdown(f"**Duration:** {slide['duration']} seconds")
                            with st.expander("Show Transcription"):
                                st.write(slide['transcription'])
                
                # Create download ZIP
                zip_path = output_dir + ".zip"
                shutil.make_archive(output_dir, 'zip', output_dir)
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📦 Download All Results",
                        data=f,
                        file_name=f"slides_{str(int(time.time()))}.zip",
                        mime="application/zip"
                    )

            finally:
                # Cleanup
                if os.path.exists(video_path):
                    os.remove(video_path)
                processor.cleanup()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
