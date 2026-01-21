# AccessEd_Ideathon
AccessEd: AI-Powered Assistive Interface
AccessEd is a comprehensive software platform designed to democratize education for students with disabilities. By integrating local Large Language Models (LLMs), computer vision, and speech processing, the system provides a multi-modal interface that adapts to the user's physical needs.

Project Overview
This application runs a local web server that manages input from various assistive hardware and software modules. It processes voice, gestures, and camera inputs locally to ensure low latency and data privacy.

Core Features
1. Context-Aware AI Assistant

->Powered by a locally hosted Mistral-7B-Instruct model.

->Operates fully offline after model download.

->Supports bi-directional voice interaction (Speech-to-Text and Text-to-Speech).

2. ASL Translation Engine

-> Real-time computer vision system that interprets American Sign Language.

-> Converts hand gestures into text or system commands.

3. Hands-Free Cursor Control

-> Allows users to control the mouse pointer using head movements or eye tracking.

-> Designed for individuals with limited upper-limb mobility.

In Development (Locked Features)
The following modules are currently in the R&D phase and are slated for the next release:

AirPen: A spatial writing interface allowing users to write in the air using finger tracking and depth estimation.

BCI (Brain-Computer Interface): A direct neural link allowing users to control the interface via EEG signals, designed for users with severe motor impairments (e.g., Locked-in Syndrome).

Technical Architecture
Backend: Flask (Python)

AI Engine: Hugging Face Transformers, BitsAndBytes (4-bit quantization), PyTorch

Audio Processing: Google Speech Recognition, PyTTSx3

Computer Vision: OpenCV

Frontend: HTML5, Jinja2

Installation Guide
1. Prerequisites
Ensure you have the following installed:

Python 3.10 or higher

NVIDIA GPU with CUDA support (Required for efficient LLM inference)

C++ Build Tools (often required for bitsandbytes)

2. Clone Repository
Bash
git clone https://github.com/your-username/AccessEd.git
cd AccessEd
3. Install Dependencies
Bash
pip install flask torch transformers bitsandbytes scipy pyttsx3 pywin32 SpeechRecognition opencv-python accelerate huggingface_hub
Model Setup (Mistral-7B)
The application requires the Mistral-7B model to be stored locally. Since the model file is large, we use a Python script to download it to the correct directory.

Step 1: Create the download script Create a file named download_model.py in your project folder and paste the following code:

Python
import os
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DOWNLOAD_DIR = os.path.join(os.getcwd(), "models", "mistral-7b")

def download_mistral():
    print(f"Starting download for {MODEL_ID}...")
    print(f"Target directory: {DOWNLOAD_DIR}")
    
    try:
        # This downloads the model snapshots to a local folder
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\nSuccess! Model downloaded to: {path}")
        print("Please update the 'MISTRAL_PATH' variable in app.py with this path.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_mistral()
Step 2: Run the script

Bash
python download_model.py
Step 3: Update app.py Copy the path output by the script and update the configuration line in app.py:

Python
# Inside app.py
MISTRAL_PATH = r"C:\path\to\your\AccessEd\models\mistral-7b"
Usage Instructions
Start the Server Run the main application file:

Bash
python app.py
Access the Dashboard Open your web browser and navigate to: http://localhost:5000

Authentication

Login: admin

Password: 123

Operation

Use the dashboard toggles to activate the ASL Camera or Cursor Control.

Use the microphone button to speak to the AI assistant.

Note: Only one camera-dependent feature (ASL or Cursor) can be active at a time.

Troubleshooting
Common Issues:

CUDA Out of Memory: If the AI fails to load, ensure you have the bitsandbytes library installed and that you are loading the model in 4-bit mode (enabled by default in the code).

Microphone Error: If PyAudio fails to install via pip, you may need to download the specific .whl file for your Python version and install it manually.

Camera Conflicts: If the camera feed is black, ensure no other application (like Zoom or Teams) is currently using the webcam.
