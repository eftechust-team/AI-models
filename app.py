from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import os
import numpy as np

# Try to import OpenCV, but make it optional for Vercel compatibility
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    # Create a dummy cv2 module for basic compatibility
    class DummyCV2:
        COLOR_RGB2GRAY = None
        COLOR_RGB2HSV = None
        COLOR_RGB2LAB = None
        THRESH_BINARY = 0
        THRESH_OTSU = 8
        RETR_EXTERNAL = 0
        RETR_TREE = 3
        CHAIN_APPROX_NONE = 2
        TERM_CRITERIA_EPS = 1
        TERM_CRITERIA_MAX_ITER = 2
        KMEANS_RANDOM_CENTERS = 0
        CC_STAT_AREA = 4
        
        @staticmethod
        def cvtColor(img, code):
            if code == DummyCV2.COLOR_RGB2GRAY:
                # Simple grayscale conversion using numpy
                if len(img.shape) == 3:
                    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                return img
            return img
        
        @staticmethod
        def threshold(img, thresh, maxval, type):
            if type & DummyCV2.THRESH_OTSU:
                # Simple Otsu-like threshold
                threshold_value = np.mean(img)
            else:
                threshold_value = thresh
            binary = (img > threshold_value).astype(np.uint8) * maxval
            return threshold_value, binary
        
        @staticmethod
        def findContours(img, mode, method):
            # Use scikit-image for contour finding (more compatible than OpenCV)
            try:
                from skimage import measure
                # Convert binary image to boolean
                binary_bool = img > 127
                # Find contours
                contours_data = measure.find_contours(binary_bool, 0.5)
                # Convert to OpenCV-like format
                contours = []
                for contour in contours_data:
                    # Convert to integer coordinates
                    contour_int = np.round(contour).astype(np.int32)
                    # Reshape to match OpenCV format: (N, 1, 2)
                    contour_cv = contour_int.reshape(-1, 1, 2)
                    contours.append(contour_cv)
                hierarchy = None  # scikit-image doesn't provide hierarchy
                return contours, hierarchy
            except ImportError:
                # Fallback: return empty
                return [], None
        
        @staticmethod
        def contourArea(contour):
            if len(contour) < 3:
                return 0
            # Shoelace formula
            x = contour[:, 0, 0] if len(contour.shape) > 2 else contour[:, 0]
            y = contour[:, 0, 1] if len(contour.shape) > 2 else contour[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        @staticmethod
        def drawContours(img, contours, idx, color, thickness):
            return img
        
        @staticmethod
        def connectedComponentsWithStats(img, connectivity=8):
            # Basic connected components using scipy
            try:
                from scipy import ndimage
                labeled, num = ndimage.label(img > 127)
                stats = []
                centroids = []
                for i in range(1, num + 1):
                    mask = labeled == i
                    stats.append([np.sum(mask), 0, 0, 0, 0])  # area, left, top, width, height
                    centroids.append([np.mean(np.where(mask)[1]), np.mean(np.where(mask)[0])])
                return num, labeled, np.array(stats), np.array(centroids)
            except ImportError:
                return 1, np.zeros_like(img), np.array([[0, 0, 0, 0, 0]]), np.array([[0, 0]])
        
        @staticmethod
        def kmeans(data, k, best_labels, criteria, attempts, flags):
            # Simple k-means using sklearn or manual
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(data)
                centers = kmeans.cluster_centers_
                return None, labels, centers
            except ImportError:
                # Fallback: random centers
                centers = data[np.random.choice(len(data), k, replace=False)]
                labels = np.argmin(np.linalg.norm(data[:, None] - centers, axis=2), axis=1)
                return None, labels, centers
        
        @staticmethod
        def bitwise_not(img):
            return 255 - img
    
    cv2 = DummyCV2()

from PIL import Image, ImageDraw, ImageFont
import io
import base64
from stl import mesh
import tempfile
import time
import requests
import sys
import json
import zipfile
from collections import defaultdict
from gcode_parser import GCodeParser
from gcode_generator import GCodeGenerator

# Safe print function for Windows console encoding issues
def safe_print(message):
    """Print message safely handling encoding issues on Windows"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII, replacing problematic characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

app = Flask(__name__)
CORS(app)

# Storage configuration
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
MODELS_DB_FILE = os.path.join(STORAGE_DIR, 'models.json')
PREVIEWS_DIR = os.path.join(STORAGE_DIR, 'previews')
STL_DIR = os.path.join(STORAGE_DIR, 'stl_files')
GCODE_DIR = os.path.join(STORAGE_DIR, 'gcode_files')
GCODE_SETTINGS_FILE = os.path.join(STORAGE_DIR, 'gcode_settings.json')

# Create storage directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PREVIEWS_DIR, exist_ok=True)
os.makedirs(STL_DIR, exist_ok=True)
os.makedirs(GCODE_DIR, exist_ok=True)

# G-code parser instance
gcode_parser = GCodeParser()

# Load learned G-code settings
def load_gcode_settings():
    """Load learned G-code settings from file"""
    if os.path.exists(GCODE_SETTINGS_FILE):
        try:
            with open(GCODE_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return gcode_parser.get_default_settings()
    return gcode_parser.get_default_settings()

def save_gcode_settings(settings):
    """Save learned G-code settings to file"""
    with open(GCODE_SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

# Initialize with default or saved settings
learned_gcode_settings = load_gcode_settings()

def load_models_db():
    """Load the models database from JSON file"""
    if os.path.exists(MODELS_DB_FILE):
        try:
            with open(MODELS_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_models_db(db):
    """Save the models database to JSON file"""
    with open(MODELS_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def save_model(name, description, preview_image_base64, stl_path, thickness, solid_infill):
    """Save a model to storage"""
    model_id = str(int(time.time() * 1000))  # Use timestamp as ID
    db = load_models_db()
    
    # Save preview image
    preview_filename = f"{model_id}.png"
    preview_path = os.path.join(PREVIEWS_DIR, preview_filename)
    preview_data = base64.b64decode(preview_image_base64)
    with open(preview_path, 'wb') as f:
        f.write(preview_data)
    
    # Copy STL file
    stl_filename = f"{model_id}.stl"
    stl_dest_path = os.path.join(STL_DIR, stl_filename)
    import shutil
    shutil.copy2(stl_path, stl_dest_path)
    
    # Add to database
    db[model_id] = {
        'id': model_id,
        'name': name,
        'description': description,
        'preview_image': preview_filename,
        'stl_file': stl_filename,
        'thickness': thickness,
        'solid_infill': solid_infill,
        'created_at': time.time()
    }
    
    save_models_db(db)
    return model_id

def delete_model(model_id):
    """Delete a model from storage"""
    db = load_models_db()
    if model_id not in db:
        return False
    
    model = db[model_id]
    
    # Delete preview image
    preview_path = os.path.join(PREVIEWS_DIR, model['preview_image'])
    if os.path.exists(preview_path):
        os.remove(preview_path)
    
    # Delete STL file
    stl_path = os.path.join(STL_DIR, model['stl_file'])
    if os.path.exists(stl_path):
        os.remove(stl_path)
    
    # Remove from database
    del db[model_id]
    save_models_db(db)
    return True

# Configuration - set these as environment variables
# DO NOT hardcode API keys in production - use environment variables for security
ARK_API_KEY = os.environ.get('ARK_API_KEY', '')
# Note: The secret access key is not needed for the OpenAI client approach

# Volcano Engine (Doubao-Seedream) API credentials
# Set these via environment variables: VOLCANO_ACCESS_KEY_ID and VOLCANO_SECRET_ACCESS_KEY
VOLCANO_ACCESS_KEY_ID = os.environ.get('VOLCANO_ACCESS_KEY_ID', '')
VOLCANO_SECRET_ACCESS_KEY = os.environ.get('VOLCANO_SECRET_ACCESS_KEY', '')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')

# Print API key status at startup
if ARK_API_KEY:
    safe_print(f"[INFO] Volcano Engine ARK API key found: {ARK_API_KEY[:15]}...")
else:
    safe_print("[INFO] Volcano Engine ARK API key not found")

if VOLCANO_ACCESS_KEY_ID:
    safe_print(f"[INFO] Volcano Engine Access Key ID found: {VOLCANO_ACCESS_KEY_ID[:15]}...")
if VOLCANO_SECRET_ACCESS_KEY:
    safe_print(f"[INFO] Volcano Engine Secret Access Key found: {VOLCANO_SECRET_ACCESS_KEY[:15]}...")

safe_print("[INFO] Will use Volcano Engine 图像生成大模型 (Doubao-Seedream) as primary service")

# Global progress tracking
progress_data = {}

def update_progress(job_id, progress, message):
    """Update progress for a job"""
    progress_data[job_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }

def get_enhanced_prompt(text_description, variation_index=None):
    """
    Generate an enhanced prompt for AI image generation.
    Each variation gets a COMPLETELY different prompt structure to ensure uniqueness.
    If user provides detailed description, use it more directly while ensuring solid infill.
    
    Args:
        text_description: Description of the shape
        variation_index: If provided (0-3), generates completely different prompt styles
    
    Returns:
        tuple: (enhanced_prompt, clean_description)
    """
    # Check if user wants hollow/outline (explicitly says no solid infill)
    is_hollow_requested = any(word in text_description.lower() for word in [
        'hollow', 'outline', 'wireframe', 'frame only', 'border only', 
        'empty inside', 'no fill', 'transparent inside'
    ])
    
    # Check if description is detailed (has multiple words or specific details)
    word_count = len(text_description.split())
    is_detailed = word_count > 3 or any(char in text_description for char in [',', ';', 'with', 'and', 'wearing', 'holding'])
    
    # Remove "a" or "an" from the beginning for better results
    clean_description = text_description.strip()
    if clean_description.lower().startswith(('a ', 'an ')):
        clean_description = clean_description.split(' ', 1)[1] if ' ' in clean_description else clean_description
    
    # Generate optimized prompt - check if it's an animal for special variations
    is_animal = any(word in clean_description.lower() for word in ['cat', 'dog', 'bird', 'bear', 'lion', 'tiger', 'elephant', 'horse', 'cow', 'pig', 'sheep', 'goat', 'rabbit', 'mouse', 'rat', 'hamster', 'guinea', 'ferret', 'chicken', 'duck', 'goose', 'turkey', 'fish', 'shark', 'whale', 'dolphin', 'seal', 'penguin', 'owl', 'eagle', 'hawk', 'fox', 'wolf', 'deer', 'moose', 'elk', 'bison', 'buffalo', 'zebra', 'giraffe', 'monkey', 'ape', 'gorilla', 'chimpanzee', 'panda', 'koala', 'kangaroo', 'camel', 'llama', 'alpaca', 'animal', 'pet', 'creature'])
    
    # Build solid infill requirement (unless user explicitly wants hollow)
    if is_hollow_requested:
        infill_requirement = "outline only, wireframe style, no fill"
    else:
        infill_requirement = (
            "COMPLETELY BLACK, SOLID BLACK, 100% BLACK FILL, NO WHITE INSIDE, "
            "solid filled BLACK shape, completely filled inside, NO hollow areas, NO empty spaces inside, NO white holes, "
            "completely opaque BLACK, solid BLACK infill, BLACK shape filled with BLACK"
        )
    
    if variation_index is not None:
        if is_animal:
            # Animal-specific variations
            if variation_index == 0:
                # Variation 1: Head/face view
                if is_detailed:
                    # Use user's detailed description directly, but add head view and solid infill
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}, but showing HEAD ONLY. "
                        f"FRONT VIEW of the head, face forward, showing head and face, NOT full body, HEAD PORTRAIT ONLY. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The HEAD shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    # Simple description - use standard prompt
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description} HEAD ONLY. "
                        f"The HEAD shape must be {infill_requirement}. "
                        f"FRONT VIEW of the head, face forward, showing head and face, NOT full body, HEAD PORTRAIT ONLY. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            elif variation_index == 1:
                # Variation 2: Full body, standing/walking
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}, showing FULL BODY in STANDING or WALKING pose. "
                        f"SIDE PROFILE VIEW, all four legs visible, full body from head to tail, NOT sitting, NOT lying down. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The BODY shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description} FULL BODY. "
                        f"The BODY shape must be {infill_requirement}. "
                        f"STANDING or WALKING pose, all four legs visible, full body from head to tail. "
                        f"SIDE PROFILE VIEW, showing complete body silhouette, NOT sitting, NOT lying down, STANDING or WALKING. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            elif variation_index == 2:
                # Variation 3: Sitting pose
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}, showing SITTING pose. "
                        f"SITTING pose, body upright, legs bent, sitting position. "
                        f"FRONT or SIDE VIEW showing sitting posture, NOT standing, NOT walking, SITTING ONLY. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The SITTING shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description} SITTING. "
                        f"The SITTING shape must be {infill_requirement}. "
                        f"SITTING pose, body upright, legs bent, sitting position. "
                        f"FRONT or SIDE VIEW showing sitting posture, NOT standing, NOT walking, SITTING ONLY. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            else:  # variation_index == 3
                # Variation 4: Standing/walking full body (alternative view)
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}, showing FULL BODY in STANDING or WALKING pose. "
                        f"FRONT VIEW or THREE-QUARTER VIEW, showing complete body, NOT sitting, NOT lying down, STANDING or WALKING. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The BODY shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description} FULL BODY. "
                        f"The BODY shape must be {infill_requirement}. "
                        f"STANDING or WALKING pose, full body view. "
                        f"FRONT VIEW or THREE-QUARTER VIEW, showing complete body, NOT sitting, NOT lying down, STANDING or WALKING. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
        else:
            # Non-animal variations
            if variation_index == 0:
                # Variation 1: Front-facing, symmetrical, logo-style
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}. "
                        f"FRONT VIEW, facing directly forward, symmetrical. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description}. "
                        f"FRONT VIEW, facing directly forward, symmetrical. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            elif variation_index == 1:
                # Variation 2: Side profile
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}. "
                        f"SIDE PROFILE VIEW ONLY, facing left, showing side of the object, NOT front view, NOT top view, SIDE VIEW. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description}. "
                        f"SIDE PROFILE VIEW ONLY, facing left, showing side of the object, NOT front view, NOT top view, SIDE VIEW. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            elif variation_index == 2:
                # Variation 3: Top-down
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}. "
                        f"TOP-DOWN VIEW ONLY, bird's eye view, looking down from above, NOT front view, NOT side view, TOP VIEW. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description}. "
                        f"TOP-DOWN VIEW ONLY, bird's eye view, looking down from above, NOT front view, NOT side view, TOP VIEW. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
            else:  # variation_index == 3
                # Variation 4: Angled view
                if is_detailed:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {text_description.strip()}. "
                        f"THREE-QUARTER VIEW ONLY, 45 degree angle, diagonal perspective, NOT front view, NOT side view, NOT top view, ANGLED VIEW. "
                        f"Generate strictly according to the description: {text_description.strip()}. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
                else:
                    enhanced_prompt = (
                        f"Create a BLACK SILHOUETTE of {clean_description}. "
                        f"THREE-QUARTER VIEW ONLY, 45 degree angle, diagonal perspective, NOT front view, NOT side view, NOT top view, ANGLED VIEW. "
                        f"The shape must be {infill_requirement}. "
                        f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                        f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                        f"NO details, NO decorations, NO patterns, "
                        f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                        f"2D flat design, high contrast, BLACK shape on WHITE background ONLY"
                    )
    else:
        # Default prompt - use user's description directly if detailed, otherwise use clean description
        if is_detailed:
            enhanced_prompt = (
                f"Create a BLACK SILHOUETTE of {text_description.strip()}. "
                f"Generate strictly according to the description: {text_description.strip()}. "
                f"The shape must be {infill_requirement}. "
                f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                f"2D flat design, high contrast, front view, centered, "
                f"BLACK shape on WHITE background ONLY"
            )
        else:
            enhanced_prompt = (
                f"Create a BLACK SILHOUETTE of {clean_description}. "
                f"The shape must be {infill_requirement}. "
                f"The background must be PURE WHITE, COMPLETELY WHITE, NO BLACK IN BACKGROUND. "
                f"NO lines, NO outlines, NO borders, NO edges, just solid BLACK shape, "
                f"NO details, NO decorations, NO patterns, "
                f"NO texture, NO shading, NO gradients, NO shadows, NO highlights, "
                f"2D flat design, high contrast, front view, centered, "
                f"BLACK shape on WHITE background ONLY"
            )
    
    return enhanced_prompt, clean_description

def generate_2d_image_ai(text_description, width=512, height=512):
    """
    Generate a 2D image using AI (multiple services with fallbacks).
    Falls back to keyword-based generation if all APIs fail.
    """
    safe_print(f"Attempting AI generation for: '{text_description}'")
    safe_print("This will generate an actual image of the object, not just text!")
    
    # Enhanced prompt for simple solid silhouette - single object
    # Remove "a" or "an" from the beginning for better results
    clean_description = text_description.strip()
    if clean_description.lower().startswith(('a ', 'an ')):
        clean_description = clean_description.split(' ', 1)[1] if ' ' in clean_description else clean_description
    
    # Get enhanced prompt (no variation for single image generation)
    enhanced_prompt, clean_description = get_enhanced_prompt(text_description)
    
    # Try OpenAI DALL-E FIRST (user has VPN and API key - best quality)
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
        safe_print(f"Trying OpenAI DALL-E first (API key found: {OPENAI_API_KEY[:15]}...)")
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            safe_print(f"Generating image with prompt: '{enhanced_prompt[:80]}...'")
            
            # Try DALL-E 3 first (best quality, highest resolution)
            try:
                safe_print("Attempting DALL-E 3...")
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                safe_print("[SUCCESS] DALL-E 3 worked!")
            except Exception as e3:
                error_msg = str(e3)
                safe_print(f"DALL-E 3 failed: {error_msg[:150]}")
                # Try DALL-E 2 as fallback
                safe_print("Trying DALL-E 2 as fallback...")
                try:
                    response = client.images.generate(
                        model="dall-e-2",
                        prompt=enhanced_prompt,
                        size="512x512",
                        n=1,
                    )
                    safe_print("[SUCCESS] DALL-E 2 worked!")
                except Exception as e2:
                    safe_print(f"DALL-E 2 also failed: {str(e2)[:150]}")
                    raise e2
            
            # Process the image response
            if hasattr(response.data[0], 'url'):
                image_url = response.data[0].url
                safe_print(f"Downloading image from OpenAI...")
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    img = Image.open(io.BytesIO(img_response.content))
                else:
                    safe_print(f"Failed to download image from URL: {img_response.status_code}")
                    raise Exception("Failed to download generated image")
            elif hasattr(response.data[0], 'b64_json'):
                # Base64 encoded image
                import base64
                img_data = base64.b64decode(response.data[0].b64_json)
                img = Image.open(io.BytesIO(img_data))
            else:
                raise Exception("Unknown response format")
            
            # Process image: resize and convert to silhouette
            img = img.convert('RGB')
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to black and white silhouette for better extrusion
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Use Otsu's method for automatic threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert if needed (we want black shape on white background)
            if np.mean(binary) < 127:  # If mostly black, invert
                binary = cv2.bitwise_not(binary)
            img = Image.fromarray(binary).convert('RGB')
            
            safe_print("[SUCCESS] Successfully generated image using OpenAI DALL-E")
            safe_print("Image converted to black silhouette for 3D extrusion")
            return img, 'dalle'
        except ImportError:
            safe_print("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            safe_print(f"[ERROR] OpenAI DALL-E failed: {error_type}")
            safe_print(f"[ERROR] Full error message: {error_msg}")
            
            # Print full traceback for debugging
            import traceback
            safe_print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            # Check for specific error types
            if "PermissionDenied" in error_type or "403" in error_msg or "unsupported_country" in error_msg.lower():
                safe_print("[WARNING] OpenAI may not be available - check VPN connection")
                safe_print("[INFO] Make sure VPN is connected and try again")
            elif "Connection" in error_type or "Connect" in error_msg:
                safe_print("[WARNING] Connection error - check VPN connection")
            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                safe_print("[WARNING] Rate limit exceeded - wait a moment and try again")
            
            safe_print("[INFO] Falling back to other AI services...")
    
    # Try Replicate API (works globally, but needs credit)
    if REPLICATE_API_TOKEN:
        safe_print("Trying Replicate API (Stable Diffusion)...")
        try:
            import replicate
            # Set the API token in environment
            import os as os_module
            os_module.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
            safe_print(f"Replicate API token found: {REPLICATE_API_TOKEN[:10]}...")
            
            safe_print(f"Generating image with prompt: '{text_description}'...")
            
            output = replicate.run(
                "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                input={
                    "prompt": enhanced_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            )
            
            if output:
                # Replicate returns a list of FileOutput objects or URLs
                # Handle FileOutput objects (newer Replicate API)
                if isinstance(output, list) and len(output) > 0:
                    first_output = output[0]
                    # Check if it's a FileOutput object
                    if hasattr(first_output, 'url'):
                        image_url = first_output.url
                    elif hasattr(first_output, '__str__'):
                        image_url = str(first_output)
                    else:
                        image_url = first_output
                else:
                    image_url = output
                
                # Convert to string if needed
                if not isinstance(image_url, str):
                    image_url = str(image_url)
                
                if image_url and image_url.startswith('http'):
                    safe_print(f"Downloading image from Replicate...")
                    img_response = requests.get(image_url, timeout=60)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        img = img.convert('RGB')
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        # Convert to silhouette
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        if np.mean(binary) < 127:
                            binary = cv2.bitwise_not(binary)
                        img = Image.fromarray(binary).convert('RGB')
                        safe_print("[SUCCESS] Successfully generated image using Replicate")
                        return img, 'replicate'
                    else:
                        safe_print(f"[ERROR] Failed to download image: {img_response.status_code}")
                else:
                    safe_print(f"[ERROR] Invalid URL format from Replicate: {image_url[:100]}")
            else:
                safe_print("[ERROR] Replicate returned no output")
        except ImportError:
            safe_print("[INFO] Replicate not installed - run: pip install replicate")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            safe_print(f"[ERROR] Replicate failed: {error_type}")
            safe_print(f"[ERROR] Full error message: {error_msg}")
            safe_print(f"[ERROR] Error type: {error_type}")
            
            # Print full traceback for debugging
            import traceback
            safe_print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            if "credit" in error_msg.lower() or "402" in error_msg or "Insufficient" in error_msg:
                safe_print("[WARNING] Replicate account needs credit!")
                safe_print("[INFO] Add credit at: https://replicate.com/account/billing")
            elif "401" in error_msg or "Unauthorized" in error_msg or "Invalid" in error_msg:
                safe_print("[WARNING] Replicate API token may be invalid!")
                safe_print("[INFO] Check your token at: https://replicate.com/account/api-tokens")
            else:
                safe_print(f"[INFO] Replicate error details: {error_msg[:500]}")
            
            safe_print("[INFO] Continuing to try other services...")
    else:
        safe_print("[INFO] Replicate API token not set, skipping...")
    
    # Try free alternative: Stability AI Free API (if available) or Hugging Face
    safe_print("Trying free AI image generation services...")
    
    # Use the same enhanced prompt for simple silhouette
    # (enhanced_prompt is already defined above with emphasis on simplicity)
    
    # Try using a free text-to-image service that doesn't require API keys
    # Option: Use Hugging Face Spaces Gradio API (public, no auth needed)
    safe_print("Trying Hugging Face Spaces (public API, no auth needed)...")
    try:
        # Use a public Stable Diffusion Space
        space_url = "https://huggingface.co/spaces/stabilityai/stable-diffusion"
        # Try using the Gradio API endpoint
        gradio_api_url = "https://stabilityai-stable-diffusion.hf.space/api/predict"
        
        payload = {
            "data": [enhanced_prompt, 512, 512, 20, 7.5, 1]
        }
        
        response = requests.post(gradio_api_url, json=payload, timeout=120)
        safe_print(f"Gradio API response: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    # Gradio returns base64 image
                    import base64
                    image_data = result['data'][0]
                    if isinstance(image_data, str):
                        if image_data.startswith('data:image'):
                            header, encoded = image_data.split(',', 1)
                        else:
                            encoded = image_data
                        img_data = base64.b64decode(encoded)
                        img = Image.open(io.BytesIO(img_data))
                        img = img.convert('RGB')
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        # Convert to silhouette
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        if np.mean(binary) < 127:
                            binary = cv2.bitwise_not(binary)
                        img = Image.fromarray(binary).convert('RGB')
                        safe_print("[SUCCESS] Generated image using Hugging Face Spaces!")
                        return img, 'huggingface-spaces'
            except Exception as e:
                safe_print(f"[INFO] Gradio API format issue: {str(e)[:100]}")
    except Exception as e:
        safe_print(f"[INFO] Hugging Face Spaces failed: {str(e)[:100]}")
    
    # Fallback to Hugging Face Inference API (requires API key or may fail)
    safe_print("Trying Hugging Face Inference API...")
    
    # Try multiple Hugging Face endpoints and formats
    hf_models = [
        {
            "url": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
            "name": "stable-diffusion-v1-5",
            "use_router": False
        },
        {
            "url": "https://router.huggingface.co/models/runwayml/stable-diffusion-v1-5",
            "name": "stable-diffusion-v1-5-router",
            "use_router": True
        },
        {
            "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1", 
            "name": "stable-diffusion-2-1",
            "use_router": False
        }
    ]
    
    for model_info in hf_models:
        API_URL = model_info["url"]
        model_name = model_info["name"]
        use_router = model_info["use_router"]
        try:
            headers = {"Content-Type": "application/json"}
            if HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
            
            safe_print(f"Attempting Hugging Face model: {model_name} ({'router' if use_router else 'direct'})")
            
            # Different payload formats for router vs direct
            if use_router:
                # Router endpoint - simpler format
                payload = {"inputs": enhanced_prompt}
            else:
                # Direct endpoint - full format
                payload = {
                    "inputs": enhanced_prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5
                    }
                }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            
            safe_print(f"Response status: {response.status_code}")
            
            if response.status_code == 410:
                # Endpoint deprecated, try router endpoint
                safe_print(f"[INFO] Endpoint deprecated (410), trying router endpoint...")
                router_url = f"https://router.huggingface.co/models/{model_name}"
                try:
                    router_response = requests.post(
                        router_url, 
                        headers=headers, 
                        json={"inputs": enhanced_prompt},
                        timeout=120
                    )
                    safe_print(f"Router response status: {router_response.status_code}")
                    if router_response.status_code == 200:
                        response = router_response
                        safe_print("[SUCCESS] Router endpoint worked!")
                    else:
                        error_text = router_response.text[:200] if hasattr(router_response, 'text') else ""
                        safe_print(f"[ERROR] Router failed: {error_text}")
                        continue
                except Exception as router_e:
                    safe_print(f"[ERROR] Router error: {str(router_e)[:100]}")
                    continue
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        error_msg = error_data.get('error', str(error_data))[:200]
                    else:
                        error_msg = str(error_data)[:200]
                except:
                    error_msg = response.text[:200] if hasattr(response, 'text') and response.text else "Unknown error"
                safe_print(f"[ERROR] API error: {error_msg}")
            
            if response.status_code == 200:
                image_bytes = response.content
                img = Image.open(io.BytesIO(image_bytes))
                img = img.convert('RGB')
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Convert to black and white silhouette for better extrusion
                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                # Use Otsu's method for automatic threshold
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Invert if needed (we want black shape on white background)
                if np.mean(binary) < 127:  # If mostly black, invert
                    binary = cv2.bitwise_not(binary)
                img = Image.fromarray(binary).convert('RGB')
                
                model_name = API_URL.split('/')[-1]
                safe_print(f"[SUCCESS] Successfully generated image using Hugging Face ({model_name})")
                return img, 'huggingface'
            elif response.status_code == 503:
                # Model is loading, wait and retry
                safe_print(f"[INFO] Model {model_name} is loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                if response.status_code == 200:
                    image_bytes = response.content
                    img = Image.open(io.BytesIO(image_bytes))
                    img = img.convert('RGB')
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    img = Image.fromarray(binary).convert('RGB')
                    model_name = API_URL.split('/')[-1]
                    safe_print(f"[SUCCESS] Successfully generated image using Hugging Face ({model_name}) after retry")
                    return img, 'huggingface'
                else:
                    safe_print(f"[ERROR] Retry failed with status {response.status_code}, trying next model...")
                    continue
            elif response.status_code == 429:
                safe_print("[ERROR] Rate limit exceeded, trying next model...")
                time.sleep(2)
                continue
            else:
                error_msg = response.text[:500] if hasattr(response, 'text') and response.text else "Unknown error"
                safe_print(f"[ERROR] API returned status {response.status_code}")
                safe_print(f"[ERROR] Response headers: {dict(response.headers)}")
                safe_print(f"[ERROR] Error message: {error_msg}")
                safe_print(f"[INFO] Trying next model...")
                continue
        except requests.exceptions.Timeout:
            safe_print(f"[ERROR] Timeout with model {API_URL.split('/')[-1]}, trying next...")
            continue
        except Exception as e:
            error_msg = str(e)
            safe_print(f"[ERROR] Error with model {API_URL.split('/')[-1]}: {error_msg}")
            safe_print(f"[INFO] Trying next model...")
            continue
    
    safe_print("="*70)
    safe_print("="*70)
    safe_print("All AI models failed!")
    safe_print("="*70)
    safe_print("DIAGNOSIS:")
    safe_print("")
    
    # Check what was tried
    if OPENAI_API_KEY:
        safe_print("  [X] OpenAI DALL-E - FAILED (check VPN connection and API key)")
    else:
        safe_print("  [ ] OpenAI DALL-E - NOT CONFIGURED")
    
    if REPLICATE_API_TOKEN:
        safe_print("  [X] Replicate - FAILED (check console above for specific error)")
        safe_print("      If you have credit, check:")
        safe_print("      - Token is correct: https://replicate.com/account/api-tokens")
        safe_print("      - Account has credit: https://replicate.com/account/billing")
    else:
        safe_print("  [ ] Replicate - NOT CONFIGURED")
    
    if HUGGINGFACE_API_KEY:
        safe_print("  [X] Hugging Face - FAILED")
    else:
        safe_print("  [ ] Hugging Face - NOT CONFIGURED (optional)")
    
    safe_print("")
    safe_print("SOLUTIONS:")
    safe_print("")
    safe_print("1. If OpenAI failed: Check VPN is connected and API key is valid")
    safe_print("2. If Replicate failed: Check the error above - may need to:")
    safe_print("   - Verify token at: https://replicate.com/account/api-tokens")
    safe_print("   - Check credit balance: https://replicate.com/account/billing")
    safe_print("   - Wait a few minutes after adding credit")
    safe_print("3. Get free Hugging Face key: https://huggingface.co/settings/tokens")
    safe_print("="*70)
    
    # Don't silently fall back - raise an error so user knows AI failed
    # Only fall back for known basic shapes
    text_lower = text_description.lower()
    basic_shapes = ['circle', 'square', 'rectangle', 'triangle', 'star', 'round']
    is_basic_shape = any(shape in text_lower for shape in basic_shapes)
    
    if is_basic_shape:
        safe_print("[WARNING] AI generation failed, but using keyword-based generation for basic shape.")
        img = generate_2d_image_keyword(text_description, width, height)
        return img, 'keyword'
    else:
        # For non-basic shapes, we need AI - raise an error with helpful message
        error_msg = (
            f"AI image generation failed for '{text_description}'.\n\n"
            "To generate images of objects, you need either:\n"
            "1. Replicate account with credit (add $5-10 at replicate.com/account/billing)\n"
            "2. OR Hugging Face API key (free at huggingface.co/settings/tokens)\n\n"
            "See console output above for detailed instructions."
        )
        safe_print(f"[ERROR] {error_msg}")
        raise Exception(error_msg)

def generate_2d_image_keyword(text_description, width=512, height=512):
    """
    Generate a 2D image from text description using keyword matching.
    This is a fallback when AI generation is not available.
    """
    # Create a white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Simple keyword-based shape generation
    text_lower = text_description.lower()
    
    if 'circle' in text_lower or 'round' in text_lower:
        # Draw a circle
        margin = 50
        draw.ellipse([margin, margin, width-margin, height-margin], 
                    fill='black', outline='black', width=2)
    elif 'square' in text_lower or 'rectangle' in text_lower:
        # Draw a square/rectangle
        margin = 50
        draw.rectangle([margin, margin, width-margin, height-margin], 
                      fill='black', outline='black', width=2)
    elif 'triangle' in text_lower:
        # Draw a triangle
        points = [
            (width//2, 50),
            (50, height-50),
            (width-50, height-50)
        ]
        draw.polygon(points, fill='black', outline='black')
    elif 'star' in text_lower:
        # Draw a star
        center_x, center_y = width // 2, height // 2
        outer_radius = min(width, height) // 3
        inner_radius = outer_radius // 2
        points = []
        for i in range(10):
            angle = i * np.pi / 5
            if i % 2 == 0:
                x = center_x + outer_radius * np.cos(angle - np.pi/2)
                y = center_y + outer_radius * np.sin(angle - np.pi/2)
            else:
                x = center_x + inner_radius * np.cos(angle - np.pi/2)
                y = center_y + inner_radius * np.sin(angle - np.pi/2)
            points.append((x, y))
        draw.polygon(points, fill='black', outline='black')
    else:
        # Default: draw text or a simple shape
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text_description[:20], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text_description[:20], fill='black', font=font)
    
    return img

def generate_2d_image(text_description, width=512, height=512):
    """
    Main function to generate 2D image - tries AI first, falls back to keywords.
    Returns the image (method info is logged but not returned for compatibility)
    """
    result = generate_2d_image_ai(text_description, width, height)
    if isinstance(result, tuple):
        return result[0]  # Return just the image
    return result

def image_to_contour(img, solid_infill=True):
    """
    Convert PIL image to OpenCV contour(s) - preserves shape details
    Simple approach: preserve black areas, white areas are holes/background
    
    If solid_infill=True: returns only the outer black outline (fills holes)
    If solid_infill=False: returns outer black outline + white holes as a list [outer, hole1, hole2, ...]
    """
    # Convert PIL to numpy array (should be RGB with black shape on white background)
    img_array = np.array(img.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold: black shape (0) on white background (255)
    # We want to find black pixels as the shape
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # After THRESH_BINARY: black pixels (0) stay black (0), white pixels (255) stay white (255)
    # So: black shape = 0, white background/holes = 255
    
    # Check if image is mostly black or mostly white to determine orientation
    mean_value = np.mean(binary)
    is_inverted = mean_value < 127  # If mostly black, image might be inverted
    
    if is_inverted:
        # Image is white shape on black background - invert it
        binary = 255 - binary
        # Now: black shape (0) on white background (255)
    
    # findContours finds white areas, so we invert to find black areas
    black_mask = 255 - binary  # Black (0) becomes white (255) for contour detection
    
    if solid_infill:
        # Only find external contours of black areas (this fills holes automatically)
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        # Find all contours with hierarchy to get both outer shape and holes
        contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Filter out very small contours (likely noise) - keep only significant shapes
    min_area = 100
    significant_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not significant_contours:
        return None
    
    if solid_infill:
        # Return only the largest external contour (fills holes)
        return max(significant_contours, key=cv2.contourArea)
    else:
        # Keep holes mode: return outer black contour + white holes inside
        # Find the outer contour (largest, no parent)
        outer_contour = max(significant_contours, key=cv2.contourArea)
        
        # Find white areas (holes) inside the black shape
        # Create a mask of the outer black area (filled)
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], -1, 255, -1)  # Fill the outer contour
        
        # Find white areas inside the mask (these are holes)
        # White areas in original binary = holes
        # We need to find white (255) pixels inside the black shape
        white_inside = cv2.bitwise_and(binary, mask)
        
        # Find contours of white holes inside
        hole_contours, _ = cv2.findContours(white_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Filter significant holes
        significant_holes = [c for c in hole_contours if cv2.contourArea(c) >= min_area]
        
        # Verify holes are actually inside the outer contour and are white
        verified_holes = []
        for hole in significant_holes:
            # Get center point of hole
            M = cv2.moments(hole)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Check if inside outer contour and is white in original binary
                if cv2.pointPolygonTest(outer_contour, (cx, cy), False) >= 0:
                    if 0 <= cy < binary.shape[0] and 0 <= cx < binary.shape[1]:
                        if binary[cy, cx] == 255:  # White = hole
                            verified_holes.append(hole)
        
        # Also check hierarchy for nested contours that might be holes
        if hierarchy is not None and len(hierarchy) > 0:
            # Find the index of outer_contour by matching area and position
            outer_idx = None
            outer_area = cv2.contourArea(outer_contour)
            for idx, c in enumerate(contours):
                if abs(cv2.contourArea(c) - outer_area) < 1.0:  # Same area (within tolerance)
                    # Check if centers are close
                    M1 = cv2.moments(outer_contour)
                    M2 = cv2.moments(c)
                    if M1["m00"] != 0 and M2["m00"] != 0:
                        cx1, cy1 = int(M1["m10"]/M1["m00"]), int(M1["m01"]/M1["m00"])
                        cx2, cy2 = int(M2["m10"]/M2["m00"]), int(M2["m01"]/M2["m00"])
                        if abs(cx1 - cx2) < 5 and abs(cy1 - cy2) < 5:
                            outer_idx = idx
                            break
            
            # Check all contours that are children of the outer contour
            if outer_idx is not None:
                for idx, c in enumerate(contours):
                    if idx != outer_idx and cv2.contourArea(c) >= min_area:
                        parent_idx = hierarchy[0][idx][3]
                        # If this contour is a child of the outer contour
                        if parent_idx == outer_idx:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if 0 <= cy < binary.shape[0] and 0 <= cx < binary.shape[1]:
                                    if binary[cy, cx] == 255:  # White = hole
                                        # Check if not already in verified_holes
                                        is_duplicate = False
                                        for existing_hole in verified_holes:
                                            if abs(cv2.contourArea(existing_hole) - cv2.contourArea(c)) < 1.0:
                                                is_duplicate = True
                                                break
                                        if not is_duplicate:
                                            verified_holes.append(c)
        
        if verified_holes:
            return [outer_contour] + verified_holes
        else:
            return outer_contour

def detect_image_components(img, max_components=10):
    """
    Detect different components by separating black outline from white areas inside
    Returns a list of images: one for the black outline, and one for each white area inside
    
    Args:
        img: PIL Image (should be black shape on white background)
        max_components: Maximum number of components to detect
    
    Returns:
        List of (component_image, component_name) tuples
    """
    # Convert to numpy array
    img_array = np.array(img.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold: black shape (0) on white background (255)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check if image is inverted (white shape on black background)
    mean_value = np.mean(binary)
    is_inverted = mean_value < 127
    
    if is_inverted:
        # Invert to get black shape on white background
        binary = 255 - binary
    
    components = []
    verified_holes = []  # Initialize to avoid scope issues
    
    # Step 1: Find the outer black contour (main shape)
    # Invert binary to find black areas (findContours finds white areas)
    black_mask = 255 - binary
    
    # Find external contours (outer shape only)
    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        # No contours found, return original as single component
        component_final = Image.fromarray(binary).convert('RGB')
        components.append((component_final, "Layer_1"))
        return components
    
    # Get the largest contour (main shape)
    outer_contour = max(contours, key=cv2.contourArea)
    
    # Create mask for the outer black shape (filled)
    outer_mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(outer_mask, [outer_contour], -1, 255, -1)  # Fill the contour
    
    # Step 2: Find white areas inside the black shape (holes/negative spaces)
    # Create a mask that is white inside the black shape, black outside
    # This is: white pixels that are inside the outer_mask
    white_inside_mask = np.zeros(binary.shape, dtype=np.uint8)
    # Set white_inside_mask to white where: binary is white (255) AND inside outer_mask
    white_inside_mask[(binary == 255) & (outer_mask == 255)] = 255
    
    # Find ALL white contours inside (use RETR_TREE to get nested holes too)
    white_contours, white_hierarchy = cv2.findContours(
        white_inside_mask, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE
    )
    
    if white_hierarchy is not None and len(white_contours) > 0:
        # Filter and verify white areas (holes)
        min_area = 100  # Minimum area for a hole to be considered
        
        for i, hole_contour in enumerate(white_contours):
            area = cv2.contourArea(hole_contour)
            if area < min_area:
                continue
            
            # Get center point of hole
            M = cv2.moments(hole_contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Verify: must be inside outer contour and actually white in original
            if 0 <= cy < binary.shape[0] and 0 <= cx < binary.shape[1]:
                if cv2.pointPolygonTest(outer_contour, (cx, cy), False) >= 0:
                    if binary[cy, cx] == 255:  # White = hole
                        # Check hierarchy: only add if it's a top-level hole (not nested inside another hole)
                        # For RETR_TREE, parent is -1 for external contours
                        if white_hierarchy[0][i][3] == -1:  # No parent = top-level hole
                            verified_holes.append(hole_contour)
        
        # Sort by area (largest first) and limit
        verified_holes.sort(key=cv2.contourArea, reverse=True)
        verified_holes = verified_holes[:max_components - 1]  # -1 because we already have outline
        
        safe_print(f"Found {len(verified_holes)} white areas (holes) inside the black outline")
        
        # Step 3: Create a separate layer for each white area (hole)
        for idx, hole in enumerate(verified_holes):
            # Create mask for this specific hole
            hole_mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(hole_mask, [hole], -1, 255, -1)  # Fill the hole
            
            # Create image with only this hole (black shape on white background)
            # This represents the hole as a positive shape that can be printed
            hole_img = np.ones_like(img_array) * 255  # White background
            hole_img[hole_mask == 255] = [0, 0, 0]  # Black shape (the hole as a positive shape)
            hole_final = Image.fromarray(hole_img).convert('RGB')
            components.append((hole_final, f"WhiteArea_{idx + 1}"))
    
    # Step 4: Create layer for the black outline (main shape)
    # The outline should be the black shape MINUS the white areas inside
    # This creates a "frame" or "outline" layer
    outline_img = np.ones_like(img_array) * 255  # White background
    
    # Draw the filled black shape
    outline_img[outer_mask == 255] = [0, 0, 0]  # Black shape
    
    # Remove white areas from the outline (make them white again)
    # This creates an outline with holes
    if len(verified_holes) > 0:
        # We have white areas, so remove them from the outline
        for hole in verified_holes:
            hole_mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(hole_mask, [hole], -1, 255, -1)
            outline_img[hole_mask == 255] = [255, 255, 255]  # Make holes white again
    
    outline_final = Image.fromarray(outline_img).convert('RGB')
    # Insert outline at the beginning
    components.insert(0, (outline_final, "BlackOutline"))
    
    # If no holes found, at least return the outline
    if len(components) == 1:
        safe_print("No white areas (holes) detected inside the black outline")
    
    safe_print(f"Generated {len(components)} layers: 1 black outline + {len(components)-1} white areas")
    
    return components

def extrude_2d_to_3d(contour, thickness, resolution=100, arc_top=False):
    """
    Extrude a 2D contour (or list of contours) to a 3D mesh
    If contour is a list, combines all contours into one mesh (preserving holes)
    Uses the user-specified thickness parameter
    
    Args:
        contour: 2D contour or list of contours
        thickness: Height of the extrusion
        resolution: Resolution parameter (unused but kept for compatibility)
        arc_top: If True, creates a curved/dome top instead of flat
    """
    # Use the thickness specified by the user
    actual_thickness = float(thickness)
    
    # Handle multiple contours (for keep holes mode)
    if isinstance(contour, list):
        if not contour:
            raise ValueError("Empty contour list")
        if len(contour) == 1:
            contour = contour[0]
        else:
            # Combine multiple contours into one mesh
            meshes = []
            for c in contour:
                if c is not None and len(c) >= 3:
                    try:
                        mesh_single = extrude_2d_to_3d(c, actual_thickness, resolution, arc_top=arc_top)
                        meshes.append(mesh_single)
                    except:
                        continue
            
            if not meshes:
                raise ValueError("No valid contours to extrude")
            
            # Combine all meshes into one
            if len(meshes) == 1:
                return meshes[0]
            
            # Merge all meshes
            combined_vectors = np.concatenate([m.vectors for m in meshes], axis=0)
            combined_mesh = mesh.Mesh(np.zeros(len(combined_vectors), dtype=mesh.Mesh.dtype))
            combined_mesh.vectors = combined_vectors
            return combined_mesh
    
    if contour is None or len(contour) < 3:
        raise ValueError("Invalid contour")
    
    # Store original contour for resampling if needed
    original_contour = contour.copy()
    
    # For smooth shapes like circles, use more points
    # Check if the contour is roughly circular by comparing area to perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # If it's roughly circular (circularity > 0.7), use finer approximation
    if circularity > 0.7:
        # Use a much smaller epsilon for circles to maintain smoothness
        epsilon = 0.005 * cv2.arcLength(contour, True)
        # Ensure minimum number of points for smooth circle
        min_points = 64
    else:
        # For complex shapes (cat, steak, etc.), use VERY small epsilon
        # This preserves the actual shape instead of simplifying with straight lines
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Much smaller - preserves curves
        min_points = 32  # More points to capture shape details
    
    # Apply minimal approximation to preserve shape
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # If we have too few points, resample from original contour to preserve shape
    if len(contour) < min_points:
        # Resample the original contour to get more points and preserve shape
        original_points = original_contour.reshape(-1, 2)
        if len(original_points) > 1:
            # Interpolate more points along the contour to preserve curves
            cumulative_distances = np.zeros(len(original_points))
            for i in range(1, len(original_points)):
                dist = np.linalg.norm(original_points[i] - original_points[i-1])
                cumulative_distances[i] = cumulative_distances[i-1] + dist
            
            # Close the loop
            if len(original_points) > 2:
                dist_to_start = np.linalg.norm(original_points[0] - original_points[-1])
                total_perimeter = cumulative_distances[-1] + dist_to_start
            else:
                total_perimeter = cumulative_distances[-1]
            
            if total_perimeter > 0:
                # Sample evenly spaced points
                target_distances = np.linspace(0, total_perimeter, min_points, endpoint=False)
                resampled_points = []
                for dist in target_distances:
                    # Handle wrap-around for closed contours
                    if dist > cumulative_distances[-1]:
                        # Interpolate between last and first point
                        t = (dist - cumulative_distances[-1]) / (dist_to_start + 1e-10)
                        point = original_points[-1] + t * (original_points[0] - original_points[-1])
                    else:
                        idx = np.searchsorted(cumulative_distances, dist)
                        if idx >= len(original_points):
                            idx = len(original_points) - 1
                        elif idx > 0:
                            # Interpolate between points
                            t = (dist - cumulative_distances[idx-1]) / (cumulative_distances[idx] - cumulative_distances[idx-1] + 1e-10)
                            point = original_points[idx-1] + t * (original_points[idx] - original_points[idx-1])
                        else:
                            point = original_points[idx]
                    resampled_points.append(point)
                
                contour = np.array(resampled_points, dtype=np.int32).reshape(-1, 1, 2)
    
    # Get contour points
    points_2d = contour.reshape(-1, 2)
    
    # Normalize points to center at origin
    center = points_2d.mean(axis=0)
    points_2d = points_2d - center
    
    # Calculate maximum distance from center for arc top
    if arc_top:
        distances_from_center = np.linalg.norm(points_2d, axis=1)
        max_distance = np.max(distances_from_center) if len(distances_from_center) > 0 else 1.0
    else:
        max_distance = 1.0  # Not used if arc_top is False
    
    # Create vertices for mesh
    num_points = len(points_2d)
    vertices = []
    
    # Bottom face vertices (z = 0)
    for point in points_2d:
        vertices.append([point[0], point[1], 0])
    
    # Top face vertices - flat or arc
    if arc_top:
        # Create arc top: height decreases from center to edges
        # Use cosine function for smooth curve: z = thickness * cos(distance/max_distance * π/2)
        # This creates a dome where center is at full thickness and edges approach 0
        for i, point in enumerate(points_2d):
            distance = distances_from_center[i]
            # Normalize distance (0 at center, 1 at edge)
            normalized_dist = distance / max_distance if max_distance > 0 else 0
            # Use cosine curve: starts at 1 (center) and goes to ~0 (edge)
            # Actually, let's use a quadratic curve for a smoother dome
            # z = thickness * (1 - normalized_dist^2) creates a smooth arch
            # Or use cosine for even smoother: z = thickness * cos(normalized_dist * π/2)
            arc_height = actual_thickness * np.cos(normalized_dist * np.pi / 2)
            # Ensure minimum height at edges (at least 5% of thickness)
            arc_height = max(arc_height, actual_thickness * 0.05)
            vertices.append([point[0], point[1], arc_height])
    else:
        # Flat top
        for point in points_2d:
            vertices.append([point[0], point[1], actual_thickness])
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    
    # Bottom face (triangulate) - use proper triangulation for any number of points
    if num_points == 3:
        # Triangle: single face
        faces.append([0, 1, 2])
    else:
        # Polygon: fan triangulation from first vertex
        for i in range(1, num_points - 1):
            faces.append([0, i, i + 1])
    
    # Top face (triangulate) - same logic
    if num_points == 3:
        # Triangle: single face
        faces.append([num_points, num_points + 1, num_points + 2])
    else:
        # Polygon: fan triangulation from first vertex
        for i in range(1, num_points - 1):
            faces.append([num_points, num_points + i, num_points + i + 1])
    
    # Side faces (quads as two triangles) - minimal for flat appearance
    for i in range(num_points):
        next_i = (i + 1) % num_points
        # First triangle of quad
        faces.append([i, next_i, num_points + i])
        # Second triangle of quad
        faces.append([next_i, num_points + next_i, num_points + i])
    
    faces = np.array(faces)
    
    # Create mesh
    shape_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            shape_mesh.vectors[i][j] = vertices[face[j]]
    
    # Scale mesh to fit within 100mm x 100mm x 100mm
    # Calculate bounding box
    min_bounds = shape_mesh.vectors.min(axis=(0, 1))
    max_bounds = shape_mesh.vectors.max(axis=(0, 1))
    dimensions = max_bounds - min_bounds
    
    # Find the maximum dimension (excluding Z for flat shapes)
    max_dimension = max(dimensions[0], dimensions[1])
    
    # Scale factor to fit within 100mm
    if max_dimension > 100.0:
        scale_factor = 100.0 / max_dimension
        # Apply scaling (preserve flat thickness)
        shape_mesh.vectors[:, :, 0:2] = shape_mesh.vectors[:, :, 0:2] * scale_factor
        safe_print(f"Scaled mesh by {scale_factor:.3f} to fit within 100mm (max dimension was {max_dimension:.2f}mm)")
    
    return shape_mesh

def process_shape(text_description, thickness, job_id, arc_top=False):
    """Process the shape generation pipeline"""
    try:
        update_progress(job_id, 5, "Initializing AI image generation...")
        time.sleep(0.3)
        
        update_progress(job_id, 10, "Generating 2D image using AI...")
        
        # Step 1: Generate 2D image using AI
        img_2d = generate_2d_image(text_description)
        
        update_progress(job_id, 40, "Processing 2D contour...")
        time.sleep(0.3)
        
        # Step 2: Extract contour
        contour = image_to_contour(img_2d, solid_infill=True)  # Default to solid for backward compatibility
        if contour is None:
            raise ValueError("Could not extract contour from image")
        
        update_progress(job_id, 60, "Extruding to 3D...")
        time.sleep(0.5)
        
        # Step 3: Extrude to 3D
        mesh_3d = extrude_2d_to_3d(contour, thickness, arc_top=arc_top)
        
        update_progress(job_id, 80, "Generating STL file...")
        time.sleep(0.5)
        
        # Step 4: Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        mesh_3d.save(temp_file.name)
        
        # Save 2D image preview
        img_buffer = io.BytesIO()
        img_2d.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        update_progress(job_id, 100, "")
        
        return {
            'success': True,
            'stl_path': temp_file.name,
            'preview_image': img_base64
        }
    except Exception as e:
        update_progress(job_id, 0, f"Error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

def generate_multiple_images_volcano(text_description, num_images=4, width=512, height=512, job_id=None):
    """Generate multiple images using Volcano Engine API with sequential_image_generation"""
    # Get enhanced prompt (no variation needed, the API will generate variations)
    enhanced_prompt, clean_description = get_enhanced_prompt(text_description, variation_index=None)
    
    if not ARK_API_KEY or len(ARK_API_KEY) < 10:
        raise Exception("Volcano Engine ARK API key not configured")
    
    try:
        from openai import OpenAI
        
        # Initialize Ark client
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=ARK_API_KEY,
        )
        
        safe_print(f"Generating {num_images} images with Volcano Engine 图像生成大模型")
        safe_print(f"Prompt: '{enhanced_prompt[:150]}...'")
        
        if job_id:
            update_progress(job_id, 15, "")
        
        # Generate images with sequential_image_generation
        imagesResponse = client.images.generate(
            model="doubao-seedream-4-0-250828",
            prompt=enhanced_prompt,
            size="2K",
            response_format="b64_json",
            stream=True,
            extra_body={
                "watermark": False,
                "sequential_image_generation": "auto",
                "sequential_image_generation_options": {
                    "max_images": num_images
                },
            },
        )
        
        images = []
        event_count = 0
        for event in imagesResponse:
            event_count += 1
            if event is None:
                continue
            
            # Log event type for debugging
            event_type = getattr(event, 'type', 'unknown')
            safe_print(f"Received event {event_count}: type={event_type}")
            
            # Check for image data in various event types
            image_data_b64 = None
            
            if hasattr(event, 'b64_json') and event.b64_json is not None:
                image_data_b64 = event.b64_json
            elif hasattr(event, 'data') and event.data:
                # Check if data contains b64_json
                if isinstance(event.data, list) and len(event.data) > 0:
                    first_data = event.data[0]
                    if hasattr(first_data, 'b64_json'):
                        image_data_b64 = first_data.b64_json
                    elif isinstance(first_data, dict) and 'b64_json' in first_data:
                        image_data_b64 = first_data['b64_json']
                elif isinstance(event.data, dict) and 'b64_json' in event.data:
                    image_data_b64 = event.data['b64_json']
            
            if image_data_b64:
                try:
                    image_data = base64.b64decode(image_data_b64)
                    img = Image.open(io.BytesIO(image_data))
                    img = img.convert('RGB')
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # Convert to silhouette
                    img_array = np.array(img)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if np.mean(binary) < 127:
                        binary = cv2.bitwise_not(binary)
                    img = Image.fromarray(binary).convert('RGB')
                    
                    images.append(img)
                    safe_print(f"Generated image {len(images)}/{num_images}")
                    if job_id:
                        progress = 15 + int((len(images) / num_images) * 35)
                        update_progress(job_id, progress, "")
                except Exception as e:
                    safe_print(f"Error processing image: {e}")
            
            # Check for completion
            if event_type == "image_generation.completed":
                if hasattr(event, 'usage') and event.usage is not None:
                    safe_print(f"Image generation completed. Usage: {event.usage}")
        
        safe_print(f"Total events received: {event_count}, Images collected: {len(images)}")
        
        # If we didn't get enough images from streaming, try making separate calls
        if len(images) < num_images:
            safe_print(f"Only got {len(images)} images from stream, making additional calls...")
            remaining = num_images - len(images)
            
            for i in range(remaining):
                try:
                    safe_print(f"Making additional API call {i+1}/{remaining}...")
                    response = client.images.generate(
                        model="doubao-seedream-4-0-250828",
                        prompt=enhanced_prompt,
                        size="2K",
                        response_format="b64_json",
                        n=1,
                        extra_body={
                            "watermark": False,
                        },
                    )
                    
                    # Extract image from response
                    if hasattr(response, 'data') and len(response.data) > 0:
                        img_data = response.data[0]
                        if hasattr(img_data, 'b64_json'):
                            image_data = base64.b64decode(img_data.b64_json)
                        elif isinstance(img_data, dict) and 'b64_json' in img_data:
                            image_data = base64.b64decode(img_data['b64_json'])
                        else:
                            continue
                        
                        img = Image.open(io.BytesIO(image_data))
                        img = img.convert('RGB')
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        
                        # Convert to silhouette
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        if np.mean(binary) < 127:
                            binary = cv2.bitwise_not(binary)
                        img = Image.fromarray(binary).convert('RGB')
                        
                        images.append(img)
                        safe_print(f"Generated additional image {len(images)}/{num_images}")
                        if job_id:
                            progress = 15 + int((len(images) / num_images) * 35)
                            update_progress(job_id, progress, "")
                except Exception as e:
                    safe_print(f"Error in additional API call: {e}")
        
        if len(images) == 0:
            raise Exception("No images were generated")
        
        safe_print(f"[SUCCESS] Generated {len(images)} images using Volcano Engine")
        return images[:num_images]  # Return only the requested number
        
    except Exception as e:
        error_msg = str(e)
        safe_print(f"[ERROR] Volcano Engine API failed: {error_msg[:300]}")
        raise Exception(f"Volcano Engine API failed: {error_msg[:200]}. Please check your API key.")

def generate_single_image_with_variation(text_description, width=512, height=512, variation_index=0, job_id=None):
    """Generate a single image with a specific variation (legacy function, kept for compatibility)"""
    # Get enhanced prompt with variation
    enhanced_prompt, clean_description = get_enhanced_prompt(text_description, variation_index)
    
    # Try Volcano Engine 图像生成大模型 FIRST
    if ARK_API_KEY and len(ARK_API_KEY) > 10:
        try:
            safe_print(f"Generating image with Volcano Engine 图像生成大模型")
            safe_print(f"Prompt: '{enhanced_prompt[:150]}...'")
            
            # Volcano Engine API implementation
            # Try multiple common endpoint patterns and authentication methods
            import json
            import hmac
            import hashlib
            from datetime import datetime
            
            # Decode the secret access key (it appears to be base64 encoded)
            try:
                secret_key_bytes = base64.b64decode(VOLCANO_SECRET_ACCESS_KEY)
            except:
                secret_key_bytes = VOLCANO_SECRET_ACCESS_KEY.encode('utf-8')
            
            # Common Volcano Engine endpoints to try
            # Based on Volcano Engine documentation, the correct endpoint is typically:
            api_endpoints = [
                "https://ark.cn-beijing.volces.com/api/v3/images/generations",  # Image generation endpoint
                "https://ark.cn-beijing.volces.com/api/v3/image/generation",
                "https://visual.volcengineapi.com/api/v1/image/generation",
                "https://visual.volcengine.com/api/v1/image/generation",
                "https://open.volcengine.com/api/v1/image/generation",
                "https://api.volcengine.com/visual/v1/image/generation"
            ]
            
            # Request payload - try common parameter names
            payload_variants = [
                {
                    "prompt": enhanced_prompt,
                    "width": width,
                    "height": height,
                    "num_images": 1
                },
                {
                    "text": enhanced_prompt,
                    "width": width,
                    "height": height,
                    "count": 1
                },
                {
                    "input": enhanced_prompt,
                    "width": width,
                    "height": height
                }
            ]
            
            # Try different authentication methods
            # Method 1: Simple header authentication
            headers_simple = {
                "Content-Type": "application/json",
                "X-Access-Key-Id": VOLCANO_ACCESS_KEY_ID,
                "X-Secret-Access-Key": VOLCANO_SECRET_ACCESS_KEY
            }
            
            # Method 2: Authorization header with access key
            headers_auth = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VOLCANO_ACCESS_KEY_ID}"
            }
            
            # Method 3: Query parameter authentication (for some APIs)
            # Will be added in the request URL if needed
            
            response = None
            last_error = None
            
            # Try each endpoint, payload, and authentication method combination
            auth_methods = [
                ("Simple headers", headers_simple),
                ("Authorization Bearer", headers_auth),
            ]
            
            for endpoint in api_endpoints:
                for auth_name, headers in auth_methods:
                    for payload in payload_variants:
                        try:
                            safe_print(f"Trying endpoint: {endpoint} with {auth_name}")
                            
                            # Try with query parameters for ark endpoint
                            if "ark.cn-beijing.volces.com" in endpoint:
                                endpoint_with_key = f"{endpoint}?access_key_id={VOLCANO_ACCESS_KEY_ID}"
                                response = requests.post(endpoint_with_key, json=payload, headers=headers, timeout=60)
                            else:
                                response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                            
                            safe_print(f"Response status: {response.status_code}")
                            safe_print(f"Response content-type: {response.headers.get('Content-Type', 'Unknown')}")
                            
                            # Check if response is HTML (error page) - skip it
                            if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                                safe_print(f"Got HTML response (likely wrong endpoint), trying next...")
                                last_error = f"Status {response.status_code}: HTML response (wrong endpoint)"
                                continue
                            
                            safe_print(f"Response text (first 300 chars): {response.text[:300]}")
                            
                            if response.status_code == 200:
                                # Check if response is actually JSON
                                if response.text and len(response.text.strip()) > 0:
                                    try:
                                        test_json = response.json()
                                        safe_print(f"Successfully got JSON response from {endpoint}")
                                        break  # Successfully got JSON response
                                    except:
                                        safe_print(f"Response is not valid JSON, trying next...")
                                        last_error = f"Status {response.status_code}: Non-JSON response"
                                else:
                                    last_error = f"Status {response.status_code}: Empty response"
                            elif response.status_code != 404:  # 404 means endpoint doesn't exist, try next
                                last_error = f"Status {response.status_code}: {response.text[:200]}"
                        except Exception as e:
                            last_error = str(e)
                            safe_print(f"Exception: {str(e)}")
                            continue
                    if response and response.status_code == 200 and response.text:
                        try:
                            response.json()  # Verify it's JSON
                            break
                        except:
                            continue
                if response and response.status_code == 200 and response.text:
                    try:
                        response.json()  # Verify it's JSON
                        break
                    except:
                        continue
            
            # If simple auth failed, try signature-based authentication
            if not response or response.status_code != 200:
                safe_print("Simple auth failed, trying signature-based authentication...")
                timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                date_stamp = datetime.utcnow().strftime('%Y%m%d')
                
                payload = payload_variants[0]  # Use first payload variant
                payload_str = json.dumps(payload, separators=(',', ':'))
                
                # Try signature method
                method = "POST"
                content_type = "application/json"
                canonical_uri = "/api/v1/image/generation"
                canonical_headers = f"content-type:{content_type}\nhost:visual.volcengineapi.com\nx-date:{timestamp}\n"
                signed_headers = "content-type;host;x-date"
                payload_hash = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
                
                canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
                algorithm = "HMAC-SHA256"
                credential_scope = f"{date_stamp}/visual/hmac-sha256"
                string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
                
                k_date = hmac.new(secret_key_bytes, date_stamp.encode('utf-8'), hashlib.sha256).digest()
                k_region = hmac.new(k_date, "visual".encode('utf-8'), hashlib.sha256).digest()
                k_service = hmac.new(k_region, "hmac-sha256".encode('utf-8'), hashlib.sha256).digest()
                k_signing = hmac.new(k_service, "request".encode('utf-8'), hashlib.sha256).digest()
                signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
                
                authorization = f"{algorithm} Credential={VOLCANO_ACCESS_KEY_ID}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
                
                headers_sig = {
                    "Content-Type": content_type,
                    "X-Date": timestamp,
                    "Authorization": authorization
                }
                
                for endpoint in api_endpoints:
                    try:
                        response = requests.post(endpoint, json=payload, headers=headers_sig, timeout=60)
                        if response.status_code == 200:
                            break
                    except:
                        continue
            
            if not response or response.status_code != 200:
                error_text = response.text[:500] if response else str(last_error)
                safe_print(f"All authentication methods failed. Last error: {error_text}")
                if response:
                    safe_print(f"Response status: {response.status_code}")
                    safe_print(f"Response headers: {dict(response.headers)}")
                    safe_print(f"Full response text: {response.text[:1000]}")
                raise Exception(f"Volcano Engine API failed: {error_text}")
            
            # Check if response is actually JSON before parsing
            if not response.text or len(response.text.strip()) == 0:
                safe_print("Volcano Engine API returned empty response")
                safe_print(f"Response status: {response.status_code}")
                safe_print(f"Response headers: {dict(response.headers)}")
                raise Exception("Volcano Engine API returned empty response")
            
            try:
                result = response.json()
            except ValueError as e:
                # Response is not JSON - show what we actually got
                safe_print(f"Response is not JSON. Status: {response.status_code}")
                safe_print(f"Response content type: {response.headers.get('Content-Type', 'Unknown')}")
                safe_print(f"Response text (first 1000 chars): {response.text[:1000]}")
                raise Exception(f"Volcano Engine API returned non-JSON response. Status: {response.status_code}, Content: {response.text[:200]}")
            
            # Extract image from response (adjust based on actual API response format)
            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                if 'image' in image_data:
                    # If image is base64 encoded
                    if isinstance(image_data['image'], str):
                        img_data = base64.b64decode(image_data['image'])
                        img = Image.open(io.BytesIO(img_data))
                    else:
                        raise Exception("Unknown image format in response")
                elif 'url' in image_data:
                    # If image URL is provided
                    image_url = image_data['url']
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                    else:
                        raise Exception("Failed to download generated image")
                else:
                    raise Exception("Unknown response format from Volcano Engine API")
            elif 'result' in result and 'images' in result['result']:
                # Alternative response format
                images = result['result']['images']
                if len(images) > 0:
                    if 'image' in images[0]:
                        img_data = base64.b64decode(images[0]['image'])
                        img = Image.open(io.BytesIO(img_data))
                    elif 'url' in images[0]:
                        image_url = images[0]['url']
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                        else:
                            raise Exception("Failed to download generated image")
                    else:
                        raise Exception("Unknown image format")
                else:
                    raise Exception("No images in response")
            else:
                # Try to find image data in any format
                safe_print(f"Response structure: {list(result.keys())}")
                raise Exception(f"Unexpected response format: {str(result)[:200]}")
            
            # Process image: resize and convert to silhouette
            img = img.convert('RGB')
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu thresholding for consistent processing
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (black shape on white background)
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)
            
            img = Image.fromarray(binary).convert('RGB')
            safe_print(f"[SUCCESS] Generated image using Volcano Engine 图像生成大模型")
            if job_id:
                update_progress(job_id, 50, "")
            return img
        except Exception as e:
            error_msg = str(e)
            safe_print(f"[ERROR] Volcano Engine API failed: {error_msg[:300]}")
            # Try OpenAI as fallback if available
            if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
                safe_print("[INFO] Trying OpenAI DALL-E as fallback...")
                try:
                    import openai
                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=enhanced_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                    )
                    if hasattr(response.data[0], 'url'):
                        image_url = response.data[0].url
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                        else:
                            raise Exception("Failed to download generated image")
                    else:
                        raise Exception("Unknown response format")
                    
                    img = img.convert('RGB')
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if np.mean(binary) < 127:
                        binary = cv2.bitwise_not(binary)
                    img = Image.fromarray(binary).convert('RGB')
                    safe_print(f"[SUCCESS] Generated image using OpenAI DALL-E (fallback)")
                    if job_id:
                        update_progress(job_id, 50, "")
                    return img
                except Exception as e2:
                    safe_print(f"[ERROR] OpenAI fallback also failed: {str(e2)[:200]}")
            
            raise Exception(f"Volcano Engine API failed: {error_msg[:200]}. Please check your credentials and API endpoint.")
    
    # Try OpenAI DALL-E as fallback
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            safe_print(f"Generating image with OpenAI DALL-E")
            safe_print(f"Prompt: '{enhanced_prompt[:150]}...'")
            
            # The prompt is already completely different for each variation
            # No need to add more elements - use the prompt as-is
            final_prompt = enhanced_prompt
            
            # Try DALL-E 3 first
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=final_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
            except Exception as e3:
                safe_print(f"DALL-E 3 failed: {str(e3)[:100]}, trying DALL-E 2...")
                # Use the same unique prompt for DALL-E 2
                # (final_prompt is already set above)
                
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=final_prompt,
                    size="512x512",
                    n=1,
                )
            
            # Process the image response
            if hasattr(response.data[0], 'url'):
                image_url = response.data[0].url
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    img = Image.open(io.BytesIO(img_response.content))
                else:
                    raise Exception("Failed to download generated image")
            elif hasattr(response.data[0], 'b64_json'):
                import base64
                img_data = base64.b64decode(response.data[0].b64_json)
                img = Image.open(io.BytesIO(img_data))
            else:
                raise Exception("Unknown response format")
            
            # Process image: resize and convert to silhouette
            # Keep processing consistent to preserve the differences from AI generation
            img = img.convert('RGB')
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu thresholding for consistent processing
            # The differences should come from the AI generation, not from processing
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (black shape on white background)
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)
            
            img = Image.fromarray(binary).convert('RGB')
            safe_print(f"[SUCCESS] Generated image using OpenAI DALL-E")
            if job_id:
                update_progress(job_id, 50, "")
            return img
        except Exception as e:
            error_msg = str(e)
            safe_print(f"[ERROR] OpenAI DALL-E failed: {error_msg[:200]}")
            # Since user wants to use OpenAI only, don't fall back to other services
            raise Exception(f"OpenAI DALL-E failed: {error_msg[:200]}. Please check your VPN connection and API key.")
    
    # Try Replicate as fallback ONLY if OpenAI is not available or explicitly failed
    # Since user has OpenAI with VPN, skip other services
    if False and REPLICATE_API_TOKEN:  # Disabled - using OpenAI only
        try:
            import replicate
            import os as os_module
            os_module.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
            
            # Use EXTREMELY different seeds
            # The prompt is already completely different for each variation, no need to add more
            seeds = [12345, 67890, 11111, 99999]
            final_replicate_prompt = enhanced_prompt
            
            if job_id:
                update_progress(job_id, 18, "Replicate is generating your image...")
            
            output = replicate.run(
                "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                input={
                    "prompt": final_replicate_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "seed": seeds[variation_index]  # Very different seeds for each variation
                }
            )
            
            if output:
                if isinstance(output, list) and len(output) > 0:
                    first_output = output[0]
                    if hasattr(first_output, 'url'):
                        image_url = first_output.url
                    else:
                        image_url = str(first_output)
                else:
                    image_url = output
                
                if not isinstance(image_url, str):
                    image_url = str(image_url)
                
                if image_url and image_url.startswith('http'):
                    img_response = requests.get(image_url, timeout=60)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        img = img.convert('RGB')
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        # Use Otsu thresholding for consistent processing
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # Invert if needed (black shape on white background)
                        if np.mean(binary) < 127:
                            binary = cv2.bitwise_not(binary)
                        img = Image.fromarray(binary).convert('RGB')
                        safe_print(f"[SUCCESS] Generated image using Replicate")
                        if job_id:
                            update_progress(job_id, 25, "Image generated successfully!")
                        return img
        except Exception as e:
            safe_print(f"[ERROR] Replicate failed for variation {variation_index + 1}: {str(e)[:100]}")
    
    # If OpenAI failed, raise exception with helpful message
    raise Exception(f"OpenAI DALL-E failed to generate image. Please check your VPN connection and API key. Error details were logged above.")

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate multiple images for user selection - streaming version"""
    data = request.json
    text_description = data.get('text', '')
    num_images = int(data.get('num_images', 4))
    
    if not text_description:
        return jsonify({'error': 'Text description is required'}), 400
    
    # Generate job ID
    job_id = str(time.time())
    
    def generate_images_stream():
        """Generator function to stream images as they're generated"""
        try:
            safe_print(f"Generating {num_images} images for: '{text_description}'")
            
            # Initialize progress
            update_progress(job_id, 5, "")
            yield f"data: {json.dumps({'type': 'progress', 'progress': 5, 'job_id': job_id})}\n\n"
            
            # Get enhanced prompt
            enhanced_prompt, clean_description = get_enhanced_prompt(text_description, variation_index=None)
            
            if not ARK_API_KEY or len(ARK_API_KEY) < 10:
                raise Exception("Volcano Engine ARK API key not configured")
            
            from openai import OpenAI
            
            # Initialize Ark client
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=ARK_API_KEY,
            )
            
            safe_print(f"Generating {num_images} images with Volcano Engine 图像生成大模型")
            safe_print(f"Prompt: '{enhanced_prompt[:150]}...'")
            
            update_progress(job_id, 10, "")
            yield f"data: {json.dumps({'type': 'progress', 'progress': 10, 'job_id': job_id})}\n\n"
            
            # Generate images with different prompts for variation
            # Use different angles/views for each image
            view_descriptions = [
                "FRONT VIEW, facing directly forward, symmetrical, centered",
                "SIDE PROFILE VIEW, facing left or right, showing side of the object",
                "TOP-DOWN VIEW, bird's eye view, looking down from above",
                "THREE-QUARTER VIEW, 45 degree angle, diagonal perspective, isometric view"
            ]
            
            images_generated = 0
            
            # Generate each image with a different view/angle
            for image_index in range(num_images):
                # Get prompt with specific view for this image
                view_prompt = get_enhanced_prompt(text_description, variation_index=image_index)[0]
                
                safe_print(f"Generating image {image_index + 1}/{num_images} with view: {view_descriptions[image_index % len(view_descriptions)]}")
                
                # Generate single image with specific view
                try:
                    response = client.images.generate(
                        model="doubao-seedream-4-0-250828",
                        prompt=view_prompt,
                        size="2K",
                        response_format="b64_json",
                        n=1,
                        extra_body={
                            "watermark": False,
                        },
                    )
                    
                    # Extract image from response
                    image_data_b64 = None
                    if hasattr(response, 'data') and len(response.data) > 0:
                        img_data = response.data[0]
                        if hasattr(img_data, 'b64_json'):
                            image_data_b64 = img_data.b64_json
                        elif isinstance(img_data, dict) and 'b64_json' in img_data:
                            image_data_b64 = img_data['b64_json']
                    
                    if not image_data_b64:
                        continue  # Skip if no image data
                    
                    # Process the image
                    try:
                        image_data = base64.b64decode(image_data_b64)
                        img = Image.open(io.BytesIO(image_data))
                        img = img.convert('RGB')
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        # Convert to silhouette with solid black infill and white background
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        
                        # Use Otsu thresholding to separate shape from background
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Find connected components to identify the main shape
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                        
                        if num_labels > 1:
                            # Find the largest component (the main shape)
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            shape_mask = (labels == largest_label).astype(np.uint8) * 255
                            
                            # Check brightness: shape should be darker than background
                            shape_pixels = gray[shape_mask == 255]
                            background_mask = (labels != largest_label).astype(np.uint8) * 255
                            background_pixels = gray[background_mask == 255]
                            
                            shape_mean = np.mean(shape_pixels) if len(shape_pixels) > 0 else 128
                            background_mean = np.mean(background_pixels) if len(background_pixels) > 0 else 128
                            
                            # We want: shape is BLACK (dark), background is WHITE (bright)
                            # If shape is brighter than background, invert
                            if shape_mean > background_mean:
                                # Shape is lighter - need to invert
                                binary = cv2.bitwise_not(binary)
                                # Re-extract after inversion
                                num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                                if num_labels2 > 1:
                                    largest_label2 = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
                                    shape_mask2 = (labels2 == largest_label2).astype(np.uint8) * 255
                                    
                                    # Create final binary: shape = 0 (black), background = 255 (white)
                                    binary = np.zeros_like(gray, dtype=np.uint8)
                                    binary[shape_mask2 == 255] = 0  # Shape is black (0)
                                    binary[shape_mask2 == 0] = 255  # Background is white (255)
                            else:
                                # Shape is darker - good, but ensure correct format
                                binary = np.zeros_like(gray, dtype=np.uint8)
                                binary[shape_mask == 255] = 0  # Shape is black (0)
                                binary[shape_mask == 0] = 255  # Background is white (255)
                        else:
                            # Fallback: check if we need to invert
                            gray_mean = np.mean(gray)
                            binary_mean = np.mean(binary)
                            
                            # If original is mostly dark and binary is mostly dark, shape is dark (good)
                            # But we need white background, so if binary is mostly 0, invert
                            if binary_mean < 50 and gray_mean < 100:
                                # Both dark - shape is dark, but need white background
                                binary = cv2.bitwise_not(binary)
                        
                        # Fill holes and ensure solid infill
                        kernel = np.ones((3, 3), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
                        
                        # Final verification: ensure we have black shape (0) on white background (255)
                        final_mean = np.mean(binary)
                        if final_mean < 30:
                            # Almost everything is black - likely wrong, invert
                            binary = cv2.bitwise_not(binary)
                        
                        # Create RGB image: black (0,0,0) for shape, white (255,255,255) for background
                        img_rgb = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                        img_rgb[binary == 0] = [0, 0, 0]  # Black shape
                        img_rgb[binary == 255] = [255, 255, 255]  # White background
                        
                        img = Image.fromarray(img_rgb)
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        images_generated += 1
                        progress = 10 + int((images_generated / num_images) * 80)
                        update_progress(job_id, progress, "")
                        
                        # Send image immediately
                        yield f"data: {json.dumps({'type': 'image', 'image': img_base64, 'index': images_generated - 1, 'progress': progress, 'job_id': job_id})}\n\n"
                        safe_print(f"Sent image {images_generated}/{num_images} to frontend")
                        
                    except Exception as e:
                        safe_print(f"Error processing image: {e}")
                        continue
                        
                except Exception as e:
                    safe_print(f"Error generating image {image_index + 1}: {e}")
                    continue
            
            # Send completion
            update_progress(job_id, 100, "")
            yield f"data: {json.dumps({'type': 'complete', 'progress': 100, 'total_images': images_generated, 'job_id': job_id})}\n\n"
            
        except Exception as e:
            safe_print(f"[ERROR] Generate failed: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'job_id': job_id})}\n\n"
    
    return Response(stream_with_context(generate_images_stream()), mimetype='text/event-stream')

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Process uploaded image and generate STL, optionally extracting described object"""
    data = request.json
    image_base64 = data.get('image', '')
    description = data.get('description', '').strip()
    thickness = float(data.get('thickness', 10.0))
    solid_infill = data.get('solid_infill', True)  # Default to True for backward compatibility
    arc_top = data.get('arc_top', False)  # Default to False for backward compatibility
    
    if not image_base64:
        return jsonify({'error': 'Image is required'}), 400
    
    # Generate job ID
    job_id = str(time.time())
    
    try:
        safe_print(f"Processing uploaded image for STL generation (solid_infill={solid_infill}, arc_top={arc_top})")
        if description:
            safe_print(f"Description provided: '{description}' - will extract this object from image")
        
        # Decode image
        img_data = base64.b64decode(image_base64)
        original_img = Image.open(io.BytesIO(img_data))
        original_img = original_img.convert('RGB')
        
        # If description provided, extract the object; otherwise process entire image
        if description:
            img = extract_object_from_image(original_img, description, job_id)
        else:
            # Process the entire image to ensure black shape on white background
            img = process_uploaded_image(original_img)
        
        # Process to STL
        multi_layer = data.get('multi_layer', False)
        result = process_shape_from_image(img, thickness, job_id, solid_infill=solid_infill, arc_top=arc_top, multi_layer=multi_layer)
        
        # Update progress to 100% for consistency
        update_progress(job_id, 100, "")
        
        if result['success']:
            response_data = {
                'job_id': job_id,
                'preview_image': result['preview_image']
            }
            if 'stl_paths' in result:
                # Multi-layer mode
                response_data['stl_paths'] = result['stl_paths']
                response_data['layer_info'] = result['layer_info']
                response_data['num_layers'] = result['num_layers']
                response_data['multi_layer'] = True
            else:
                # Single STL mode
                response_data['stl_path'] = result['stl_path']
                response_data['multi_layer'] = False
            return jsonify(response_data)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        safe_print(f"[ERROR] Upload image processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reverse-image', methods=['POST'])
def reverse_image():
    """Reverse (invert) image colors and regenerate STL"""
    data = request.json
    image_base64 = data.get('image', '')
    thickness = float(data.get('thickness', 10.0))
    solid_infill = data.get('solid_infill', True)  # Default to True for backward compatibility
    arc_top = data.get('arc_top', False)  # Default to False for backward compatibility
    
    if not image_base64:
        return jsonify({'error': 'Image is required'}), 400
    
    # Generate job ID
    job_id = str(time.time())
    
    try:
        safe_print(f"Reversing image colors and regenerating STL (solid_infill={solid_infill}, arc_top={arc_top})")
        
        # Decode image
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        
        # Invert the image (swap black and white)
        img_array = np.array(img)
        # Invert: black (0) becomes white (255), white (255) becomes black (0)
        inverted_array = 255 - img_array
        inverted_img = Image.fromarray(inverted_array)
        
        # Process to STL
        multi_layer = data.get('multi_layer', False)
        result = process_shape_from_image(inverted_img, thickness, job_id, solid_infill=solid_infill, arc_top=arc_top, multi_layer=multi_layer)
        
        # Update progress to 100% for consistency
        update_progress(job_id, 100, "")
        
        if result['success']:
            response_data = {
                'job_id': job_id,
                'preview_image': result['preview_image']
            }
            if 'stl_paths' in result:
                # Multi-layer mode
                response_data['stl_paths'] = result['stl_paths']
                response_data['layer_info'] = result['layer_info']
                response_data['num_layers'] = result['num_layers']
                response_data['multi_layer'] = True
            else:
                # Single STL mode
                response_data['stl_path'] = result['stl_path']
                response_data['multi_layer'] = False
            return jsonify(response_data)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        safe_print(f"[ERROR] Reverse image failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-stl', methods=['POST'])
def generate_stl():
    """Generate STL file from selected image"""
    data = request.json
    image_base64 = data.get('image', '')
    thickness = float(data.get('thickness', 10.0))
    solid_infill = data.get('solid_infill', True)  # Default to True for backward compatibility
    arc_top = data.get('arc_top', False)  # Default to False for backward compatibility
    
    if not image_base64:
        return jsonify({'error': 'Image is required'}), 400
    
    # Generate job ID
    job_id = str(time.time())
    
    try:
        # Decode image
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        
        # Process to STL
        multi_layer = data.get('multi_layer', False)
        result = process_shape_from_image(img, thickness, job_id, solid_infill=solid_infill, arc_top=arc_top, multi_layer=multi_layer)
        
        if result['success']:
            response_data = {
                'job_id': job_id,
                'preview_image': result['preview_image']
            }
            if 'stl_paths' in result:
                # Multi-layer mode
                response_data['stl_paths'] = result['stl_paths']
                response_data['layer_info'] = result['layer_info']
                response_data['num_layers'] = result['num_layers']
                response_data['multi_layer'] = True
            else:
                # Single STL mode
                response_data['stl_path'] = result['stl_path']
                response_data['multi_layer'] = False
            return jsonify(response_data)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        safe_print(f"[ERROR] Generate STL failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

def extract_object_from_image(img, description, job_id=None):
    """Extract a specific object from the uploaded image based on description using image-to-image AI"""
    try:
        safe_print(f"Extracting object from uploaded image based on description: '{description}'")
        if job_id:
            update_progress(job_id, 30, "")
        
        # First, try image-to-image generation using Volcano Engine
        try:
            if job_id:
                update_progress(job_id, 40, "")
            ai_img = generate_image_from_image(img, description, job_id=job_id)
            if ai_img:
                safe_print("Successfully generated silhouette using image-to-image AI")
                return ai_img
        except Exception as e:
            safe_print(f"Image-to-image AI generation failed: {str(e)[:100]}, trying intelligent extraction...")
        
        # Fallback: try intelligent extraction from the uploaded image
        safe_print("Attempting intelligent extraction from uploaded image...")
        if job_id:
            update_progress(job_id, 50, "")
        
        # Resize for processing
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # Use intelligent segmentation to extract object
        extracted_img = intelligent_extract_from_image(img_array, description)
        
        # Process extracted region to silhouette
        processed_img = process_uploaded_image(Image.fromarray(extracted_img))
        
        # Verify extraction quality by checking if we got a reasonable shape
        # (not just a tiny blob or the entire image)
        img_array_check = np.array(processed_img.convert('L'))
        binary_check = (img_array_check < 127).astype(np.uint8) * 255
        contour_area = np.sum(binary_check > 0)
        total_area = binary_check.shape[0] * binary_check.shape[1]
        fill_ratio = contour_area / total_area
        
        if fill_ratio < 0.05 or fill_ratio > 0.95:
            safe_print(f"Extraction quality low (fill ratio: {fill_ratio:.2%}), trying text-to-image AI...")
            try:
                if job_id:
                    update_progress(job_id, 60, "")
                ai_img = generate_single_image_with_variation(description, width=512, height=512, variation_index=0, job_id=job_id)
                if ai_img:
                    safe_print("Successfully generated silhouette using text-to-image AI")
                    return ai_img
            except Exception as e:
                safe_print(f"Text-to-image AI generation failed: {str(e)[:100]}, using extracted image anyway...")
        else:
            safe_print(f"Extraction successful (fill ratio: {fill_ratio:.2%})")
        
        return processed_img
        
    except Exception as e:
        safe_print(f"[ERROR] Object extraction failed: {str(e)}")
        # Fallback: try AI generation based on description
        try:
            safe_print("Trying AI generation as fallback...")
            if job_id:
                update_progress(job_id, 40, "")
            ai_img = generate_single_image_with_variation(description, width=512, height=512, variation_index=0, job_id=job_id)
            if ai_img:
                return ai_img
        except:
            pass
        # Final fallback: process entire image
        return process_uploaded_image(img)

def generate_image_from_image(input_img, description, width=512, height=512, job_id=None):
    """Generate image from image using Volcano Engine image-to-image model"""
    try:
        # Get enhanced prompt for the description
        enhanced_prompt, _ = get_enhanced_prompt(description, variation_index=0)
        
        # Resize input image to standard size
        input_img = input_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert image to base64 and keep buffer for file-like object
        img_buffer = io.BytesIO()
        input_img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_buffer.seek(0)  # Reset for potential reuse
        
        # Use Volcano Engine API for image-to-image generation
        if ARK_API_KEY and len(ARK_API_KEY) > 10:
            try:
                from openai import OpenAI
                
                safe_print(f"Generating image from image using Volcano Engine")
                safe_print(f"Description: '{description}'")
                
                # Initialize client with Volcano Engine endpoint
                client = OpenAI(
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                    api_key=ARK_API_KEY,
                )
                
                # Generate image from image
                # Try different ways to pass the image parameter
                imagesResponse = None
                last_error = None
                
                # Method 1: Try image as base64 string in extra_body
                try:
                    imagesResponse = client.images.generate(
                        model="doubao-seedream-4-0-250828",
                        prompt=enhanced_prompt,
                        size="2K",
                        response_format="b64_json",
                        extra_body={
                            "watermark": False,
                            "image": img_base64,  # Base64 encoded image
                        },
                    )
                    safe_print("Successfully called API with image in extra_body")
                except Exception as e1:
                    last_error = e1
                    safe_print(f"Method 1 failed: {str(e1)[:100]}")
                    
                    # Method 2: Try image as file-like object
                    try:
                        img_buffer.seek(0)  # Reset buffer
                        imagesResponse = client.images.generate(
                            model="doubao-seedream-4-0-250828",
                            prompt=enhanced_prompt,
                            size="2K",
                            response_format="b64_json",
                            image=img_buffer,  # File-like object
                            extra_body={
                                "watermark": False,
                            },
                        )
                        safe_print("Successfully called API with image as file")
                    except Exception as e2:
                        last_error = e2
                        safe_print(f"Method 2 failed: {str(e2)[:100]}")
                        
                        # Method 3: Try image as direct parameter (base64)
                        try:
                            imagesResponse = client.images.generate(
                                model="doubao-seedream-4-0-250828",
                                prompt=enhanced_prompt,
                                size="2K",
                                response_format="b64_json",
                                image=img_base64,  # Direct parameter
                                extra_body={
                                    "watermark": False,
                                },
                            )
                            safe_print("Successfully called API with image as direct parameter")
                        except Exception as e3:
                            last_error = e3
                            safe_print(f"Method 3 failed: {str(e3)[:100]}")
                            raise Exception(f"All image parameter methods failed. Last error: {str(last_error)}")
                
                if not imagesResponse:
                    raise Exception("Failed to generate image from image")
                
                # Extract image from response
                image_data_b64 = None
                if hasattr(imagesResponse, 'data') and len(imagesResponse.data) > 0:
                    img_data = imagesResponse.data[0]
                    if hasattr(img_data, 'b64_json'):
                        image_data_b64 = img_data.b64_json
                    elif isinstance(img_data, dict) and 'b64_json' in img_data:
                        image_data_b64 = img_data['b64_json']
                
                if image_data_b64:
                    image_data = base64.b64decode(image_data_b64)
                    img = Image.open(io.BytesIO(image_data))
                    img = img.convert('RGB')
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # Process to silhouette (black shape on white background)
                    img = process_uploaded_image(img)
                    
                    safe_print("Successfully generated image from image using Volcano Engine")
                    return img
                else:
                    raise Exception("No image data in response")
                    
            except Exception as e:
                safe_print(f"Volcano Engine image-to-image failed: {str(e)[:100]}")
                raise
        else:
            raise Exception("ARK_API_KEY not configured")
            
    except Exception as e:
        safe_print(f"[ERROR] Image-to-image generation failed: {str(e)}")
        raise

def intelligent_extract_from_image(img_array, description):
    """Use OpenCV to intelligently extract object based on description"""
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use K-means clustering to identify distinct regions
    # Reshape image to a list of pixels
    pixel_values = img_array.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5  # Number of clusters (objects + background variations)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_array.shape)
    
    # Find the most prominent object (largest non-background cluster)
    # Background is usually the largest cluster at edges
    edge_mask = np.zeros(gray.shape, dtype=np.uint8)
    edge_mask[0:10, :] = 1  # Top edge
    edge_mask[-10:, :] = 1  # Bottom edge
    edge_mask[:, 0:10] = 1  # Left edge
    edge_mask[:, -10:] = 1  # Right edge
    
    # Find which cluster is most common at edges (likely background)
    edge_labels = labels.flatten()[edge_mask.flatten() == 1]
    if len(edge_labels) > 0:
        background_label = np.bincount(edge_labels).argmax()
    else:
        background_label = 0
    
    # Get all non-background regions
    object_mask = (labels.flatten() != background_label).reshape(gray.shape).astype(np.uint8) * 255
    
    # Find the largest connected component (main object)
    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(object_mask, connectivity=8)
    
    if num_labels > 1:
        # Find largest component (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_object_mask = (labels_cc == largest_label).astype(np.uint8) * 255
        
        # Apply mask to original image
        result = img_array.copy()
        result[main_object_mask == 0] = [255, 255, 255]  # White background
        
        return result
    else:
        # Fallback: return original image
        return img_array

def process_uploaded_image(img):
    """Process uploaded image to ensure black shape on white background"""
    # Resize to standard size
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:
        # Find largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        shape_mask = (labels == largest_label).astype(np.uint8) * 255
        
        # Check brightness
        shape_pixels = gray[shape_mask == 255]
        background_mask = (labels != largest_label).astype(np.uint8) * 255
        background_pixels = gray[background_mask == 255]
        
        shape_mean = np.mean(shape_pixels) if len(shape_pixels) > 0 else 128
        background_mean = np.mean(background_pixels) if len(background_pixels) > 0 else 128
        
        # If shape is lighter than background, invert
        if shape_mean > background_mean:
            binary = cv2.bitwise_not(binary)
            # Re-extract after inversion
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)
            num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels2 > 1:
                largest_label2 = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
                shape_mask2 = (labels2 == largest_label2).astype(np.uint8) * 255
                binary = np.zeros_like(gray, dtype=np.uint8)
                binary[shape_mask2 == 255] = 0
                binary[shape_mask2 == 0] = 255
        else:
            binary = np.zeros_like(gray, dtype=np.uint8)
            binary[shape_mask == 255] = 0
            binary[shape_mask == 0] = 255
    else:
        # Fallback
        binary_mean = np.mean(binary)
        gray_mean = np.mean(gray)
        if binary_mean < 50 and gray_mean < 100:
            binary = cv2.bitwise_not(binary)
    
    # Fill holes and ensure solid infill
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Final verification
    final_mean = np.mean(binary)
    if final_mean < 30:
        binary = cv2.bitwise_not(binary)
    
    # Create RGB image: black shape, white background
    img_rgb = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    img_rgb[binary == 0] = [0, 0, 0]  # Black shape
    img_rgb[binary == 255] = [255, 255, 255]  # White background
    
    return Image.fromarray(img_rgb)

def process_shape_from_image(img_2d, thickness, job_id, solid_infill=True, arc_top=False, multi_layer=False):
    """Process STL generation from an existing image
    
    Args:
        img_2d: PIL Image
        thickness: Thickness in mm
        job_id: Job ID for progress tracking
        solid_infill: Whether to fill holes
        arc_top: Whether to use arc top
        multi_layer: If True, generate multiple STL files (one per component/layer)
    
    Returns:
        dict with 'success', 'stl_path' or 'stl_paths' (list), 'preview_image', 'layer_info'
    """
    try:
        if multi_layer:
            # Multi-layer mode: detect components and generate separate STL for each
            update_progress(job_id, 20, "Analyzing image components...")
            components = detect_image_components(img_2d, max_components=10)
            
            if not components:
                raise ValueError("Could not detect components in image")
            
            safe_print(f"Detected {len(components)} components/layers")
            update_progress(job_id, 30, f"Generating {len(components)} STL layers...")
            
            stl_paths = []
            layer_info = []
            
            for idx, (component_img, layer_name) in enumerate(components):
                try:
                    # Extract contour for this component
                    contour = image_to_contour(component_img, solid_infill=solid_infill)
                    if contour is None:
                        safe_print(f"Warning: Could not extract contour for {layer_name}, skipping")
                        continue
                    
                    # Extrude to 3D
                    mesh_3d = extrude_2d_to_3d(contour, thickness, arc_top=arc_top)
                    
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
                    mesh_3d.save(temp_file.name)
                    
                    stl_paths.append(temp_file.name)
                    layer_info.append({
                        'name': layer_name,
                        'stl_path': temp_file.name,
                        'index': idx
                    })
                    
                    progress = 30 + int((idx + 1) / len(components) * 60)
                    update_progress(job_id, progress, f"Generated layer {idx+1}/{len(components)}")
                    
                except Exception as e:
                    safe_print(f"Error processing layer {layer_name}: {str(e)}")
                    continue
            
            if not stl_paths:
                raise ValueError("Could not generate any STL layers")
            
            # Save 2D image preview (original image)
            img_buffer = io.BytesIO()
            img_2d.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            update_progress(job_id, 100, f"Generated {len(stl_paths)} layers successfully")
            
            return {
                'success': True,
                'stl_paths': stl_paths,
                'layer_info': layer_info,
                'preview_image': img_base64,
                'num_layers': len(stl_paths)
            }
        else:
            # Single STL mode (original behavior)
            # Extract contour(s) - may be single contour or list of contours
            contour = image_to_contour(img_2d, solid_infill=solid_infill)
            if contour is None:
                raise ValueError("Could not extract contour from image")
            
            # Use exact thickness as specified by user (in mm)
            # Extrude to 3D (handles both single contour and list of contours)
            mesh_3d = extrude_2d_to_3d(contour, thickness, arc_top=arc_top)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
            mesh_3d.save(temp_file.name)
            
            # Save 2D image preview
            img_buffer = io.BytesIO()
            img_2d.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            update_progress(job_id, 100, "")
            
            return {
                'success': True,
                'stl_path': temp_file.name,
                'preview_image': img_base64
            }
    except Exception as e:
        update_progress(job_id, 0, f"Error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/progress/<job_id>')
def get_progress(job_id):
    if job_id in progress_data:
        return jsonify(progress_data[job_id])
    return jsonify({'progress': 0, 'message': 'Job not found'}), 404

@app.route('/api/download/<path:filename>')
def download_stl(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True, download_name='shape.stl')
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/download-layers', methods=['POST'])
def download_layers_zip():
    """Download multiple STL files as a zip archive"""
    try:
        data = request.json
        stl_paths = data.get('stl_paths', [])
        layer_info = data.get('layer_info', [])
        
        if not stl_paths:
            return jsonify({'error': 'No STL files provided'}), 400
        
        # Verify all files exist
        missing_files = []
        for stl_path in stl_paths:
            if not os.path.exists(stl_path):
                missing_files.append(stl_path)
        
        if missing_files:
            safe_print(f"[ERROR] Missing STL files: {missing_files}")
            return jsonify({'error': f'STL file path not available. Missing files: {len(missing_files)}'}), 404
        
        # Create a temporary zip file
        zip_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(zip_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, stl_path in enumerate(stl_paths):
                # Get layer name from layer_info if available
                layer_name = f"Layer_{i+1}"
                if layer_info and i < len(layer_info):
                    layer_name = layer_info[i].get('name', layer_name)
                
                # Clean layer name for filename (remove invalid characters)
                safe_layer_name = "".join(c for c in layer_name if c.isalnum() or c in ('_', '-', ' '))
                safe_layer_name = safe_layer_name.replace(' ', '_')
                
                # Add file to zip with a clean name
                zipf.write(stl_path, f"{safe_layer_name}.stl")
        
        return send_file(zip_file.name, as_attachment=True, download_name='layers.zip', mimetype='application/zip')
    except Exception as e:
        safe_print(f"[ERROR] Download layers zip failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Storage API endpoints
@app.route('/api/models/save', methods=['POST'])
def save_model_api():
    """Save a model to storage"""
    try:
        data = request.json
        name = data.get('name', 'Untitled Model')
        description = data.get('description', '')
        preview_image = data.get('preview_image', '')
        stl_path = data.get('stl_path', '')
        thickness = data.get('thickness', 10.0)
        solid_infill = data.get('solid_infill', True)
        
        if not preview_image or not stl_path:
            return jsonify({'error': 'Preview image and STL path are required'}), 400
        
        if not os.path.exists(stl_path):
            return jsonify({'error': 'STL file not found'}), 404
        
        model_id = save_model(name, description, preview_image, stl_path, thickness, solid_infill)
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'message': 'Model saved successfully'
        })
    except Exception as e:
        safe_print(f"[ERROR] Save model failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all saved models"""
    try:
        db = load_models_db()
        models = list(db.values())
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        safe_print(f"[ERROR] List models failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get a specific model"""
    try:
        db = load_models_db()
        if model_id not in db:
            return jsonify({'error': 'Model not found'}), 404
        
        model = db[model_id].copy()
        
        # Load preview image as base64
        preview_path = os.path.join(PREVIEWS_DIR, model['preview_image'])
        if os.path.exists(preview_path):
            with open(preview_path, 'rb') as f:
                preview_data = base64.b64encode(f.read()).decode()
            model['preview_image_base64'] = preview_data
        
        return jsonify({'success': True, 'model': model})
    except Exception as e:
        safe_print(f"[ERROR] Get model failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/preview', methods=['GET'])
def get_model_preview(model_id):
    """Get model preview image"""
    try:
        db = load_models_db()
        if model_id not in db:
            return jsonify({'error': 'Model not found'}), 404
        
        preview_path = os.path.join(PREVIEWS_DIR, db[model_id]['preview_image'])
        if not os.path.exists(preview_path):
            return jsonify({'error': 'Preview image not found'}), 404
        
        return send_file(preview_path, mimetype='image/png')
    except Exception as e:
        safe_print(f"[ERROR] Get preview failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/stl', methods=['GET'])
def get_model_stl(model_id):
    """Download model STL file"""
    try:
        db = load_models_db()
        if model_id not in db:
            return jsonify({'error': 'Model not found'}), 404
        
        stl_path = os.path.join(STL_DIR, db[model_id]['stl_file'])
        if not os.path.exists(stl_path):
            return jsonify({'error': 'STL file not found'}), 404
        
        model_name = db[model_id].get('name', 'model').replace(' ', '_')
        return send_file(stl_path, as_attachment=True, download_name=f'{model_name}.stl')
    except Exception as e:
        safe_print(f"[ERROR] Get STL failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model_api(model_id):
    """Delete a model"""
    try:
        if delete_model(model_id):
            return jsonify({'success': True, 'message': 'Model deleted successfully'})
        else:
            return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        safe_print(f"[ERROR] Delete model failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-ai')
def test_ai():
    """Test endpoint to check AI API connectivity"""
    result = {
        'openai_key_set': bool(OPENAI_API_KEY and len(OPENAI_API_KEY) > 10),
        'openai_key_preview': OPENAI_API_KEY[:15] + '...' if OPENAI_API_KEY else 'Not set',
        'huggingface_key_set': bool(HUGGINGFACE_API_KEY),
        'test_result': 'Testing...'
    }
    
    # Try a simple test generation
    try:
        test_img, method = generate_2d_image_ai("a circle", 256, 256)
        result['test_result'] = f'Success using {method}'
        result['success'] = True
    except Exception as e:
        result['test_result'] = f'Failed: {str(e)}'
        result['success'] = False
        result['error'] = str(e)
    
    return jsonify(result)

@app.route('/api/gcode/learn', methods=['POST'])
def learn_gcode():
    """Learn printer and printing settings from uploaded G-code file"""
    try:
        if 'gcode_file' not in request.files:
            return jsonify({'error': 'No G-code file provided'}), 400
        
        gcode_file = request.files['gcode_file']
        if gcode_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read G-code content
        gcode_content = gcode_file.read().decode('utf-8')
        
        # Parse settings
        parsed_settings = gcode_parser.parse_gcode_file(gcode_content)
        
        # Merge with existing settings (new settings take precedence)
        global learned_gcode_settings
        learned_gcode_settings = gcode_parser.merge_settings(learned_gcode_settings, parsed_settings)
        
        # Save to file
        save_gcode_settings(learned_gcode_settings)
        
        safe_print(f"[SUCCESS] Learned settings from G-code file: {gcode_file.filename}")
        
        return jsonify({
            'success': True,
            'message': 'Settings learned successfully',
            'settings': learned_gcode_settings
        })
    except Exception as e:
        safe_print(f"[ERROR] Learn G-code failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gcode/settings', methods=['GET'])
def get_gcode_settings():
    """Get current G-code settings"""
    global learned_gcode_settings
    return jsonify({
        'success': True,
        'settings': learned_gcode_settings
    })

@app.route('/api/gcode/settings', methods=['POST'])
def update_gcode_settings():
    """Update G-code settings"""
    try:
        data = request.json
        if 'settings' not in data:
            return jsonify({'error': 'Settings not provided'}), 400
        
        global learned_gcode_settings
        learned_gcode_settings = gcode_parser.merge_settings(learned_gcode_settings, data['settings'])
        save_gcode_settings(learned_gcode_settings)
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': learned_gcode_settings
        })
    except Exception as e:
        safe_print(f"[ERROR] Update G-code settings failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gcode/generate', methods=['POST'])
def generate_gcode():
    """Generate G-code from STL file"""
    try:
        data = request.json
        stl_path = data.get('stl_path')
        job_id = data.get('job_id')
        
        if not stl_path or not os.path.exists(stl_path):
            return jsonify({'error': 'STL file not found'}), 400
        
        # Get settings (use provided settings or learned settings)
        settings = data.get('settings', learned_gcode_settings)
        
        # Generate G-code
        generator = GCodeGenerator(settings)
        gcode_content = generator.generate_from_stl(stl_path)
        
        # Save G-code file
        gcode_filename = f"{job_id}.gcode"
        gcode_path = os.path.join(GCODE_DIR, gcode_filename)
        with open(gcode_path, 'w', encoding='utf-8') as f:
            f.write(gcode_content)
        
        safe_print(f"[SUCCESS] Generated G-code: {gcode_filename}")
        
        return jsonify({
            'success': True,
            'gcode_path': gcode_path,
            'gcode_filename': gcode_filename
        })
    except Exception as e:
        safe_print(f"[ERROR] Generate G-code failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gcode/download/<filename>')
def download_gcode(filename):
    """Download generated G-code file"""
    try:
        gcode_path = os.path.join(GCODE_DIR, filename)
        if not os.path.exists(gcode_path):
            return jsonify({'error': 'G-code file not found'}), 404
        
        return send_file(gcode_path, as_attachment=True, download_name=filename)
    except Exception as e:
        safe_print(f"[ERROR] Download G-code failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on all network interfaces (0.0.0.0) to make it accessible from other devices
    # Access from other devices using: http://<your-ip-address>:5000
    import socket
    
    # Get local IP address
    def get_local_ip():
        try:
            # Connect to a remote address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    
    safe_print("="*70)
    safe_print("Flask server starting...")
    safe_print("="*70)
    safe_print(f"Local access:    http://localhost:5000")
    safe_print(f"Network access:  http://{local_ip}:5000")
    safe_print("="*70)
    safe_print("If you cannot access from other devices:")
    safe_print("1. Make sure devices are on the same Wi-Fi/network")
    safe_print("2. Check Windows Firewall - allow port 5000")
    safe_print("3. Try disabling firewall temporarily to test")
    safe_print("="*70)
    
    app.run(host='0.0.0.0', debug=True, port=5000)

