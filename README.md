# 🎨 AI Text to 3D STL Generator

Transform your ideas into 3D printable models! This web application uses AI to generate 2D images from text descriptions or uploaded images, then converts them into 3D STL files ready for 3D printing.

## ✨ Features

### 🤖 AI-Powered Image Generation
- **Text-to-Image**: Describe any shape or object in natural language (e.g., "a cat", "a flower", "a heart")
- **Multiple Variations**: Generate 4 different views/poses at once and choose your favorite
- **Smart Processing**: Automatically converts images to clean black silhouettes on white backgrounds
- **Animal-Specific Views**: For animals, generates head, full body, sitting, and standing/walking poses

### 📤 Image Upload & Extraction
- **Upload Your Own Images**: Upload PNG, JPG, or other image formats
- **AI Object Extraction**: Describe what you want to extract from the image, and AI will identify and extract it
- **Smart Background Removal**: Automatically processes images to create clean silhouettes

### 🎨 Drawing Tools
- **Edit Your Shapes**: Use black and white pens to adjust shapes before generating STL
- **Adjustable Brush Size**: Control the thickness of your drawing strokes
- **Undo/Redo**: Full history support for your drawing edits
- **Real-time Preview**: See your changes instantly on the canvas

### 🔄 Advanced Controls
- **Reverse Selection**: Invert colors (black ↔ white) if the AI generates the wrong colors
- **Undo Reverse**: Restore the original image if you change your mind
- **Custom Thickness**: Set exact thickness in millimeters (not proportional)
- **Progress Tracking**: Real-time progress updates during generation

### 📦 Export & Print
- **STL File Export**: Download ready-to-print 3D models
- **High-Quality Meshes**: Optimized for smooth curves and accurate shapes
- **Network Access**: Access from any device on your local network

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/angrycat0-0/AI-stl.git
   cd AI-stl
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up AI API key** for image generation
   
   The app uses **Volcano Engine (Doubao-Seedream)** by default. Set your API key:
   
   **Windows (PowerShell):**
   ```powershell
   $env:ARK_API_KEY="your-api-key-here"
   ```
   
   **Windows (CMD):**
   ```cmd
   set ARK_API_KEY=your-api-key-here
   ```
   
   **Linux/Mac:**
   ```bash
   export ARK_API_KEY=your-api-key-here
   ```
   
   > **Note**: You can also use other AI services like OpenAI DALL-E or Hugging Face. See [API Configuration](#api-configuration) below.

4. **Start the server**
   ```bash
   python app.py
   ```
   
   The server will display your IP address when it starts. You'll see something like:
   ```
   * Running on http://0.0.0.0:5000
   * Access from other devices: http://192.168.31.76:5000
   ```

5. **Open in your browser**
   - **Local access**: `http://localhost:5000`
   - **Network access**: Use the IP address shown when the server starts

---

## 📖 Usage Guide

### Method 1: AI Generate Image

1. Select **"AI Generate Image"** mode
2. Enter a description in the text box (e.g., "a cat", "a flower", "a heart")
3. Set the thickness in millimeters (default: 10.0mm)
4. Click **"Generate STL"**
5. Wait for the AI to generate 4 image variations
6. Images will appear as they're generated - you can see them in real-time
7. The first image will automatically be used to generate the STL
8. Click **"Download STL File"** when ready

**Tips:**
- Be specific: "a sitting cat" vs "a cat"
- For animals, the AI automatically generates different poses
- If the result has white infill and black background, use the **"Reverse Selection"** button

### Method 2: Upload Image

1. Select **"Upload Image"** mode
2. Click **"Choose File"** and select an image (PNG, JPG, etc.)
3. **(Optional)** Describe what you want to extract (e.g., "a dog", "a flower", "the main object")
   - If you provide a description, AI will intelligently extract that object
   - If left empty, the system will extract the main object automatically
4. Set the thickness in millimeters
5. Click **"Generate STL"**
6. Download the STL file when ready

### Editing Your Shape

After generating or uploading an image, you can edit it:

1. **Drawing Tools** will appear below the preview
2. Select **"Black Pen"** or **"White Pen"**
3. Adjust **Brush Size** slider
4. Draw on the canvas to modify the shape
5. Use **"Undo"** and **"Redo"** to manage your edits
6. Click **"Apply Drawing"** to regenerate the STL with your changes
7. Use **"Clear"** to remove all drawing

### Reverse Selection

If the AI generates white infill with a black background:

1. Click **"Reverse Selection"** to invert the colors
2. The STL will automatically regenerate with correct colors
3. Click **"Undo Reverse"** to restore the original if needed

---

## 🎯 Examples

### Text Descriptions
- `"a cat"` - Generates a cat silhouette
- `"a sitting cat"` - Generates a cat in sitting pose
- `"a flower"` - Generates a flower shape
- `"a heart"` - Generates a heart shape
- `"a star"` - Generates a star shape
- `"a circle"` - Generates a circular shape
- `"a bear standing"` - Generates a standing bear
- `"a simple logo of a mountain"` - Generates a minimalist mountain logo

### Animal Variations
When you describe an animal, the AI automatically generates:
- **Head view** - Close-up of the head
- **Full body standing/walking** - Complete animal standing or walking
- **Sitting pose** - Animal in sitting position
- **Alternative standing/walking** - Different angle of standing/walking

### Upload Examples
- Upload a photo of a logo → Extract and convert to 3D
- Upload a drawing → Convert to 3D model
- Upload a photo with multiple objects → Describe which one to extract

---

## ⚙️ API Configuration

### Volcano Engine (Default - Currently Used)

The app uses **Volcano Engine's Doubao-Seedream** model by default.

1. Get your API key from [Volcano Engine](https://www.volcengine.com/)
2. Set the environment variable:
   ```bash
   set ARK_API_KEY=your-api-key-here
   ```
3. Restart the server

### Alternative: OpenAI DALL-E

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Install: `pip install openai`
3. Set environment variable:
   ```bash
   set OPENAI_API_KEY=your-api-key-here
   ```

### Alternative: Hugging Face

1. Get a free API key from [Hugging Face](https://huggingface.co/settings/tokens)
2. Set environment variable:
   ```bash
   set HUGGINGFACE_API_KEY=your-api-key-here
   ```

### Alternative: Replicate

1. Sign up at [Replicate](https://replicate.com)
2. Get your API token from your account settings
3. Set environment variable:
   ```bash
   set REPLICATE_API_TOKEN=your-token-here
   ```

> **Note**: The app will automatically try these services in order if one fails. Volcano Engine is the primary service.

---

## 🌐 Network Access

### Accessing from Other Devices

The server is configured to be accessible on your local network by default.

1. **Find your IP address**:
   - **Windows**: Open Command Prompt → Type `ipconfig` → Look for "IPv4 Address"
   - **Mac/Linux**: Open Terminal → Type `ifconfig` or `ip addr`
   
2. **Access from another device**:
   - Make sure both devices are on the same Wi-Fi network
   - Open browser on the other device
   - Navigate to: `http://<your-ip>:5000`
   - Examples: 
     - `http://192.168.31.76:5000` (common local network IP)
     - `http://192.168.1.100:5000` (alternative format)

### Troubleshooting Network Access

**If you cannot access from other devices:**

#### Windows Firewall Configuration

1. Open **Windows Defender Firewall**
2. Click **"Advanced settings"**
3. Click **"Inbound Rules"** → **"New Rule"**
4. Select **"Port"** → Click **Next**
5. Select **"TCP"** and enter port **"5000"** → Click **Next**
6. Select **"Allow the connection"** → Click **Next**
7. Check all profiles (Domain, Private, Public) → Click **Next**
8. Name it **"Flask App"** → Click **Finish**

#### Alternative: Temporarily Disable Firewall (Testing Only)

1. Open **Windows Defender Firewall**
2. Click **"Turn Windows Defender Firewall on or off"**
3. Turn off for **Private networks** (temporarily for testing)
4. **Remember to turn it back on after testing!**

#### Verify Network Connection

1. **Check same network**: Both devices must be on the same Wi-Fi
2. **Check IP addresses**: Both should start with the same numbers (e.g., `192.168.1.x`)
3. **Test connection**: From another device, try: `http://<your-ip>:5000`
   - Example: `http://192.168.31.76:5000`

#### Network Diagnostic Tool

Run the diagnostic script to check your network setup:
```bash
python check_network.py
```

This will:
- Check if port 5000 is accessible
- Display all your IP addresses
- Help identify connection issues

---

## 🔧 Troubleshooting

### Image Generation Issues

**Problem**: "AI image generation failed"
- **Solution**: Check your API key is set correctly and has sufficient credits
- **Solution**: Try a different AI service (see API Configuration above)

**Problem**: Images have white infill and black background
- **Solution**: Click the **"Reverse Selection"** button to invert colors

**Problem**: Generated images are too complex
- **Solution**: Be more specific in your description (e.g., "a simple silhouette of a cat")
- **Solution**: The system automatically processes images to create clean silhouettes

### STL Generation Issues

**Problem**: STL file is a square or wrong shape
- **Solution**: The image might not have been processed correctly. Try using **"Reverse Selection"**
- **Solution**: Use the drawing tools to manually adjust the shape

**Problem**: STL thickness is wrong
- **Solution**: The thickness setting is now exact (in millimeters), not proportional. Adjust the value directly.

### Drawing Tools Issues

**Problem**: Can't select white pen
- **Solution**: Make sure you've generated or uploaded an image first
- **Solution**: Refresh the page and try again

**Problem**: Drawing doesn't appear
- **Solution**: Make sure you're clicking and dragging on the canvas
- **Solution**: Check that the brush size is not too small

### Network Issues

**Problem**: Can't access from other devices
- **Solution**: See [Network Access](#network-access) section above
- **Solution**: Check Windows Firewall settings
- **Solution**: Verify both devices are on the same network

---

## 📁 Project Structure

```
text-2d-stl/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Frontend HTML
├── static/
│   ├── style.css          # Stylesheet
│   └── script.js          # Frontend JavaScript
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── check_network.py      # Network diagnostic tool
```

---

## 🛠️ Technical Details

- **Backend**: Flask (Python web framework)
- **Image Processing**: OpenCV, PIL (Pillow)
- **3D Mesh Generation**: numpy-stl
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **AI Services**: Volcano Engine, OpenAI DALL-E, Hugging Face, Replicate

### Image Processing Pipeline

1. **AI Generation/Upload**: Get 2D image from AI or user upload
2. **Object Extraction**: If description provided, extract specific object from image
3. **Color Processing**: Ensure black shape on white background
4. **Morphological Operations**: Fill holes and clean edges
5. **Contour Extraction**: Find the main shape contour
6. **3D Extrusion**: Extrude contour to 3D mesh with specified thickness
7. **STL Export**: Generate and download STL file

---

## 📝 License

MIT License - feel free to use and modify as needed!

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 💡 Tips & Best Practices

1. **For best results**: Use simple, clear descriptions
2. **For animals**: The AI automatically generates multiple poses
3. **For complex shapes**: Use the drawing tools to refine after generation
4. **For logos**: Upload your logo image and let AI extract it
5. **For thickness**: Start with 5-10mm for most shapes, adjust as needed
6. **For printing**: Ensure your STL has sufficient thickness for your printer

---

## 📞 Support

If you encounter any issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the console output when running `python app.py`
3. Check your API keys are set correctly
4. Verify your network configuration

---

**Happy 3D Printing! 🎉**
