# AI Image Generation Setup Guide

This application uses **AI image generation** to create 2D shapes from text descriptions, which are then extruded into 3D STL files.

## How It Works

The app automatically tries AI generation in this order:

1. **OpenAI DALL-E 3** (if API key is set) - Best quality, fastest
2. **Hugging Face Stable Diffusion** (free) - Good quality, may be slower
3. **Keyword-based fallback** - Basic shapes only (circle, square, triangle, star)

## Setup Options

### Option 1: Use Free Hugging Face API (No Setup Required)

The app works **immediately** without any setup! It uses Hugging Face's free public API.

**Note**: The first request may take 10-30 seconds as the model loads. Subsequent requests are faster.

### Option 2: Use OpenAI DALL-E (Best Quality)

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Install the OpenAI library:
   ```bash
   pip install openai
   ```
3. Set the environment variable:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Windows CMD
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY=your-api-key-here
   ```
4. Restart the Flask server

### Option 3: Use Hugging Face with API Key (Faster)

1. Get a free API key from [Hugging Face](https://huggingface.co/settings/tokens)
2. Set the environment variable:
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACE_API_KEY="your-api-key-here"
   
   # Windows CMD
   set HUGGINGFACE_API_KEY=your-api-key-here
   
   # Linux/Mac
   export HUGGINGFACE_API_KEY=your-api-key-here
   ```
3. Restart the Flask server

## Testing AI Generation

1. Start the server:
   ```bash
   python app.py
   ```

2. Open `http://localhost:5000` in your browser

3. Try these examples:
   - **"a steak"** - Should generate a steak silhouette
   - **"a cat"** - Should generate a cat silhouette
   - **"a heart"** - Should generate a heart shape
   - **"a circle"** - Should generate a circular shape
   - **"a star"** - Should generate a star shape

4. Check the console output - you'll see messages like:
   - `✓ Successfully generated image using OpenAI DALL-E`
   - `✓ Successfully generated image using Hugging Face (stable-diffusion-v1-5)`
   - `⚠ Falling back to keyword-based generation (AI unavailable)`

## Troubleshooting

### AI Generation Not Working

1. **Check your internet connection** - AI generation requires internet access
2. **Check console output** - Look for error messages
3. **Try again** - Hugging Face models may need to load on first request
4. **Check rate limits** - Free APIs have rate limits

### Slow Generation

- First request is always slowest (model loading)
- Use OpenAI DALL-E for faster generation
- Use Hugging Face API key for faster access

### Poor Quality Images

- The app automatically converts images to black/white silhouettes for 3D extrusion
- Try more descriptive prompts (e.g., "a detailed steak silhouette")
- OpenAI DALL-E generally produces better quality

## How to Verify AI is Working

When you enter a description like "a steak" or "a cat", you should see:
- A generated image that matches your description (not just text)
- Console messages indicating which AI service was used
- The image converted to a black silhouette for extrusion

If you see basic shapes (circle, square, etc.) for non-basic descriptions, AI generation may have failed and it fell back to keyword matching.

