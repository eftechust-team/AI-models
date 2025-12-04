# Vercel Function Size Fix

## Problem
The serverless function exceeds Vercel's 250 MB limit due to large dependencies:
- numpy (~150 MB)
- scipy (~100+ MB)
- scikit-image (~50+ MB)
- scikit-learn (~50+ MB)
- numpy-stl (depends on numpy)

## Solution Applied

1. **Created `requirements-vercel-minimal.txt`** - Minimal dependencies:
   - Flask
   - flask-cors
   - Pillow (for basic image processing)
   - requests, replicate, openai (for AI APIs)

2. **Updated `vercel.json`** - Uses minimal requirements

3. **Removed heavy packages**:
   - numpy (can be added back if needed, but increases size significantly)
   - scipy
   - scikit-image
   - scikit-learn
   - numpy-stl

## Limitations

Without numpy and related packages:
- ❌ STL file generation won't work (requires numpy-stl)
- ❌ Advanced image processing limited
- ✅ Basic Flask app will work
- ✅ AI image generation APIs will work
- ✅ Basic image upload/display will work

## Alternative Solutions

### Option 1: Use Different Platform (Recommended)
For full functionality, use platforms that support larger functions:
- **Railway** - No size limits, better for Flask apps
- **Render** - Supports larger functions
- **Fly.io** - Container-based, full control
- **Heroku** - Classic option (paid)

### Option 2: Split into Multiple Functions
- Create separate functions for different features
- Use API routes to call different functions
- Each function stays under 250MB

### Option 3: Use External Services
- Move STL generation to a separate service
- Use external API for image processing
- Keep main function minimal

## Current Status
✅ Minimal requirements configured
✅ Function should now be under 250MB
⚠️ Some features (STL generation) won't work without numpy

