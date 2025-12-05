# ⚠️ VERCEL DEPLOYMENT SIZE WARNING

## Current Configuration

Your `vercel.json` is now configured to use **full requirements** (`requirements.txt`) which includes:

- ✅ All packages restored (numpy, scipy, Pillow, scikit-image, etc.)
- ✅ All application files restored (app.py, gcode_generator.py, etc.)
- ✅ Full functionality available

## ⚠️ **IMPORTANT: Size Limit Issue**

**This configuration will likely EXCEED Vercel's 250 MB limit!**

The full requirements include:
- `numpy` (~150 MB)
- `scipy` (~100+ MB)
- `scikit-image` (~50+ MB)
- `scikit-learn` (~50+ MB)
- `Pillow` (~50 MB)
- `numpy-stl` (depends on numpy)
- Other dependencies

**Total estimated size: ~500+ MB** ❌

## Expected Error

When you deploy, you will likely see:
```
Error: A Serverless Function has exceeded the unzipped maximum size of 250 MB.
```

## Solutions

### Option 1: Use a Different Platform (Recommended) ✅

For full functionality, deploy to platforms without strict size limits:

#### **Railway** (Recommended)
- No size limits
- Better for Flask apps
- Easy deployment
- Free tier available
- [railway.app](https://railway.app)

#### **Render**
- Supports up to 512 MB
- Good for Python apps
- Free tier available
- [render.com](https://render.com)

#### **Fly.io**
- Container-based (full control)
- No size limits
- [fly.io](https://fly.io)

### Option 2: Keep Minimal for Vercel, Full for Local

1. **For Vercel**: Use `requirements-vercel-minimal.txt`
   - Update `vercel.json` to use minimal requirements
   - Limited functionality but stays under 250 MB

2. **For Local Development**: Use `requirements.txt`
   - Full functionality
   - All features work

### Option 3: Hybrid Architecture

- **Vercel**: Host API endpoints (minimal requirements)
- **External Service**: Handle heavy processing
  - Image processing: Cloudinary, Imgix
  - STL generation: Separate service
  - GCode generation: Separate service

### Option 4: Optimize Dependencies

Try to reduce package sizes:
- Use lighter alternatives where possible
- Remove unused dependencies
- Use pre-compiled wheels
- Split into multiple functions

## Current Status

- ✅ All files restored
- ✅ Full requirements configured
- ⚠️ Will likely fail on Vercel due to size
- 💡 Consider using Railway/Render for full functionality

## Quick Switch Back to Minimal

If you want to switch back to minimal requirements for Vercel:

```json
"installCommand": "pip install --upgrade pip && pip install --no-cache-dir -r requirements-vercel-minimal.txt"
```

## Recommendation

**For production with full features**: Use **Railway** or **Render**
**For Vercel**: Keep minimal requirements and accept limited functionality

