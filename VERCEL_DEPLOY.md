# Vercel Deployment Guide

## Important Notes

⚠️ **Vercel Limitations:**
- Vercel is designed for serverless functions, not long-running Flask apps
- File storage (`saved_models/`) won't persist between deployments
- Functions have a 60-second timeout (can be extended to 300s on Pro plan)
- Binary dependencies like OpenCV may have issues

## Setup Steps

1. **Install Vercel CLI** (if deploying via CLI):
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**:
   ```bash
   vercel
   ```

   Or connect your GitHub repo directly in Vercel dashboard.

3. **Set Environment Variables** in Vercel Dashboard:
   - `VOLCANO_ACCESS_KEY_ID` - Your Volcano Engine Access Key ID
   - `VOLCANO_SECRET_ACCESS_KEY` - Your Volcano Engine Secret Access Key
   - `ARK_API_KEY` - Your ARK API Key (optional)
   - `OPENAI_API_KEY` - Your OpenAI API Key (optional)
   - `REPLICATE_API_TOKEN` - Your Replicate API Token (optional)
   - `HUGGINGFACE_API_KEY` - Your Hugging Face API Key (optional)

4. **Storage Considerations**:
   - Files saved to `saved_models/` will be lost on each deployment
   - Consider using external storage (S3, Cloudinary, etc.) for production
   - Or use Vercel's Blob Storage for file persistence

## Alternative Deployment Options

For better compatibility with Flask apps, consider:

1. **Railway** - Easy Flask deployment with persistent storage
2. **Render** - Free tier available, good for Flask apps
3. **Fly.io** - Good for containerized apps
4. **Heroku** - Classic option (paid now)

## Troubleshooting

If you encounter errors:

1. **OpenCV issues**: The `opencv-python-headless` package should work better than `opencv-python`
2. **Timeout errors**: Increase function timeout in `vercel.json`
3. **Import errors**: Check that all dependencies are in `requirements.txt`
4. **File storage**: Use external storage service instead of local filesystem

