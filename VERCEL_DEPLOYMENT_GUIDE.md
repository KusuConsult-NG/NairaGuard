# 🚀 Vercel Deployment Guide for NairaGuard

## 📋 Overview

This guide covers deploying the NairaGuard Fake Naira Detection app to Vercel, including both frontend and backend API deployment.

## 🏗️ Project Structure for Vercel

```
nairaguard/
├── vercel.json                 # Vercel configuration
├── api/                        # Vercel API routes
│   ├── index.py               # Main API handler
│   └── predict.py             # Prediction endpoint
├── frontend/                   # React frontend
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
├── backend/                    # FastAPI backend (for reference)
└── vercel.env.example         # Environment variables template
```

## 🔧 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code must be on GitHub
3. **Node.js**: For local development (v18+)
4. **Python**: For local development (v3.9+)

## 📦 Installation & Setup

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Link Your Project

```bash
cd /path/to/nairaguard
vercel link
```

## ⚙️ Configuration

### 1. Environment Variables

Copy `vercel.env.example` and configure in Vercel dashboard:

```bash
# Essential variables
APP_NAME=NairaGuard
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
DATABASE_URL=sqlite:///./fake_naira_detection.db
```

### 2. Vercel Configuration (`vercel.json`)

The `vercel.json` file is already configured with:
- Frontend build settings
- API route configuration
- Python runtime settings
- CORS configuration

## 🚀 Deployment Methods

### Method 1: Vercel CLI (Recommended)

```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Method 2: GitHub Integration

1. **Connect Repository**:
   - Go to Vercel Dashboard
   - Click "New Project"
   - Import from GitHub
   - Select your repository

2. **Configure Build Settings**:
   - Framework Preset: `Other`
   - Root Directory: `./`
   - Build Command: `cd frontend && npm run build`
   - Output Directory: `frontend/dist`

3. **Environment Variables**:
   - Add variables from `vercel.env.example`
   - Set production values

4. **Deploy**:
   - Click "Deploy"
   - Wait for build to complete

## 🌐 API Endpoints

After deployment, your API will be available at:

```
https://your-app.vercel.app/api/
```

### Available Endpoints:

- `GET /api/` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/predict` - Mock prediction (Vercel-compatible)
- `GET /api/model/status` - Model status
- `GET /api/predictions/history` - Mock prediction history

## 🔧 Frontend Configuration

### 1. Update API Base URL

Update `frontend/src/services/api.ts`:

```typescript
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-app.vercel.app/api'
  : 'http://localhost:8000';
```

### 2. Build Configuration

The `vite.config.ts` is configured for Vercel deployment:

```typescript
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

## 📊 Vercel-Specific Considerations

### ✅ What Works on Vercel:

- **Frontend**: Full React app deployment
- **API Routes**: Serverless functions
- **Static Assets**: Images, CSS, JS
- **Environment Variables**: Secure configuration
- **Custom Domains**: Professional URLs
- **SSL**: Automatic HTTPS

### ⚠️ Limitations:

- **ML Models**: Large models may not work (use mock predictions)
- **Database**: No persistent storage (use external DB)
- **File Uploads**: Limited to 4.5MB
- **Execution Time**: 10-second limit for serverless functions
- **Memory**: 1GB limit per function

### 🔄 Workarounds:

1. **ML Models**: Use mock predictions or external ML service
2. **Database**: Use Vercel Postgres or external database
3. **File Storage**: Use Vercel Blob or external storage
4. **Long Tasks**: Use background jobs or external services

## 🧪 Testing Deployment

### 1. Local Testing

```bash
# Test frontend
cd frontend
npm run build
npm run preview

# Test API
cd api
python index.py
```

### 2. Production Testing

```bash
# Test deployed app
curl https://your-app.vercel.app/api/health
curl https://your-app.vercel.app/api/
```

## 🔍 Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check Node.js version (use 18+)
   - Verify all dependencies in `package.json`
   - Check build logs in Vercel dashboard

2. **API Errors**:
   - Verify environment variables
   - Check function logs
   - Test endpoints individually

3. **CORS Issues**:
   - Verify CORS configuration in `vercel.json`
   - Check frontend API base URL

4. **Import Errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

### Debug Commands:

```bash
# Check Vercel status
vercel ls

# View deployment logs
vercel logs

# Inspect deployment
vercel inspect
```

## 📈 Performance Optimization

### 1. Frontend Optimization:

- Enable Vercel's automatic optimizations
- Use Vercel's Image Optimization
- Implement code splitting
- Enable compression

### 2. API Optimization:

- Use Vercel's Edge Functions for simple operations
- Implement caching where possible
- Optimize payload sizes
- Use streaming for large responses

## 🔐 Security Considerations

### 1. Environment Variables:

- Never commit secrets to Git
- Use Vercel's environment variable system
- Rotate keys regularly
- Use different keys for different environments

### 2. API Security:

- Implement rate limiting
- Use HTTPS only
- Validate all inputs
- Implement proper error handling

## 📊 Monitoring & Analytics

### 1. Vercel Analytics:

- Enable Vercel Analytics
- Monitor performance metrics
- Track user behavior
- Set up alerts

### 2. Custom Monitoring:

- Implement health checks
- Log important events
- Monitor API response times
- Track error rates

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] Custom domain set up (optional)
- [ ] SSL certificate active
- [ ] Analytics enabled
- [ ] Error monitoring set up
- [ ] Performance monitoring active
- [ ] Backup strategy implemented
- [ ] Documentation updated

## 📞 Support

- **Vercel Documentation**: [vercel.com/docs](https://vercel.com/docs)
- **Vercel Community**: [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)
- **Project Issues**: [github.com/KusuConsult-NG/NairaGuard/issues](https://github.com/KusuConsult-NG/NairaGuard/issues)

## 🎯 Next Steps

1. **Deploy**: Follow the deployment steps above
2. **Test**: Verify all functionality works
3. **Monitor**: Set up monitoring and analytics
4. **Optimize**: Improve performance based on usage
5. **Scale**: Add more features as needed

---

**🎉 Your NairaGuard app is now ready for Vercel deployment!**
