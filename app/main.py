from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import summary

app = FastAPI(
    title="Course Summary AI",
    description="AI-powered course material summarization service",
    version="1.0.0"
)

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(summary.router)

@app.get("/")
async def root():
    return {
        "service": "Course Summary AI",
        "version": "1.0.0",
        "endpoints": {
            "generate_summary": "/api/v1/summary/generate (POST)"
        },
        "notes": "Using Hugging Face Inference API - no local models required"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)