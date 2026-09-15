import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Medical OCR API", 
    description="Dual-brain microservice for printed and handwritten prescriptions."
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)