from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.ocr_engine import process_prescription

router = APIRouter()

@router.post("/extract-medicine")
async def extract_medicine(
    file: UploadFile = File(...),
    mode: str = Form(...) 
):
    if mode not in ["printed", "handwritten"]:
        raise HTTPException(status_code=400, detail="Mode must be 'printed' or 'handwritten'")
        
    try:
        image_bytes = await file.read()
        results = process_prescription(image_bytes, mode)
        return {
            "status": "success", 
            "mode": mode, 
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))