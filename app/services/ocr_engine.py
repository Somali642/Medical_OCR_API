import io
import re
import string
import difflib
import pandas as pd
from PIL import Image
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1. Load Local TrOCR Models (Dual-Brain Pipeline)
processor_printed = TrOCRProcessor.from_pretrained('./trocr-model-flat')
model_printed = VisionEncoderDecoderModel.from_pretrained('./trocr-model-flat')

processor_hw = TrOCRProcessor.from_pretrained('./trocr-handwritten-flat')
model_hw = VisionEncoderDecoderModel.from_pretrained('./trocr-handwritten-flat')

# 2. Text Detection with EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# 3. Load CSV Database for Validation
try:
    med_db = pd.read_csv('medicine_dataset.csv')
    med_list = med_db['medicine_name'].dropna().astype(str).tolist()
except Exception as e:
    med_list = []

def scrub_punctuation(text: str) -> str:
    """Scrub punctuation and normalize casing to eliminate fuzzy match false positives."""
    return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

def extract_strength(text: str) -> str:
    """Extract standard dosage strengths using regex."""
    match = re.search(r'[0-9]+[ ]*(mg|ml|g|mcg)', text, re.IGNORECASE)
    return match.group(0) if match else None

def validate_medicine(ocr_text: str):
    """Perform fuzzy matching against database and pull substitutes."""
    clean_text = scrub_punctuation(ocr_text)
    
    matches = difflib.get_close_matches(clean_text, med_list, n=3, cutoff=0.6)
    
    if not matches:
        return {"matched": False, "raw_text": ocr_text}
        
    primary_match = matches[0]
    substitutes = matches[1:]
    strength = extract_strength(ocr_text)
    
    return {
        "matched": True,
        "raw_text": ocr_text,
        "validated_name": primary_match,
        "strength": strength,
        "substitutes": substitutes
    }

def process_prescription(image_bytes: bytes, mode: str):
    """Pipeline: EasyOCR BBoxes -> TrOCR Generation -> Fuzzy Validation"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bounds = reader.readtext(image_bytes, detail=1)
    results = []
    
    processor = processor_hw if mode == 'handwritten' else processor_printed
    model = model_hw if mode == 'handwritten' else model_printed
    
    for (bbox, text, prob) in bounds:
        x_min = min([pt[0] for pt in bbox])
        y_min = min([pt[1] for pt in bbox])
        x_max = max([pt[0] for pt in bbox])
        y_max = max([pt[1] for pt in bbox])
        
        crop_img = image.crop((x_min, y_min, x_max, y_max))
        
        pixel_values = processor(images=crop_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        validation = validate_medicine(trocr_text)
        results.append(validation)
        
    return results