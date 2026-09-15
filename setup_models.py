import os
from transformers import (
    VisionEncoderDecoderModel, 
    TrOCRProcessor, 
    RobertaTokenizer, 
    AutoImageProcessor
)

def download_and_save_models():
    print("Downloading Printed Text Model...")
    
    # 1. Manually initialize the exact tokenizer and image processor 
    tokenizer_printed = RobertaTokenizer.from_pretrained('microsoft/trocr-base-printed')
    image_processor_printed = AutoImageProcessor.from_pretrained('microsoft/trocr-base-printed')
    model_printed = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    
    # 2. Combine them into the TrOCRProcessor
    processor_printed = TrOCRProcessor(image_processor=image_processor_printed, tokenizer=tokenizer_printed)
    
    # 3. Save to local directory
    os.makedirs('./trocr-model-flat', exist_ok=True)
    processor_printed.save_pretrained('./trocr-model-flat')
    model_printed.save_pretrained('./trocr-model-flat')

    print("Downloading Handwritten Text Model...")
    
    tokenizer_hw = RobertaTokenizer.from_pretrained('microsoft/trocr-base-handwritten')
    image_processor_hw = AutoImageProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    processor_hw = TrOCRProcessor(image_processor=image_processor_hw, tokenizer=tokenizer_hw)
    
    os.makedirs('./trocr-handwritten-flat', exist_ok=True)
    processor_hw.save_pretrained('./trocr-handwritten-flat')
    model_hw.save_pretrained('./trocr-handwritten-flat')
    
    print("Models successfully saved to local directories!")

if __name__ == "__main__":
    download_and_save_models()