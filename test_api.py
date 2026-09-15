import requests

# The URL where your FastAPI server is running
API_URL = "http://127.0.0.1:8000/api/v1/extract-medicine"

# 1. Provide the path to a sample image of a prescription on your computer
# (Make sure to put a real image in your folder and change this name if needed)
image_path = "sample_prescription.jpg" 
mode = "printed" # Change to "handwritten" for cursive text

def test_ocr_api():
    try:
        with open(image_path, "rb") as image_file:
            # We must send it as multipart/form-data
            files = {"file": (image_path, image_file, "image/jpeg")}
            data = {"mode": mode}
            
            print(f"Sending {mode} image to API...")
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                print("\nSuccess! API Response:")
                print(response.json())
            else:
                print(f"Failed with status {response.status_code}: {response.text}")
                
    except FileNotFoundError:
        print(f"Error: Could not find the image '{image_path}'. Please add one to the folder.")

if __name__ == "__main__":
    test_ocr_api()