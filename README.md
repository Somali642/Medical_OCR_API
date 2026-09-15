# Medical OCR API 🏥

A dual-brain Optical Character Recognition (OCR) microservice built with FastAPI. This API processes medical prescriptions by using **EasyOCR** for bounding-box text detection and Microsoft's **TrOCR** (Transformer-based OCR) for highly accurate text recognition of both printed and handwritten medical text. 

Extracted text is fuzzy-matched against a large-scale medicine database to return validated drug names, strengths, and standard substitutes.

## Tech Stack
* **Framework:** FastAPI / Uvicorn
* **Detection:** EasyOCR
* **Recognition:** HuggingFace `transformers` (TrOCR)
* **Validation:** Python `difflib` and `pandas`

## 🚀 Installation & Setup

### 1. Clone the repository

git clone https://github.com/Somali642/Medical_OCR_API.git
cd "Medical OCR API"


### 2. Set up the Python Environment
*Note: Built and tested using Python 3.12.*

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt


### 3. Download the TrOCR Models
This project requires Microsoft's Base Printed and Handwritten TrOCR models. Run the setup script to download them to your local directory:

python setup_models.py


### 4. Build the Medicine Database
Add a medicine dataset (e.g., from Kaggle) as a `.csv` file to the root directory, then run the formatter to generate the expected `medicine_dataset.csv`:

python format_dataset.py


## ⚙️ Running the Server
Start the FastAPI server:

python -m uvicorn main:app --reload

The API will be available at `http://127.0.0.1:8000`. You can view the interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.

## 🧪 Testing the API
You can test the extraction endpoint using the provided test script. Ensure you have a sample image named `sample_prescription.jpg` in the root folder.

python test_api.py