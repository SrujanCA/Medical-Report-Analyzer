import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
import requests
import json
import logging

import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Auto-discover Tesseract path on Windows
def init_tesseract():
    tess_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        os.getenv('TESSERACT_PATH', '')
    ]
    for path in tess_paths:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract found at: {path}")
            return path
    which_p = shutil.which("tesseract")
    if which_p:
        pytesseract.pytesseract.tesseract_cmd = which_p
        logger.info(f"Tesseract found via PATH: {which_p}")
        return which_p
    logger.warning("Tesseract not found in standard system paths.")
    return None

init_tesseract()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# LLM Providers Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "deepseek-r1:14b"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

def get_available_ollama_model():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get('name', '') for m in r.json().get('models', [])]
            if models:
                for m in models:
                    if 'deepseek-r1:14b' in m:
                        return m
                for m in models:
                    if 'deepseek' in m:
                        return m
                return models[0]
    except Exception:
        pass
    return DEFAULT_MODEL_NAME

def generate_ai_analysis(prompt, system_prompt="You are an expert medical AI advisor."):
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    
    # 1. NVIDIA Nemotron / NIM API
    if nvidia_key:
        try:
            from openai import OpenAI
            model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct").strip()
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)
            logger.info(f"Using NVIDIA NIM API (Model: {model})")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"NVIDIA API Error: {e}. Falling back to Ollama local AI.")

    # 2. Local Ollama AI
    model = get_available_ollama_model()
    data = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2000
        }
    }
    logger.info(f"Using Ollama local AI (Model: {model})")
    response = requests.post(OLLAMA_ENDPOINT, json=data, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get('response', 'No analysis generated')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from win_ocr import run_win_ocr

def extract_text_from_image(image_path):
    # 1. Try PyTesseract if available
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        if text and text.strip():
            return text
    except Exception as e:
        logger.warning(f"PyTesseract extraction warning: {e}")

    # 2. Try Windows Native OCR (WinRT)
    try:
        win_text = run_win_ocr(image_path)
        if win_text and win_text.strip():
            logger.info("Successfully extracted text using Windows Native OCR!")
            return win_text
    except Exception as e:
        logger.warning(f"Windows Native OCR warning: {e}")

    return "Error: Tesseract OCR is not installed and built-in image OCR could not read this file. Please paste your report text directly into the text field below or upload a text PDF."

def extract_text_from_pdf(pdf_path):
    # Try native PDF text extraction with PyPDF first
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        extracted = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                extracted.append(t)
        if extracted:
            return "\n".join(extracted)
    except Exception as e:
        logger.warning(f"PyPDF extraction fallback: {e}")

    # Fallback to pdf2image + pytesseract OCR
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        if text and text.strip():
            return text
    except Exception as e:
        logger.warning(f"PDF OCR warning: {e}")

    return "Error: Could not extract text from PDF. Please upload a text-based PDF or paste the report text directly."

def translate_to_bangla(text):
    """Return a placeholder translation message for preview mode."""
    if not text:
        return "No text to translate"
    return "Translation feature is unavailable in this preview environment."

def is_connection_error(e):
    err_str = str(e).lower()
    return isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) or \
        'connection refused' in err_str or '10061' in err_str or 'actively refused' in err_str or 'max retries exceeded' in err_str

def analyze_medical_report(text):
    try:
        system_prompt = """You are an expert medical report analyzer AI. Analyze the provided medical test report and format your response strictly using the following clear sections with highlighted key statements:

# Medical Report Analysis

## 1. Test Result Analysis
- **Test Performed**: State the exact test name (e.g., Glycosylated Haemoglobin [HbA1c] via HPLC).
- **Result**: State the exact value and unit (e.g., 121.6 mg/dl Derived Mean Blood Glucose over 6-8 weeks).
- **Reference Ranges for HbA1c**:
  > **Normal**: < 5.7% (equivalent to ~95 mg/dl average glucose)
  > **Prediabetes**: 5.7% – 6.4% (equivalent to ~117–137 mg/dl average glucose)
  > **Diabetes**: ≥ 6.5% (equivalent to ≥ 140 mg/dl average glucose)

## AI Summary
Provide a concise, 1-2 sentence summary explaining the core test result in plain language (e.g. "The derived average blood glucose level is 121.6 mg/dl, which falls within the prediabetes range. Please consult a healthcare provider for further interpretation."). Keep it brief, direct, and free of repetitive technical disclaimers.

## What to Reduce
- **Sugary Foods & Drinks**: Limit sweets, soft drinks, sweetened tea/coffee, and fruit juices.
- **Refined Carbohydrates**: Reduce portions of white rice, refined flour (maida), sweets, and refined snacks.

## Recommended Health Actions
- Provide clear, actionable dietary, exercise, and follow-up guidance."""

        prompt = f"Here is the medical report to analyze:\n\n{text}"
        
        english_analysis = generate_ai_analysis(prompt, system_prompt)

        return {
            'english': english_analysis,
            'bangla': None
        }

    except Exception as e:
        if is_connection_error(e):
            logger.warning("AI Connection failed. Returning diagnostic & fallback response.")
            extracted_snippet = (text[:300] + "...") if len(text) > 300 else text
            fallback_text = f"""### ⚠️ AI Server Connection Warning

**Notice:** Could not connect to local AI or NVIDIA API endpoint.

To enable live AI analysis:
- **NVIDIA Nemotron API**: Set `NVIDIA_API_KEY` in your `.env` file.
- **Local Ollama**: Ensure Ollama is running (`ollama serve`).

---

### 📋 Extracted Report Content
**Extracted Text:**
> {extracted_snippet if extracted_snippet.strip() else "No text extracted."}

### 📊 Preliminary Summary & Advice
1. **Report Status:** Text extracted from document.
2. **General Recommendation:** Please consult a healthcare professional to evaluate your official medical lab results."""
            return {
                'english': fallback_text,
                'bangla': None
            }
        return {
            'success': False,
            'error': f"Error analyzing report: {str(e)}"
        }

def analyze_symptoms(symptoms):
    try:
        system_prompt = """You are a medical advisor. Based on the symptoms, please:
1. Analyze the symptoms and provide potential conditions
2. Rate the urgency level (Low/Medium/High)
3. Suggest immediate steps or precautions
4. Recommend when to seek professional medical help

Please note this is for informational purposes only and not a substitute for professional medical advice."""

        prompt = f"Symptoms:\n{symptoms}"
        
        english_analysis = generate_ai_analysis(prompt, system_prompt)
        medline_info = "\n\nFor more detailed medical information, please visit: https://medlineplus.gov/"

        return {
            'english': english_analysis + medline_info,
            'bangla': None
        }

    except Exception as e:
        if is_connection_error(e):
            logger.warning("AI Connection failed for symptoms analysis.")
            fallback_text = f"""### ⚠️ AI Server Connection Warning

**Notice:** Could not connect to local AI or NVIDIA API endpoint.

---

### 🔍 Symptoms Assessment (Fallback)
**Entered Symptoms:** "{symptoms}"

1. **Urgency Assessment:** 🟡 **Medium** (Monitor closely)
2. **General Precautions:** Rest adequately and maintain proper hydration.
3. **Medical Disclaimer:** Seek emergency care if symptoms worsen or severe pain/fever occurs.

For more detailed medical information, please visit: https://medlineplus.gov/"""
            return {
                'english': fallback_text,
                'bangla': None
            }
        return {
            'success': False,
            'error': f"Error analyzing symptoms: {str(e)}"
        }

def analyze_medicine(medicine_name, dosage, patient):
    try:
        dosage_str = []
        if dosage.get('morning', 0) > 0:
            dosage_str.append(f"{dosage['morning']} tablet(s) in the morning")
        if dosage.get('evening', 0) > 0:
            dosage_str.append(f"{dosage['evening']} tablet(s) in the evening")
        if dosage.get('night', 0) > 0:
            dosage_str.append(f"{dosage['night']} tablet(s) at night")
        
        formatted_dosage = ", ".join(dosage_str) if dosage_str else "As directed"

        system_prompt = "You are a medical information advisor."
        prompt = f"""Please analyze the following medicine and dosage for a patient:

Patient Information:
- Age: {patient['age']} years old
- Gender: {patient['gender']}

Medicine Name: {medicine_name}
Current Dosage: {formatted_dosage}

Provide:
1. Primary uses
2. Side effects (common to severe)
3. Recommended dosage comparison
4. Warnings & drug interactions
5. When to seek medical attention"""

        english_analysis = generate_ai_analysis(prompt, system_prompt)
        english_analysis += "\n\nFor more detailed medical information, please visit: https://medlineplus.gov/druginformation.html"

        return {
            'english': english_analysis,
            'bangla': None
        }

    except Exception as e:
        if is_connection_error(e):
            logger.warning("AI Connection failed for medicine analysis.")
            fallback_text = f"""### ⚠️ AI Server Connection Warning

**Notice:** Could not connect to local AI or NVIDIA API endpoint.

---

### 💊 Medicine Information (Fallback)
**Medicine:** {medicine_name}  
**Patient Profile:** {patient.get('age')} yrs, {patient.get('gender')}  
**Dosage:** {formatted_dosage}

1. **General Usage:** Consult your pharmacist or prescribing doctor regarding `{medicine_name}`.

For more detailed medical information, please visit: https://medlineplus.gov/druginformation.html"""
            return {
                'english': fallback_text,
                'bangla': None
            }
        return {
            'success': False,
            'error': f"Error analyzing medicine: {str(e)}"
        }



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved: {filepath}")

            try:
                # Determine extraction method from file extension
                ext = filename.rsplit('.', 1)[1].lower()
                if ext == 'pdf':
                    text = extract_text_from_pdf(filepath)
                else:
                    text = extract_text_from_image(filepath)

                # Clean up the uploaded file
                os.remove(filepath)
                logger.info("File processed and removed")

                # Analyze the extracted text
                logger.info("Starting text analysis")
                analysis = analyze_medical_report(text)
                logger.info("Analysis completed")
                
                if isinstance(analysis, dict) and 'error' in analysis:
                    return jsonify(analysis), 500
                
                return jsonify({
                    'success': True,
                    'analysis': analysis
                })
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500

        logger.error("Invalid file type from extension check")
        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/analyze-symptoms', methods=['POST'])
def process_symptoms():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            logger.error("No symptoms provided in request")
            return jsonify({'error': 'No symptoms provided'}), 400

        symptoms = data['symptoms']
        if not symptoms.strip():
            logger.error("Empty symptoms string provided")
            return jsonify({'error': 'Symptoms cannot be empty'}), 400

        logger.info("Starting symptoms analysis")
        analysis = analyze_symptoms(symptoms)
        logger.info("Symptoms analysis completed")
        
        if isinstance(analysis, dict) and 'error' in analysis:
            return jsonify(analysis), 500
            
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Error in process_symptoms: {str(e)}")
        return jsonify({'error': f'Error processing symptoms: {str(e)}'}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            logger.error("No text provided for translation")
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text.strip():
            logger.error("Empty text provided for translation")
            return jsonify({'error': 'Text cannot be empty'}), 400

        logger.info("Starting translation")
        translated_text = translate_to_bangla(text)
        logger.info("Translation completed")
        
        return jsonify({
            'success': True,
            'translation': translated_text
        })
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error during translation: {str(e)}'
        }), 500

@app.route('/analyze-medicine', methods=['POST'])
def process_medicine():
    try:
        data = request.get_json()
        if not data or 'medicine' not in data or 'dosage' not in data or 'patient' not in data:
            logger.error("Missing medicine information")
            return jsonify({'error': 'Missing required information'}), 400

        medicine = data['medicine'].strip()
        dosage = data['dosage']
        patient = data['patient']

        if not medicine:
            logger.error("Empty medicine name provided")
            return jsonify({'error': 'Medicine name cannot be empty'}), 400

        # Validate dosage format
        required_fields = ['morning', 'evening', 'night']
        if not all(field in dosage for field in required_fields):
            logger.error("Invalid dosage format")
            return jsonify({'error': 'Invalid dosage format'}), 400

        # Validate patient information
        if not isinstance(patient.get('age'), int) or patient['age'] <= 0:
            logger.error("Invalid patient age")
            return jsonify({'error': 'Invalid patient age'}), 400

        if not patient.get('gender') or patient['gender'] not in ['male', 'female', 'other']:
            logger.error("Invalid patient gender")
            return jsonify({'error': 'Invalid patient gender'}), 400

        logger.info(f"Starting medicine analysis for: {medicine} (Patient: {patient['age']}y, {patient['gender']})")
        analysis = analyze_medicine(medicine, dosage, patient)
        logger.info("Medicine analysis completed")
        
        if isinstance(analysis, dict) and 'error' in analysis:
            return jsonify(analysis), 500
            
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Error in process_medicine: {str(e)}")
        return jsonify({'error': f'Error processing medicine information: {str(e)}'}), 500

@app.route('/analyze-report-text', methods=['POST'])
def process_report_text():
    try:
        data = request.get_json()
        if not data or 'report_text' not in data:
            logger.error("No report text provided in request")
            return jsonify({'error': 'No report text provided'}), 400

        report_text = data['report_text'].strip()
        if not report_text:
            logger.error("Empty report text provided")
            return jsonify({'error': 'Report text cannot be empty'}), 400

        logger.info("Starting direct report text analysis")
        analysis = analyze_medical_report(report_text)
        logger.info("Report text analysis completed")
        
        if isinstance(analysis, dict) and 'error' in analysis:
            return jsonify(analysis), 500
            
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Error in process_report_text: {str(e)}")
        return jsonify({'error': f'Error processing report text: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 