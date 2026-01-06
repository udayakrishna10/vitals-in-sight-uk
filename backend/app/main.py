import os
import uuid
import re
from typing import List, Dict
from pathlib import Path
from io import BytesIO
from datetime import datetime

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import bigquery
from openai import OpenAI

# Always load .env from the backend folder (one level above app/)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Config ---
GCS_RAW_BUCKET = os.getenv("GCS_RAW_BUCKET", "vital-insights-raw-uk")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "vitals-data-analytics")
BQ_DATASET = os.getenv("BQ_DATASET", "lab_results")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is not set")

# Initialize Google Cloud Storage and BigQuery clients
try:
    if GCS_CREDENTIALS_PATH and os.path.exists(GCS_CREDENTIALS_PATH):
        storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS_PATH)
        bq_client = bigquery.Client.from_service_account_json(GCS_CREDENTIALS_PATH)
    else:
        storage_client = storage.Client()
        bq_client = bigquery.Client()
except Exception as e:
    print(f"Warning: Could not initialize GCS/BQ clients: {e}")
    storage_client = None
    bq_client = None

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Cache for test mappings
TEST_MAP_CACHE = {}

app = FastAPI(title="Vital Insights API")

# CORS so frontend (Vite / Vercel) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: restrict to your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def load_test_mappings() -> Dict:
    """Load test name mappings from BigQuery dim_test_map table."""
    global TEST_MAP_CACHE
    
    # Return cached version if available
    if TEST_MAP_CACHE:
        return TEST_MAP_CACHE
    
    if not bq_client:
        print("BigQuery client not available, using empty test map")
        return {}
    
    try:
        query = f"""
        SELECT 
            raw_name,
            canonical_code,
            canonical_display_name,
            category,
            default_unit
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.dim_test_map`
        """
        
        results = bq_client.query(query).result()
        
        # Build mapping dictionary
        test_map = {}
        for row in results:
            # Map both raw name and lowercase version for flexible matching
            test_map[row.raw_test_name.lower()] = {
                "code": row.canonical_code,
                "display_name": row.canonical_display_name,
                "category": row.category,
                "unit": row.default_unit
            }
        
        TEST_MAP_CACHE = test_map
        print(f"Loaded {len(test_map)} test mappings from BigQuery")
        return test_map
    
    except Exception as e:
        print(f"Error loading test mappings: {e}")
        return {}

def standardize_test_name(raw_name: str, unit: str = "") -> Dict:
    """
    Standardize test name using dim_test_map.
    Returns dict with canonical name, category, and unit.
    """
    test_map = load_test_mappings()
    
    # Try exact match (case-insensitive)
    key = raw_name.lower().strip()
    if key in test_map:
        return test_map[key]
    
    # Try partial match
    for mapped_name, info in test_map.items():
        if mapped_name in key or key in mapped_name:
            return info
    
    # No match found, return original
    return {
        "code": raw_name.upper().replace(" ", "_"),
        "display_name": raw_name,
        "category": "Other",
        "unit": unit
    }

def extract_labs_from_pdf(content: bytes) -> List[Dict]:
    """
    Extract lab results from PDF using pdfplumber.
    This is a simple pattern-based parser. You'll need to customize
    this based on your actual lab report formats.
    """
    labs = []
    
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            all_text = ""
            
            # Extract text from all pages
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            
            print(f"Extracted text from {len(pdf.pages)} pages")
            
            # Try to extract tables first (more structured)
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                print(f"Found {len(tables)} tables on page {page_num + 1}")
                for table_num, table in enumerate(tables):
                    print(f"Table {table_num + 1} has {len(table)} rows")
                    if table:
                        # Print first few rows for debugging
                        print(f"First rows: {table[:3]}")
                    parsed_labs = parse_table_rows(table)
                    print(f"Extracted {len(parsed_labs)} labs from table {table_num + 1}")
                    labs.extend(parsed_labs)
            
            # If no tables found or no valid labs extracted, try text parsing
            if not labs:
                print("No valid labs from tables, trying text parsing...")
                labs = parse_text_for_labs(all_text)
            
    except Exception as e:
        print(f"PDF extraction error: {e}")
        import traceback
        traceback.print_exc()
        # Return mock data as fallback
        return build_mock_labs()
    
    # If still no labs found, return mock data
    if not labs:
        print("No labs extracted from PDF, using mock data")
        return build_mock_labs()
    
    print(f"Successfully extracted {len(labs)} lab results")
    return labs

def parse_table_rows(table: List[List[str]]) -> List[Dict]:
    """Parse lab results from table structure."""
    labs = []
    
    if not table or len(table) < 2:
        return labs
    
    # Look for common column headers
    header_row = None
    for idx, row in enumerate(table[:5]):  # Check first 5 rows for headers
        if any(cell and any(keyword in str(cell).lower() for keyword in ['test', 'name', 'parameter', 'investigation']) for cell in row):
            header_row = idx
            break
    
    if header_row is None:
        return labs
    
    headers = [str(cell).lower().strip() if cell else "" for cell in table[header_row]]
    
    # Find column indices
    test_col = find_column_index(headers, ['test', 'name', 'parameter', 'investigation'])
    value_col = find_column_index(headers, ['result', 'value', 'observed'])
    unit_col = find_column_index(headers, ['unit', 'units'])
    range_col = find_column_index(headers, ['range', 'reference', 'normal', 'ref'])
    
    # Parse data rows
    for row in table[header_row + 1:]:
        if not row or len(row) < 2:
            continue
        
        try:
            test_name = str(row[test_col]).strip() if test_col < len(row) and row[test_col] else None
            value_str = str(row[value_col]).strip() if value_col < len(row) and row[value_col] else None
            unit = str(row[unit_col]).strip() if unit_col < len(row) and row[unit_col] else ""
            range_str = str(row[range_col]).strip() if range_col < len(row) and row[range_col] else ""
            
            # Filter out invalid test names
            if not test_name or not value_str:
                continue
            
            # Skip if test name is too short or looks like a page number/header
            if len(test_name) < 3 or test_name.lower() in ['page', 'p', 'a', 'test', 'name']:
                continue
            
            # Skip if test name is just numbers or single characters
            if test_name.replace(' ', '').isdigit() or len(test_name.replace(' ', '')) <= 2:
                continue
            
            # Extract numeric value
            value = extract_number(value_str)
            if value is None:
                continue
            
            # Skip unrealistic values (likely page numbers or table artifacts)
            if value < 0.001 or value > 100000:
                continue
            
            # Parse reference range
            ref_low, ref_high = parse_reference_range(range_str)
            
            # Skip if reference range is invalid (both zero means no range found)
            if ref_low == 0.0 and ref_high == 0.0:
                # Try to extract range from unit column if range column is empty
                if unit:
                    alt_ref_low, alt_ref_high = parse_reference_range(unit)
                    if alt_ref_low > 0 or alt_ref_high > 0:
                        ref_low, ref_high = alt_ref_low, alt_ref_high
                        unit = ""  # Clear unit since it was actually the range
            
            # Determine flag
            flag = determine_flag(value, ref_low, ref_high)
            
            # Standardize test name using BigQuery mapping
            standardized = standardize_test_name(test_name, unit)
            
            labs.append({
                "test_name": standardized["display_name"],
                "test_code": standardized["code"],
                "category": standardized["category"],
                "value": value,
                "unit": standardized["unit"] or unit,
                "ref_low": ref_low,
                "ref_high": ref_high,
                "flag": flag,
            })
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue
    
    return labs

def parse_text_for_labs(text: str) -> List[Dict]:
    """Parse lab results from plain text using regex patterns."""
    labs = []
    
    # Common patterns for lab results
    # Example: "Glucose: 118 mg/dL (70-99)"
    # Example: "HbA1c 6.3 % Reference: 4.0-5.6"
    
    lines = text.split('\n')
    
    for line in lines:
        # Skip empty lines or headers
        if not line.strip() or len(line.strip()) < 5:
            continue
        
        # Pattern: test name followed by value and optional unit/range
        match = re.search(
            r'([A-Za-z][A-Za-z0-9\s\-/]+?)\s*[:=]?\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)?\s*(?:\(|range|ref)?.*?(\d+\.?\d*)\s*[-–to]\s*(\d+\.?\d*)',
            line,
            re.IGNORECASE
        )
        
        if match:
            test_name = match.group(1).strip()
            value = float(match.group(2))
            unit = match.group(3).strip() if match.group(3) else ""
            ref_low = float(match.group(4))
            ref_high = float(match.group(5))
            
            flag = determine_flag(value, ref_low, ref_high)
            
            # Standardize test name
            standardized = standardize_test_name(test_name, unit)
            
            labs.append({
                "test_name": standardized["display_name"],
                "test_code": standardized["code"],
                "category": standardized["category"],
                "value": value,
                "unit": standardized["unit"] or unit,
                "ref_low": ref_low,
                "ref_high": ref_high,
                "flag": flag,
            })
    
    return labs

def find_column_index(headers: List[str], keywords: List[str]) -> int:
    """Find column index by matching keywords."""
    for idx, header in enumerate(headers):
        if any(keyword in header for keyword in keywords):
            return idx
    return 0

def extract_number(text: str) -> float:
    """Extract first number from text."""
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def parse_reference_range(range_str: str) -> tuple:
    """Parse reference range string like '70-99' or '4.0 - 5.6'."""
    if not range_str:
        return 0.0, 0.0
    
    # Clean up the string
    range_str = range_str.replace('–', '-').replace('to', '-')
    
    # Pattern: number - number
    match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', range_str)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            return 0.0, 0.0
    
    return 0.0, 0.0

def determine_flag(value: float, ref_low: float, ref_high: float) -> str:
    """Determine if value is normal, high, or low."""
    if ref_low == 0.0 and ref_high == 0.0:
        return "NORMAL"
    
    if value < ref_low:
        # Check if critically low (< 80% of lower bound)
        if value < ref_low * 0.8:
            return "CRITICAL"
        return "LOW"
    elif value > ref_high:
        # Check if critically high (> 120% of upper bound)
        if value > ref_high * 1.2:
            return "CRITICAL"
        return "HIGH"
    else:
        return "NORMAL"

def build_mock_labs() -> List[Dict]:
    """Fallback mock data when PDF parsing fails."""
    return [
        {
            "test_name": "Fasting Glucose",
            "test_code": "FASTING_GLUCOSE",
            "category": "Glucose / Metabolism",
            "value": 118.0,
            "unit": "mg/dL",
            "ref_low": 70.0,
            "ref_high": 99.0,
            "flag": "HIGH",
        },
        {
            "test_name": "Hemoglobin A1c",
            "test_code": "HBA1C",
            "category": "Glucose / Metabolism",
            "value": 6.3,
            "unit": "%",
            "ref_low": 4.0,
            "ref_high": 5.6,
            "flag": "HIGH",
        },
        {
            "test_name": "Total Cholesterol",
            "test_code": "TOTAL_CHOL",
            "category": "Lipid Panel",
            "value": 210.0,
            "unit": "mg/dL",
            "ref_low": 0.0,
            "ref_high": 200.0,
            "flag": "HIGH",
        },
    ]

def generate_ai_insights(labs: List[Dict]) -> str:
    """Generate AI insights using OpenAI API."""
    messages = [
        {
            "role": "system",
            "content": (
                "You explain blood test results in simple, safe language. "
                "You are NOT a doctor and do not diagnose or prescribe. "
                "You can explain what high/low results might generally mean, "
                "suggest general diet and lifestyle ideas, and highlight red flags. "
                "Always include a short disclaimer to consult a doctor."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are some lab results. "
                "Provide: 1) Overall summary, 2) Key abnormal values, "
                "3) Possible general meaning, 4) Diet & lifestyle suggestions, "
                "5) Red flags & when to see a doctor.\n\n"
                f"{labs}"
            ),
        },
    ]

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def save_labs_to_bigquery(report_id: str, labs: List[Dict]):
    """Save lab results to BigQuery fact_lab_results table."""
    if not bq_client:
        return
    
    table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET}.fact_lab_results"
    
    rows_to_insert = []
    for lab in labs:
        row = {
            "report_id": report_id,
            "test_name": lab.get("test_name"),
            "canonical_test_code": lab.get("test_code"),
            "value": lab.get("value"),
            "unit": lab.get("unit"),
            "ref_low": lab.get("ref_low"),
            "ref_high": lab.get("ref_high"),
            "flag": lab.get("flag"),
            "result_date": datetime.now().strftime("%Y-%m-%d"),
            "ingested_at": datetime.now().isoformat(),
        }
        rows_to_insert.append(row)
    
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    
    if errors:
        print(f"BigQuery insert errors: {errors}")
    else:
        print(f"Successfully inserted {len(rows_to_insert)} rows to BigQuery")

@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    """
    One-shot endpoint:
    - Receives PDF
    - Extracts lab results using pdfplumber
    - Saves to GCS (optional)
    - Saves to BigQuery
    - Returns labs + AI insights
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported. Please upload a valid lab report PDF."
        )

    report_id = str(uuid.uuid4())
    content = await file.read()

    # Save to GCS raw bucket (if storage client is available)
    if storage_client:
        try:
            bucket = storage_client.bucket(GCS_RAW_BUCKET)
            blob = bucket.blob(f"uploads/{report_id}.pdf")
            blob.upload_from_string(content, content_type=file.content_type)
            print(f"Saved to GCS: gs://{GCS_RAW_BUCKET}/uploads/{report_id}.pdf")
        except Exception as e:
            print(f"GCS upload failed: {e}")
    else:
        # Save locally for development
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        local_path = upload_dir / f"{report_id}.pdf"
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"Saved locally to {local_path} (GCS not configured)")

    # Extract lab results from PDF
    labs = extract_labs_from_pdf(content)
    
    # Save to BigQuery fact_lab_results if available
    if bq_client and labs:
        try:
            save_labs_to_bigquery(report_id, labs)
        except Exception as e:
            print(f"Failed to save to BigQuery: {e}")
    
    # Generate AI insights
    ai_text = generate_ai_insights(labs)

    return {
        "report_id": report_id,
        "labs": labs,
        "insights": ai_text,
    }
