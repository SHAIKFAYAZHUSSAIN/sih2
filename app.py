"""
LMPC SmartInspector - FastAPI Application
Legal Metrology Compliance Checking System for Packaged Commodities
Problem Statement: SIH 26034
"""

import os
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import database
from lmpc_rule_engine import LMPCRuleEngine
from lmpc_extractor import LMPCExtractor
from report_generator import generate_inspection_pdf

# Setup directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Determine writable directory for uploads and reports
def _get_writable_dir(name: str) -> str:
    # On serverless (Vercel, AWS Lambda), always use /tmp
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        p = os.path.join("/tmp", name)
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            return "/tmp"
    
    # Try local base directory first
    p = os.path.join(BASE_DIR, name)
    try:
        os.makedirs(p, exist_ok=True)
        return p
    except Exception:
        p = os.path.join("/tmp", name)
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            return "/tmp"

UPLOADS_DIR = _get_writable_dir("uploads")
REPORTS_DIR = _get_writable_dir("reports")

# Initialize Database safely
try:
    database.init_db()
except Exception as e:
    print(f"Warning: Database init error ({e}). Continuing with fallback.")

# Initialize AI & Rule Services
rule_engine = LMPCRuleEngine()
extractor = LMPCExtractor()

app = FastAPI(
    title="LMPC SmartInspector",
    description="Automated Legal Metrology Compliance Inspection Platform",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def vercel_path_normalization(request: Request, call_next):
    """
    Normalizes ASGI request paths across various Vercel deployment modes:
    - Zero-config api/index.py routing
    - vercel.json routes and rewrites
    - Injected edge headers (x-matched-path, x-forwarded-uri)
    """
    # 1. Check if Vercel edge injected the original matched URL path
    for header_name in ("x-matched-path", "x-forwarded-uri", "x-original-uri", "x-rewrite-url"):
        header_val = request.headers.get(header_name)
        if header_val and not header_val.startswith("/api/index"):
            request.scope["path"] = header_val.split("?")[0]
            break
            
    # 2. Normalize function path prefixes if request arrived via api/index.py
    p = request.scope.get("path", "")
    if p in ("/api/index.py", "/api/index", "/api/index/", ""):
        request.scope["path"] = "/"
    elif p.startswith("/api/index.py/"):
        request.scope["path"] = p[len("/api/index.py"):]
    elif p.startswith("/api/index/"):
        request.scope["path"] = p[len("/api/index"):]

    return await call_next(request)

# Mount static directory for sample assets if physical folder exists
if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except Exception as e:
        print(f"Notice: Could not mount /static ({e})")

def get_dashboard_html() -> str:
    # 1. Search filesystem paths
    for p in [
        os.path.join(TEMPLATES_DIR, "index.html"),
        os.path.join(BASE_DIR, "templates", "index.html"),
        os.path.join(os.getcwd(), "templates", "index.html")
    ]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    # 2. Use embedded fallback template
    try:
        from embedded_template import DASHBOARD_HTML
        return DASHBOARD_HTML
    except Exception:
        return "<h1>LMPC SmartInspector</h1><p>Dashboard template loaded in minimal fallback mode.</p>"

@app.get("/api/health")
async def health_check():
    """Diagnostic health check endpoint"""
    return {
        "status": "healthy",
        "system": "LMPC SmartInspector",
        "env_vercel": bool(os.environ.get("VERCEL")),
        "uploads_dir": UPLOADS_DIR,
        "reports_dir": REPORTS_DIR,
        "samples_loaded": len(extractor.preloaded_samples)
    }

@app.get("/", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=get_dashboard_html())

@app.get("/api/samples")
async def get_samples():
    """Returns catalog of preloaded packaging inspection samples"""
    samples = []
    for sid, sdata in extractor.preloaded_samples.items():
        samples.append({
            "id": sdata.get("id", sid),
            "title": sdata.get("title", sid),
            "category": sdata.get("category", "Packaged Commodity"),
            "description": sdata.get("description", ""),
            "expected_verdict": sdata.get("expected_verdict", "INSPECT"),
            "image_url": sdata.get("image_rel_path", "")
        })
    return {"samples": samples}

@app.get("/api/scan/sample/{sample_id}")
@app.post("/api/scan/demo/{sample_id}")
@app.get("/api/scan/demo/{sample_id}")
async def scan_sample(sample_id: str):
    """Instant execution for preset demonstration benchmark labels"""
    extraction = extractor.extract_from_sample_id(sample_id)
    extracted_data = extraction["extracted_data"]
    is_electronic = sample_id == "sample_05" or "electronics" in sample_id or extracted_data.get("has_qr_code", False)
    
    # Run Statutory Rule Engine
    eval_result = rule_engine.evaluate(extracted_data, is_electronic=is_electronic)
    
    # Persist to Database
    scan_id = database.save_scan(
        extracted=extracted_data,
        result=eval_result.dict(),
        image_path=extraction["image_url"]
    )
    
    return {
        "scan_id": scan_id,
        "image_url": extraction["image_url"],
        "extracted_data": extracted_data,
        "bounding_boxes": extraction["bounding_boxes"],
        "compliance_result": eval_result.dict()
    }

@app.post("/api/scan/upload")
async def scan_uploaded_image(
    image: UploadFile = File(...),
    ocr_json: Optional[str] = Form(None)
):
    """Uploads an arbitrary product package label image and executes compliance verification"""
    try:
        filename = f"upload_{os.urandom(6).hex()}_{image.filename}"
        dest_path = os.path.join(UPLOADS_DIR, filename)
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        extraction = extractor.extract_from_image(dest_path, client_ocr_json=ocr_json)
        extracted_data = extraction["extracted_data"]
        
        # Run Statutory Rule Engine
        eval_result = rule_engine.evaluate(extracted_data)
        
        # Persist to Database
        scan_id = database.save_scan(
            extracted=extracted_data,
            result=eval_result.dict(),
            image_path=extraction["image_url"]
        )
        
        return {
            "scan_id": scan_id,
            "image_url": extraction["image_url"],
            "extracted_data": extracted_data,
            "bounding_boxes": extraction["bounding_boxes"],
            "compliance_result": eval_result.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inspection failed: {str(e)}")

@app.get("/api/scan/{scan_id}")
async def get_scan(scan_id: int):
    record = database.get_scan_by_id(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    
    return {
        "scan_id": record["id"],
        "scan_uid": record["scan_uid"],
        "image_url": record["image_path"],
        "extracted_data": record.get("raw_extracted", {}),
        "bounding_boxes": [],
        "compliance_result": record.get("raw_result", {})
    }

@app.get("/api/history")
async def get_history(limit: int = 50):
    return database.get_all_scans(limit=limit)

@app.get("/api/metrics")
async def get_metrics():
    return database.get_dashboard_metrics()

@app.get("/api/report/{scan_id}")
async def download_report(scan_id: int):
    record = database.get_scan_by_id(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    
    pdf_path = os.path.join(REPORTS_DIR, f"Inspection_Report_{record['scan_uid']}.pdf")
    if not os.path.exists(pdf_path):
        generate_inspection_pdf(record, pdf_path)
        
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Inspection_Memo_{record['scan_uid']}.pdf"
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting LMPC SmartInspector Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
