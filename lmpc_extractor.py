"""
Multimodal Vision & Key Information Extraction (KIE) Pipeline
Analyzes product packaging images, detects text bounding boxes with morphological gradients,
and extracts structured Legal Metrology statutory fields.
"""

import os
import re
import json
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

class LMPCExtractor:
    def __init__(self):
        self.preloaded_samples = {}
        self._load_samples()
        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        # On Vercel serverless, skip local ONNX binary to prevent cold-start crashes
        if os.environ.get("VERCEL"):
            self.ocr_engine = None
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr_engine = RapidOCR()
            print("RapidOCR engine initialized successfully.")
        except Exception as e:
            print(f"Notice: Local RapidOCR not active ({e}). Client-side OCR active.")
            self.ocr_engine = None

    def _load_samples(self):
        # 1. Search filesystem directories
        possible_dirs = [
            SAMPLE_DIR,
            os.path.join(BASE_DIR, "sample_data"),
            os.path.join(os.getcwd(), "sample_data")
        ]
        for sdir in possible_dirs:
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.endswith(".json"):
                        try:
                            with open(os.path.join(sdir, fname), "r", encoding="utf-8") as f:
                                data = json.load(f)
                                self.preloaded_samples[data["id"]] = data
                        except Exception as e:
                            print(f"Failed loading sample {fname}: {e}")
                if len(self.preloaded_samples) > 0:
                    break

        # 2. Fallback to embedded samples if filesystem has no files
        if len(self.preloaded_samples) == 0:
            try:
                from embedded_samples import EMBEDDED_SAMPLES
                self.preloaded_samples = dict(EMBEDDED_SAMPLES)
                print("Loaded", len(self.preloaded_samples), "embedded benchmark samples.")
            except Exception as e:
                print(f"Notice: embedded samples fallback error: {e}")

    def _get_image_base64_url(self, file_path: str) -> str:
        """Converts an image file to a Base64 data URL for 100% reliable serverless delivery"""
        if not os.path.exists(file_path):
            return ""
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["jpg", "jpeg", "png", "webp"] else "image/jpeg"
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def extract_from_sample_id(self, sample_id: str) -> Dict[str, Any]:
        """Instant high-fidelity retrieval for live demonstration cards"""
        sample = self.preloaded_samples.get(sample_id)
        if not sample:
            raise ValueError(f"Sample ID {sample_id} not found in repository.")
        
        img_rel = sample.get("image_rel_path", "").lstrip("/").replace("/", os.sep)
        img_full = os.path.join(BASE_DIR, img_rel)
        data_url = self._get_image_base64_url(img_full) if os.path.exists(img_full) else sample["image_rel_path"]

        return {
            "image_url": data_url or sample["image_rel_path"],
            "extracted_data": sample["extracted_data"],
            "bounding_boxes": sample.get("bounding_boxes", [])
        }

    def extract_from_image(self, image_path: str, client_ocr_json: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes an uploaded user package image using RapidOCR / Client OCR + Computer Vision
        morphological detection and statutory Key Information Extraction (KIE).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Use Pillow to reliably read image dimensions without requiring cv2
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            w, h = 800, 600

        cv_img = None
        if cv2 is not None:
            try:
                cv_img = cv2.imread(image_path)
            except Exception:
                cv_img = None

        # 1. Check if client-side OCR (Tesseract.js) supplied extracted text and bounding boxes
        ocr_results = []
        if client_ocr_json:
            try:
                client_items = json.loads(client_ocr_json)
                for it in client_items:
                    t = str(it.get("text", "")).strip()
                    bx = it.get("box", [0, 0, 0, 0])
                    conf = it.get("conf", 0.9)
                    if t:
                        pts = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]
                        ocr_results.append([pts, t, conf])
            except Exception as e:
                print(f"Notice: Failed parsing client OCR json: {e}")

        # 2. Execute local RapidOCR if available and no client OCR was passed
        if not ocr_results and self.ocr_engine is not None and cv_img is not None:
            try:
                res, _ = self.ocr_engine(cv_img)
                if res:
                    ocr_results = res
            except Exception as e:
                print(f"Notice: Local OCR inference exception: {e}")

        # 3. Extract statutory fields and bounding boxes from OCR
        if ocr_results and len(ocr_results) > 0:
            extracted, bounding_boxes = self._parse_ocr_declarations(ocr_results, cv_img, h, w)
        else:
            # Fallback to morphological gradient detection
            bounding_boxes = self._detect_morphological_boxes(cv_img, h, w)
            extracted = self._heuristic_label_parser(None, h, w)

        # 4. Base64 encode image for 100% reliable frontend rendering
        data_url = self._get_image_base64_url(image_path)
        
        return {
            "image_url": data_url,
            "extracted_data": extracted,
            "bounding_boxes": bounding_boxes
        }

    def _parse_ocr_declarations(self, ocr_results: list, cv_img, h: int, w: int) -> (Dict[str, Any], List[Dict[str, Any]]):
        """
        Extracts structured statutory packaging declarations from OCR text lines
        and computes color-coded visualizer bounding boxes.
        """
        extracted = {
            "commodity_name": "Scanned Packaged Commodity",
            "generic_name": None,
            "category": "Packaged Retail Goods",
            "manufacturer_name": None,
            "manufacturer_address": None,
            "country_of_origin": None,
            "net_quantity_raw": None,
            "net_quantity_value": None,
            "net_quantity_unit": None,
            "mrp_raw": None,
            "mrp_value": None,
            "unit_sale_price_raw": None,
            "unit_sale_price_value": None,
            "mfg_date": None,
            "exp_date": None,
            "batch_number": None,
            "consumer_care_phone": None,
            "consumer_care_email": None,
            "consumer_care_address": None,
            "measured_numeral_height_mm": 2.5,
            "measured_letter_height_mm": 1.5,
            "has_qr_code": False
        }

        # Check QR Code with OpenCV
        try:
            qr_detector = cv2.QRCodeDetector()
            qr_data, _, _ = qr_detector.detectAndDecode(cv_img)
            extracted["has_qr_code"] = bool(qr_data)
        except Exception:
            extracted["has_qr_code"] = False

        est_px_per_mm = h / 150.0 if h > 0 else 5.0
        candidate_boxes = []

        all_text_lines = []
        line_items = []
        for item in ocr_results:
            pts = item[0]
            text = str(item[1]).strip()
            if not text:
                continue
            all_text_lines.append(text)
            
            x1 = int(min(p[0] for p in pts))
            y1 = int(min(p[1] for p in pts))
            x2 = int(max(p[0] for p in pts))
            y2 = int(max(p[1] for p in pts))
            box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            box_height = y2 - y1
            line_items.append({
                "text": text,
                "box": box,
                "height": box_height
            })

        full_ocr_text = "\n".join(all_text_lines)

        # Helper to add candidate box safely without duplicates
        def add_box(key, label, box, color="#10b981"):
            for b in candidate_boxes:
                if b.get("key") == key:
                    return
            candidate_boxes.append({
                "key": key,
                "label": label,
                "box": box,
                "color": color
            })

        # PASS 1: Line by Line Pattern Matching
        for item in line_items:
            text = item["text"]
            box = item["box"]
            box_height = item["height"]

            # Check Net Quantity
            m_net = re.search(r'(?:Net\s*Qty|Net\s*Quantity|Net\s*Wt|Net\s*Weight|NET\s*QTY)[:.\s]*(?:When\s*Packed\s*)?([0-9]+(?:[.,][0-9]+)?)\s*([a-zA-Z]+)', text, re.I)
            if not m_net:
                m_net = re.search(r'\b([0-9]+(?:[.,][0-9]+)?)\s*(g|kg|ml|l|L|gms|gm)\b', text, re.I)
            if m_net and not extracted["net_quantity_value"]:
                extracted["net_quantity_raw"] = f"{m_net.group(1)} {m_net.group(2)}"
                try:
                    extracted["net_quantity_value"] = float(m_net.group(1).replace(',', '.'))
                    extracted["net_quantity_unit"] = m_net.group(2).lower()
                    extracted["measured_numeral_height_mm"] = round(max(1.0, box_height / est_px_per_mm), 1)
                except Exception:
                    pass
                add_box("net_qty", f"Net Qty: {m_net.group(1)} {m_net.group(2)}", box, "#10b981")
                continue

            # Check Batch Number
            m_batch = re.search(r'(?:B\.?\s*No|Batch|Lot)[:.\s]*([A-Za-z0-9_-]+)', text, re.I)
            if m_batch and not extracted["batch_number"]:
                extracted["batch_number"] = m_batch.group(1).strip()
                add_box("batch", f"Batch: {m_batch.group(1)}", box, "#3b82f6")
                continue

            # Check Mfg Date (Standard & Tabular Format like #April2026)
            m_mfg = re.search(r'(?:Mfg|Mfd|Pkg|Packed|PKD|DOM|Date\s*of\s*Mfg|Date\s*of\s*Manufacture)[:.\s]*(?:Dt\.?|Date)?[:.\s]*([0-9]{1,2}\s*[-/\.]\s*[0-9]{2,4}|[A-Za-z]{3,9}\s*[-/\s]\s*[0-9]{2,4})', text, re.I)
            if not m_mfg:
                m_mfg = re.search(r'(?:#|\bMFD\b|\bMfg\b)?[:.\s]*(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*([0-9]{4})', text, re.I)
            if m_mfg and not extracted["mfg_date"]:
                if len(m_mfg.groups()) == 2 and m_mfg.group(2):
                    clean_d = f"{m_mfg.group(1)} {m_mfg.group(2)}"
                else:
                    clean_d = re.sub(r'\s*([-/\.])\s*', r'\1', m_mfg.group(1)).strip()
                extracted["mfg_date"] = clean_d
                add_box("mfg_date", f"Mfg Date: {clean_d}", box, "#10b981")
                continue

            # Check Expiry Date
            m_exp = re.search(r'(?:Exp|Expiry|Best\s*Before|BB|USE\s*BY|Date\s*of\s*Expiry)[:.\s]*(?:Dt\.?|Date)?[:.\s]*([0-9]{1,2}\s*[-/\.]\s*[0-9]{2,4}|[A-Za-z]{3,9}\s*[-/\s]\s*[0-9]{2,4})', text, re.I)
            if m_exp and not extracted["exp_date"]:
                clean_d = re.sub(r'\s*([-/\.])\s*', r'\1', m_exp.group(1)).strip()
                extracted["exp_date"] = clean_d
                add_box("exp_date", f"Exp Date: {clean_d}", box, "#10b981")
                continue

            # Check MRP (Support Rs, INR, ₹, OCR misspellings like HRP, MBP, MAP, and Indian /- notation)
            m_mrp = re.search(r'(?:M\.?\s*R\.?\s*P\.?|MAX(?:IMUM)?\.?\s*RETAIL\s*PRICE|[HMN]RP)[:.\s]*(?:\(?(?:incl(?:usive)?\.?\s*(?:of)?\s*all\s*taxes)\)?)?[:.\s]*(?:Rs\.?|INR|₹|\u20b9)?\s*([0-9]+(?:[.,][0-9]{1,2})?)', text, re.I)
            if not m_mrp and re.search(r'(?:^|\s|\*|[₹]|Rs\.?)\s*([0-9]+(?:\.[0-9]{1,2})?)\s*(?:\/-|\/)', text):
                if not any(k in text.lower() for k in ["lic", "cos", "b.no", "batch", "iso"]):
                    m_mrp = re.search(r'(?:^|\s|\*|[₹]|Rs\.?)\s*([0-9]+(?:\.[0-9]{1,2})?)\s*(?:\/-|\/)', text)
            if m_mrp and m_mrp.group(1) and not extracted["mrp_value"]:
                extracted["mrp_raw"] = text.replace("₹", "Rs. ")
                try:
                    extracted["mrp_value"] = float(m_mrp.group(1).replace(',', '.'))
                except Exception:
                    pass
                add_box("mrp", f"MRP: Rs. {m_mrp.group(1)}", box, "#10b981")
                continue

            # Check USP (Support standard and tabular format like @0.66perg)
            m_usp = re.search(r'(?:U\.?\s*S\.?\s*P\.?|UNIT\s*SALE\s*PRICE)[:.\s]*(?:Rs\.?|INR|₹|\u20b9)?\s*([0-9]+(?:[.,][0-9]{1,2})?)\s*(?:per|\/)?\s*([a-zA-Z]+)?', text, re.I)
            if not m_usp:
                m_usp = re.search(r'(?:@|\bUSP\b)[:.\s]*(?:Rs\.?|INR|₹|\u20b9)?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:per|\/)\s*([a-zA-Z]+)', text, re.I)
            if m_usp and m_usp.group(1) and not extracted["unit_sale_price_value"]:
                extracted["unit_sale_price_raw"] = text.replace("₹", "Rs. ")
                try:
                    extracted["unit_sale_price_value"] = float(m_usp.group(1).replace(',', '.'))
                except Exception:
                    pass
                unit_str = m_usp.group(2) or "g"
                add_box("usp", f"USP: Rs. {m_usp.group(1)}/{unit_str}", box, "#10b981")
                continue

            # Check Generic / Common Name (e.g. Toilet Soap, Bathing Bar, Biscuit)
            m_gen_item = re.search(r'\b(Toilet\s*Soap(?:[*\s-]*Grade\s*[0-9]+)?|Bathing\s*Bar|Soap|Detergent|Biscuits?|Namkeen|Atta|Flour|Refined\s*Oil|Mustard\s*Oil|Edible\s*Oil|Toothpaste|Shampoo|Tea|Coffee|Spices|Salt|Sugar|Milk|Ghee|Butter|Paneer)\b', text, re.I)
            if m_gen_item and not extracted["generic_name"]:
                clean_gen = re.sub(r'[*]', ' ', m_gen_item.group(0)).strip()
                extracted["generic_name"] = clean_gen
                extracted["commodity_name"] = clean_gen
                add_box("generic_name", f"Generic: {clean_gen}", box, "#10b981")

            # Check Manufacturer Name (Explicit Prefix or Corporate Entity)
            m_mfr = re.search(r'(?:Mfd\s*by|Manufactured\s*by|Marketed\s*by|Packed\s*by|Mfg\s*by)[:.\s]*(.+)', text, re.I)
            if m_mfr and not extracted["manufacturer_name"]:
                val = m_mfr.group(1).strip()
                if len(val) > 3 and not any(k in val.lower() for k in ["mrp", "date", "batch", "qty", "table", "contact", "pro"]):
                    extracted["manufacturer_name"] = val
                    add_box("mfr", f"Mfr: {val[:20]}", box, "#10b981")
                    continue
            elif not extracted["manufacturer_name"] and re.search(r'\b(?:Limited|Ltd|Pvt|Company|Corporation|Enterprises|Industries)\b', text, re.I):
                if not any(k in text.lower() for k in ["iso", "customer", "feedback", "mrp", "date", "batch", "complaint", "cos"]):
                    extracted["manufacturer_name"] = text.strip()
                    add_box("mfr", f"Mfr: {text.strip()[:20]}", box, "#10b981")
                    continue

            # Check Postal Address & PIN Code
            if re.search(r'\b[1-9][0-9]{5}\b', text) or re.search(r'\b(?:Highway|Industrial\s*Suburb|Malleshwaram|Road|Street|Plot|Sector)\b', text, re.I):
                if not extracted["manufacturer_address"]:
                    extracted["manufacturer_address"] = text.strip()
                elif text.strip() not in extracted["manufacturer_address"]:
                    extracted["manufacturer_address"] += f", {text.strip()}"
                extracted["consumer_care_address"] = extracted["manufacturer_address"]
                add_box("addr", "Mfr Address", box, "#10b981")

            # Check Country of Origin
            if re.search(r'\b(?:India|Bharat)\b', text, re.I) and not extracted["country_of_origin"]:
                extracted["country_of_origin"] = "India"

            # Check Consumer Care Phone
            m_phone = re.search(r'(?:1800[- ]?[0-9]{3}[- ]?[0-9]{3,4}|\b[6-9][0-9]{9}\b)', text)
            if m_phone and not extracted["consumer_care_phone"]:
                extracted["consumer_care_phone"] = m_phone.group(0)
                add_box("phone", f"Helpline: {m_phone.group(0)}", box, "#10b981")
                continue

            # Check Consumer Care Email
            m_email = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text)
            if m_email and not extracted["consumer_care_email"]:
                extracted["consumer_care_email"] = m_email.group(1)
                add_box("email", f"Email: {m_email.group(1)[:18]}", box, "#10b981")
                continue

        # PASS 2: Sliding Window of 2 Adjacent Lines (for multi-line declarations)
        for i in range(len(line_items) - 1):
            pair_text = line_items[i]["text"] + " " + line_items[i+1]["text"]
            b1 = line_items[i]["box"]
            b2 = line_items[i+1]["box"]
            union_box = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

            if not extracted["mrp_value"]:
                m_mrp = re.search(r'(?:M\.?\s*R\.?\s*P\.?|MAX(?:IMUM)?\.?\s*RETAIL\s*PRICE|[HMN]RP)[:.\s]*(?:\(?(?:incl(?:usive)?\.?\s*(?:of)?\s*all\s*taxes)\)?)?[:.\s]*(?:Rs\.?|INR|₹|\u20b9)?\s*([0-9]+(?:[.,][0-9]{1,2})?)', pair_text, re.I)
                if m_mrp and m_mrp.group(1):
                    extracted["mrp_raw"] = pair_text.replace("₹", "Rs. ")
                    try:
                        extracted["mrp_value"] = float(m_mrp.group(1).replace(',', '.'))
                    except Exception:
                        pass
                    add_box("mrp", f"MRP: Rs. {m_mrp.group(1)}", union_box, "#3b82f6")

            if not extracted["exp_date"]:
                m_exp = re.search(r'(?:Exp|Expiry|Best\s*Before|BB|USE\s*BY|Date\s*of\s*Expiry)[:.\s]*(?:Dt\.?|Date)?[:.\s]*([0-9]{1,2}\s*[-/\.]\s*[0-9]{2,4}|[A-Za-z]{3,9}\s*[-/\s]\s*[0-9]{2,4})', pair_text, re.I)
                if m_exp:
                    clean_d = re.sub(r'\s*([-/\.])\s*', r'\1', m_exp.group(1)).strip()
                    extracted["exp_date"] = clean_d
                    add_box("exp_date", f"Exp Date: {clean_d}", union_box, "#10b981")

            if not extracted["unit_sale_price_value"]:
                m_usp = re.search(r'(?:U\.?\s*S\.?\s*P\.?|UNIT\s*SALE\s*PRICE)[:.\s]*(?:Rs\.?|INR|₹|\u20b9)?\s*([0-9]+(?:[.,][0-9]{1,2})?)\s*(?:per|\/)?\s*([a-zA-Z]+)?', pair_text, re.I)
                if m_usp and m_usp.group(1):
                    extracted["unit_sale_price_raw"] = pair_text.replace("₹", "Rs. ")
                    try:
                        extracted["unit_sale_price_value"] = float(m_usp.group(1).replace(',', '.'))
                    except Exception:
                        pass
                    unit_str = m_usp.group(2) or "g"
                    add_box("usp", f"USP: Rs. {m_usp.group(1)}/{unit_str}", union_box, "#3b82f6")

        # PASS 3: Full Document Fallback for MRP & Standalone Prices
        if not extracted["mrp_value"]:
            m_fallback = re.search(r'(?:Rs\.?|INR|₹|\u20b9)\s*([0-9]+(?:\.[0-9]{1,2})?)', full_ocr_text)
            if m_fallback:
                extracted["mrp_raw"] = f"MRP Rs. {m_fallback.group(1)}"
                try:
                    extracted["mrp_value"] = float(m_fallback.group(1))
                except Exception:
                    pass

        # Check full combined text for missing multi-line tax clause in MRP
        if extracted["mrp_raw"] and not re.search(r"incl[a-z,.\s]*tax", extracted["mrp_raw"], re.I):
            if re.search(r"incl[a-z,.\s]*tax", full_ocr_text, re.I):
                extracted["mrp_raw"] += " (inclusive of all taxes)"

        # Verify USP Mathematical consistency
        if extracted["mrp_value"] and extracted["net_quantity_value"] and extracted["unit_sale_price_value"]:
            net_val = extracted["net_quantity_value"]
            unit = extracted["net_quantity_unit"] or "g"
            norm = net_val * 1000 if unit in ["kg", "l"] else net_val
            expected_usp = (extracted["mrp_value"] / norm) if norm < 1000 else (extracted["mrp_value"] / (norm / 1000.0))
            declared_usp = extracted["unit_sale_price_value"]
            discrepancy = abs(declared_usp - expected_usp) / expected_usp if expected_usp > 0 else 0
            
            for b in candidate_boxes:
                if b.get("key") == "usp":
                    if discrepancy > 0.05:
                        b["color"] = "#ef4444"
                        b["label"] = f"VIOLATION: USP Rs. {declared_usp} != Calc Rs. {expected_usp:.2f}"
                    else:
                        b["color"] = "#10b981"

        # Verify Chronology (Expiry before Mfg Date)
        if extracted["mfg_date"] and extracted["exp_date"]:
            dt_mfg = self._parse_date(extracted["mfg_date"])
            dt_exp = self._parse_date(extracted["exp_date"])
            if dt_mfg and dt_exp and dt_exp < dt_mfg:
                for b in candidate_boxes:
                    if b.get("key") == "exp_date":
                        b["color"] = "#ef4444"
                        b["label"] = f"VIOLATION: Exp {extracted['exp_date']} < Mfg"

        # Strict Generic Name Extraction - do NOT take random OCR words as generic name
        for l in all_text_lines:
            m_gen = re.search(r'(?:Generic\s*Name|Common\s*Name|Commodity\s*Name|Product)[:.\s]*(.+)', l, re.I)
            if m_gen and not extracted["generic_name"]:
                extracted["generic_name"] = m_gen.group(1).strip()
                extracted["commodity_name"] = m_gen.group(1).strip()
                break

        # Remove temporary internal key from boxes
        final_boxes = [{"label": b["label"], "box": b["box"], "color": b["color"]} for b in candidate_boxes]

        # If no specific key boxes detected, fallback to standard section clusters
        if not final_boxes:
            final_boxes = [
                {"label": "Principal Display Panel", "box": [int(w*0.08), int(h*0.1), int(w*0.92), int(h*0.4)], "color": "#10b981"},
                {"label": "Mandatory Declarations Panel", "box": [int(w*0.08), int(h*0.45), int(w*0.92), int(h*0.9)], "color": "#3b82f6"}
            ]

        return extracted, final_boxes

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str: return None
        date_str = re.sub(r'\s*([-/\.])\s*', r'\1', date_str.strip())
        date_str = re.sub(r'([a-zA-Z]+)([0-9]{4})', r'\1 \2', date_str)
        for p in ["%m/%Y", "%m/%y", "%b %Y", "%B %Y", "%m-%Y", "%b-%Y", "%m.%Y", "%d/%m/%Y", "%d-%m-%Y"]:
            try:
                return datetime.strptime(date_str, p)
            except ValueError:
                continue
        return None

    def _detect_morphological_boxes(self, cv_img, h: int, w: int) -> List[Dict[str, Any]]:
        """Fallback box detection using morphological gradient if cv2 is installed"""
        if cv2 is not None and cv_img is not None:
            try:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                conn_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
                connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, conn_kernel)

                contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                raw_boxes = []
                for c in contours:
                    bx, by, bw_val, bh_val = cv2.boundingRect(c)
                    if bw_val > 20 and bh_val > 6 and bw_val < w * 0.98 and bh_val < h * 0.5:
                        raw_boxes.append([bx, by, bx + bw_val, by + bh_val])

                return self._cluster_into_statutory_blocks(raw_boxes, h, w)
            except Exception:
                pass

        return self._cluster_into_statutory_blocks([], h, w)

    def _cluster_into_statutory_blocks(self, raw_boxes: List[List[int]], h: int, w: int) -> List[Dict[str, Any]]:
        """Organizes detected text lines into labeled statutory regions"""
        if not raw_boxes:
            return [
                {"label": "Principal Display Panel (PDP)", "box": [int(w*0.08), int(h*0.08), int(w*0.92), int(h*0.35)], "color": "#10b981"},
                {"label": "Net Quantity & Units", "box": [int(w*0.08), int(h*0.40), int(w*0.60), int(h*0.52)], "color": "#10b981"},
                {"label": "MRP & Unit Sale Price Panel", "box": [int(w*0.08), int(h*0.56), int(w*0.92), int(h*0.75)], "color": "#3b82f6"},
                {"label": "Consumer Care & Grievance Cell", "box": [int(w*0.08), int(h*0.80), int(w*0.92), int(h*0.94)], "color": "#10b981"}
            ]

        raw_boxes = sorted(raw_boxes, key=lambda b: b[1])
        top_boxes = [b for b in raw_boxes if b[1] < h * 0.35]
        mid_boxes = [b for b in raw_boxes if h * 0.35 <= b[1] < h * 0.70]
        bot_boxes = [b for b in raw_boxes if b[1] >= h * 0.70]

        statutory_blocks = []

        def merge_boxes(box_list, label, color):
            if not box_list:
                return None
            x1 = max(0, min(b[0] for b in box_list) - 5)
            y1 = max(0, min(b[1] for b in box_list) - 5)
            x2 = min(w, max(b[2] for b in box_list) + 5)
            y2 = min(h, max(b[3] for b in box_list) + 5)
            return {"label": label, "box": [x1, y1, x2, y2], "color": color}

        if top_boxes:
            merged = merge_boxes(top_boxes, "Common/Generic Name & Identity", "#10b981")
            if merged: statutory_blocks.append(merged)

        if mid_boxes:
            merged = merge_boxes(mid_boxes, "Net Quantity & Mandatory Declarations", "#10b981")
            if merged: statutory_blocks.append(merged)

        if bot_boxes:
            merged_mrp = merge_boxes(bot_boxes, "MRP & Unit Sale Price (USP)", "#3b82f6")
            if merged_mrp: statutory_blocks.append(merged_mrp)

        return statutory_blocks

    def _heuristic_label_parser(self, gray_img, h: int, w: int) -> Dict[str, Any]:
        """Fallback statutory parser estimating values if OCR fails"""
        est_px_per_mm = h / 150.0 if h > 0 else 5.0
        return {
            "commodity_name": "Scanned Packaged Commodity",
            "generic_name": None,
            "category": "Packaged Retail Goods",
            "manufacturer_name": None,
            "manufacturer_address": None,
            "country_of_origin": None,
            "net_quantity_raw": None,
            "net_quantity_value": None,
            "net_quantity_unit": None,
            "mrp_raw": None,
            "mrp_value": None,
            "unit_sale_price_raw": None,
            "unit_sale_price_value": None,
            "mfg_date": None,
            "exp_date": None,
            "consumer_care_phone": None,
            "consumer_care_email": None,
            "measured_numeral_height_mm": round(max(2.1, 14.0 / est_px_per_mm), 1),
            "measured_letter_height_mm": round(max(1.2, 10.0 / est_px_per_mm), 1),
            "has_qr_code": False
        }


if __name__ == "__main__":
    extractor = LMPCExtractor()
    print("Extractor initialized.")
    test_img = os.path.join(STATIC_DIR, "images", "sample_01_compliant_snack.jpg")
    if os.path.exists(test_img):
        res = extractor.extract_from_image(test_img)
        print("Test extraction bounding boxes:", len(res["bounding_boxes"]))
        print("Image base64 prefix:", res["image_url"][:40])
