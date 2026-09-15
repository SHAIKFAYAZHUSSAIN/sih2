"""
Legal Metrology (Packaged Commodities) Rules, 2011 - Rule Engine
Codified from official Gazette Notifications:
- G.S.R. 202(E) (Principal Rules, 2011)
- G.S.R. 779(E) (Amendment Rules, 2021 - Unit Sale Price & Multi-packs)
- G.S.R. 577(E) (Second Amendment Rules, 2022 - QR Code for Electronics)
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Violation(BaseModel):
    rule_id: str
    rule_reference: str
    severity: str  # 'CRITICAL', 'MAJOR', 'MINOR'
    title: str
    description: str
    statutory_penalty_ref: str = "Legal Metrology Act, 2009 - Section 36(1)"


class ComplianceResult(BaseModel):
    is_compliant: bool
    compliance_score: float  # 0 to 100%
    inspected_at: str
    commodity_category: str
    violations: List[Violation] = []
    warnings: List[str] = []
    extracted_data_summary: Dict[str, Any] = {}


class LMPCRuleEngine:
    # Statutory SI Units under Rule 13
    VALID_MASS_UNITS = {"g", "kg"}
    VALID_VOLUME_UNITS = {"ml", "l", "L"}
    VALID_LENGTH_UNITS = {"mm", "cm", "m"}
    VALID_NUMBER_UNITS = {"N", "U"}
    
    ILLEGAL_UNITS_MAP = {
        "gms": "g",
        "gm": "g",
        "kgr": "kg",
        "kilo": "kg",
        "ltr": "l",
        "ltrs": "l",
        "ml.": "ml",
        "doz": "N",
        "dozen": "N",
        "gross": "N"
    }

    PROHIBITED_APPROXIMATION_WORDS = [
        "minimum", "not less than", "approx", "approximate",
        "approximately", "average", "when packed"
    ]

    def __init__(self):
        pass

    def evaluate(self, extracted: Dict[str, Any], is_electronic: bool = False, pdp_area_cm2: Optional[float] = None) -> ComplianceResult:
        violations: List[Violation] = []
        warnings: List[str] = []
        
        # 1. Check Exemptions (Rule 26 & Rule 3)
        net_qty_val = extracted.get("net_quantity_value")
        net_qty_unit = str(extracted.get("net_quantity_unit", "")).lower()

        if net_qty_val is not None:
            if (net_qty_unit in ["g", "ml"] and net_qty_val <= 10.0) or (net_qty_unit in ["kg", "l"] and net_qty_val > 25.0):
                warnings.append(f"Package with net quantity {net_qty_val} {net_qty_unit} falls under Rule 26/Rule 3 exemption thresholds.")

        # 2. Rule 6(1)(a) & Rule 10: Manufacturer / Packer / Importer Name and Address
        mfg_name = extracted.get("manufacturer_name")
        mfg_address = extracted.get("manufacturer_address")
        has_qr_code = extracted.get("has_qr_code", False)

        if not mfg_name:
            violations.append(Violation(
                rule_id="RULE_6_1_A_NAME",
                rule_reference="Rule 6(1)(a) & Rule 10(1)",
                severity="CRITICAL",
                title="Missing Manufacturer/Packer Name",
                description="The name of the manufacturer, packer, or importer is completely missing from the package."
            ))
        
        if not mfg_address:
            if is_electronic and has_qr_code:
                # G.S.R. 577(E) 2022 allows address via QR code if declaration instructs to scan
                qr_instruction = extracted.get("qr_scan_instruction", "")
                if not re.search(r"scan.*qr", qr_instruction, re.IGNORECASE):
                    violations.append(Violation(
                        rule_id="RULE_6_1_A_QR_INSTRUCTION",
                        rule_reference="Rule 6(1)(a) Proviso (G.S.R. 577(E))",
                        severity="MAJOR",
                        title="Missing Instruction to Scan QR Code for Address",
                        description="For electronic products providing address via QR code, package must inform consumers to scan QR code."
                    ))
            else:
                violations.append(Violation(
                    rule_id="RULE_6_1_A_ADDRESS",
                    rule_reference="Rule 6(1)(a) & Rule 10(1)",
                    severity="CRITICAL",
                    title="Missing Manufacturer/Packer Complete Address",
                    description="Physical postal address with city/state and PIN code is not declared."
                ))
        elif isinstance(mfg_address, str):
            # Check for PIN code
            if not re.search(r"\b[1-9][0-9]{5}\b", mfg_address):
                warnings.append("PIN Code not explicitly detected in manufacturer address.")

        # 3. Rule 6(1)(b): Generic / Common Name
        generic_name = extracted.get("generic_name")
        if not generic_name:
            if is_electronic and has_qr_code:
                pass
            else:
                violations.append(Violation(
                    rule_id="RULE_6_1_B_GENERIC_NAME",
                    rule_reference="Rule 6(1)(b)",
                    severity="MAJOR",
                    title="Missing Generic/Common Name",
                    description="The common or generic name of the packaged commodity is not declared."
                ))

        # 4. Rule 6(1)(c), 11, 12, 13: Net Quantity & Standard Units
        if net_qty_val is None or not net_qty_unit:
            violations.append(Violation(
                rule_id="RULE_6_1_C_NET_QTY",
                rule_reference="Rule 6(1)(c) & Rule 11",
                severity="CRITICAL",
                title="Missing Net Quantity Declaration",
                description="Net quantity in standard weight, measure, or number is missing."
            ))
        else:
            # Check illegal unit symbols (e.g. 'gms', 'ltrs')
            raw_unit_str = str(extracted.get("net_quantity_raw", "")).lower()
            for illegal in self.ILLEGAL_UNITS_MAP:
                if re.search(r"\b" + re.escape(illegal) + r"\b", raw_unit_str):
                    violations.append(Violation(
                        rule_id="RULE_13_5_NON_STANDARD_UNIT",
                        rule_reference="Rule 13(5) & Second Schedule",
                        severity="MAJOR",
                        title=f"Non-Standard Unit Symbol '{illegal}'",
                        description=f"Unit '{illegal}' is non-standard. Rule 13 mandates SI symbols: '{self.ILLEGAL_UNITS_MAP[illegal]}'."
                    ))
                    break

            # Check deceptive / approximating words (Rule 12(6))
            for word in self.PROHIBITED_APPROXIMATION_WORDS:
                if word in raw_unit_str:
                    violations.append(Violation(
                        rule_id="RULE_12_6_DECEPTIVE_QTY",
                        rule_reference="Rule 12(6)",
                        severity="CRITICAL",
                        title=f"Prohibited Exaggerated Expression '{word}'",
                        description=f"Quantity cannot be qualified with deceptive/approximating terms like '{word}'."
                    ))

        # 5. Rule 6(1)(d): Month & Year of Manufacture / Packing
        mfg_date_str = extracted.get("mfg_date")
        if not mfg_date_str:
            violations.append(Violation(
                rule_id="RULE_6_1_D_DATE",
                rule_reference="Rule 6(1)(d)",
                severity="CRITICAL",
                title="Missing Month & Year of Manufacture/Packing",
                description="Month and year of manufacture or pre-packing is absent."
            ))
        else:
            # Check for future post-dating
            parsed_date = self._parse_month_year(mfg_date_str)
            if parsed_date:
                now = datetime.now()
                # If manufacture month is later than current month + 1
                if parsed_date.year > now.year or (parsed_date.year == now.year and parsed_date.month > now.month + 1):
                    violations.append(Violation(
                        rule_id="RULE_6_1_D_FUTURE_DATE",
                        rule_reference="Rule 6(1)(d)",
                        severity="CRITICAL",
                        title="Post-Dated / Future Packaging Date",
                        description=f"Manufacture date '{mfg_date_str}' is set in the future beyond allowable threshold."
                    ))

        # Check Expiry Date vs Manufacturing Date Chronology
        exp_date_str = extracted.get("exp_date")
        if mfg_date_str and exp_date_str:
            parsed_mfg = self._parse_month_year(mfg_date_str)
            parsed_exp = self._parse_month_year(exp_date_str)
            if parsed_mfg and parsed_exp:
                if parsed_exp < parsed_mfg:
                    violations.append(Violation(
                        rule_id="RULE_6_1_D_INVERTED_EXPIRY",
                        rule_reference="Rule 6(1)(d) & LM Act",
                        severity="CRITICAL",
                        title="Expiry Date Precedes Manufacturing Date",
                        description=f"Expiry date '{exp_date_str}' is earlier than manufacture date '{mfg_date_str}' (Chronological Inversion / Label Tampering)."
                    ))

        # 6. Rule 6(1)(e): Maximum Retail Price (MRP)
        mrp_val = extracted.get("mrp_value")
        mrp_raw = str(extracted.get("mrp_raw", ""))
        if mrp_val is None:
            violations.append(Violation(
                rule_id="RULE_6_1_E_MRP_MISSING",
                rule_reference="Rule 6(1)(e)",
                severity="CRITICAL",
                title="Missing Maximum Retail Price (MRP)",
                description="MRP declaration is completely missing from the package."
            ))
        else:
            # Check mandatory "inclusive of all taxes" clause
            if not re.search(r"incl[a-z,.\s]*tax", mrp_raw, re.IGNORECASE):
                violations.append(Violation(
                    rule_id="RULE_6_1_E_TAX_CLAUSE",
                    rule_reference="Rule 6(1)(e)",
                    severity="MAJOR",
                    title="Missing 'Inclusive of all taxes' in MRP",
                    description="MRP must explicitly state 'incl. of all taxes' or 'inclusive of all taxes'."
                ))

        # 7. Rule 6(11): Unit Sale Price (USP) - 2021 Amendment G.S.R. 779(E)
        usp_val = extracted.get("unit_sale_price_value")
        usp_raw = str(extracted.get("unit_sale_price_raw", ""))
        if mrp_val and net_qty_val and net_qty_val > 0:
            if not usp_val:
                violations.append(Violation(
                    rule_id="RULE_6_11_USP_MISSING",
                    rule_reference="Rule 6(11) (G.S.R. 779(E))",
                    severity="CRITICAL",
                    title="Missing Unit Sale Price (USP)",
                    description="Mandatory Unit Sale Price (e.g. Rs./g, Rs./kg, Rs./ml, Rs./l) is not declared."
                ))
            else:
                # Verify standard USP denominator unit format
                self._verify_usp_unit_format(net_qty_val, net_qty_unit, usp_raw, violations)
                
                # Check mathematical consistency: MRP / Net Quantity == USP
                expected_usp = self._calculate_expected_usp(mrp_val, net_qty_val, net_qty_unit)
                if expected_usp is not None and usp_val > 0:
                    discrepancy = abs(usp_val - expected_usp) / expected_usp
                    if discrepancy > 0.05:  # 5% tolerance for rounding
                        violations.append(Violation(
                            rule_id="RULE_6_11_USP_MISMATCH",
                            rule_reference="Rule 6(11) (G.S.R. 779(E))",
                            severity="CRITICAL",
                            title="Mathematical Discrepancy in Unit Sale Price",
                            description=f"Declared USP (Rs. {usp_val}) does not match calculated USP (Rs. {expected_usp:.2f}) from MRP Rs. {mrp_val} & Net Qty {net_qty_val} {net_qty_unit}."
                        ))

        # 8. Rule 6(2): Consumer Care Details
        care_phone = extracted.get("consumer_care_phone")
        care_email = extracted.get("consumer_care_email")
        care_address = extracted.get("consumer_care_address")

        if not care_phone and not care_email and not care_address:
            violations.append(Violation(
                rule_id="RULE_6_2_CONSUMER_CARE",
                rule_reference="Rule 6(2)",
                severity="CRITICAL",
                title="Missing Consumer Grievance / Care Details",
                description="No customer care telephone, email, or physical address declared for consumer complaints."
            ))
        elif not care_phone or not care_email:
            # Rule 6(2) mandates phone and email
            violations.append(Violation(
                rule_id="RULE_6_2_INCOMPLETE_CARE",
                rule_reference="Rule 6(2)",
                severity="MAJOR",
                title="Incomplete Consumer Care Channels",
                description="Both telephone number and email address must be declared under Rule 6(2)."
            ))

        # 9. Rule 7 & Schedule II: Font Height Compliance
        measured_numeral_height_mm = extracted.get("measured_numeral_height_mm")
        measured_letter_height_mm = extracted.get("measured_letter_height_mm")
        
        required_numeral_height = self._get_required_font_height(net_qty_val, net_qty_unit, pdp_area_cm2)
        if measured_numeral_height_mm is not None and required_numeral_height is not None:
            if measured_numeral_height_mm < required_numeral_height:
                violations.append(Violation(
                    rule_id="RULE_7_NUMERAL_HEIGHT",
                    rule_reference="Rule 7 & Table I/II of Schedule II",
                    severity="MAJOR",
                    title="Numeral Font Height Below Statutory Minimum",
                    description=f"Measured numeral height is {measured_numeral_height_mm}mm, but statutory minimum is {required_numeral_height}mm."
                ))

        if measured_letter_height_mm is not None and measured_letter_height_mm < 1.0:
            violations.append(Violation(
                rule_id="RULE_7_LETTER_HEIGHT",
                rule_reference="Rule 7(3)",
                severity="MAJOR",
                title="Letter Font Height Below 1mm",
                description=f"Measured letter height is {measured_letter_height_mm}mm. Rule 7(3) prohibits letters under 1.0mm."
            ))

        # Calculate compliance score
        penalty_points = sum(15 if v.severity == "CRITICAL" else (8 if v.severity == "MAJOR" else 3) for v in violations)
        score = max(0.0, 100.0 - penalty_points)
        is_compliant = len(violations) == 0

        return ComplianceResult(
            is_compliant=is_compliant,
            compliance_score=round(score, 1),
            inspected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            commodity_category=extracted.get("category", "Packaged Commodity"),
            violations=violations,
            warnings=warnings,
            extracted_data_summary=extracted
        )

    def _get_required_font_height(self, net_qty: Optional[float], unit: str, pdp_area: Optional[float]) -> float:
        """Determines minimum statutory numeral height in mm as per Rule 7 Table I & II"""
        if pdp_area is not None:
            if pdp_area <= 100:
                return 1.0
            elif pdp_area <= 500:
                return 2.0
            elif pdp_area <= 2500:
                return 4.0
            else:
                return 6.0
        
        # Fallback to Net Quantity Table I
        if net_qty is not None:
            norm_g = net_qty * 1000 if unit in ["kg", "l"] else net_qty
            if norm_g <= 200:
                return 1.0
            elif norm_g <= 500:
                return 2.0
            else:
                return 4.0
        return 1.0

    def _verify_usp_unit_format(self, net_qty: float, unit: str, usp_raw: str, violations: List[Violation]):
        """Verifies statutory unit format for Unit Sale Price under Rule 6(11)"""
        norm_g_or_ml = net_qty * 1000 if unit in ["kg", "l"] else net_qty
        usp_lower = usp_raw.lower()
        
        if unit in ["g", "kg"]:
            has_per_g = bool(re.search(r'(?:per|\/)\s*g\b', usp_lower) or "perg" in usp_lower)
            has_per_kg = bool(re.search(r'(?:per|\/)\s*kg\b', usp_lower) or "perkg" in usp_lower)
            if norm_g_or_ml < 1000 and not has_per_g:
                violations.append(Violation(
                    rule_id="RULE_6_11_USP_UNIT",
                    rule_reference="Rule 6(11)(i) (G.S.R. 779(E))",
                    severity="MAJOR",
                    title="Incorrect USP Denominator Unit (< 1kg)",
                    description="For net quantity less than 1kg, Unit Sale Price must be declared as 'Rs. __ per g'."
                ))
            elif norm_g_or_ml >= 1000 and not has_per_kg:
                violations.append(Violation(
                    rule_id="RULE_6_11_USP_UNIT",
                    rule_reference="Rule 6(11)(ii) (G.S.R. 779(E))",
                    severity="MAJOR",
                    title="Incorrect USP Denominator Unit (>= 1kg)",
                    description="For net quantity 1kg or more, Unit Sale Price must be declared as 'Rs. __ per kg'."
                ))
        elif unit in ["ml", "l"]:
            has_per_ml = bool(re.search(r'(?:per|\/)\s*ml\b', usp_lower) or "perml" in usp_lower)
            has_per_l = bool(re.search(r'(?:per|\/)\s*(?:l|litre)\b', usp_lower) or "perl" in usp_lower or "perlitre" in usp_lower)
            if norm_g_or_ml < 1000 and not has_per_ml:
                violations.append(Violation(
                    rule_id="RULE_6_11_USP_UNIT",
                    rule_reference="Rule 6(11)(vi) (G.S.R. 779(E))",
                    severity="MAJOR",
                    title="Incorrect USP Denominator Unit (< 1L)",
                    description="For net volume less than 1 litre, Unit Sale Price must be declared as 'Rs. __ per ml'."
                ))
            elif norm_g_or_ml >= 1000 and not has_per_l:
                violations.append(Violation(
                    rule_id="RULE_6_11_USP_UNIT",
                    rule_reference="Rule 6(11)(vii) (G.S.R. 779(E))",
                    severity="MAJOR",
                    title="Incorrect USP Denominator Unit (>= 1L)",
                    description="For net volume 1 litre or more, Unit Sale Price must be declared as 'Rs. __ per litre'."
                ))

    def _calculate_expected_usp(self, mrp: float, net_qty: float, unit: str) -> Optional[float]:
        norm_g_or_ml = net_qty * 1000 if unit in ["kg", "l"] else net_qty
        if norm_g_or_ml <= 0:
            return None
        
        if norm_g_or_ml < 1000:
            # USP is per g or per ml
            return mrp / norm_g_or_ml
        else:
            # USP is per kg or per litre
            return mrp / (norm_g_or_ml / 1000.0)

    def _parse_month_year(self, date_str: str) -> Optional[datetime]:
        if not date_str: return None
        # Clean whitespace inside date like '12 - 2024' -> '12-2024'
        date_str = re.sub(r'\s*([-/\.])\s*', r'\1', date_str.strip())
        date_str = re.sub(r'([a-zA-Z]+)([0-9]{4})', r'\1 \2', date_str)
        patterns = ["%m/%Y", "%m/%y", "%b %Y", "%B %Y", "%m-%Y", "%b-%Y", "%m.%Y", "%d/%m/%Y", "%d-%m-%Y"]
        for p in patterns:
            try:
                return datetime.strptime(date_str, p)
            except ValueError:
                continue
        return None


if __name__ == "__main__":
    # Quick self-test demonstration
    engine = LMPCRuleEngine()
    test_sample = {
        "manufacturer_name": "Haldiram Snacks Pvt Ltd",
        "manufacturer_address": "Sector 62, Noida, Uttar Pradesh 201301",
        "generic_name": "Bhujia Sev",
        "net_quantity_raw": "400 gms",
        "net_quantity_value": 400.0,
        "net_quantity_unit": "gms",  # Violation: 'gms' instead of 'g'
        "mrp_raw": "MRP Rs. 100.00 (inclusive of all taxes)",
        "mrp_value": 100.0,
        "unit_sale_price_raw": "Rs. 0.25 / g",
        "unit_sale_price_value": 0.25,
        "mfg_date": "06/2026",
        "consumer_care_phone": "1800-102-0000",
        "consumer_care_email": "customercare@haldiram.com",
        "measured_numeral_height_mm": 1.2  # Violation: for 400g, min height is 2.0mm
    }
    
    result = engine.evaluate(test_sample)
    print("--- LMPC EVALUATION REPORT ---")
    print(f"Compliant: {result.is_compliant}")
    print(f"Score: {result.compliance_score}%")
    print(f"Total Violations: {len(result.violations)}")
    for v in result.violations:
        print(f"[{v.severity}] {v.rule_reference}: {v.title} -> {v.description}")
