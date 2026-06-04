"""
Cardiology Decision Support Module
====================================
Implements five clinical tools:
  1. ECG interpretation helper
  2. Heart failure staging (NYHA + AHA/ACC)
  3. Anticoagulation decision tool (CHA₂DS₂-VASc + HAS-BLED + OAC recommendation)
  4. Shock classification
  5. Cardiac biomarker calculator (TnI kinetics, Delta TnI rule)

Each tool accepts structured input and returns a structured dictionary.
All calculations follow current clinical guidelines (ESC 2021, ACC/AHA 2022).
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ===========================================================================
# 1. ECG Interpretation Helper
# ===========================================================================

# Pattern → finding / urgency mapping
ECG_PATTERNS = [
    # STEMI / ischemia
    (r"ST上昇\s*(?:V[1-6]|aVL|aVF|II|III|I)", "ST elevation (potential STEMI)"),
    (r"ST低下", "ST depression (possible NSTEMI/ischemia)"),
    (r"T波逆転|T波陰性", "T-wave inversion (ischemia/strain pattern)"),
    (r"ST上昇.*下壁|下壁.*ST上昇", "Inferior ST elevation — evaluate RV involvement"),
    # Conduction
    (r"LBBB|左脚ブロック", "Left bundle branch block (LBBB)"),
    (r"RBBB|右脚ブロック", "Right bundle branch block (RBBB)"),
    (r"AVブロック.*[23]度|[23]度.*AVブロック|完全房室ブロック", "High-degree AV block"),
    (r"1度.*AVブロック|AVブロック.*1度", "1st-degree AV block"),
    (r"WPW|ウォルフパーキンソンホワイト", "WPW pattern — risk of rapid conduction in AF"),
    # Arrhythmia
    (r"\bAF\b|心房細動", "Atrial fibrillation (AF)"),
    (r"\bAFL\b|心房粗動", "Atrial flutter"),
    (r"\bVT\b|心室頻拍", "Ventricular tachycardia (VT) — EMERGENCY"),
    (r"\bVF\b|心室細動", "Ventricular fibrillation (VF) — CARDIAC ARREST"),
    (r"SVT|上室性頻拍", "Supraventricular tachycardia (SVT)"),
    (r"PVC|心室性期外収縮|VPC", "Premature ventricular complexes (PVCs)"),
    (r"PAC|心房性期外収縮", "Premature atrial complexes (PACs)"),
    (r"洞不全症候群|SSS", "Sick sinus syndrome"),
    # Intervals
    (r"QTc\s+(\d+)\s*ms", "QTc interval"),
    (r"HR\s+(\d+)|HR(\d+)", "Heart rate"),
]

URGENT_CONDITIONS = {
    "ST elevation (potential STEMI)",
    "Inferior ST elevation — evaluate RV involvement",
    "High-degree AV block",
    "Ventricular tachycardia (VT) — EMERGENCY",
    "Ventricular fibrillation (VF) — CARDIAC ARREST",
    "WPW pattern — risk of rapid conduction in AF",
}

QTC_PROLONGED_THRESHOLD = 450  # ms (men); use 460 for women
TACHYCARDIA_THRESHOLD = 100
BRADYCARDIA_THRESHOLD = 60


def parse_ecg_text(ecg_text: str) -> Dict[str, Any]:
    """
    Parse free-text ECG findings and return structured interpretation.

    Args:
        ecg_text: Free-text ECG report, e.g. "ST上昇 V1-V4, LBBB, QTc 480ms, AF HR 120"

    Returns:
        {
          "findings": [...],
          "urgent_flags": [...],
          "qtc_ms": int | None,
          "heart_rate": int | None,
          "rhythm_summary": str,
          "interpretation": str,
          "action": str
        }
    """
    findings: List[str] = []
    urgent_flags: List[str] = []
    qtc_ms: Optional[int] = None
    heart_rate: Optional[int] = None

    for pattern, finding in ECG_PATTERNS:
        m = re.search(pattern, ecg_text, re.IGNORECASE)
        if m:
            if "QTc" in finding:
                try:
                    qtc_ms = int(m.group(1))
                    label = f"QTc {qtc_ms} ms"
                    if qtc_ms >= QTC_PROLONGED_THRESHOLD:
                        label += " — PROLONGED (risk of Torsades de Pointes)"
                        urgent_flags.append(label)
                    findings.append(label)
                except (IndexError, ValueError):
                    findings.append(finding)
            elif "Heart rate" in finding:
                # Try to extract HR value
                hr_match = re.search(r"HR\s*(\d+)", ecg_text, re.IGNORECASE)
                if hr_match:
                    heart_rate = int(hr_match.group(1))
                    if heart_rate >= TACHYCARDIA_THRESHOLD:
                        rate_label = f"HR {heart_rate} bpm — tachycardia"
                    elif heart_rate < BRADYCARDIA_THRESHOLD:
                        rate_label = f"HR {heart_rate} bpm — bradycardia"
                    else:
                        rate_label = f"HR {heart_rate} bpm — normal rate"
                    findings.append(rate_label)
            else:
                if finding not in findings:
                    findings.append(finding)
                if finding in URGENT_CONDITIONS:
                    urgent_flags.append(finding)

    # Classify STEMI territory
    stemi_interpretation = ""
    if re.search(r"ST上昇\s*(?:V[1-4])", ecg_text, re.IGNORECASE):
        stemi_interpretation = "STEMI anterior (LAD territory)"
    elif re.search(r"ST上昇\s*(?:V[5-6]|aVL|I)", ecg_text, re.IGNORECASE):
        stemi_interpretation = "STEMI lateral (LCx territory)"
    elif re.search(r"ST上昇\s*(?:II|III|aVF)", ecg_text, re.IGNORECASE):
        stemi_interpretation = "STEMI inferior (RCA territory)"
    elif "ST elevation" in " ".join(findings):
        stemi_interpretation = "STEMI — territory unclear, check all leads"

    # LBBB with symptoms = STEMI-equivalent
    if "Left bundle branch block (LBBB)" in findings and re.search(r"胸痛|ST上昇", ecg_text):
        stemi_interpretation = "LBBB with chest pain — STEMI-equivalent"
        urgent_flags.append("LBBB + chest pain = STEMI-equivalent per Sgarbossa criteria")

    interpretation = stemi_interpretation if stemi_interpretation else (
        "No STEMI pattern detected" if "ST elevation" not in " ".join(findings) else "ST elevation without clear territory"
    )

    # Rhythm summary
    rhythm_parts = []
    if "Atrial fibrillation (AF)" in findings:
        rhythm_parts.append("AF")
    elif "Ventricular tachycardia (VT) — EMERGENCY" in findings:
        rhythm_parts.append("VT")
    elif "Ventricular fibrillation (VF) — CARDIAC ARREST" in findings:
        rhythm_parts.append("VF")
    elif "Supraventricular tachycardia (SVT)" in findings:
        rhythm_parts.append("SVT")
    elif heart_rate:
        rhythm_parts.append(f"Rate {heart_rate} bpm")
    rhythm_summary = ", ".join(rhythm_parts) if rhythm_parts else "Sinus rhythm (assumed)"

    # Recommended action
    if "Ventricular fibrillation (VF) — CARDIAC ARREST" in urgent_flags:
        action = "IMMEDIATE: Start CPR, defibrillation, call code team"
    elif stemi_interpretation and "STEMI" in stemi_interpretation:
        action = "URGENT: Activate cath lab, dual antiplatelet + anticoagulation, PCI within 90 min"
    elif "Ventricular tachycardia (VT) — EMERGENCY" in urgent_flags:
        action = "URGENT: Assess hemodynamic stability. Synchronized cardioversion if unstable. Amiodarone if stable"
    elif "High-degree AV block" in urgent_flags:
        action = "URGENT: Transcutaneous pacing, prepare for transvenous pacing, call cardiology"
    elif urgent_flags:
        action = "PRIORITY: Urgent cardiology review required"
    else:
        action = "Routine: Correlate with clinical findings and prior ECGs"

    return {
        "findings": findings,
        "urgent_flags": urgent_flags,
        "qtc_ms": qtc_ms,
        "heart_rate": heart_rate,
        "rhythm_summary": rhythm_summary,
        "interpretation": interpretation,
        "action": action,
    }


# ===========================================================================
# 2. Heart Failure Staging (NYHA + AHA/ACC)
# ===========================================================================

@dataclass
class HeartFailureInput:
    # Symptoms
    dyspnea_rest: bool = False          # Dyspnea at rest
    dyspnea_minimal_activity: bool = False  # Dyspnea with minimal activity
    dyspnea_moderate_activity: bool = False  # Dyspnea with moderate activity
    dyspnea_vigorous_activity: bool = False  # Dyspnea only with vigorous activity
    orthopnea: bool = False
    paroxysmal_nocturnal_dyspnea: bool = False
    edema: bool = False
    fatigue: bool = False
    # Objective
    ef_percent: Optional[float] = None  # ejection fraction
    nt_probnp_pgml: Optional[float] = None  # NT-proBNP pg/mL
    # History
    prior_hf_hospitalization: bool = False
    structural_heart_disease: bool = False  # e.g., LVH, prior MI
    hf_risk_factors_only: bool = False      # e.g., hypertension, DM, no structural disease
    asymptomatic_structural: bool = False   # structural disease but no symptoms


def classify_heart_failure(inp: HeartFailureInput) -> Dict[str, Any]:
    """
    Returns AHA/ACC Stage (A-D) and NYHA Class (I-IV) with medication targets.
    """
    # AHA/ACC Stage
    if inp.dyspnea_rest and (inp.nt_probnp_pgml is not None and inp.nt_probnp_pgml > 2000):
        acc_stage = "D"
        acc_desc = "Advanced HF — refractory symptoms despite GDMT"
    elif inp.dyspnea_rest or inp.dyspnea_minimal_activity or inp.prior_hf_hospitalization:
        acc_stage = "C"
        acc_desc = "Structural heart disease with prior or current HF symptoms"
    elif inp.structural_heart_disease or inp.asymptomatic_structural:
        acc_stage = "B"
        acc_desc = "Structural heart disease without HF symptoms"
    else:
        acc_stage = "A"
        acc_desc = "At risk for HF, no structural disease or symptoms"

    # NYHA Class
    if inp.dyspnea_rest:
        nyha = "IV"
        nyha_desc = "Symptoms at rest, unable to carry on any physical activity"
    elif inp.dyspnea_minimal_activity:
        nyha = "III"
        nyha_desc = "Symptoms with less-than-ordinary activity, marked limitation"
    elif inp.dyspnea_moderate_activity or inp.orthopnea or inp.paroxysmal_nocturnal_dyspnea:
        nyha = "II"
        nyha_desc = "Symptoms with ordinary activity, slight limitation"
    elif inp.structural_heart_disease or inp.asymptomatic_structural:
        nyha = "I"
        nyha_desc = "No symptoms with ordinary activity (underlying disease present)"
    else:
        nyha = "N/A"
        nyha_desc = "No structural disease — stage A"

    # EF category
    if inp.ef_percent is not None:
        if inp.ef_percent < 40:
            ef_cat = "HFrEF (EF < 40%)"
        elif inp.ef_percent < 50:
            ef_cat = "HFmrEF (EF 40-49%)"
        else:
            ef_cat = "HFpEF (EF ≥ 50%)"
    else:
        ef_cat = "EF not provided"

    # NT-proBNP interpretation
    nt_interp = "Not measured"
    if inp.nt_probnp_pgml is not None:
        if inp.nt_probnp_pgml < 125:
            nt_interp = f"NT-proBNP {inp.nt_probnp_pgml:.0f} pg/mL — likely not HF"
        elif inp.nt_probnp_pgml < 900:
            nt_interp = f"NT-proBNP {inp.nt_probnp_pgml:.0f} pg/mL — elevated, possible HF"
        else:
            nt_interp = f"NT-proBNP {inp.nt_probnp_pgml:.0f} pg/mL — significantly elevated, supports HF"

    # Medication targets based on stage + EF
    med_targets = []
    if acc_stage in ("C", "D") and inp.ef_percent is not None and inp.ef_percent < 40:
        med_targets = [
            "ACE inhibitor/ARB or ARNI (sacubitril/valsartan)",
            "Beta-blocker (carvedilol, metoprolol succinate, bisoprolol)",
            "MRA (spironolactone/eplerenone) if tolerated",
            "SGLT2 inhibitor (dapagliflozin/empagliflozin)",
            "Loop diuretic for congestion control",
        ]
    elif acc_stage == "B":
        med_targets = [
            "ACE inhibitor or ARB if prior MI or EF < 40%",
            "Beta-blocker if prior MI or EF < 40%",
            "Treat underlying risk factors aggressively",
        ]
    elif acc_stage == "A":
        med_targets = [
            "Blood pressure control",
            "Diabetes management",
            "Lipid-lowering therapy",
            "Lifestyle modification (exercise, diet, smoking cessation)",
        ]

    return {
        "acc_stage": acc_stage,
        "acc_description": acc_desc,
        "nyha_class": nyha,
        "nyha_description": nyha_desc,
        "ef_category": ef_cat,
        "nt_probnp_interpretation": nt_interp,
        "medication_targets": med_targets,
    }


# ===========================================================================
# 3. Anticoagulation Decision Tool
# ===========================================================================

@dataclass
class PatientProfile:
    age: int
    sex: str  # "M" or "F"
    # CHA₂DS₂-VASc components
    chf: bool = False           # Congestive heart failure (1 pt)
    hypertension: bool = False  # (1 pt)
    diabetes: bool = False      # (1 pt)
    stroke_tia_history: bool = False  # Prior stroke/TIA (2 pts)
    vascular_disease: bool = False    # Prior MI, PAD (1 pt)
    # HAS-BLED components
    uncontrolled_htn: bool = False   # SBP > 160 (1 pt)
    renal_dysfunction: bool = False  # Cr > 200 µmol/L or dialysis (1 pt)
    liver_dysfunction: bool = False  # Cirrhosis or bilirubin > 2x (1 pt)
    prior_stroke: bool = False       # (1 pt)
    prior_bleeding: bool = False     # or predisposition (1 pt)
    labile_inr: bool = False         # TTR < 60% (1 pt)
    alcohol_or_drugs: bool = False   # (1 pt)
    # Renal function for drug dosing
    egfr_ml_min: Optional[float] = None
    creatinine_umol_l: Optional[float] = None


def calculate_cha2ds2_vasc(p: PatientProfile) -> Tuple[int, str]:
    score = 0
    if p.chf:
        score += 1
    if p.hypertension:
        score += 1
    if p.age >= 75:
        score += 2
    elif p.age >= 65:
        score += 1
    if p.diabetes:
        score += 1
    if p.stroke_tia_history:
        score += 2
    if p.vascular_disease:
        score += 1
    if p.sex == "F":
        score += 1

    if score == 0 and p.sex == "M":
        risk = "Low — no anticoagulation required"
    elif score == 1 and p.sex == "M":
        risk = "Low-moderate — consider anticoagulation"
    elif score == 1 and p.sex == "F":
        risk = "Low — sex category only, no anticoagulation"
    else:
        risk = "High — anticoagulation recommended"
    return score, risk


def calculate_has_bled(p: PatientProfile) -> Tuple[int, str]:
    score = 0
    if p.uncontrolled_htn:
        score += 1
    if p.renal_dysfunction:
        score += 1
    if p.liver_dysfunction:
        score += 1
    if p.prior_stroke:
        score += 1
    if p.prior_bleeding:
        score += 1
    if p.labile_inr:
        score += 1
    if p.alcohol_or_drugs:
        score += 1
    if p.age >= 65:
        score += 1

    if score >= 3:
        risk = "High bleeding risk — address modifiable factors; do not withhold OAC"
    elif score >= 2:
        risk = "Moderate bleeding risk"
    else:
        risk = "Low bleeding risk"
    return score, risk


def _oac_recommendation(p: PatientProfile, cha2ds2_score: int) -> Dict[str, str]:
    """Recommend OAC type and dose based on renal function."""
    if cha2ds2_score < 2 and p.sex == "M":
        return {"oac": "None", "reason": "Low stroke risk — OAC not indicated"}
    if cha2ds2_score == 1 and p.sex == "F":
        return {"oac": "None", "reason": "Female sex only — OAC not indicated"}

    egfr = p.egfr_ml_min

    if egfr is None or egfr >= 30:
        # Prefer NOAC over warfarin
        if egfr is not None and egfr >= 50:
            options = [
                "Apixaban 5mg BID (or 2.5mg BID if ≥2 of: age≥80, weight≤60kg, Cr≥133µmol/L)",
                "Rivaroxaban 20mg OD with evening meal",
                "Dabigatran 150mg BID (or 110mg BID if age≥80 or bleeding risk)",
                "Edoxaban 60mg OD (or 30mg OD if CrCl 15-50 mL/min)",
            ]
        elif egfr is not None and 30 <= egfr < 50:
            options = [
                "Apixaban 5mg BID (preferred — least renal clearance)",
                "Rivaroxaban 15mg OD with meal (CrCl 15-49)",
                "Edoxaban 30mg OD",
                "Warfarin with careful INR monitoring (INR target 2.0-3.0)",
            ]
        else:
            options = [
                "Apixaban 5mg BID (can be used to CrCl ~15)",
                "Warfarin with INR 2.0-3.0 (traditional choice for severe renal impairment)",
                "Consult nephrology/haematology",
            ]
        oac_type = "NOAC preferred over VKA"
    else:
        # eGFR < 30 — NOACs largely contraindicated except apixaban
        options = [
            "Apixaban 5mg BID (limited data but most evidence in advanced CKD)",
            "Warfarin with careful INR monitoring (INR 2.0-3.0)",
            "Discuss with nephrology — high-risk decision",
        ]
        oac_type = "Severe CKD — limited options"

    return {
        "oac": oac_type,
        "options": options,
        "note": "Prefer NOAC over warfarin unless CrCl < 15 or on dialysis (individual assessment required)",
    }


def anticoagulation_recommendation(p: PatientProfile) -> Dict[str, Any]:
    cha2ds2_score, stroke_risk = calculate_cha2ds2_vasc(p)
    has_bled_score, bleed_risk = calculate_has_bled(p)
    oac = _oac_recommendation(p, cha2ds2_score)

    return {
        "cha2ds2_vasc": {"score": cha2ds2_score, "interpretation": stroke_risk},
        "has_bled": {"score": has_bled_score, "interpretation": bleed_risk},
        "oac_recommendation": oac,
        "key_message": (
            "High bleeding risk should prompt correction of modifiable factors "
            "(BP control, INR optimization, avoid NSAIDs/alcohol), "
            "but should NOT automatically lead to withholding OAC in high-stroke-risk patients."
        ),
    }


# ===========================================================================
# 4. Shock Classification
# ===========================================================================

@dataclass
class ShockInput:
    sbp_mmhg: float               # Systolic blood pressure
    dbp_mmhg: float               # Diastolic blood pressure
    hr_bpm: float                 # Heart rate
    map_mmhg: Optional[float] = None  # Mean arterial pressure (auto-calculated if None)
    # Clinical findings
    cool_clammy_skin: bool = False    # Cardiogenic / hypovolemic
    warm_flushed_skin: bool = False   # Distributive
    distended_neck_veins: bool = False  # Cardiogenic / obstructive
    absent_breath_sounds: bool = False  # Tension pneumothorax (obstructive)
    pulsus_paradoxus: bool = False    # Cardiac tamponade (obstructive)
    pulmonary_edema: bool = False     # Cardiogenic
    fever: bool = False               # Distributive (septic)
    trauma_or_blood_loss: bool = False  # Hypovolemic
    dehydration: bool = False         # Hypovolemic
    # Hemodynamic
    cardiac_output_low: Optional[bool] = None   # measured/estimated
    svr_high: Optional[bool] = None             # systemic vascular resistance
    svr_low: Optional[bool] = None


def classify_shock(inp: ShockInput) -> Dict[str, Any]:
    """
    Classify shock type and return management priorities.
    """
    if inp.map_mmhg is None:
        inp.map_mmhg = inp.dbp_mmhg + (inp.sbp_mmhg - inp.dbp_mmhg) / 3

    hypotension = inp.map_mmhg < 65 or inp.sbp_mmhg < 90
    tachycardia = inp.hr_bpm > 100

    shock_present = hypotension and tachycardia
    severity_label = "Shock" if shock_present else (
        "Hemodynamic instability" if hypotension or tachycardia else "No shock criteria met"
    )

    # Classification scoring
    scores = {
        "cardiogenic": 0,
        "distributive": 0,
        "hypovolemic": 0,
        "obstructive": 0,
    }

    if inp.cool_clammy_skin:
        scores["cardiogenic"] += 2
        scores["hypovolemic"] += 2
    if inp.warm_flushed_skin:
        scores["distributive"] += 3
    if inp.distended_neck_veins:
        scores["cardiogenic"] += 2
        scores["obstructive"] += 2
    if inp.pulmonary_edema:
        scores["cardiogenic"] += 3
    if inp.absent_breath_sounds:
        scores["obstructive"] += 3  # tension pneumo
    if inp.pulsus_paradoxus:
        scores["obstructive"] += 3  # tamponade
    if inp.fever:
        scores["distributive"] += 3
    if inp.trauma_or_blood_loss or inp.dehydration:
        scores["hypovolemic"] += 4
    if inp.cardiac_output_low:
        scores["cardiogenic"] += 2
        scores["hypovolemic"] += 1
        scores["obstructive"] += 1
    if inp.svr_high:
        scores["cardiogenic"] += 1
        scores["hypovolemic"] += 1
    if inp.svr_low:
        scores["distributive"] += 2

    shock_type = max(scores, key=lambda k: scores[k])
    confidence = scores[shock_type]

    management: Dict[str, List[str]] = {
        "cardiogenic": [
            "Minimize IV fluids (unless initial small bolus for preload assessment)",
            "Vasopressor: Norepinephrine first-line to MAP ≥65 mmHg",
            "Inotrope: Dobutamine if low cardiac output persists",
            "Consider intra-aortic balloon pump (IABP) or mechanical circulatory support",
            "Urgent echocardiography",
            "If STEMI: emergent PCI",
            "Target: MAP ≥65, UO ≥0.5 mL/kg/h, lactate trending down",
        ],
        "distributive": [
            "Aggressive IV fluid resuscitation (30 mL/kg crystalloid in first 3 hours if septic)",
            "Blood cultures ×2 before antibiotics",
            "Broad-spectrum antibiotics within 1 hour of sepsis recognition",
            "Vasopressor: Norepinephrine (first-line) to MAP ≥65 mmHg",
            "Add vasopressin if norepinephrine >0.25 µg/kg/min",
            "Hydrocortisone 200 mg/day if vasopressor-refractory",
            "ICU monitoring, source control",
        ],
        "hypovolemic": [
            "Aggressive IV crystalloid or colloid resuscitation",
            "If hemorrhagic: activate massive transfusion protocol (1:1:1 pRBC:FFP:PLT)",
            "Direct pressure / tourniquet for external hemorrhage",
            "Surgery/IR for internal hemorrhage control",
            "Tranexamic acid within 3 hours of injury (hemorrhagic)",
            "Target: MAP ≥65, Hgb ≥7-10 g/dL, correct coagulopathy",
        ],
        "obstructive": [
            "If tension pneumothorax: immediate needle decompression (2nd ICS, MCL), then chest tube",
            "If cardiac tamponade: urgent pericardiocentesis",
            "If massive PE: thrombolysis (alteplase 100 mg IV) or surgical embolectomy",
            "IV fluids cautiously (right heart preload dependent)",
            "Avoid vasodilators",
            "Emergent cardiothoracic/vascular surgery consult",
        ],
    }

    differentials = {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0}

    return {
        "hemodynamic_status": {
            "sbp": inp.sbp_mmhg,
            "dbp": inp.dbp_mmhg,
            "map": round(inp.map_mmhg, 1),
            "hr": inp.hr_bpm,
            "shock_present": shock_present,
            "severity": severity_label,
        },
        "type": shock_type,
        "confidence_score": confidence,
        "differential_scores": differentials,
        "management": management[shock_type],
        "immediate_actions": [
            "Ensure IV access ×2 (large bore)",
            "Supplemental O₂ / airway protection",
            "Continuous cardiac monitoring",
            "Point-of-care labs: ABG, lactate, BMP, CBC, coags, troponin",
            "POCUS (bedside echo + lung ultrasound) if available",
        ],
    }


# ===========================================================================
# 5. Cardiac Biomarker Calculator
# ===========================================================================

@dataclass
class TroponinSample:
    time_h: float       # Hours since symptom onset (or ED arrival)
    tni_ngl: float      # High-sensitivity TnI in ng/L


# Reference: ESC 0/1h algorithm (hs-cTnI, Architect i-STAT)
# Rule-out: TnI < 5 ng/L at 0h, or TnI < 12 ng/L at 0h + absolute change < 3 ng/L at 1h
# Rule-in:  TnI ≥ 52 ng/L at 0h, or absolute change ≥ 6 ng/L at 1h
RULE_OUT_THRESHOLD_0H = 5.0      # ng/L (very low)
RULE_IN_THRESHOLD_0H = 52.0     # ng/L
RULE_OUT_DELTA = 3.0            # ng/L change over 1-2h
RULE_IN_DELTA = 6.0             # ng/L change over 1-2h
NORMAL_UPPER = 26.0             # 99th percentile (sex-neutral approximation)


def interpret_troponin_kinetics(samples: List[TroponinSample]) -> Dict[str, Any]:
    """
    Interpret TnI samples: rising/falling pattern, delta TnI rule, AMI probability.

    Args:
        samples: list of TroponinSample (sorted by time_h)

    Returns:
        structured interpretation with AMI likelihood and recommendation
    """
    if not samples:
        return {"error": "No samples provided"}

    samples = sorted(samples, key=lambda s: s.time_h)

    values = [s.tni_ngl for s in samples]
    times = [s.time_h for s in samples]

    # Pattern analysis
    if len(values) >= 2:
        delta_first = values[1] - values[0]
        time_delta = times[1] - times[0]
        delta_per_hour = delta_first / time_delta if time_delta > 0 else 0
        if delta_first > 0 and values[-1] > values[0]:
            pattern = "Rising (evolving MI or acute injury)"
        elif delta_first < -0 and values[-1] < values[0]:
            pattern = "Falling (resolving MI or prior injury)"
        elif all(v < NORMAL_UPPER for v in values):
            pattern = "Persistently normal — myocardial injury unlikely"
        else:
            pattern = "Elevated but stable (chronic elevation — consider non-ischemic cause)"
    else:
        delta_first = None
        delta_per_hour = None
        pattern = "Single sample — cannot assess kinetics"

    # Delta TnI 0h/3h (or 0h/1h) rule
    ami_likelihood = "Indeterminate"
    recommendation = "Repeat TnI at 1-3 hours and correlate with symptoms/ECG"

    sample_0h = next((s for s in samples if s.time_h == 0), None) or samples[0]

    if sample_0h.tni_ngl < RULE_OUT_THRESHOLD_0H:
        ami_likelihood = "Very Low — Rule Out (ESC 0h criterion)"
        recommendation = "Discharge consideration if low pre-test probability and no symptoms; follow-up with PCP"
    elif sample_0h.tni_ngl >= RULE_IN_THRESHOLD_0H:
        ami_likelihood = "High — Rule In (ESC 0h criterion)"
        recommendation = "Admit, cardiology consult, early invasive strategy (coronary angiography within 24h)"

    if delta_first is not None:
        if abs(delta_first) <= RULE_OUT_DELTA and sample_0h.tni_ngl < 12:
            ami_likelihood = "Low — Rule Out (ESC 0/1h delta criterion)"
            recommendation = "Safe discharge if low pre-test probability; follow up in 4-6 weeks"
        elif abs(delta_first) >= RULE_IN_DELTA:
            ami_likelihood = "High — Rule In (ESC delta criterion)"
            recommendation = "Admit for NSTEMI management; dual antiplatelet, anticoagulation, early angiography"

    # Elevation level
    peak = max(values)
    if peak < NORMAL_UPPER:
        elevation_grade = "Normal (< 99th percentile)"
    elif peak < NORMAL_UPPER * 3:
        elevation_grade = "Mildly elevated (1-3× URL)"
    elif peak < NORMAL_UPPER * 10:
        elevation_grade = "Moderately elevated (3-10× URL)"
    else:
        elevation_grade = "Markedly elevated (> 10× URL) — major myocardial injury"

    return {
        "samples": [{"time_h": s.time_h, "tni_ngl": s.tni_ngl} for s in samples],
        "kinetics_pattern": pattern,
        "peak_tni_ngl": peak,
        "elevation_grade": elevation_grade,
        "delta_tni_ngl": round(delta_first, 2) if delta_first is not None else None,
        "delta_per_hour": round(delta_per_hour, 2) if delta_per_hour else None,
        "ami_likelihood": ami_likelihood,
        "recommendation": recommendation,
    }


# ===========================================================================
# Demo / Main
# ===========================================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def demo_ecg():
    print_section("1. ECG Interpretation Helper")
    cases = [
        "ST上昇 V1-V4, LBBB, QTc 480ms, AF HR 120",
        "ST低下 II III aVF, T波逆転 V4-V6, HR 95, RBBB",
        "VT HR 180, QTc 460ms",
        "正常洞調律 HR 72, QTc 420ms",
    ]
    for case in cases:
        print(f"\nECG Input: {case}")
        result = parse_ecg_text(case)
        print(f"  Findings       : {result['findings']}")
        print(f"  Urgent Flags   : {result['urgent_flags']}")
        print(f"  Rhythm         : {result['rhythm_summary']}")
        print(f"  Interpretation : {result['interpretation']}")
        print(f"  Action         : {result['action']}")


def demo_heart_failure():
    print_section("2. Heart Failure Staging")
    patient = HeartFailureInput(
        dyspnea_moderate_activity=True,
        orthopnea=True,
        edema=True,
        ef_percent=32.0,
        nt_probnp_pgml=1800,
        structural_heart_disease=True,
    )
    result = classify_heart_failure(patient)
    print(f"  AHA/ACC Stage : {result['acc_stage']} — {result['acc_description']}")
    print(f"  NYHA Class    : {result['nyha_class']} — {result['nyha_description']}")
    print(f"  EF Category   : {result['ef_category']}")
    print(f"  NT-proBNP     : {result['nt_probnp_interpretation']}")
    print(f"  Med Targets   :")
    for med in result["medication_targets"]:
        print(f"    - {med}")


def demo_anticoagulation():
    print_section("3. Anticoagulation Decision Tool")
    patient = PatientProfile(
        age=73,
        sex="F",
        chf=True,
        hypertension=True,
        diabetes=True,
        stroke_tia_history=True,
        vascular_disease=False,
        uncontrolled_htn=False,
        renal_dysfunction=False,
        liver_dysfunction=False,
        prior_stroke=True,
        prior_bleeding=False,
        labile_inr=False,
        alcohol_or_drugs=False,
        egfr_ml_min=55,
    )
    result = anticoagulation_recommendation(patient)
    c = result["cha2ds2_vasc"]
    h = result["has_bled"]
    oac = result["oac_recommendation"]
    print(f"  CHA₂DS₂-VASc : Score {c['score']} — {c['interpretation']}")
    print(f"  HAS-BLED     : Score {h['score']} — {h['interpretation']}")
    print(f"  OAC Type     : {oac['oac']}")
    if "options" in oac:
        print(f"  Options      :")
        for opt in oac["options"]:
            print(f"    - {opt}")


def demo_shock():
    print_section("4. Shock Classification")
    cardiogenic = ShockInput(
        sbp_mmhg=78, dbp_mmhg=50, hr_bpm=118,
        cool_clammy_skin=True, distended_neck_veins=True,
        pulmonary_edema=True, cardiac_output_low=True, svr_high=True,
    )
    result = classify_shock(cardiogenic)
    hd = result["hemodynamic_status"]
    print(f"  MAP: {hd['map']} mmHg  HR: {hd['hr']} bpm  Shock: {hd['shock_present']}")
    print(f"  Classification : {result['type'].upper()} shock (confidence={result['confidence_score']})")
    print(f"  Differentials  : {result['differential_scores']}")
    print(f"  Management     :")
    for step in result["management"][:4]:
        print(f"    - {step}")


def demo_troponin():
    print_section("5. Cardiac Biomarker Calculator (Delta TnI)")
    samples = [
        TroponinSample(time_h=0.0, tni_ngl=48.0),
        TroponinSample(time_h=1.0, tni_ngl=89.0),
        TroponinSample(time_h=3.0, tni_ngl=312.0),
    ]
    result = interpret_troponin_kinetics(samples)
    print(f"  Samples        : {[(s['time_h'], s['tni_ngl']) for s in result['samples']]}")
    print(f"  Kinetics       : {result['kinetics_pattern']}")
    print(f"  Peak TnI       : {result['peak_tni_ngl']} ng/L — {result['elevation_grade']}")
    print(f"  Delta TnI      : {result['delta_tni_ngl']} ng/L over first interval")
    print(f"  AMI Likelihood : {result['ami_likelihood']}")
    print(f"  Recommendation : {result['recommendation']}")


if __name__ == "__main__":
    print("Cardiology Decision Support Module — Demo")
    demo_ecg()
    demo_heart_failure()
    demo_anticoagulation()
    demo_shock()
    demo_troponin()
    print("\nDemo complete.")
