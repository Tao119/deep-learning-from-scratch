"""
Pulmonology Decision Support Module
=====================================
Implements five clinical tools:
  1. Spirometry interpretation (obstructive / restrictive / mixed + severity)
  2. Asthma severity + GINA stepwise treatment
  3. COPD severity — GOLD 2023 (CAT + mMRC + exacerbation history → GOLD 1-4 + ABCD groups)
  4. Oxygen therapy calculator
  5. Pulmonary Embolism probability (Wells + Revised Geneva + YEARS criteria)

All calculations follow published clinical guidelines (GINA 2023, GOLD 2023,
ESC/ERS 2019 PE guidelines, ATS/ERS spirometry standards).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# ===========================================================================
# 1. Spirometry Interpretation
# ===========================================================================

@dataclass
class SpirometryInput:
    fev1_percent_predicted: float   # FEV1 % predicted
    fvc_percent_predicted: float    # FVC % predicted
    fev1_fvc_ratio: float           # Post-bronchodilator ratio (fraction, e.g. 0.62)
    # Optional
    tlc_percent_predicted: Optional[float] = None  # Total lung capacity
    dlco_percent_predicted: Optional[float] = None  # Diffusing capacity
    post_bronchodilator: bool = True  # Whether values are post-BD


def interpret_spirometry(inp: SpirometryInput) -> Dict[str, Any]:
    """
    Classify pattern (obstructive / restrictive / mixed / normal) and severity.
    Uses ATS/ERS lower limit of normal (LLN) concept: FEV1/FVC < 0.70 threshold.
    """
    ratio = inp.fev1_fvc_ratio
    fev1_pred = inp.fev1_percent_predicted
    fvc_pred = inp.fvc_percent_predicted

    # Pattern classification
    obstructive = ratio < 0.70
    # Restriction suggested by FVC < 80% when obstruction is absent,
    # confirmed by TLC < 80% if available
    restrictive_suspected = (not obstructive) and (fvc_pred < 80)
    restrictive_confirmed = (
        restrictive_suspected
        and inp.tlc_percent_predicted is not None
        and inp.tlc_percent_predicted < 80
    )
    mixed = obstructive and (fvc_pred < 80)

    if mixed:
        pattern = "Mixed (obstructive + restrictive)"
    elif obstructive:
        pattern = "Obstructive"
    elif restrictive_confirmed:
        pattern = "Restrictive (confirmed by TLC)"
    elif restrictive_suspected:
        pattern = "Restrictive (suspected — TLC measurement recommended)"
    else:
        pattern = "Normal spirometry"

    # Obstruction severity (GOLD / ATS-ERS grade based on FEV1)
    if obstructive:
        if fev1_pred >= 80:
            obs_severity = "GOLD 1 — Mild (FEV1 ≥ 80%)"
        elif fev1_pred >= 50:
            obs_severity = "GOLD 2 — Moderate (FEV1 50-79%)"
        elif fev1_pred >= 30:
            obs_severity = "GOLD 3 — Severe (FEV1 30-49%)"
        else:
            obs_severity = "GOLD 4 — Very Severe (FEV1 < 30%)"
    else:
        obs_severity = "N/A"

    # Restriction severity
    if restrictive_suspected or restrictive_confirmed:
        if fvc_pred >= 70:
            rest_severity = "Mild restriction (FVC 70-79%)"
        elif fvc_pred >= 60:
            rest_severity = "Moderate restriction (FVC 60-69%)"
        elif fvc_pred >= 50:
            rest_severity = "Severe restriction (FVC 50-59%)"
        else:
            rest_severity = "Very severe restriction (FVC < 50%)"
    else:
        rest_severity = "N/A"

    # DLCO interpretation
    dlco_interp = "Not measured"
    if inp.dlco_percent_predicted is not None:
        d = inp.dlco_percent_predicted
        if d >= 80:
            dlco_interp = f"DLCO {d:.0f}% — Normal"
        elif d >= 60:
            dlco_interp = f"DLCO {d:.0f}% — Mildly reduced"
        elif d >= 40:
            dlco_interp = f"DLCO {d:.0f}% — Moderately reduced"
        else:
            dlco_interp = f"DLCO {d:.0f}% — Severely reduced"

    # Bronchodilator test flag
    bd_note = (
        "Post-bronchodilator values used (recommended)" if inp.post_bronchodilator
        else "Pre-bronchodilator values — repeat post-BD for definitive assessment"
    )

    # Clinical suggestions
    suggestions = []
    if obstructive:
        suggestions.append("Consider COPD or asthma — clinical correlation required")
        suggestions.append("Bronchodilator reversibility: ≥200 mL AND ≥12% FEV1 increase suggests asthma")
    if restrictive_suspected and not restrictive_confirmed:
        suggestions.append("Order full lung volumes (body plethysmography) to confirm restriction")
    if inp.dlco_percent_predicted is not None and inp.dlco_percent_predicted < 60:
        suggestions.append("Low DLCO suggests emphysema, interstitial lung disease, or pulmonary vascular disease")
    if mixed:
        suggestions.append("Mixed pattern may represent COPD + fibrosis or severe asthma + remodeling")

    return {
        "pattern": pattern,
        "obstruction_severity": obs_severity,
        "restriction_severity": rest_severity,
        "fev1_fvc_ratio": round(ratio, 3),
        "fev1_percent": fev1_pred,
        "fvc_percent": fvc_pred,
        "dlco_interpretation": dlco_interp,
        "bronchodilator_note": bd_note,
        "clinical_suggestions": suggestions,
    }


# ===========================================================================
# 2. Asthma Severity + GINA Stepwise Treatment
# ===========================================================================

@dataclass
class AsthmaInput:
    # Symptom frequency
    daytime_symptoms_per_week: int    # 0, 1-2, 3-4, daily
    nocturnal_awakenings_per_month: int  # 0, 1-2, ≥3
    saba_use_per_week: int            # symptom-driven SABA puffs per week
    activity_limitation: bool = False
    # Objective
    fev1_percent_predicted: Optional[float] = None
    exacerbations_past_year: int = 0
    oral_corticosteroid_courses_past_year: int = 0
    # Current treatment step (1-5), if known
    current_step: Optional[int] = None


def classify_asthma(inp: AsthmaInput) -> Dict[str, Any]:
    """
    GINA 2023 severity and step classification.
    """
    # Determine GINA-based symptom burden
    # Intermittent: ≤2 days/week, no nocturnal, FEV1 ≥80%
    # Mild persistent: >2 days/week but not daily
    # Moderate persistent: daily symptoms, ≥1 nocturnal/week
    # Severe persistent: throughout day, frequent nocturnal

    severity_score = 0
    if inp.daytime_symptoms_per_week > 4:
        severity_score += 4
    elif inp.daytime_symptoms_per_week > 2:
        severity_score += 2
    elif inp.daytime_symptoms_per_week > 0:
        severity_score += 1

    if inp.nocturnal_awakenings_per_month >= 4:
        severity_score += 3
    elif inp.nocturnal_awakenings_per_month > 1:
        severity_score += 2
    elif inp.nocturnal_awakenings_per_month > 0:
        severity_score += 1

    if inp.saba_use_per_week > 7:
        severity_score += 2
    elif inp.saba_use_per_week > 2:
        severity_score += 1

    if inp.activity_limitation:
        severity_score += 1

    if inp.fev1_percent_predicted is not None:
        if inp.fev1_percent_predicted < 60:
            severity_score += 3
        elif inp.fev1_percent_predicted < 80:
            severity_score += 1

    if inp.exacerbations_past_year >= 2:
        severity_score += 3
    elif inp.exacerbations_past_year == 1:
        severity_score += 1

    # Map score to severity label
    if severity_score <= 1:
        severity = "Intermittent"
    elif severity_score <= 3:
        severity = "Mild Persistent"
    elif severity_score <= 6:
        severity = "Moderate Persistent"
    else:
        severity = "Severe Persistent"

    # GINA Step recommendation
    gina_step_map = {
        "Intermittent": 1,
        "Mild Persistent": 2,
        "Moderate Persistent": 3,
        "Severe Persistent": 4,
    }
    recommended_step = gina_step_map[severity]
    # If on current treatment and uncontrolled, step up
    if inp.current_step is not None and inp.current_step >= recommended_step and severity_score >= 5:
        recommended_step = min(inp.current_step + 1, 5)

    step_treatments = {
        1: {
            "controller": "None (SABA PRN or low-dose ICS-formoterol PRN)",
            "reliever": "SABA (albuterol/salbutamol) PRN — or low-dose ICS-formoterol PRN",
            "note": "GINA 2023 no longer recommends SABA-only for any step",
        },
        2: {
            "controller": "Low-dose ICS daily (e.g., budesonide 200-400 µg/day or equivalent)",
            "reliever": "SABA PRN or low-dose ICS-formoterol PRN",
            "note": "Adherence monitoring critical; consider MART (maintenance + reliever) with ICS-formoterol",
        },
        3: {
            "controller": "Low-dose ICS + LABA (e.g., budesonide/formoterol or fluticasone/salmeterol)",
            "reliever": "Low-dose ICS-formoterol (preferred) or SABA PRN",
            "note": "Referral to specialist if still uncontrolled after 3 months",
        },
        4: {
            "controller": "Medium/high-dose ICS + LABA; add LAMA (tiotropium) if uncontrolled",
            "reliever": "Low-dose ICS-formoterol or SABA PRN",
            "note": "Consider phenotyping: eosinophilic (add anti-IL-5), allergic (omalizumab), TSLP (tezepelumab)",
        },
        5: {
            "controller": "High-dose ICS + LABA + additional biologics (anti-IL-5: mepolizumab, benralizumab; anti-IgE: omalizumab; anti-TSLP: tezepelumab)",
            "reliever": "Low-dose ICS-formoterol or SABA PRN",
            "note": "Refer to severe asthma specialist. Consider maintenance OCS only as last resort.",
        },
    }

    exacerbation_risk = "Low"
    if inp.exacerbations_past_year >= 2 or inp.oral_corticosteroid_courses_past_year >= 2:
        exacerbation_risk = "High — consider biologic therapy evaluation, ensure ICS adherence"
    elif inp.exacerbations_past_year == 1:
        exacerbation_risk = "Moderate — review triggers, technique, adherence"

    return {
        "severity": severity,
        "severity_score": severity_score,
        "recommended_gina_step": recommended_step,
        "treatment": step_treatments[recommended_step],
        "exacerbation_risk": exacerbation_risk,
        "monitoring": [
            "Asthma Control Test (ACT) every visit",
            "Peak expiratory flow monitoring if poor perception of obstruction",
            "Inhaler technique review at every visit",
            "Trigger identification and avoidance",
        ],
    }


# ===========================================================================
# 3. COPD Severity — GOLD 2023
# ===========================================================================

@dataclass
class COPDInput:
    # Spirometry
    fev1_percent_predicted: float   # Post-BD FEV1 % predicted
    fev1_fvc_ratio: float           # Post-BD ratio (< 0.70 required for COPD diagnosis)
    # Symptom burden
    cat_score: int                  # COPD Assessment Test (0-40)
    mmrc_score: int                 # mMRC dyspnea scale (0-4)
    # Exacerbation history (past 12 months)
    moderate_exacerbations: int = 0  # requiring antibiotics/OCS (outpatient)
    severe_exacerbations: int = 0    # requiring hospitalization


def classify_copd(inp: COPDInput) -> Dict[str, Any]:
    """
    GOLD 2023 classification: Grades 1-4 (spirometry) + Groups A/B/E (ABE model).
    2023 update replaces ABCD with ABE groups.
    """
    # Spirometric grade (requires FEV1/FVC < 0.70)
    confirmed = inp.fev1_fvc_ratio < 0.70
    if not confirmed:
        return {
            "error": f"FEV1/FVC = {inp.fev1_fvc_ratio:.2f} — does not meet COPD spirometric criterion (< 0.70)",
            "suggestion": "Re-evaluate diagnosis; consider asthma, restriction, or normal variant",
        }

    fev1 = inp.fev1_percent_predicted
    if fev1 >= 80:
        gold_grade = 1
        grade_label = "GOLD 1 — Mild (FEV1 ≥ 80%)"
    elif fev1 >= 50:
        gold_grade = 2
        grade_label = "GOLD 2 — Moderate (FEV1 50-79%)"
    elif fev1 >= 30:
        gold_grade = 3
        grade_label = "GOLD 3 — Severe (FEV1 30-49%)"
    else:
        gold_grade = 4
        grade_label = "GOLD 4 — Very Severe (FEV1 < 30%)"

    # Symptom burden
    high_symptoms = inp.cat_score >= 10 or inp.mmrc_score >= 2

    # Exacerbation history
    total_exacerbations = inp.moderate_exacerbations + inp.severe_exacerbations
    high_exacerbation_risk = (total_exacerbations >= 2) or (inp.severe_exacerbations >= 1)

    # GOLD 2023 ABE model
    if high_exacerbation_risk:
        gold_group = "E"
        group_desc = "High exacerbation risk (≥2 moderate or ≥1 hospitalized)"
    elif high_symptoms:
        gold_group = "B"
        group_desc = "High symptom burden (CAT ≥10 or mMRC ≥2)"
    else:
        gold_group = "A"
        group_desc = "Low symptom burden and low exacerbation risk"

    # Initial pharmacotherapy recommendations (GOLD 2023)
    pharma_map = {
        "A": {
            "initial": ["Short-acting bronchodilator PRN (SABA or SAMA)"],
            "escalation": ["Regular LABA or LAMA if persistent symptoms"],
            "note": "Reassess after 3 months",
        },
        "B": {
            "initial": ["LAMA (tiotropium, umeclidinium, glycopyrronium) — preferred",
                        "OR LABA (indacaterol, olodaterol, formoterol)",
                        "LAMA + LABA dual if highly symptomatic"],
            "escalation": ["LAMA + LABA if single agent insufficient"],
            "note": "LAMA preferred for exacerbation prevention; LABA for dyspnea",
        },
        "E": {
            "initial": ["LAMA + LABA dual bronchodilator therapy",
                        "Add ICS if blood eosinophils ≥300 cells/µL"],
            "escalation": ["Triple therapy (ICS + LABA + LAMA) if frequent exacerbations on LAMA+LABA",
                           "Roflumilast (PDE4 inhibitor) if FEV1 < 50% + chronic bronchitis",
                           "Azithromycin maintenance in ex-smokers with frequent exacerbations"],
            "note": "ICS should NOT be used in monotherapy; blood eos guides ICS use",
        },
    }
    pharma = pharma_map[gold_group]

    # Non-pharmacological
    non_pharma = [
        "Smoking cessation (most important intervention)",
        "Pulmonary rehabilitation (all grades, especially GOLD 3-4)",
        "Annual influenza vaccine + pneumococcal vaccine",
        "COVID-19 vaccine",
    ]
    if gold_grade >= 3:
        non_pharma.append("Assess for LTOT (long-term oxygen therapy) if PaO₂ ≤ 55 mmHg or SpO₂ ≤ 88%")
    if gold_grade == 4:
        non_pharma.append("Consider lung volume reduction surgery or lung transplant evaluation")

    return {
        "copd_confirmed": True,
        "gold_grade": gold_grade,
        "grade_label": grade_label,
        "gold_group": gold_group,
        "group_description": group_desc,
        "cat_score": inp.cat_score,
        "mmrc_score": inp.mmrc_score,
        "exacerbations": {
            "moderate": inp.moderate_exacerbations,
            "severe": inp.severe_exacerbations,
            "total": total_exacerbations,
        },
        "pharmacotherapy": pharma,
        "non_pharmacological": non_pharma,
    }


# ===========================================================================
# 4. Oxygen Therapy Calculator
# ===========================================================================

@dataclass
class OxygenInput:
    spo2_percent: float                # Pulse oximetry
    respiratory_rate: int              # breaths per minute
    # Risk of hypercapnic failure (CO₂ retention)
    copd_or_known_co2_retention: bool = False
    # Available PaO2 if measured
    pao2_mmhg: Optional[float] = None
    paco2_mmhg: Optional[float] = None
    # Clinical context
    acute_illness: bool = True


def calculate_oxygen_therapy(inp: OxygenInput) -> Dict[str, Any]:
    """
    Recommend O₂ flow rate, delivery device, target SpO₂, and monitoring.
    Follows BTS O₂ guidelines and ESC/ERS COPD emergency management.
    """
    spo2 = inp.spo2_percent

    # Target SpO₂
    if inp.copd_or_known_co2_retention:
        target_spo2 = "88-92%"
        hypercapnia_risk = "High — risk of hypercapnic respiratory failure with excess O₂"
        max_fio2_guidance = "Start low (24-28% FiO₂); titrate carefully"
    else:
        target_spo2 = "94-98%"
        hypercapnia_risk = "Low — standard O₂ therapy"
        max_fio2_guidance = "Titrate to maintain SpO₂ 94-98%"

    # Severity assessment
    if spo2 < 85:
        severity = "Critical hypoxemia"
        urgency = "Immediate O₂ therapy required — consider NIV/intubation"
    elif spo2 < 90:
        severity = "Severe hypoxemia"
        urgency = "Urgent O₂ therapy; prepare for escalation"
    elif spo2 < 94:
        severity = "Moderate hypoxemia"
        urgency = "O₂ therapy indicated; close monitoring"
    elif spo2 >= 94 and not inp.copd_or_known_co2_retention:
        severity = "Mild or no hypoxemia"
        urgency = "Supplemental O₂ not routinely indicated unless symptomatic"
    else:
        severity = "Borderline for COPD target"
        urgency = "Monitor closely; titrate O₂ carefully"

    # Device and flow recommendation
    if inp.copd_or_known_co2_retention:
        if spo2 < 88:
            device = "Venturi mask 28% (4 L/min) or 24% (2 L/min) — start low"
            flow_lpm = "2-4 L/min via Venturi"
        elif spo2 < 92:
            device = "Nasal cannula 1-2 L/min (≈ 24-28% FiO₂)"
            flow_lpm = "1-2 L/min"
        else:
            device = "Nasal cannula 1 L/min — only if symptomatic"
            flow_lpm = "0-1 L/min"
        escalation = "If SpO₂ < 88% or respiratory distress despite O₂ → NIV (BiPAP)"
    else:
        if spo2 < 85 or (inp.respiratory_rate > 30 and inp.acute_illness):
            device = "Non-rebreather mask (NRB) 10-15 L/min (FiO₂ ~60-80%)"
            flow_lpm = "10-15 L/min"
            escalation = "If no improvement within 5-10 min → prepare for intubation"
        elif spo2 < 90:
            device = "Venturi mask 40-60% OR high-flow nasal cannula (HFNC) 20-40 L/min"
            flow_lpm = "8-12 L/min (mask) or 20-40 L/min (HFNC)"
            escalation = "If SpO₂ < 90% after 30 min → escalate to NIV or intubation"
        elif spo2 < 94:
            device = "Nasal cannula 2-4 L/min (FiO₂ ≈ 28-36%)"
            flow_lpm = "2-4 L/min"
            escalation = "Reassess in 15-30 min; increase flow if no response"
        else:
            device = "No supplemental O₂ needed (SpO₂ ≥ 94% without COPD)"
            flow_lpm = "0 L/min"
            escalation = "Monitor; O₂ if SpO₂ drops below 94%"

    # PaO₂/PaCO₂ interpretation
    gas_interp = []
    if inp.pao2_mmhg is not None:
        if inp.pao2_mmhg < 60:
            gas_interp.append(f"PaO₂ {inp.pao2_mmhg} mmHg — significant hypoxemia (LTOT threshold)")
        elif inp.pao2_mmhg < 80:
            gas_interp.append(f"PaO₂ {inp.pao2_mmhg} mmHg — mild hypoxemia")
        else:
            gas_interp.append(f"PaO₂ {inp.pao2_mmhg} mmHg — adequate oxygenation")

    if inp.paco2_mmhg is not None:
        if inp.paco2_mmhg > 50:
            gas_interp.append(f"PaCO₂ {inp.paco2_mmhg} mmHg — hypercapnia (type 2 respiratory failure)")
        elif inp.paco2_mmhg > 45:
            gas_interp.append(f"PaCO₂ {inp.paco2_mmhg} mmHg — upper normal / early hypercapnia")
        else:
            gas_interp.append(f"PaCO₂ {inp.paco2_mmhg} mmHg — normal")

    return {
        "spo2_measured": spo2,
        "severity": severity,
        "urgency": urgency,
        "target_spo2": target_spo2,
        "hypercapnia_risk": hypercapnia_risk,
        "recommended_device": device,
        "flow_rate": flow_lpm,
        "fio2_guidance": max_fio2_guidance,
        "escalation_trigger": escalation,
        "arterial_blood_gas": gas_interp if gas_interp else ["Not measured"],
        "monitoring": [
            "Continuous pulse oximetry",
            "ABG within 30-60 min of O₂ initiation (especially if COPD)",
            "Respiratory rate and work of breathing",
            "Reassess device/flow every 15-30 min during acute phase",
        ],
    }


# ===========================================================================
# 5. PE Probability Calculator
# ===========================================================================

@dataclass
class PEInput:
    # Clinical features
    age_over_65: bool = False
    prior_dvt_or_pe: bool = False
    recent_surgery_or_immobilization: bool = False  # last 4 weeks (Wells) or last month
    active_cancer: bool = False
    hemoptysis: bool = False
    clinical_signs_dvt: bool = False    # leg swelling + tenderness
    tachycardia: bool = False           # HR > 100
    pe_more_likely_than_alternative: bool = False  # clinical judgment (Wells)
    # Additional for Revised Geneva
    unilateral_leg_pain: bool = False
    pain_on_deep_palpation_with_edema: bool = False
    age_over_75: bool = False           # Revised Geneva
    heart_rate_75_94: bool = False      # Revised Geneva — moderate tachycardia
    heart_rate_ge_95: bool = False      # Revised Geneva — severe tachycardia
    # YEARS criteria
    years_dvt_signs: bool = False       # Clinical signs of DVT
    years_pe_diagnosis_most_likely: bool = False
    years_hemoptysis: bool = False
    # D-dimer
    ddimer_ug_ml: Optional[float] = None  # D-dimer in µg/mL (mg/L equivalent)
    patient_age: Optional[int] = None     # For age-adjusted D-dimer


def calculate_pe_probability(inp: PEInput) -> Dict[str, Any]:
    """
    Calculate PE pretest probability using Wells, Revised Geneva, and YEARS.
    Determine D-dimer threshold and CTPA need.
    """
    # -----------------------------------------------------------------------
    # Wells Score
    # -----------------------------------------------------------------------
    wells = 0
    wells_details = []
    if inp.clinical_signs_dvt:
        wells += 3
        wells_details.append("+3: Clinical signs of DVT")
    if inp.pe_more_likely_than_alternative:
        wells += 3
        wells_details.append("+3: PE more likely than alternative diagnosis")
    if inp.tachycardia:
        wells += 1.5
        wells_details.append("+1.5: HR > 100 bpm")
    if inp.recent_surgery_or_immobilization:
        wells += 1.5
        wells_details.append("+1.5: Immobilization/surgery last 4 weeks")
    if inp.prior_dvt_or_pe:
        wells += 1.5
        wells_details.append("+1.5: Prior DVT/PE")
    if inp.hemoptysis:
        wells += 1
        wells_details.append("+1: Hemoptysis")
    if inp.active_cancer:
        wells += 1
        wells_details.append("+1: Active cancer")

    if wells <= 1:
        wells_prob = "Low (< 10%)"
        wells_action = "D-dimer; if negative → PE excluded"
    elif wells <= 4:
        wells_prob = "Moderate (10-40%)"
        wells_action = "D-dimer; if positive → CTPA"
    elif wells > 6:
        wells_prob = "High (> 40%)"
        wells_action = "Proceed directly to CTPA (D-dimer insufficient)"
    else:
        wells_prob = "Moderate-High"
        wells_action = "CTPA recommended"

    # -----------------------------------------------------------------------
    # Revised Geneva Score
    # -----------------------------------------------------------------------
    rev_geneva = 0
    rg_details = []
    if inp.age_over_75:
        rev_geneva += 4
        rg_details.append("+4: Age > 75")
    elif inp.age_over_65:
        rev_geneva += 1
        rg_details.append("+1: Age 65-74")
    if inp.prior_dvt_or_pe:
        rev_geneva += 3
        rg_details.append("+3: Prior DVT/PE")
    if inp.recent_surgery_or_immobilization:
        rev_geneva += 3
        rg_details.append("+3: Surgery/immobilization last month")
    if inp.active_cancer:
        rev_geneva += 2
        rg_details.append("+2: Active cancer")
    if inp.unilateral_leg_pain:
        rev_geneva += 3
        rg_details.append("+3: Unilateral leg pain")
    if inp.hemoptysis:
        rev_geneva += 2
        rg_details.append("+2: Hemoptysis")
    if inp.heart_rate_ge_95:
        rev_geneva += 5
        rg_details.append("+5: HR ≥ 95 bpm")
    elif inp.heart_rate_75_94:
        rev_geneva += 3
        rg_details.append("+3: HR 75-94 bpm")
    if inp.pain_on_deep_palpation_with_edema:
        rev_geneva += 4
        rg_details.append("+4: Pain on deep palpation + unilateral edema")

    if rev_geneva <= 3:
        rg_prob = "Low (< 10%)"
        rg_action = "D-dimer; if negative → PE excluded"
    elif rev_geneva <= 10:
        rg_prob = "Intermediate (10-50%)"
        rg_action = "D-dimer if score ≤6; CTPA if score 7-10"
    else:
        rg_prob = "High (> 50%)"
        rg_action = "Proceed directly to CTPA"

    # -----------------------------------------------------------------------
    # YEARS Algorithm
    # -----------------------------------------------------------------------
    years_items = 0
    if inp.years_dvt_signs:
        years_items += 1
    if inp.years_pe_diagnosis_most_likely:
        years_items += 1
    if inp.years_hemoptysis:
        years_items += 1

    # YEARS + D-dimer interpretation
    years_result = {}
    if inp.ddimer_ug_ml is not None:
        dd = inp.ddimer_ug_ml
        if years_items == 0:
            threshold = 1.0  # µg/mL (1000 ng/mL)
            years_result["threshold_applied"] = "1.0 µg/mL (no YEARS criteria)"
        else:
            threshold = 0.5  # Standard threshold
            years_result["threshold_applied"] = "0.5 µg/mL (≥1 YEARS criterion)"

        if dd < threshold:
            years_result["conclusion"] = f"PE excluded (D-dimer {dd:.2f} < {threshold:.1f} µg/mL)"
        else:
            years_result["conclusion"] = f"CTPA indicated (D-dimer {dd:.2f} ≥ {threshold:.1f} µg/mL)"
    else:
        years_result["conclusion"] = "D-dimer not provided — measure to complete YEARS algorithm"
        years_result["threshold_applied"] = f"Threshold: {'1.0' if years_items == 0 else '0.5'} µg/mL"

    years_result["criteria_met"] = years_items

    # -----------------------------------------------------------------------
    # Age-adjusted D-dimer threshold
    # -----------------------------------------------------------------------
    age_adjusted_threshold = 0.5  # default µg/mL
    age_threshold_note = ""
    if inp.patient_age is not None and inp.patient_age > 50:
        age_adjusted_threshold = inp.patient_age * 0.01
        age_threshold_note = (
            f"Age-adjusted threshold: {age_adjusted_threshold:.2f} µg/mL "
            f"(age {inp.patient_age} × 0.01) — use for low/intermediate pre-test probability"
        )

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    # Rough consensus
    high_risk_count = sum([
        wells > 6,
        rev_geneva > 10,
        years_items >= 2,
    ])
    if high_risk_count >= 2:
        consensus = "High pre-test probability — proceed to CTPA without D-dimer"
    elif (wells <= 1 and rev_geneva <= 3):
        consensus = "Low pre-test probability — D-dimer; if negative → PE excluded"
    else:
        consensus = "Intermediate pre-test probability — D-dimer with age adjustment; CTPA if positive"

    return {
        "wells": {
            "score": wells,
            "probability": wells_prob,
            "details": wells_details,
            "action": wells_action,
        },
        "revised_geneva": {
            "score": rev_geneva,
            "probability": rg_prob,
            "details": rg_details,
            "action": rg_action,
        },
        "years": years_result,
        "d_dimer": {
            "value_ugml": inp.ddimer_ug_ml,
            "standard_threshold_ugml": 0.5,
            "age_adjusted_threshold": age_adjusted_threshold,
            "age_threshold_note": age_threshold_note if age_threshold_note else "Not applicable (age ≤ 50)",
        },
        "consensus_recommendation": consensus,
        "if_pe_confirmed_management": [
            "Anticoagulation: LMWH or fondaparinux bridge to NOAC, or direct NOAC (rivaroxaban/apixaban)",
            "Massive PE (hemodynamic instability): systemic thrombolysis (alteplase) or catheter-directed therapy",
            "Submassive PE (RV strain, no shock): consider catheter-directed thrombolysis",
            "Duration: 3 months if provoked; 6+ months if unprovoked; indefinite if cancer/recurrent",
        ],
    }


# ===========================================================================
# Demo / Main
# ===========================================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def demo_spirometry():
    print_section("1. Spirometry Interpretation")
    cases = [
        SpirometryInput(fev1_percent_predicted=55, fvc_percent_predicted=82,
                        fev1_fvc_ratio=0.55, dlco_percent_predicted=48),
        SpirometryInput(fev1_percent_predicted=72, fvc_percent_predicted=65,
                        fev1_fvc_ratio=0.78, tlc_percent_predicted=70),
        SpirometryInput(fev1_percent_predicted=96, fvc_percent_predicted=98,
                        fev1_fvc_ratio=0.80),
    ]
    labels = ["Case 1 (COPD)", "Case 2 (Restriction)", "Case 3 (Normal)"]
    for label, case in zip(labels, cases):
        result = interpret_spirometry(case)
        print(f"\n  {label}")
        print(f"    Pattern    : {result['pattern']}")
        print(f"    Obstruction: {result['obstruction_severity']}")
        print(f"    Restriction: {result['restriction_severity']}")
        if result['clinical_suggestions']:
            print(f"    Suggestions: {result['clinical_suggestions'][0]}")


def demo_asthma():
    print_section("2. Asthma Severity + GINA Step")
    patient = AsthmaInput(
        daytime_symptoms_per_week=5,
        nocturnal_awakenings_per_month=4,
        saba_use_per_week=8,
        activity_limitation=True,
        fev1_percent_predicted=62,
        exacerbations_past_year=2,
        oral_corticosteroid_courses_past_year=2,
        current_step=3,
    )
    result = classify_asthma(patient)
    print(f"\n  Severity      : {result['severity']}")
    print(f"  GINA Step     : Step {result['recommended_gina_step']}")
    print(f"  Controller    : {result['treatment']['controller']}")
    print(f"  Reliever      : {result['treatment']['reliever']}")
    print(f"  Exacer. Risk  : {result['exacerbation_risk']}")


def demo_copd():
    print_section("3. COPD Severity — GOLD 2023")
    patient = COPDInput(
        fev1_percent_predicted=42,
        fev1_fvc_ratio=0.58,
        cat_score=18,
        mmrc_score=3,
        moderate_exacerbations=1,
        severe_exacerbations=1,
    )
    result = classify_copd(patient)
    print(f"\n  GOLD Grade   : {result['grade_label']}")
    print(f"  GOLD Group   : {result['gold_group']} — {result['group_description']}")
    print(f"  Initial Rx   :")
    for rx in result["pharmacotherapy"]["initial"]:
        print(f"    - {rx}")
    print(f"  Non-Pharma   :")
    for item in result["non_pharmacological"][:3]:
        print(f"    - {item}")


def demo_oxygen():
    print_section("4. Oxygen Therapy Calculator")
    cases = [
        OxygenInput(spo2_percent=82, respiratory_rate=28, copd_or_known_co2_retention=False, acute_illness=True),
        OxygenInput(spo2_percent=86, respiratory_rate=24, copd_or_known_co2_retention=True,
                    paco2_mmhg=55, acute_illness=True),
    ]
    labels = ["Acute Hypoxemia (no COPD)", "COPD Exacerbation with Hypercapnia"]
    for label, case in zip(labels, cases):
        result = calculate_oxygen_therapy(case)
        print(f"\n  {label}")
        print(f"    SpO₂      : {result['spo2_measured']}%  →  Target: {result['target_spo2']}")
        print(f"    Severity  : {result['severity']}")
        print(f"    Device    : {result['recommended_device']}")
        print(f"    Flow      : {result['flow_rate']}")
        print(f"    Escalation: {result['escalation_trigger']}")
        if result["arterial_blood_gas"] != ["Not measured"]:
            print(f"    ABG       : {result['arterial_blood_gas']}")


def demo_pe():
    print_section("5. PE Probability Calculator (Wells + Revised Geneva + YEARS)")
    patient = PEInput(
        tachycardia=True,
        recent_surgery_or_immobilization=True,
        clinical_signs_dvt=True,
        pe_more_likely_than_alternative=True,
        heart_rate_ge_95=True,
        unilateral_leg_pain=True,
        years_dvt_signs=True,
        years_pe_diagnosis_most_likely=True,
        ddimer_ug_ml=2.8,
        patient_age=68,
        age_over_65=True,
    )
    result = calculate_pe_probability(patient)
    print(f"\n  Wells Score          : {result['wells']['score']} — {result['wells']['probability']}")
    print(f"  Wells Action         : {result['wells']['action']}")
    print(f"  Revised Geneva Score : {result['revised_geneva']['score']} — {result['revised_geneva']['probability']}")
    print(f"  YEARS Criteria Met   : {result['years']['criteria_met']}/3")
    print(f"  YEARS Conclusion     : {result['years']['conclusion']}")
    print(f"  D-dimer              : {result['d_dimer']['value_ugml']} µg/mL")
    print(f"  Age-adj Threshold    : {result['d_dimer']['age_adjusted_threshold']:.2f} µg/mL")
    print(f"  Consensus            : {result['consensus_recommendation']}")
    print(f"  If PE confirmed:")
    for step in result["if_pe_confirmed_management"][:2]:
        print(f"    - {step}")


if __name__ == "__main__":
    print("Pulmonology Decision Support Module — Demo")
    demo_spirometry()
    demo_asthma()
    demo_copd()
    demo_oxygen()
    demo_pe()
    print("\nDemo complete.")
