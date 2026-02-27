"""Shared financial-planning computation engine.

Pure Python — no Streamlit imports.  Used by both the existing retired/accum
pages and the new combined Financial Plan page.
"""

import datetime as dt
import numpy as np
from functools import lru_cache
from itertools import permutations
from fpdf import FPDF
import math, io

# ═══════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# --- From retired engine ---
UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
    88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
    104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4,
    112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
}

ADAPTIVE_STRATEGIES = {
    "aggressive_depletion": {
        "label": "Adaptive: Aggressive Depletion",
        "description": "Before 73: fill 24% bracket conversions + pre-tax first. After 73: brokerage first, no conversions.",
        "phases": [
            {"until_age": 73, "pt_cap": 999999, "conversion_strategy": "fill_bracket_24",
             "blend_mode": False, "prorata": False},
            {"from_age": 73, "pt_cap": 0, "conversion_strategy": "none",
             "blend_mode": False, "prorata": False},
        ],
    },
    "moderate_depletion": {
        "label": "Adaptive: Moderate Depletion",
        "description": "Before 73: fill 22% bracket + $50k/yr pre-tax cap. After 73: brokerage first.",
        "phases": [
            {"until_age": 73, "pt_cap": 50000, "conversion_strategy": "fill_bracket_22",
             "blend_mode": False, "prorata": False},
            {"from_age": 73, "pt_cap": 0, "conversion_strategy": "none",
             "blend_mode": False, "prorata": False},
        ],
    },
    "bracket_equalize_22": {
        "label": "Adaptive: Bracket Equalize 22%/24%",
        "description": "Dynamic withdrawals + fill 22% bracket conversions every year.",
        "phases": [
            {"until_age": 999, "pt_cap": None, "conversion_strategy": "fill_bracket_22",
             "blend_mode": True, "prorata": False},
        ],
    },
    "irmaa_aware": {
        "label": "Adaptive: IRMAA-Aware",
        "description": "Before 73: convert to IRMAA floor. After 73: dynamic withdrawals, no conversions.",
        "phases": [
            {"until_age": 73, "pt_cap": 50000, "conversion_strategy": "fill_irmaa_0",
             "blend_mode": False, "prorata": False},
            {"from_age": 73, "pt_cap": None, "conversion_strategy": "none",
             "blend_mode": True, "prorata": False},
        ],
    },
    "roth_sprint": {
        "label": "Adaptive: Roth Sprint",
        "description": "Before 73: max conversions (fill 24%) + spend from brokerage. After 73: dynamic, no conversions.",
        "phases": [
            {"until_age": 73, "pt_cap": 0, "conversion_strategy": "fill_bracket_24",
             "blend_mode": False, "prorata": False},
            {"from_age": 73, "pt_cap": None, "conversion_strategy": "none",
             "blend_mode": True, "prorata": False},
        ],
    },
}

# --- From accum engine ---
LIMITS_401K = 23500
LIMITS_401K_CATCHUP = 7500
LIMITS_IRA = 7000
LIMITS_IRA_CATCHUP = 1000
LIMITS_HSA_SINGLE = 4300
LIMITS_HSA_FAMILY = 8550
LIMITS_HSA_CATCHUP = 1000

# 2025 IRA income phase-out limits (MAGI)
# Roth IRA
ROTH_PHASE_SINGLE = (150000, 165000)
ROTH_PHASE_MFJ = (236000, 246000)
ROTH_PHASE_HOH = (150000, 165000)
# Traditional IRA deductibility (if covered by employer plan)
TRAD_PHASE_SINGLE = (79000, 89000)
TRAD_PHASE_MFJ = (126000, 146000)
TRAD_PHASE_HOH = (79000, 89000)

SS_WAGE_BASE_2025 = 176100
MEDICARE_ADDITIONAL_THRESHOLD = {"Married Filing Jointly": 250000, "Single": 200000, "Head of Household": 200000}

SS_TAXABLE_MAX_2025 = SS_WAGE_BASE_2025

RMD_TABLE = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2,
    81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7,
    89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4,
    97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9,
    105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3,
    113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}

# Single Life Expectancy Table (IRS Pub 590-B) — for inherited IRA lifetime RMDs
SINGLE_LIFE_TABLE = {
    1: 82.4, 2: 81.4, 3: 80.4, 4: 79.5, 5: 78.5, 6: 77.5, 7: 76.5, 8: 75.5, 9: 74.5, 10: 73.5,
    11: 72.5, 12: 71.5, 13: 70.5, 14: 69.5, 15: 68.5, 16: 67.5, 17: 66.5, 18: 65.5, 19: 64.5, 20: 63.5,
    21: 62.5, 22: 61.5, 23: 60.5, 24: 59.5, 25: 58.5, 26: 57.5, 27: 56.5, 28: 55.5, 29: 54.5, 30: 53.5,
    31: 52.5, 32: 51.5, 33: 50.5, 34: 49.5, 35: 48.5, 36: 47.5, 37: 46.5, 38: 45.5, 39: 44.5, 40: 43.5,
    41: 42.5, 42: 41.5, 43: 40.5, 44: 39.5, 45: 38.5, 46: 37.5, 47: 36.5, 48: 35.5, 49: 34.5, 50: 33.5,
    51: 32.5, 52: 31.5, 53: 30.5, 54: 29.5, 55: 28.5, 56: 27.5, 57: 26.5, 58: 25.5, 59: 24.5, 60: 23.5,
    61: 22.5, 62: 21.5, 63: 20.5, 64: 19.5, 65: 18.5, 66: 17.5, 67: 16.5, 68: 15.5, 69: 14.5, 70: 13.5,
    71: 12.5, 72: 11.5, 73: 10.5, 74: 9.5, 75: 8.5, 76: 7.5, 77: 6.5, 78: 5.5, 79: 4.5, 80: 3.5,
    81: 3.4, 82: 3.3, 83: 3.1, 84: 3.0, 85: 2.9, 86: 2.8, 87: 2.7, 88: 2.5, 89: 2.3, 90: 2.0,
    91: 1.9, 92: 1.8, 93: 1.7, 94: 1.6, 95: 1.5, 96: 1.4, 97: 1.3, 98: 1.2, 99: 1.1, 100: 1.0,
}

_FED_BRACKET_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]


# ═══════════════════════════════════════════════════════════════════════
# 2. UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def age_at_date(dob, asof):
    if dob is None: return None
    years = asof.year - dob.year
    if (asof.month, asof.day) < (dob.month, dob.day): years -= 1
    return years

def money(x): return f"${float(x):,.0f}"

def money2(x): return f"${float(x):,.2f}"

def calc_mortgage_interest_for_year(balance, annual_rate, annual_payment):
    """Calculate interest paid and ending balance for one year of mortgage payments."""
    if balance <= 0 or annual_payment <= 0:
        return 0.0, 0.0
    monthly_rate = annual_rate / 12
    monthly_pmt = annual_payment / 12
    total_interest = 0.0
    bal = balance
    for _ in range(12):
        if bal <= 0:
            break
        month_interest = bal * monthly_rate
        total_interest += month_interest
        principal = min(monthly_pmt - month_interest, bal)
        bal = max(0.0, bal - principal)
    return total_interest, bal

def calc_mortgage_payment(balance, annual_rate, years):
    """Calculate annual P&I payment from balance, rate, years."""
    if balance <= 0 or years <= 0: return 0.0
    if annual_rate <= 0: return balance / years
    r = annual_rate / 12
    n = years * 12
    monthly = balance * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(monthly * 12, 0)

def calc_mortgage_years(balance, annual_rate, annual_payment):
    """Calculate years to payoff from balance, rate, annual payment."""
    if balance <= 0 or annual_payment <= 0: return 0
    if annual_rate <= 0: return max(0, math.ceil(balance / annual_payment))
    r = annual_rate / 12
    monthly_pmt = annual_payment / 12
    if monthly_pmt <= balance * r: return 99  # payment doesn't cover interest
    n = math.log(monthly_pmt / (monthly_pmt - balance * r)) / math.log(1 + r)
    return max(0, math.ceil(n / 12))


# ═══════════════════════════════════════════════════════════════════════
# 3. SS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def ss_claim_factor(choice):
    """SS benefit as fraction of PIA based on claim age (FRA=67).
    Before FRA: 5/9 of 1% per month for first 36 months (ages 64-67),
                5/12 of 1% per month for additional months (ages 62-64).
    After FRA:  2/3 of 1% per month (8% per year) delayed retirement credits."""
    c = str(choice or "").strip().lower()
    if c == "fra":
        return 1.00
    try:
        age = int(c)
    except (ValueError, TypeError):
        return 1.00
    if age <= 62: return 0.70
    if age >= 70: return 1.24
    if age >= 67:
        return 1.00 + (age - 67) * 0.08
    months_early = (67 - age) * 12
    if months_early <= 36:
        return 1.00 - months_early * (5 / 9 / 100)
    return 1.00 - 36 * (5 / 9 / 100) - (months_early - 36) * (5 / 12 / 100)

def ss_first_year_fraction(dob):
    """Fraction of the claim year with SS benefits based on birthday month.
    Benefits start the first full month at claiming age.
    Born on the 1st: benefits start in birth month.
    Born on any other day: benefits start the month after birthday month."""
    if dob is None:
        return 1.0
    first_month = dob.month if dob.day == 1 else dob.month + 1
    if first_month > 12:
        return 0.0
    return (13 - first_month) / 12.0

def _ss_pia_from_aime(aime):
    """Calculate monthly PIA from AIME using 2025 bend points."""
    bp1, bp2 = 1174, 7078
    if aime <= bp1:
        return aime * 0.90
    elif aime <= bp2:
        return bp1 * 0.90 + (aime - bp1) * 0.32
    else:
        return bp1 * 0.90 + (bp2 - bp1) * 0.32 + (aime - bp2) * 0.15

def _build_earnings_history(salary, current_age, retire_age, salary_growth_rate, work_start_age=20):
    """Build earnings history from work_start_age to retire_age.
    Ages 20-21: part-time/summer work (~$12,000/yr in today's dollars)
    Ages 22+: career earnings deflated from current salary by growth rate"""
    PART_TIME_PRECOLLEGE = 12000  # pre-career part-time/summer (today's $)

    work_end_age = max(current_age, retire_age)
    earnings = []
    for yr_idx in range(max(0, work_end_age - work_start_age)):
        work_age = work_start_age + yr_idx
        if work_age < 22:
            yr_salary = PART_TIME_PRECOLLEGE
        else:
            years_from_now = work_age - current_age
            if years_from_now < 0:
                yr_salary = salary / ((1 + salary_growth_rate) ** (-years_from_now))
            else:
                yr_salary = salary
        taxable_earnings = min(yr_salary, SS_TAXABLE_MAX_2025)
        earnings.append(taxable_earnings)
    return earnings

def estimate_ss_pia(salary, current_age, retire_age, salary_growth_rate, work_start_age=20, detailed=False):
    """Estimate annual PIA at FRA using projected earnings through retirement.
    All earnings expressed in today's dollars (consistent with 2025 bend points).
    Past years deflated by salary growth; future years use current salary
    (real wage growth ≈ SS indexing, so today's dollars is the right basis).
    Assumes work begins at age 20 (part-time before college grad, then career at 22).

    If detailed=True, returns dict with PIA, work years, zero years, AIME,
    and PIA at alternative retirement ages."""
    earnings = _build_earnings_history(salary, current_age, retire_age, salary_growth_rate, work_start_age)
    total_work_years = len(earnings)

    earnings.sort(reverse=True)
    top_35 = (earnings[:35] + [0] * 35)[:35]
    zero_years = top_35.count(0)
    aime = sum(top_35) / (35 * 12)
    pia_monthly = _ss_pia_from_aime(aime)
    pia_annual = round(pia_monthly * 12, 0)

    if not detailed:
        return pia_annual

    # Calculate PIA at alternative retirement ages
    alt_ages = {}
    for alt_ret in range(max(50, current_age), 71):
        if alt_ret == retire_age:
            alt_ages[alt_ret] = pia_annual
            continue
        alt_earnings = _build_earnings_history(salary, current_age, alt_ret, salary_growth_rate, work_start_age)
        alt_earnings.sort(reverse=True)
        alt_top35 = (alt_earnings[:35] + [0] * 35)[:35]
        alt_aime = sum(alt_top35) / (35 * 12)
        alt_ages[alt_ret] = round(_ss_pia_from_aime(alt_aime) * 12, 0)

    return {
        "pia_annual": pia_annual,
        "work_years": total_work_years,
        "zero_years": zero_years,
        "aime": round(aime, 0),
        "alt_ages": alt_ages,
    }

def annual_ss_in_year(*, dob, tax_year, cola, already_receiving, current_annual, start_year, fra_annual, claim_choice, current_year):
    tax_year = int(tax_year); current_year = int(current_year); cola = float(cola)
    if already_receiving:
        if current_annual <= 0: return 0.0
        if tax_year < int(start_year): return 0.0
        return float(current_annual) * ((1.0 + cola) ** max(0, tax_year - current_year))
    if dob is None or fra_annual <= 0: return 0.0
    fra_age = 67
    claim_age = fra_age if (claim_choice or "").strip().lower() == "fra" else int(claim_choice)
    age_eoy = age_at_date(dob, dt.date(tax_year, 12, 31))
    if age_eoy is None or age_eoy < claim_age: return 0.0
    base = float(fra_annual) * ss_claim_factor(claim_choice)
    claim_year = dob.year + claim_age
    return base * ((1.0 + cola) ** max(0, tax_year - claim_year))


# ═══════════════════════════════════════════════════════════════════════
# 4. RMD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_rmd_uniform_start73(balance, age):
    if balance <= 0 or age is None or age < 73: return 0.0
    factor = UNIFORM_LIFETIME.get(age, UNIFORM_LIFETIME[120])
    return float(balance) / float(factor)

def calc_rmd(age, pretax_balance):
    """Calculate required minimum distribution for the year."""
    if age < 73 or pretax_balance <= 0:
        return 0.0
    divisor = RMD_TABLE.get(age, 2.0)
    return pretax_balance / divisor


# ═══════════════════════════════════════════════════════════════════════
# 5. TAX CORE — Retired Engine (comprehensive version)
# ═══════════════════════════════════════════════════════════════════════

def get_federal_base_std(status, inf=1.0):
    s = status.lower()
    val = 15750.0
    if "joint" in s: val = 31500.0
    elif "head" in s or "hoh" in s: val = 23625.0
    return val * inf

def get_federal_traditional_extra(status, filer_65, spouse_65, inf=1.0):
    extra = 0.0
    is_joint = "joint" in status.lower()
    base_add = 1600.0 if is_joint else 2000.0
    if filer_65: extra += base_add
    if is_joint and spouse_65: extra += 1600.0
    return extra * inf

def get_federal_enhanced_extra(agi, filer_65, spouse_65, status, inf=1.0, tax_year=None):
    # Enhanced elderly deduction expires after 2028
    if tax_year is not None and tax_year > 2028:
        return 0.0
    extra = 0.0
    is_joint = "joint" in status.lower()
    threshold = (150000.0 if is_joint else 75000.0) * inf
    if filer_65: extra += 6000.0 * inf
    if is_joint and spouse_65: extra += 6000.0 * inf
    if agi > threshold: extra = max(0.0, extra - 0.06 * (agi - threshold))
    return extra

def get_preferential_brackets(status, inf=1.0):
    key = "joint" if "joint" in status.lower() else "hoh" if "head" in status.lower() else "single"
    raw = {
        "single": [(0, 48350, 0.0), (48350, 533400, 0.15), (533400, float("inf"), 0.20)],
        "joint": [(0, 96700, 0.0), (96700, 600050, 0.15), (600050, float("inf"), 0.20)],
        "hoh": [(0, 64750, 0.0), (64750, 566700, 0.15), (566700, float("inf"), 0.20)],
    }[key]
    return [(l * inf, h * inf if h != float("inf") else h, r) for l, h, r in raw]

def calculate_federal_tax(taxable_income, preferential_amount, status, inf=1.0):
    key = "joint" if "joint" in status.lower() else "hoh" if "head" in status.lower() else "single"
    raw_ordinary = {
        "single": [(0,11925,0.10),(11925,48475,0.12),(48475,103350,0.22),(103350,197300,0.24),(197300,250525,0.32),(250525,626350,0.35),(626350,float("inf"),0.37)],
        "joint": [(0,23850,0.10),(23850,96950,0.12),(96950,206700,0.22),(206700,394600,0.24),(394600,501050,0.32),(501050,751600,0.35),(751600,float("inf"),0.37)],
        "hoh": [(0,17000,0.10),(17000,64850,0.12),(64850,103350,0.22),(103350,197300,0.24),(197300,250500,0.32),(250500,626350,0.35),(626350,float("inf"),0.37)],
    }[key]
    ordinary_brackets = [(l*inf, h*inf if h != float("inf") else h, r) for l,h,r in raw_ordinary]
    taxable_income = max(0.0, float(taxable_income))
    preferential_amount = max(0.0, float(preferential_amount))
    ordinary_amount = max(0.0, taxable_income - preferential_amount)
    ordinary_tax = 0.0; prev = 0.0
    for _, upper, rate in ordinary_brackets:
        seg = min(ordinary_amount, upper) - prev
        if seg > 0: ordinary_tax += seg * rate
        prev = upper
    pref_tax = 0.0
    for low, high, rate in get_preferential_brackets(status, inf):
        eff_low = max(low, ordinary_amount)
        seg = min(high, ordinary_amount + preferential_amount) - eff_low
        if seg > 0: pref_tax += seg * rate
    total = ordinary_tax + pref_tax
    eff = round((total / taxable_income * 100.0) if taxable_income > 0 else 0.0, 2)
    return {"federal_tax": round(total, 2), "effective_fed": eff}

def calculate_taxable_ss(base_non_ss, tax_exempt, gross_ss, status):
    provisional = float(base_non_ss) + float(tax_exempt) + 0.5 * float(gross_ss)
    is_joint = "joint" in status.lower()
    base1 = 32000 if is_joint else 25000; base2 = 44000 if is_joint else 34000
    if provisional <= base1: return 0.0
    taxable = 0.0
    tier1 = min(provisional - base1, base2 - base1)
    if tier1 > 0: taxable += tier1 * 0.5
    if provisional > base2: taxable += (provisional - base2) * 0.85
    return min(taxable, 0.85 * float(gross_ss))

def calculate_sc_tax(fed_taxable, dependents, taxable_ss, out_of_state_gain, filer_65_plus, spouse_65_plus, retirement_income, cap_gain_loss, enhanced_elderly_addback=0.0):
    sc_start = max(0.0, float(fed_taxable)) + max(0.0, float(enhanced_elderly_addback))
    # Retirement deduction: auto-compute based on age
    ret_ded_limit = 10000.0 if filer_65_plus else 3000.0
    ret_ded = min(ret_ded_limit, max(0.0, float(retirement_income)))
    sub = max(0.0, float(taxable_ss)) + ret_ded
    if filer_65_plus: sub += 15000.0
    if spouse_65_plus: sub += 15000.0
    sub += 0.44 * max(0.0, float(cap_gain_loss)) + max(0.0, float(out_of_state_gain)) + 4930.0 * max(0, int(dependents))
    sc_taxable = max(0.0, sc_start - sub)
    brackets = [(0,3560,0.00),(3560,17830,0.03),(17830,float("inf"),0.06)]
    tax = 0.0; prev = 0.0
    for _, upper, rate in brackets:
        seg = min(sc_taxable, upper) - prev
        if seg > 0: tax += seg * rate
        prev = upper
    eff = round((tax / sc_taxable * 100.0) if sc_taxable > 0 else 0.0, 2)
    return {"sc_tax": round(tax, 2), "effective_sc": eff, "sc_taxable": round(sc_taxable, 2)}

def estimate_medicare_premiums(agi, filing_status, inf=1.0, medicare_inf=None,
                               filer_65=True, spouse_65=False):
    """2026 Medicare Part B + Part D premiums with full 5-tier IRMAA.
    Thresholds are MAGI cliffs (based on 2024 tax return for 2026).
    inf: bracket/threshold inflation factor.
    medicare_inf: premium growth factor (defaults to inf if not provided).
    filer_65/spouse_65: who is actually on Medicare (65+)."""
    _med_inf = medicare_inf if medicare_inf is not None else inf
    is_joint = "joint" in filing_status.lower()
    a = float(agi)
    people = int(bool(filer_65)) + (int(bool(spouse_65)) if is_joint else 0)
    # 2026 base Part B + avg Part D = $202.90 + $46.50 = $249.40/mo
    base_monthly = 249.40 * _med_inf  # grown by medicare inflation
    # IRMAA tiers: (joint_threshold, single_threshold, partB_surcharge, partD_surcharge) per month
    irmaa_tiers = [
        (750000, 500000, 487.00, 91.00),   # Tier 5 (top, frozen)
        (410000, 205000, 446.30, 83.30),   # Tier 4
        (342000, 171000, 324.60, 60.40),   # Tier 3
        (274000, 137000, 202.90, 37.50),   # Tier 2
        (218000, 109000,  81.20, 14.50),   # Tier 1
    ]
    surcharge_monthly = 0.0
    for jt, st_thresh, partb_s, partd_s in irmaa_tiers:
        threshold = (jt if is_joint else st_thresh) * inf
        if a > threshold:
            surcharge_monthly = (partb_s + partd_s) * _med_inf
            break
    annual = (base_monthly + surcharge_monthly) * 12 * people
    return annual, surcharge_monthly > 0

def annuity_gains(val, basis): return max(0.0, float(val) - float(basis))


# ═══════════════════════════════════════════════════════════════════════
# 6. TAX CORE — Accum Engine (simpler version)
# ═══════════════════════════════════════════════════════════════════════

def get_fed_brackets(status, inf=1.0):
    if "joint" in status.lower():
        raw = [(0,23850,0.10),(23850,96950,0.12),(96950,206700,0.22),(206700,394600,0.24),
               (394600,501050,0.32),(501050,751600,0.35),(751600,float("inf"),0.37)]
    elif "head" in status.lower():
        raw = [(0,17000,0.10),(17000,64850,0.12),(64850,103350,0.22),(103350,197300,0.24),
               (197300,250500,0.32),(250500,626350,0.35),(626350,float("inf"),0.37)]
    else:
        raw = [(0,11925,0.10),(11925,48475,0.12),(48475,103350,0.22),(103350,197300,0.24),
               (197300,250525,0.32),(250525,626350,0.35),(626350,float("inf"),0.37)]
    return [(l*inf, h*inf if h != float("inf") else h, r) for l,h,r in raw]

def get_std_deduction(status, filer_65, spouse_65, inf=1.0):
    if "joint" in status.lower():
        base = 31500.0
        extra = (1600.0 if filer_65 else 0) + (1600.0 if spouse_65 else 0)
    elif "head" in status.lower():
        base = 23625.0
        extra = 2000.0 if filer_65 else 0
    else:
        base = 15750.0
        extra = 2000.0 if filer_65 else 0
    return (base + extra) * inf

def calc_taxable_ss(other_income, gross_ss, status):
    """IRS Pub 915 worksheet: how much of SS is included in taxable income.
    Thresholds are NOT inflation-indexed (per IRC §86)."""
    if gross_ss <= 0:
        return 0.0
    provisional = other_income + 0.5 * gross_ss
    is_joint = "joint" in status.lower()
    b1 = 32000 if is_joint else 25000
    b2 = 44000 if is_joint else 34000
    if provisional <= b1:
        return 0.0
    # Tier 1: 50% of lesser of (excess over b1) or (band width b2-b1)
    band = b2 - b1  # $12,000 MFJ / $9,000 single
    tier1 = 0.5 * min(provisional - b1, band)
    if provisional <= b2:
        return min(tier1, 0.5 * gross_ss)
    # Tier 2: tier1 + 85% of excess over b2, capped at 85% of benefits
    tier2 = tier1 + 0.85 * (provisional - b2)
    return min(tier2, 0.85 * gross_ss)

def calc_federal_tax(taxable_income, status, inf=1.0):
    brackets = get_fed_brackets(status, inf)
    tax = 0.0
    prev = 0.0
    for _, upper, rate in brackets:
        seg = min(max(0, taxable_income), upper) - prev
        if seg > 0: tax += seg * rate
        prev = upper
    return tax

def calc_state_tax(taxable_income, state_rate):
    return max(0.0, taxable_income) * state_rate

def get_marginal_fed_rate(taxable_income, status, inf=1.0):
    brackets = get_fed_brackets(status, inf)
    if taxable_income <= 0:
        return brackets[0][2]
    for low, high, rate in brackets:
        if taxable_income <= high:
            return rate
    return brackets[-1][2]

def _heir_rate_from_offset(my_marginal, offset):
    """Return heir's marginal rate given retiree's rate and bracket offset (-1, 0, +1)."""
    try:
        idx = _FED_BRACKET_RATES.index(my_marginal)
    except ValueError:
        idx = 3  # default to 24% if not found
    new_idx = max(0, min(len(_FED_BRACKET_RATES) - 1, idx + offset))
    return _FED_BRACKET_RATES[new_idx]

def calc_cg_tax(cap_gains, ordinary_taxable, status, inf=1.0):
    """Calculate capital gains tax using 0%/15%/20% brackets.
    Cap gains stack on top of ordinary income."""
    if cap_gains <= 0:
        return 0.0
    if "joint" in status.lower():
        b0 = 96700 * inf   # top of 0% bracket
        b1 = 600050 * inf  # top of 15% bracket
    elif "head" in status.lower():
        b0 = 64750 * inf
        b1 = 566700 * inf
    else:
        b0 = 48350 * inf
        b1 = 533400 * inf
    cg_start = ordinary_taxable
    cg_end = ordinary_taxable + cap_gains
    cg_at_0 = max(0.0, min(cg_end, b0) - cg_start)
    cg_at_15 = max(0.0, min(cg_end, b1) - max(cg_start, b0))
    cg_at_20 = max(0.0, cg_end - max(cg_start, b1))
    return cg_at_15 * 0.15 + cg_at_20 * 0.20

def calc_year_taxes(gross_ss, pretax_income, cap_gains=0.0, ord_invest_income=0.0, status="Single",
                    filer_65=False, spouse_65=False, inf=1.0, state_rate=0.05, sc_params=None):
    """Calculate federal + state tax for a retirement year. All inputs in nominal dollars.
    cap_gains: long-term capital gains + qualified dividends (0%/15%/20% brackets)
    ord_invest_income: cash interest and other ordinary investment income
    sc_params: optional dict for SC bracket-based tax instead of flat rate"""
    base_other = pretax_income + cap_gains + ord_invest_income
    taxable_ss = calc_taxable_ss(base_other, gross_ss, status)
    agi = base_other + taxable_ss
    std_ded = get_std_deduction(status, filer_65, spouse_65, inf)
    fed_taxable = max(0.0, agi - std_ded)
    # Cap gains sit on top of the income stack; standard deduction reduces ordinary first
    cg_in_taxable = min(cap_gains, fed_taxable)
    ordinary_taxable = max(0.0, fed_taxable - cg_in_taxable)
    fed_ord_tax = calc_federal_tax(ordinary_taxable, status, inf)
    fed_cg_tax = calc_cg_tax(cg_in_taxable, ordinary_taxable, status, inf)
    fed_tax = fed_ord_tax + fed_cg_tax
    if sc_params is not None:
        _tax_yr = sc_params.get("tax_year")
        _enh = get_federal_enhanced_extra(agi, filer_65, spouse_65, status, inf, tax_year=_tax_yr)
        _ret_inc = sc_params.get("retirement_income", 0.0)
        sc_result = calculate_sc_tax(
            fed_taxable, sc_params.get("dependents", 0), taxable_ss,
            sc_params.get("out_of_state_gain", 0.0),
            filer_65, spouse_65,
            _ret_inc, cap_gains,
            enhanced_elderly_addback=_enh)
        st_tax = sc_result["sc_tax"]
    else:
        st_tax = calc_state_tax(fed_taxable, state_rate)
    marginal = get_marginal_fed_rate(ordinary_taxable, status, inf)
    return {"fed_tax": fed_tax, "state_tax": st_tax, "total_tax": fed_tax + st_tax,
            "agi": agi, "taxable_ss": taxable_ss, "fed_taxable": fed_taxable,
            "cap_gains_tax": fed_cg_tax, "marginal_rate": marginal}


# ═══════════════════════════════════════════════════════════════════════
# 7. FICA & ACCUMULATION TAX HELPERS
# ═══════════════════════════════════════════════════════════════════════

def ira_phase_out(magi, phase_range, full_limit):
    """Calculate allowed contribution after income phase-out."""
    low, high = phase_range
    if magi <= low:
        return full_limit
    if magi >= high:
        return 0.0
    # Pro-rata reduction, rounded up to nearest $10 per IRS rules
    reduction = full_limit * (magi - low) / (high - low)
    allowed = math.ceil((full_limit - reduction) / 10) * 10
    return max(0.0, min(full_limit, float(allowed)))

def calc_fica(wages_filer, wages_spouse, status, inf=1.0,
              se_filer=False, se_spouse=False):
    """Calculate FICA taxes.

    Employee-side: SS 6.2% + Medicare 1.45% + Additional Medicare 0.9%.
    Self-employed: both halves (12.4% SS + 2.9% Medicare) on 92.35% of net SE income,
    plus 0.9% Additional Medicare on amounts over threshold.
    Returns (total_fica, se_deduction) — se_deduction is the 50% SE tax deduction from AGI.
    """
    wage_base = SS_WAGE_BASE_2025 * inf
    threshold = MEDICARE_ADDITIONAL_THRESHOLD.get(status, 200000) * inf

    def _employee_fica(w):
        return min(w, wage_base) * 0.062 + w * 0.0145

    def _se_fica(w):
        se_base = w * 0.9235
        ss = min(se_base, wage_base) * 0.124
        med = se_base * 0.029
        return ss + med, se_base

    fica_f = 0.0; se_base_f = 0.0
    if wages_filer > 0:
        if se_filer:
            fica_f, se_base_f = _se_fica(wages_filer)
        else:
            fica_f = _employee_fica(wages_filer)

    fica_s = 0.0; se_base_s = 0.0
    if wages_spouse > 0:
        if se_spouse:
            fica_s, se_base_s = _se_fica(wages_spouse)
        else:
            fica_s = _employee_fica(wages_spouse)

    # Additional Medicare on combined wages/SE-base over threshold
    combined = (se_base_f if se_filer else wages_filer) + (se_base_s if se_spouse else wages_spouse)
    add_med = max(0.0, combined - threshold) * 0.009

    total = fica_f + fica_s + add_med

    # SE deduction = 50% of base SE tax (SS + Medicare, excluding Additional Medicare)
    # This represents the employer-equivalent share
    se_tax_portion = 0.0
    if se_filer and wages_filer > 0:
        _f_se, _f_base = _se_fica(wages_filer)
        se_tax_portion += _f_se
    if se_spouse and wages_spouse > 0:
        _s_se, _s_base = _se_fica(wages_spouse)
        se_tax_portion += _s_se
    se_deduction = se_tax_portion / 2.0

    return total, se_deduction


# ═══════════════════════════════════════════════════════════════════════
# 8. BRACKET FILL / CONVERSION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _get_bracket_top(target_bracket_rate, filing_status, inf=1.0):
    """Return the taxable-income ceiling for a given marginal bracket rate."""
    key = "joint" if "joint" in filing_status.lower() else "hoh" if "head" in filing_status.lower() else "single"
    raw = {
        "single": [(0,11925,0.10),(11925,48475,0.12),(48475,103350,0.22),(103350,197300,0.24),(197300,250525,0.32),(250525,626350,0.35),(626350,float("inf"),0.37)],
        "joint": [(0,23850,0.10),(23850,96950,0.12),(96950,206700,0.22),(206700,394600,0.24),(394600,501050,0.32),(501050,751600,0.35),(751600,float("inf"),0.37)],
        "hoh": [(0,17000,0.10),(17000,64850,0.12),(64850,103350,0.22),(103350,197300,0.24),(197300,250500,0.32),(250500,626350,0.35),(626350,float("inf"),0.37)],
    }[key]
    for _lo, hi, rate in raw:
        if abs(rate - target_bracket_rate) < 0.001:
            return hi * inf
    return 0.0


def compute_bracket_fill_amount(target_bracket_rate, base_non_ss_income, gross_ss,
                                 filing_status, filer_65, spouse_65, retirement_deduction,
                                 preferential_amount=0.0, inf_factor=1.0, tax_year=None):
    """Binary search for the Roth conversion amount that fills ordinary income to the
    top of a specified federal bracket (e.g. 0.12, 0.22, 0.24).

    Binary search is needed because adding conversion income changes SS taxability
    (non-linear feedback) and the enhanced-elderly deduction phases out with AGI.
    Returns the conversion amount (>=0). Pure function — no side effects.
    """
    bracket_top = _get_bracket_top(target_bracket_rate, filing_status, inf_factor)
    if bracket_top <= 0 or bracket_top == float("inf"):
        return 0.0

    lo, hi = 0.0, 2_000_000.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        total_non_ss = base_non_ss_income + mid
        taxable_ss = calculate_taxable_ss(total_non_ss, 0.0, gross_ss, filing_status)
        agi = total_non_ss + taxable_ss
        base_std = get_federal_base_std(filing_status, inf_factor)
        trad_extra = get_federal_traditional_extra(filing_status, filer_65, spouse_65, inf_factor)
        enh_extra = get_federal_enhanced_extra(agi, filer_65, spouse_65, filing_status, inf_factor, tax_year=tax_year)
        deduction = base_std + trad_extra + enh_extra
        ordinary_taxable = max(0.0, agi - deduction) - max(0.0, preferential_amount)
        if ordinary_taxable < bracket_top:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2.0
    return max(0.0, result)


def compute_irmaa_safe_amount(target_irmaa_tier, base_non_ss_income, gross_ss,
                               filing_status, inf_factor=1.0):
    """Binary search for the max Roth conversion amount that keeps AGI below an IRMAA
    threshold.

    target_irmaa_tier:
        0 = stay below IRMAA Tier 1 (no surcharge)
        1 = stay below Tier 2, etc.

    Returns the max safe conversion amount (>=0). Pure function.
    """
    is_joint = "joint" in filing_status.lower()
    # IRMAA thresholds (ascending): Tier 1, Tier 2, Tier 3, Tier 4, Tier 5
    irmaa_thresholds_joint = [218000, 274000, 342000, 410000, 750000]
    irmaa_thresholds_single = [109000, 137000, 171000, 205000, 500000]
    thresholds = irmaa_thresholds_joint if is_joint else irmaa_thresholds_single

    if target_irmaa_tier < 0 or target_irmaa_tier >= len(thresholds):
        return 0.0
    agi_ceiling = thresholds[target_irmaa_tier] * inf_factor

    lo, hi = 0.0, 2_000_000.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        total_non_ss = base_non_ss_income + mid
        taxable_ss = calculate_taxable_ss(total_non_ss, 0.0, gross_ss, filing_status)
        agi = total_non_ss + taxable_ss
        if agi < agi_ceiling:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2.0
    return max(0.0, result)


def compute_gains_harvest_amount(base_non_ss_income, existing_preferential, gross_ss,
                                  filing_status, filer_65, spouse_65, retirement_deduction,
                                  inf_factor=1.0, tax_year=None):
    """Binary search for the max additional LTCG that can be realized at the 0% rate.

    Accounts for SS taxability feedback and deduction phaseouts (gains add to AGI).
    Returns the max additional gains that fit entirely in the 0% LTCG bracket.
    """
    pref_brackets = get_preferential_brackets(filing_status, inf_factor)
    zero_pct_ceiling = pref_brackets[0][1]  # e.g., $96,700 joint

    lo, hi = 0.0, 2_000_000.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        total_non_ss = base_non_ss_income + mid
        taxable_ss = calculate_taxable_ss(total_non_ss, 0.0, gross_ss, filing_status)
        agi = total_non_ss + taxable_ss
        base_std = get_federal_base_std(filing_status, inf_factor)
        trad_extra = get_federal_traditional_extra(filing_status, filer_65, spouse_65, inf_factor)
        enh_extra = get_federal_enhanced_extra(agi, filer_65, spouse_65, filing_status, inf_factor, tax_year=tax_year)
        deduction = base_std + trad_extra + enh_extra
        taxable_income = max(0.0, agi - deduction)
        total_pref = existing_preferential + mid
        ordinary_taxable = max(0.0, taxable_income - total_pref)
        pref_room = max(0.0, zero_pct_ceiling - ordinary_taxable)
        if total_pref < pref_room:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2.0
    return max(0.0, result)


# ═══════════════════════════════════════════════════════════════════════
# 9. CENTRAL TAX CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

def compute_taxes_only(gross_ss, taxable_pensions, rmd_amount, taxable_ira, conversion_amount,
                       ordinary_income, cap_gains, filing_status, filer_65, spouse_65,
                       retirement_deduction, inf_factor=1.0, tax_year=None):
    base_non_ss = taxable_pensions + rmd_amount + taxable_ira + conversion_amount + ordinary_income + cap_gains
    taxable_ss = calculate_taxable_ss(base_non_ss, 0.0, gross_ss, filing_status)
    agi = base_non_ss + taxable_ss
    base_std = get_federal_base_std(filing_status, inf_factor)
    trad_extra = get_federal_traditional_extra(filing_status, filer_65, spouse_65, inf_factor)
    enh_extra = get_federal_enhanced_extra(agi, filer_65, spouse_65, filing_status, inf_factor, tax_year=tax_year)
    deduction = base_std + trad_extra + enh_extra
    fed_taxable = max(0.0, agi - deduction)
    preferential = max(0.0, cap_gains)
    fed = calculate_federal_tax(fed_taxable, preferential, filing_status, inf_factor)
    retirement_income = taxable_ira + rmd_amount + taxable_pensions
    sc = calculate_sc_tax(fed_taxable, 0, taxable_ss, 0.0, filer_65, spouse_65,
                          retirement_income * inf_factor, cap_gains,
                          enhanced_elderly_addback=enh_extra)
    medicare, has_irmaa = estimate_medicare_premiums(agi, filing_status, inf_factor,
                                                     filer_65=filer_65, spouse_65=spouse_65)
    total_tax = fed["federal_tax"] + sc["sc_tax"]
    return {
        "agi": agi, "fed_taxable": fed_taxable, "fed_tax": fed["federal_tax"],
        "sc_tax": sc["sc_tax"], "total_tax": total_tax, "medicare": medicare,
        "has_irmaa": has_irmaa, "total_outflow": total_tax + medicare
    }

def compute_case(inputs, inflation_factor=1.0, medicare_inflation_factor=None):
    wages = float(inputs["wages"]); tax_exempt_interest = float(inputs["tax_exempt_interest"])
    interest_taxable = float(inputs["interest_taxable"]); total_ordinary_dividends = float(inputs["total_ordinary_dividends"])
    qualified_dividends = float(inputs["qualified_dividends"]); taxable_ira = float(inputs["taxable_ira"])
    rmd_amount = float(inputs["rmd_amount"]); taxable_pensions = float(inputs["taxable_pensions"])
    gross_ss = float(inputs["gross_ss"]); cap_gain_loss = float(inputs["cap_gain_loss"])
    ordinary_tax_only = float(inputs.get("ordinary_tax_only", 0.0)); other_income = float(inputs["other_income"])
    cashflow_taxfree = float(inputs.get("cashflow_taxfree", 0.0))
    brokerage_proceeds = float(inputs.get("brokerage_proceeds", 0.0))
    annuity_proceeds = float(inputs.get("annuity_proceeds", 0.0))
    filing_status = inputs["filing_status"]; filer_65_plus = bool(inputs["filer_65_plus"])
    spouse_65_plus = bool(inputs["spouse_65_plus"]); adjustments = float(inputs["adjustments"])
    dependents = int(inputs["dependents"]); retirement_deduction = float(inputs["retirement_deduction"])
    out_of_state_gain = float(inputs["out_of_state_gain"])
    mtg_balance = float(inputs.get("mortgage_balance", 0.0))
    mtg_rate = float(inputs.get("mortgage_rate", 0.0))
    mtg_payment = float(inputs.get("mortgage_payment", 0.0))
    prop_tax = float(inputs.get("property_tax", 0.0))
    medical_exp = float(inputs.get("medical_expenses", 0.0))
    charitable_amt = float(inputs.get("charitable", 0.0))
    qcd_amt = float(inputs.get("qcd_annual", 0.0))

    # QCD: excluded from taxable income, reduces charitable deduction (no double-dipping)
    taxable_rmd = max(0.0, rmd_amount - qcd_amt)
    charitable_amt = max(0.0, charitable_amt - qcd_amt)

    base_non_ss = wages + interest_taxable + total_ordinary_dividends + taxable_ira + taxable_rmd + taxable_pensions + ordinary_tax_only + cap_gain_loss + other_income
    taxable_ss = calculate_taxable_ss(base_non_ss, tax_exempt_interest, gross_ss, filing_status)
    total_income_for_tax = base_non_ss + taxable_ss
    agi = max(0.0, total_income_for_tax - adjustments)
    base_std = get_federal_base_std(filing_status, inflation_factor)
    traditional_extra = get_federal_traditional_extra(filing_status, filer_65_plus, spouse_65_plus, inflation_factor)
    _tax_year = inputs.get("tax_year", None)
    enhanced_extra = get_federal_enhanced_extra(agi, filer_65_plus, spouse_65_plus, filing_status, inflation_factor, tax_year=_tax_year)
    fed_std = base_std + traditional_extra + enhanced_extra

    # SC retirement income = IRA/401k distributions + pension
    retirement_income = taxable_ira + rmd_amount + taxable_pensions

    # Calculate itemized deductions from components
    mortgage_interest = calc_mortgage_interest_for_year(mtg_balance, mtg_rate, mtg_payment)[0] if mtg_balance > 0 else 0.0
    # SALT: estimate state tax using standard deduction, add property tax, cap at $10k
    est_sc = calculate_sc_tax(max(0.0, agi - fed_std), dependents, taxable_ss, out_of_state_gain,
                              filer_65_plus, spouse_65_plus, retirement_income, cap_gain_loss,
                              enhanced_elderly_addback=enhanced_extra)
    salt = min(10000.0 * inflation_factor, est_sc["sc_tax"] + prop_tax)
    # Medical: amount exceeding 7.5% of AGI
    medical_deduction = max(0.0, medical_exp - agi * 0.075)
    itemized_total = mortgage_interest + salt + medical_deduction + charitable_amt
    # Deduction method override
    _ded_method = inputs.get("force_deduction_method", "auto")
    if _ded_method == "force_standard":
        is_itemizing = False
        deduction_used = fed_std
    elif _ded_method == "force_itemized":
        is_itemizing = True
        deduction_used = itemized_total
    elif _ded_method == "custom_itemized":
        is_itemizing = True
        itemized_total = float(inputs.get("custom_deduction_amount", itemized_total))
        deduction_used = itemized_total
    else:  # auto
        is_itemizing = itemized_total > fed_std
        deduction_used = itemized_total if is_itemizing else fed_std

    fed_taxable = max(0.0, agi - deduction_used)
    preferential_amount = qualified_dividends + max(0.0, cap_gain_loss)
    fed = calculate_federal_tax(fed_taxable, preferential_amount, filing_status, inflation_factor)
    sc = calculate_sc_tax(fed_taxable, dependents, taxable_ss, out_of_state_gain, filer_65_plus, spouse_65_plus, retirement_income, cap_gain_loss, enhanced_elderly_addback=enhanced_extra)
    total_tax = fed["federal_tax"] + sc["sc_tax"]
    if filer_65_plus or spouse_65_plus:
        _med_inf = medicare_inflation_factor if medicare_inflation_factor is not None else None
        medicare_premiums, has_irmaa = estimate_medicare_premiums(
            agi, filing_status, inflation_factor, _med_inf,
            filer_65=filer_65_plus, spouse_65=spouse_65_plus)
    else:
        medicare_premiums, has_irmaa = 0.0, False
    reinvest_int = bool(inputs.get("reinvest_interest", False))
    reinvest_div = bool(inputs.get("reinvest_dividends", False))
    reinvest_cg = bool(inputs.get("reinvest_cap_gains", False))
    realized_cg_from_sales = float(inputs.get("_realized_cap_gains", 0.0))
    reinvested_amount = 0.0
    if reinvest_int:
        reinvested_amount += interest_taxable
    if reinvest_div:
        reinvested_amount += total_ordinary_dividends
    if reinvest_cg:
        # Only reinvest base investment income cap gains, not realized gains from brokerage sales
        base_cg = max(0.0, cap_gain_loss - realized_cg_from_sales)
        if base_cg > 0:
            reinvested_amount += base_cg
    spendable_gross = wages + gross_ss + taxable_pensions + total_ordinary_dividends + interest_taxable + tax_exempt_interest + taxable_ira + rmd_amount + other_income + cashflow_taxfree + brokerage_proceeds + annuity_proceeds
    spendable_gross -= reinvested_amount
    net_before_tax = spendable_gross - medicare_premiums
    net_after_tax = spendable_gross - medicare_premiums - total_tax
    return {
        # Income components
        "wages": wages, "interest_taxable": interest_taxable, "tax_exempt_interest": tax_exempt_interest,
        "total_ordinary_dividends": total_ordinary_dividends, "qualified_dividends": qualified_dividends,
        "taxable_ira": taxable_ira, "rmd_amount": rmd_amount, "taxable_rmd": taxable_rmd, "qcd": qcd_amt, "taxable_pensions": taxable_pensions,
        "gross_ss": gross_ss, "taxable_ss": taxable_ss,
        "cap_gain_loss": cap_gain_loss, "other_income": other_income,
        "ordinary_tax_only": ordinary_tax_only,
        # Computed totals
        "total_income_for_tax": total_income_for_tax, "adjustments": adjustments, "agi": agi,
        # Deductions
        "deduction_used": deduction_used, "is_itemizing": is_itemizing,
        "itemized_total": itemized_total, "fed_std": fed_std,
        "mortgage_interest": mortgage_interest, "salt": salt,
        "medical_expenses": medical_exp, "medical_deduction": medical_deduction,
        "charitable": charitable_amt, "property_tax": prop_tax,
        # Tax
        "fed_taxable": fed_taxable, "fed_tax": fed["federal_tax"], "effective_fed": fed["effective_fed"],
        "sc_tax": sc["sc_tax"], "effective_sc": sc["effective_sc"], "sc_taxable": sc["sc_taxable"],
        "total_tax": total_tax,
        # SC detail
        "retirement_deduction": retirement_deduction, "dependents": dependents,
        # Medicare / cashflow
        "medicare_premiums": medicare_premiums, "has_irmaa": has_irmaa,
        "spendable_gross": spendable_gross, "net_before_tax": net_before_tax, "net_after_tax": net_after_tax,
        "cashflow_taxfree": cashflow_taxfree, "brokerage_proceeds": brokerage_proceeds,
        "annuity_proceeds": annuity_proceeds, "reinvested_amount": reinvested_amount,
    }

def _serialize_inputs_for_cache(d):
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (float, int, bool, str, type(None))): items.append((k, v))
        else: items.append((k, str(v)))
    return tuple(items)

@lru_cache(maxsize=4096)
def compute_case_cached(serialized_inputs, inflation_factor=1.0, medicare_inflation_factor=None):
    inputs = {k: v for k, v in serialized_inputs}
    return compute_case(inputs, inflation_factor, medicare_inflation_factor=medicare_inflation_factor)


# ═══════════════════════════════════════════════════════════════════════
# 10. WITHDRAWAL & INCOME NEEDS
# ═══════════════════════════════════════════════════════════════════════

def apply_withdrawal(base_inputs, base_assets, source, amount, gain_pct):
    inp = dict(base_inputs); assets = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_assets.items()}
    for k in ["cashflow_taxfree", "brokerage_proceeds", "annuity_proceeds", "ordinary_tax_only"]:
        inp[k] = float(inp.get(k, 0.0))
    amt = max(0.0, float(amount))
    if amt == 0.0: return inp, assets
    if source == "Taxable \u2013 Cash":
        amt = min(amt, assets["taxable"]["cash"]); assets["taxable"]["cash"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Taxable \u2013 Brokerage":
        amt = min(amt, assets["taxable"]["brokerage"]); assets["taxable"]["brokerage"] -= amt
        _realized_cg = amt * max(0.0, min(1.0, float(gain_pct)))
        inp["cap_gain_loss"] = float(inp.get("cap_gain_loss", 0.0)) + _realized_cg
        inp["brokerage_proceeds"] += amt
        inp["_realized_cap_gains"] = float(inp.get("_realized_cap_gains", 0.0)) + _realized_cg
    elif source == "Pre-Tax \u2013 IRA/401k":
        amt = min(amt, assets["pretax"]["balance"]); assets["pretax"]["balance"] -= amt
        inp["taxable_ira"] = float(inp.get("taxable_ira", 0.0)) + amt
    elif source == "Roth":
        amt = min(amt, assets["taxfree"]["roth_filer"]); assets["taxfree"]["roth_filer"] -= amt; assets["taxfree"]["roth"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Roth \u2014 Spouse":
        amt = min(amt, assets["taxfree"]["roth_spouse"]); assets["taxfree"]["roth_spouse"] -= amt; assets["taxfree"]["roth"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Life Insurance (loan)":
        amt = min(amt, assets["taxfree"]["life_cash_value"]); assets["taxfree"]["life_cash_value"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Annuity":
        value = assets["annuity"]["value_filer"]; basis = assets["annuity"]["basis_filer"]; amt = min(amt, value)
        if amt > 0:
            gains = annuity_gains(value, basis); taxable_ordinary = min(amt, gains); nontax_basis = amt - taxable_ordinary
            value -= amt; basis = max(0.0, basis - nontax_basis)
            assets["annuity"]["value_filer"] = value; assets["annuity"]["basis_filer"] = basis
            assets["annuity"]["value"] -= amt; assets["annuity"]["basis"] = max(0.0, assets["annuity"]["basis"] - nontax_basis)
            inp["ordinary_tax_only"] += taxable_ordinary; inp["annuity_proceeds"] += amt
    elif source == "Annuity \u2014 Spouse":
        value = assets["annuity"]["value_spouse"]; basis = assets["annuity"]["basis_spouse"]; amt = min(amt, value)
        if amt > 0:
            gains = annuity_gains(value, basis); taxable_ordinary = min(amt, gains); nontax_basis = amt - taxable_ordinary
            value -= amt; basis = max(0.0, basis - nontax_basis)
            assets["annuity"]["value_spouse"] = value; assets["annuity"]["basis_spouse"] = basis
            assets["annuity"]["value"] -= amt; assets["annuity"]["basis"] = max(0.0, assets["annuity"]["basis"] - nontax_basis)
            inp["ordinary_tax_only"] += taxable_ordinary; inp["annuity_proceeds"] += amt
    return inp, assets

def max_withdrawable(assets, source):
    if source == "Taxable \u2013 Cash": return assets["taxable"]["cash"]
    if source == "Taxable \u2013 Brokerage": return assets["taxable"]["brokerage"]
    if source == "Pre-Tax \u2013 IRA/401k": return assets["pretax"]["balance"]
    if source == "Roth": return assets["taxfree"]["roth_filer"]
    if source == "Roth \u2014 Spouse": return assets["taxfree"]["roth_spouse"]
    if source == "Life Insurance (loan)": return assets["taxfree"]["life_cash_value"]
    if source == "Annuity": return assets["annuity"]["value_filer"]
    if source == "Annuity \u2014 Spouse": return assets["annuity"]["value_spouse"]
    return 0.0

def solve_gross_up_with_assets(base_inputs, base_assets, source, gain_pct, target_net_needed, taxes_paid_by_cash, max_iter=60):
    base_case = compute_case(base_inputs, 1.0)
    def metric(res): return res["net_before_tax"] if taxes_paid_by_cash else res["net_after_tax"]
    def f(withdraw_amt):
        trial_inputs, trial_assets = apply_withdrawal(base_inputs, base_assets, source, withdraw_amt, gain_pct)
        res = compute_case(trial_inputs, 1.0)
        return metric(res) - target_net_needed, res, trial_assets, trial_inputs
    v0, r0, a0, i0 = f(0.0)
    if v0 >= 0: return 0.0, base_case, r0, base_assets, base_inputs
    cap = max_withdrawable(base_assets, source)
    if cap <= 0: return None, base_case, r0, base_assets, base_inputs
    lo, hi = 0.0, cap
    last_rr, last_aa, last_ii = r0, base_assets, base_inputs
    vh, rh, ah, ih = f(hi)
    if vh < 0: return None, base_case, rh, ah, ih
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        v, rr, aa, ii = f(mid)
        if v >= 0: hi = mid; last_rr, last_aa, last_ii = rr, aa, ii
        else: lo = mid
    return hi, base_case, last_rr, last_aa, last_ii


# ═══════════════════════════════════════════════════════════════════════
# 11. DYNAMIC SHORTFALL FILLER
# ═══════════════════════════════════════════════════════════════════════

def _fill_shortfall_dynamic(total_spend_need, cash_received, balances,
                             base_year_inp, p_base_cap_gain, inf_factor,
                             conversion_this_year, medicare_inf_factor=None):
    """Fill spending shortfall using tax-optimal source blending.

    Probes marginal tax cost of each withdrawal source via compute_case_cached,
    pulls from the cheapest source first, and uses binary search to detect
    bracket boundaries so multiple sources are blended within a single year.

    Args:
        total_spend_need: Total spending needed this year (living + mortgage)
        cash_received: Cash from fixed sources (SS + pensions + RMDs)
        balances: dict with keys cash, brokerage, pretax, roth, life,
                  annuity_value, annuity_basis, dyn_gain_pct
        base_year_inp: Base tax input dict (taxable_ira set to conversion only,
                       cap_gain_loss set to investment income only, other_income=0)
        p_base_cap_gain: Base capital gains from investment income
        inf_factor: Inflation factor for this year
        conversion_this_year: Roth conversion amount this year

    Returns:
        (wd_cash, wd_brokerage, wd_pretax, wd_roth, wd_life, wd_annuity,
         ann_gains_withdrawn, cap_gains_realized, final_tax_result)
    """
    wd_cash = wd_brokerage = wd_pretax = wd_roth = wd_life = wd_annuity = 0.0
    ann_gains_withdrawn = cap_gains_realized = 0.0

    def _build_inp(wd_pt, cg, ag):
        inp = dict(base_year_inp)
        inp["taxable_ira"] = wd_pt + conversion_this_year
        inp["cap_gain_loss"] = p_base_cap_gain + cg
        inp["other_income"] = ag
        return inp

    def _tax_total(res):
        return res["total_tax"] + res["medicare_premiums"]

    ann_total_gains = max(0.0, balances["annuity_value"] - balances["annuity_basis"])
    dyn_gain_pct = balances["dyn_gain_pct"]

    for conv_iter in range(10):
        # Compute current tax with current withdrawals
        cap_gains_realized = wd_brokerage * dyn_gain_pct
        curr_inp = _build_inp(wd_pretax, cap_gains_realized, ann_gains_withdrawn)
        curr_res = compute_case_cached(_serialize_inputs_for_cache(curr_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)
        taxes = curr_res["total_tax"]
        medicare = curr_res["medicare_premiums"]
        curr_tax = taxes + medicare

        # Compute shortfall
        total_wd = wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
        cash_available = cash_received + total_wd
        cash_needed = total_spend_need + taxes + medicare
        shortfall = cash_needed - cash_available

        if shortfall <= 100.0:
            break

        # Step 1: Cash first ($0 cost)
        avail = balances["cash"] - wd_cash
        if avail > 0:
            pull = min(shortfall, avail)
            wd_cash += pull
            shortfall -= pull

        if shortfall <= 1.0:
            continue

        # Steps 2-5: Fill from cheapest taxable source with bracket detection
        for fill_round in range(20):
            if shortfall <= 1.0:
                break

            PROBE = min(500.0, shortfall)
            candidates = []

            avail_brok = balances["brokerage"] - wd_brokerage
            if avail_brok > 0:
                p = min(PROBE, avail_brok)
                test_cg = (wd_brokerage + p) * dyn_gain_pct
                test_inp = _build_inp(wd_pretax, test_cg, ann_gains_withdrawn)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)
                cost = (_tax_total(test_res) - curr_tax) / p
                candidates.append(("brokerage", cost, avail_brok))

            avail_pre = balances["pretax"] - wd_pretax
            if avail_pre > 0:
                p = min(PROBE, avail_pre)
                test_inp = _build_inp(wd_pretax + p, cap_gains_realized, ann_gains_withdrawn)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)
                cost = (_tax_total(test_res) - curr_tax) / p
                candidates.append(("pretax", cost, avail_pre))

            avail_ann = balances["annuity_value"] - wd_annuity
            if avail_ann > 0:
                p = min(PROBE, avail_ann)
                rem_gains = max(0.0, ann_total_gains - ann_gains_withdrawn)
                test_ag = ann_gains_withdrawn + min(p, rem_gains)
                test_inp = _build_inp(wd_pretax, cap_gains_realized, test_ag)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)
                cost = (_tax_total(test_res) - curr_tax) / p
                candidates.append(("annuity", cost, avail_ann))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1])
            best_src, best_cost, best_avail = candidates[0]
            next_cost = candidates[1][1] if len(candidates) > 1 else best_cost + 0.10

            # Determine pull amount — detect bracket boundary via binary search
            pull_full = min(shortfall, best_avail)

            if pull_full > PROBE * 2:
                # Check if average cost at full pull is significantly higher
                if best_src == "brokerage":
                    test_inp_full = _build_inp(wd_pretax, (wd_brokerage + pull_full) * dyn_gain_pct, ann_gains_withdrawn)
                elif best_src == "pretax":
                    test_inp_full = _build_inp(wd_pretax + pull_full, cap_gains_realized, ann_gains_withdrawn)
                else:
                    rem = max(0.0, ann_total_gains - ann_gains_withdrawn)
                    test_inp_full = _build_inp(wd_pretax, cap_gains_realized, ann_gains_withdrawn + min(pull_full, rem))
                test_res_full = compute_case_cached(_serialize_inputs_for_cache(test_inp_full), inf_factor, medicare_inflation_factor=medicare_inf_factor)
                cost_full = (_tax_total(test_res_full) - curr_tax) / pull_full

                threshold = next_cost + 0.01
                if cost_full > threshold:
                    # Binary search for the bracket boundary
                    lo, hi = PROBE, pull_full
                    for _ in range(12):
                        mid = (lo + hi) / 2
                        if best_src == "brokerage":
                            test_inp_mid = _build_inp(wd_pretax, (wd_brokerage + mid) * dyn_gain_pct, ann_gains_withdrawn)
                        elif best_src == "pretax":
                            test_inp_mid = _build_inp(wd_pretax + mid, cap_gains_realized, ann_gains_withdrawn)
                        else:
                            rem = max(0.0, ann_total_gains - ann_gains_withdrawn)
                            test_inp_mid = _build_inp(wd_pretax, cap_gains_realized, ann_gains_withdrawn + min(mid, rem))
                        test_res_mid = compute_case_cached(_serialize_inputs_for_cache(test_inp_mid), inf_factor, medicare_inflation_factor=medicare_inf_factor)
                        cost_mid = (_tax_total(test_res_mid) - curr_tax) / mid
                        if cost_mid <= threshold:
                            lo = mid
                        else:
                            hi = mid
                    pull_full = lo

            # Apply the pull
            if best_src == "brokerage":
                wd_brokerage += pull_full
                cap_gains_realized = wd_brokerage * dyn_gain_pct
            elif best_src == "pretax":
                wd_pretax += pull_full
            elif best_src == "annuity":
                rem = max(0.0, ann_total_gains - ann_gains_withdrawn)
                ann_gains_withdrawn += min(pull_full, rem)
                wd_annuity += pull_full

            shortfall -= pull_full

            # Update curr_tax for next fill_round probe
            cap_gains_realized = wd_brokerage * dyn_gain_pct
            curr_inp = _build_inp(wd_pretax, cap_gains_realized, ann_gains_withdrawn)
            curr_res = compute_case_cached(_serialize_inputs_for_cache(curr_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)
            curr_tax = _tax_total(curr_res)

        # Step 6: Tax-free sources (last resort)
        if shortfall > 1.0:
            if conversion_this_year == 0:
                avail = balances["roth"] - wd_roth
                pull = min(shortfall, max(0.0, avail))
                if pull > 0:
                    wd_roth += pull
                    shortfall -= pull
            if shortfall > 1.0:
                avail = balances["life"] - wd_life
                pull = min(shortfall, max(0.0, avail))
                if pull > 0:
                    wd_life += pull
                    shortfall -= pull

    # Final tax result
    cap_gains_realized = wd_brokerage * dyn_gain_pct
    final_inp = _build_inp(wd_pretax, cap_gains_realized, ann_gains_withdrawn)
    final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), inf_factor, medicare_inflation_factor=medicare_inf_factor)

    return (wd_cash, wd_brokerage, wd_pretax, wd_roth, wd_life, wd_annuity,
            ann_gains_withdrawn, cap_gains_realized, final_res)


# ═══════════════════════════════════════════════════════════════════════
# 12. ESTATE VALUATION
# ═══════════════════════════════════════════════════════════════════════

def heir_value_pretax(balance, heir_tax_rate, growth_rate, dist_years=10):
    """Model 10-year inherited IRA distribution (SECURE Act).

    Heirs must empty within dist_years. Undistributed balance grows.
    Returns the net after-tax value received by heirs.
    """
    if balance <= 0 or dist_years <= 0:
        return 0.0
    remaining = float(balance)
    total_net = 0.0
    for yr in range(dist_years):
        yrs_left = dist_years - yr
        distribution = remaining / yrs_left  # even spread over remaining years
        total_net += distribution * (1.0 - heir_tax_rate)
        remaining -= distribution
        remaining *= (1.0 + growth_rate)
    return total_net


def heir_value_roth(balance, growth_rate, dist_years=10):
    """Model 10-year inherited Roth IRA (SECURE Act).

    Heirs can let it compound tax-free for up to 10 years, then take it all out tax-free.
    Returns the total tax-free value (balance * (1 + growth)^years).
    """
    if balance <= 0 or dist_years <= 0:
        return 0.0
    return float(balance) * ((1.0 + growth_rate) ** dist_years)


def heir_value_annuity(value, basis, heir_tax_rate, growth_rate, dist_years=10):
    """Model inherited annuity — gains taxed as ordinary income over dist_years, basis returned tax-free."""
    if value <= 0 or dist_years <= 0:
        return 0.0
    gains = max(0.0, value - basis)
    # Gains portion taxed like inherited IRA
    net_gains = heir_value_pretax(gains, heir_tax_rate, growth_rate, dist_years)
    # Basis returned tax-free
    return basis + net_gains


# ═══════════════════════════════════════════════════════════════════════
# 12b. ESTATE TAX COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_estate_tax(gross_estate, federal_exemption, exemption_growth, years_elapsed,
                       is_joint, use_portability,
                       state_rate=0.0, state_exemption=0.0):
    """Compute combined federal + state estate tax.

    Federal: 40% on amount over exemption (inflation-adjusted).
    State: flat rate on amount over state exemption.
    """
    adj_exemption = federal_exemption * ((1 + exemption_growth / 100) ** years_elapsed)
    if is_joint and use_portability:
        adj_exemption *= 2

    taxable_federal = max(0, gross_estate - adj_exemption)
    federal_tax = taxable_federal * 0.40

    adj_state_ex = state_exemption * ((1 + exemption_growth / 100) ** years_elapsed)
    taxable_state = max(0, gross_estate - adj_state_ex)
    state_tax = taxable_state * (state_rate / 100) if state_rate > 0 else 0.0

    return federal_tax + state_tax


# ═══════════════════════════════════════════════════════════════════════
# 13. WEALTH PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_wealth_projection(initial_assets, params, spending_order, conversion_strategy="none",
                           target_agi=0, stop_conversion_age=100, conversion_years_limit=0,
                           blend_mode=False, pretax_annual_cap=None, prorata_blend=False,
                           prorata_weights=None, adaptive_strategy=None,
                           extra_pretax_bracket=None, annuity_depletion_years=None,
                           annuity_gains_only=False, harvest_gains_bracket=None):
    """Unified wealth projection engine used by both Tab 3 and Tab 4.

    Combines:
    - Full compute_case() tax calc with itemized deductions (from Tab 3)
    - Separate cash/brokerage with dynamic basis tracking (from Pre-Ret/Tab 4)
    - Annuity gains/basis tracking (from Tab 4)
    - Surplus reinvestment with basis update (from Tab 3/Pre-Ret)
    - Negative balance protection (from Tab 4)
    - Roth conversion guard (from Tab 4)
    - Home value tracking (from Tab 3)
    - Medicare/IRMAA (from Tab 3/Tab 4)
    - Investment income reinvestment flags (from Tab 3)
    """
    spending_goal = params["spending_goal"]
    start_year = params["start_year"]
    years = params["years"]
    inflation = params["inflation"]
    bracket_growth = params.get("bracket_growth", inflation)
    medicare_growth = params.get("medicare_growth", inflation)
    pension_cola = params["pension_cola"]
    heir_tax_rate = params["heir_tax_rate"]
    r_cash = params.get("r_cash", 0.02)
    r_taxable = params["r_taxable"]
    r_pretax = params["r_pretax"]
    r_roth = params["r_roth"]
    r_annuity = params["r_annuity"]
    r_life = params["r_life"]
    gross_ss_total = params["gross_ss_total"]
    taxable_pensions_total = params["taxable_pensions_total"]
    filing_status = params["filing_status"]
    current_age_filer = params["current_age_filer"]
    current_age_spouse = params["current_age_spouse"]
    pretax_filer_ratio = params["pretax_filer_ratio"]
    brokerage_gain_pct = params["brokerage_gain_pct"]

    # Per-spouse SS detail params for year-by-year calculation
    _ss_filer_dob = params.get("filer_dob")
    _ss_spouse_dob = params.get("spouse_dob")
    _ss_filer_already = params.get("filer_ss_already", False)
    _ss_filer_current = params.get("filer_ss_current", 0.0)
    _ss_filer_start_year = params.get("filer_ss_start_year", 9999)
    _ss_filer_fra = params.get("filer_ss_fra", 0.0)
    _ss_filer_claim = params.get("filer_ss_claim", "FRA")
    _ss_spouse_already = params.get("spouse_ss_already", False)
    _ss_spouse_current = params.get("spouse_ss_current", 0.0)
    _ss_spouse_start_year = params.get("spouse_ss_start_year", 9999)
    _ss_spouse_fra = params.get("spouse_ss_fra", 0.0)
    _ss_spouse_claim = params.get("spouse_ss_claim", "FRA")
    _ss_current_year = params.get("current_year", start_year)
    # Whether we have detail params to compute SS year-by-year
    _ss_have_detail = _ss_filer_dob is not None
    # Fallback: static base amounts (backward compatible)
    ss_filer_base = params.get("gross_ss_filer")
    ss_spouse_base = params.get("gross_ss_spouse", 0.0)
    if ss_filer_base is None:
        ss_filer_base = gross_ss_total
        ss_spouse_base = 0.0

    pen_filer_base = params.get("pension_filer")
    pen_spouse_base = params.get("pension_spouse", 0.0)
    if pen_filer_base is None:
        pen_filer_base = taxable_pensions_total
        pen_spouse_base = 0.0

    filer_plan_age = params.get("filer_plan_through_age")
    spouse_plan_age = params.get("spouse_plan_through_age")
    survivor_spending_pct = params.get("survivor_spending_pct", 100) / 100.0
    pension_survivor_pct_val = params.get("pension_survivor_pct", 100) / 100.0

    # Compute first-death year index
    if filer_plan_age and current_age_spouse and spouse_plan_age:
        filer_death_idx = filer_plan_age - current_age_filer
        spouse_death_idx = spouse_plan_age - current_age_spouse
        first_death_idx = min(filer_death_idx, spouse_death_idx)
        filer_dies_first = filer_death_idx <= spouse_death_idx
    else:
        first_death_idx = None
        filer_dies_first = None

    # Sidebar income/deduction items carried into projection
    p_wages = params.get("wages", 0.0)
    p_tax_exempt_interest = params.get("tax_exempt_interest", 0.0)
    p_other_income = params.get("other_income", 0.0)
    p_adjustments = params.get("adjustments", 0.0)

    # Investment income detail (for full tax calc)
    p_interest_taxable = params.get("interest_taxable", 0.0)
    p_total_ordinary_div = params.get("total_ordinary_dividends", 0.0)
    p_qualified_div = params.get("qualified_dividends", 0.0)
    p_base_cap_gain = params.get("cap_gain_loss", 0.0)
    p_reinvest_int = params.get("reinvest_interest", False)
    p_reinvest_div = params.get("reinvest_dividends", False)
    p_reinvest_cg = params.get("reinvest_cap_gains", False)

    # Compute implied yields so investment income scales with brokerage balance
    _init_brok = initial_assets["taxable"]["brokerage"]
    if _init_brok > 0:
        _div_yield = p_total_ordinary_div / _init_brok
        _qual_ratio = p_qualified_div / p_total_ordinary_div if p_total_ordinary_div > 0 else 0.0
        _cg_yield = max(0.0, p_base_cap_gain) / _init_brok
    else:
        _div_yield = 0.0
        _qual_ratio = 0.0
        _cg_yield = 0.0

    # Derive brokerage interest yield: total interest minus estimated cash interest
    _cash_int_est = params.get("emergency_fund", 0.0) * r_cash
    _int_yield = max(0.0, p_interest_taxable - _cash_int_est) / _init_brok if _init_brok > 0 else 0.0

    # Growth rate = total return (includes dividend + CG + interest yield).
    # When not reinvesting, reduce effective growth by distributed yields.
    _div_drag = ((_div_yield if not p_reinvest_div else 0.0)
                 + (_cg_yield if not p_reinvest_cg else 0.0)
                 + (_int_yield if not p_reinvest_int else 0.0))

    # Deduction inputs
    p_retirement_deduction = params.get("retirement_deduction", 0.0)
    p_out_of_state_gain = params.get("out_of_state_gain", 0.0)
    p_dependents = params.get("dependents", 0)
    p_property_tax = params.get("property_tax", 0.0)
    p_medical_expenses = params.get("medical_expenses", 0.0)
    p_charitable = params.get("charitable", 0.0)
    p_qcd_annual = params.get("qcd_annual", 0.0)

    # Mortgage
    mtg_balance = params.get("mortgage_balance", 0.0)
    mtg_rate = params.get("mortgage_rate", 0.0)
    mtg_payment = params.get("mortgage_payment", 0.0)

    # Home
    home_val = params.get("home_value", 0.0)
    home_appr = params.get("home_appreciation", 0.0)

    # Surplus destination
    surplus_dest = params.get("surplus_destination", "brokerage")

    # Estate tax params
    _estate_tax_enabled = params.get("estate_tax_enabled", False)
    _estate_fed_exemption = params.get("federal_estate_exemption", 15000000.0)
    _estate_exemption_growth = params.get("exemption_inflation", 2.5)
    _estate_use_portability = params.get("use_portability", True)
    _estate_state_rate = params.get("state_estate_tax_rate", 0.0)
    _estate_state_exemption = params.get("state_estate_exemption", 0.0)
    _estate_is_joint = "joint" in filing_status.lower()

    # Initialize balances
    curr_cash = initial_assets["taxable"]["cash"]
    curr_brokerage = initial_assets["taxable"]["brokerage"]
    curr_ef = initial_assets["taxable"].get("emergency_fund", 0.0)
    brokerage_basis = curr_brokerage * (1.0 - brokerage_gain_pct)
    total_pre = initial_assets["pretax"]["balance"]
    curr_pre_filer = total_pre * pretax_filer_ratio
    curr_pre_spouse = total_pre * (1.0 - pretax_filer_ratio)
    curr_roth = initial_assets["taxfree"]["roth"]
    curr_life = initial_assets["taxfree"]["life_cash_value"]
    curr_ann = initial_assets["annuity"]["value"]
    curr_ann_basis = initial_assets["annuity"]["basis"]
    curr_mtg_bal = mtg_balance
    curr_home_val = home_val

    # Spending: mortgage stays fixed, rest inflates
    initial_mtg_pmt = mtg_payment if curr_mtg_bal > 0 else 0.0
    base_non_mtg = max(0.0, spending_goal - initial_mtg_pmt)

    total_taxes_paid = 0.0
    total_converted = 0.0
    year_details = []

    do_conversions = conversion_strategy != "none" and conversion_strategy != 0 and conversion_strategy != 0.0

    # Resolve adaptive strategy definition
    _adaptive_phases = None
    if adaptive_strategy and adaptive_strategy in ADAPTIVE_STRATEGIES:
        _adaptive_phases = ADAPTIVE_STRATEGIES[adaptive_strategy]["phases"]

    for i in range(years):
        yr = start_year + i
        age_f = current_age_filer + i
        age_s = (current_age_spouse + i) if current_age_spouse else None
        inf_factor = (1 + inflation) ** i
        bracket_inf = (1 + bracket_growth) ** i
        medicare_inf = (1 + medicare_growth) ** i
        filer_65 = age_f >= 65
        spouse_65 = age_s >= 65 if age_s else False

        # Adaptive strategy: resolve per-year overrides from active phase
        _yr_blend_mode = blend_mode
        _yr_pt_cap = pretax_annual_cap
        _yr_prorata = prorata_blend
        _yr_conv_strat = conversion_strategy
        if _adaptive_phases:
            for _phase in _adaptive_phases:
                _phase_from = _phase.get("from_age", 0)
                _phase_until = _phase.get("until_age", 999)
                if _phase_from <= age_f < _phase_until:
                    _yr_blend_mode = _phase.get("blend_mode", False)
                    _yr_pt_cap = _phase.get("pt_cap", None)
                    _yr_prorata = _phase.get("prorata", False)
                    # If caller provided an explicit conversion_strategy (Phase 2 override),
                    # use it instead of the adaptive phase's built-in conversion
                    if conversion_strategy == "none" or conversion_strategy == 0 or conversion_strategy == 0.0:
                        _yr_conv_strat = _phase.get("conversion_strategy", "none")
                    break

        # Investment income: scale with current brokerage balance (yield-based)
        yr_ordinary_div = curr_brokerage * _div_yield
        yr_qualified_div = yr_ordinary_div * _qual_ratio
        yr_cap_gain = curr_brokerage * _cg_yield
        yr_interest = curr_brokerage * _int_yield + curr_cash * r_cash

        reinvested_base = 0.0
        if p_reinvest_int:
            reinvested_base += yr_interest
        if p_reinvest_div:
            reinvested_base += yr_ordinary_div
        if p_reinvest_cg and yr_cap_gain > 0:
            reinvested_base += yr_cap_gain
        spendable_inv = (yr_interest + yr_ordinary_div + yr_cap_gain) - reinvested_base

        # Spending
        yr_mtg_payment = mtg_payment if curr_mtg_bal > 0 else 0.0
        total_spend_need = base_non_mtg * inf_factor + yr_mtg_payment

        # Additional expenses from Tab 2
        yr_addl_expense = 0.0
        _active_addl_expenses = []
        for _fe in params.get("additional_expenses", []):
            _fe_end = _fe["end_age"] if _fe["end_age"] > _fe["start_age"] else _fe["start_age"] + 1
            if _fe["start_age"] <= age_f < _fe_end:
                _fe_yrs = age_f - _fe["start_age"]
                _fe_amt = _fe["net_amount"] * ((1 + inflation) ** _fe_yrs) if _fe["inflates"] else _fe["net_amount"]
                yr_addl_expense += _fe_amt
                _active_addl_expenses.append({"amount": _fe_amt, "source": _fe["source"]})
        total_spend_need += yr_addl_expense

        # Charitable giving as real spending
        yr_charitable = p_charitable * inf_factor
        total_spend_need += yr_charitable

        # Survivor mode: reduce spending after first spouse death
        _survivor_mode = first_death_idx is not None and i >= first_death_idx
        # Filing status: year of death can still file joint; year AFTER switches to single
        if first_death_idx is not None and i > first_death_idx and "joint" in filing_status.lower():
            _yr_filing_status = "Single"
            _yr_filer_65 = (age_f >= 65) if not filer_dies_first else (age_s >= 65 if age_s else False)
            _yr_spouse_65 = False
        else:
            _yr_filing_status = filing_status
            _yr_filer_65 = filer_65
            _yr_spouse_65 = spouse_65
        if _survivor_mode:
            total_spend_need *= survivor_spending_pct

        # Fixed income — per-spouse SS with survivor logic
        if _ss_have_detail:
            _ss_filer_yr = annual_ss_in_year(dob=_ss_filer_dob, tax_year=yr, cola=inflation,
                already_receiving=_ss_filer_already, current_annual=_ss_filer_current,
                start_year=_ss_filer_start_year, fra_annual=_ss_filer_fra,
                claim_choice=_ss_filer_claim, current_year=_ss_current_year)
            _ss_spouse_yr = annual_ss_in_year(dob=_ss_spouse_dob, tax_year=yr, cola=inflation,
                already_receiving=_ss_spouse_already, current_annual=_ss_spouse_current,
                start_year=_ss_spouse_start_year, fra_annual=_ss_spouse_fra,
                claim_choice=_ss_spouse_claim, current_year=_ss_current_year) if _ss_spouse_dob else 0.0
        else:
            _ss_filer_yr = ss_filer_base * inf_factor
            _ss_spouse_yr = ss_spouse_base * inf_factor
        if _survivor_mode:
            # Survivor gets higher of the two benefits; deceased gets zero
            _ss_filer_full = _ss_filer_yr
            _ss_spouse_full = _ss_spouse_yr
            if filer_dies_first:
                _ss_filer_yr = 0.0
                _ss_spouse_yr = max(_ss_spouse_full, _ss_filer_full)
            else:
                _ss_spouse_yr = 0.0
                _ss_filer_yr = max(_ss_filer_full, _ss_spouse_full)
        ss_now = _ss_filer_yr + _ss_spouse_yr

        # Per-spouse pension with survivor logic
        _pen_cola = (1 + pension_cola) ** i
        _pen_filer_yr = pen_filer_base * _pen_cola
        _pen_spouse_yr = pen_spouse_base * _pen_cola
        if _survivor_mode:
            if filer_dies_first:
                _pen_filer_yr *= pension_survivor_pct_val
            else:
                _pen_spouse_yr *= pension_survivor_pct_val
        pen_now = _pen_filer_yr + _pen_spouse_yr

        # Pre-tax: RMD
        boy_pretax = curr_pre_filer + curr_pre_spouse
        rmd_f = compute_rmd_uniform_start73(curr_pre_filer, age_f)
        rmd_s = compute_rmd_uniform_start73(curr_pre_spouse, age_s)
        rmd_total = rmd_f + rmd_s
        curr_pre_filer -= rmd_f
        curr_pre_spouse -= rmd_s

        # QCD: direct IRA-to-charity transfer
        yr_qcd = 0.0
        qcd_beyond_rmd = 0.0
        if p_qcd_annual > 0 and age_f >= 70:
            is_joint_yr = "joint" in _yr_filing_status.lower()
            qcd_cap = (210000.0 if is_joint_yr else 105000.0) * inf_factor
            yr_qcd_requested = min(p_qcd_annual * inf_factor, qcd_cap)
            # QCD within RMD: no additional IRA withdrawal needed (already withdrawn via RMD)
            qcd_within_rmd = min(yr_qcd_requested, rmd_total)
            # QCD beyond RMD: additional withdrawal from pre-tax
            qcd_beyond_rmd = max(0.0, yr_qcd_requested - rmd_total)
            avail_pt = curr_pre_filer + curr_pre_spouse
            qcd_beyond_rmd = min(qcd_beyond_rmd, max(0.0, avail_pt))
            yr_qcd = qcd_within_rmd + qcd_beyond_rmd
            # Withdraw beyond-RMD portion from pre-tax balances directly
            if qcd_beyond_rmd > 0 and avail_pt > 0:
                filer_share = curr_pre_filer / avail_pt
                curr_pre_filer -= qcd_beyond_rmd * filer_share
                curr_pre_spouse -= qcd_beyond_rmd * (1 - filer_share)

        # QCD adjustments: reduce taxable RMD, avoid double-dipping deduction, reduce spending need
        taxable_rmd = rmd_total - min(yr_qcd, rmd_total)
        yr_charitable_deduction = max(0.0, yr_charitable - yr_qcd)
        total_spend_need -= yr_qcd  # QCD pays this portion of charitable directly from IRA

        # QCD reserve: when QCD is active, the IRA is occupied funding charitable gifts.
        # Protect the full remaining balance from voluntary withdrawals (conversions, waterfall, accel PT).
        # The overflow / last-resort path can still tap it if all other accounts are exhausted.
        qcd_reserve = 0.0
        if p_qcd_annual > 0 and age_f >= 70:
            qcd_reserve = max(0.0, curr_pre_filer + curr_pre_spouse)

        # Roth conversion (after RMD, before waterfall)
        conversion_this_year = 0.0
        _do_conv = (_yr_conv_strat != "none" and _yr_conv_strat != 0 and _yr_conv_strat != 0.0) if _adaptive_phases else do_conversions
        if _do_conv and (i < conversion_years_limit or _adaptive_phases) and age_f < stop_conversion_age:
            avail_pretax = max(0.0, curr_pre_filer + curr_pre_spouse - qcd_reserve)
            if avail_pretax > 0:
                _yr_conv_strategy = _yr_conv_strat
                # Estimate spending-driven pre-tax withdrawal so bracket fill accounts for it
                _est_fixed = ss_now + pen_now + taxable_rmd + spendable_inv + p_wages + p_other_income
                _est_spending_wd = max(0.0, total_spend_need - _est_fixed)
                base_taxable = pen_now + taxable_rmd + yr_interest + yr_ordinary_div + yr_cap_gain + _est_spending_wd
                if _yr_conv_strategy == "fill_to_target":
                    room = max(0.0, target_agi * bracket_inf - base_taxable - ss_now * 0.85)
                    conversion_this_year = min(room, avail_pretax)
                elif isinstance(_yr_conv_strategy, str) and _yr_conv_strategy.startswith("fill_bracket_"):
                    _br_rate = {"fill_bracket_12": 0.12, "fill_bracket_22": 0.22,
                                "fill_bracket_24": 0.24, "fill_bracket_32": 0.32}.get(_yr_conv_strategy, 0.0)
                    if _br_rate > 0:
                        pref_amt = max(0.0, yr_cap_gain) + max(0.0, yr_qualified_div)
                        fill_amt = compute_bracket_fill_amount(
                            _br_rate, base_taxable, ss_now, _yr_filing_status,
                            _yr_filer_65, _yr_spouse_65, p_retirement_deduction,
                            preferential_amount=pref_amt, inf_factor=bracket_inf, tax_year=yr)
                        conversion_this_year = min(max(0.0, fill_amt), avail_pretax)
                elif isinstance(_yr_conv_strategy, str) and _yr_conv_strategy.startswith("fill_irmaa_"):
                    _irmaa_tier = int(_yr_conv_strategy.split("_")[-1])
                    safe_amt = compute_irmaa_safe_amount(
                        _irmaa_tier, base_taxable, ss_now, _yr_filing_status, inf_factor=bracket_inf)
                    conversion_this_year = min(max(0.0, safe_amt), avail_pretax)
                elif isinstance(_yr_conv_strategy, (int, float)):
                    conversion_this_year = min(float(_yr_conv_strategy), avail_pretax)
                else:
                    try:
                        conversion_this_year = min(float(_yr_conv_strategy), avail_pretax)
                    except (ValueError, TypeError):
                        conversion_this_year = 0.0

        if conversion_this_year > 0:
            avail_pretax = curr_pre_filer + curr_pre_spouse
            if avail_pretax > 0:
                filer_share = curr_pre_filer / avail_pretax
                curr_pre_filer -= conversion_this_year * filer_share
                curr_pre_spouse -= conversion_this_year * (1 - filer_share)
            curr_roth += conversion_this_year
            total_converted += conversion_this_year

        # Cash received from fixed sources (QCD portion of RMD goes to charity, not spendable)
        cash_received = ss_now + pen_now + taxable_rmd + spendable_inv + p_wages + p_other_income + p_tax_exempt_interest

        # Future income from Tab 2
        yr_extra_taxable = 0.0
        yr_extra_taxfree = 0.0
        for _fi in params.get("future_income", []):
            _fi_end = _fi["end_age"] if _fi["end_age"] > _fi["start_age"] else 999
            if _fi["start_age"] <= age_f < _fi_end:
                _fi_yrs = age_f - _fi["start_age"]
                _fi_amt = _fi["amount"] * ((1 + inflation) ** _fi_yrs) if _fi["inflates"] else _fi["amount"]
                if _fi["taxable"]:
                    yr_extra_taxable += _fi_amt
                else:
                    yr_extra_taxfree += _fi_amt
        yr_extra_income = yr_extra_taxable + yr_extra_taxfree
        cash_received += yr_extra_income

        # Withdrawal loop
        wd_cash = 0.0
        wd_brokerage = 0.0
        wd_pretax = 0.0
        wd_roth = 0.0
        wd_life = 0.0
        wd_annuity = 0.0
        ann_gains_withdrawn = 0.0
        cap_gains_realized = 0.0

        # Forced annuity withdrawal: pre-seed wd_annuity over N years
        if annuity_depletion_years is not None and curr_ann > 0 and i < annuity_depletion_years:
            _remaining_depl_yrs = annuity_depletion_years - i
            _total_ann_gains = max(0.0, curr_ann - curr_ann_basis)
            if annuity_gains_only:
                # Only withdraw gains — leave basis intact for tax-free heir transfer
                if _total_ann_gains > 0:
                    _forced_ann_wd = _total_ann_gains / _remaining_depl_yrs
                    _forced_gains = _forced_ann_wd  # all gains (LIFO)
                else:
                    _forced_ann_wd = 0.0
                    _forced_gains = 0.0
            else:
                _forced_ann_wd = curr_ann / _remaining_depl_yrs
                _forced_gains = min(_forced_ann_wd, _total_ann_gains)
            wd_annuity = _forced_ann_wd
            ann_gains_withdrawn = _forced_gains

        # Pre-pull additional expenses from their designated sources
        for _aex in _active_addl_expenses:
            _aex_amt = _aex["amount"]
            _aex_src = _aex["source"]
            if _aex_src == "Taxable \u2013 Cash":
                _pull = min(_aex_amt, max(0.0, curr_cash - wd_cash))
                wd_cash += _pull
            elif _aex_src == "Taxable \u2013 Brokerage":
                _pull = min(_aex_amt, max(0.0, curr_brokerage - wd_brokerage))
                wd_brokerage += _pull
            elif _aex_src == "Pre-Tax \u2013 IRA/401k":
                _pull = min(_aex_amt, max(0.0, curr_pre_filer + curr_pre_spouse - wd_pretax))
                wd_pretax += _pull
            elif _aex_src == "Roth":
                _pull = min(_aex_amt, max(0.0, curr_roth - wd_roth))
                wd_roth += _pull
            elif _aex_src == "Life Insurance (loan)":
                _pull = min(_aex_amt, max(0.0, curr_life - wd_life))
                wd_life += _pull
            elif _aex_src == "Annuity":
                _pull = min(_aex_amt, max(0.0, curr_ann - wd_annuity))
                if _pull > 0:
                    _total_gains = max(0.0, curr_ann - curr_ann_basis)
                    _rem_gains = max(0.0, _total_gains - ann_gains_withdrawn)
                    ann_gains_withdrawn += min(_pull, _rem_gains)
                wd_annuity += _pull

        yr_agi = 0.0  # initialized before branch; set in each path below

        if _yr_blend_mode:
            # Dynamic blend: tax-optimal source selection
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            base_year_inp = {
                "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                "rmd_amount": taxable_rmd,
                "taxable_ira": conversion_this_year,
                "total_ordinary_dividends": yr_ordinary_div,
                "qualified_dividends": yr_qualified_div,
                "tax_exempt_interest": p_tax_exempt_interest,
                "interest_taxable": yr_interest,
                "cap_gain_loss": yr_cap_gain,
                "other_income": yr_extra_taxable + p_other_income,
                "ordinary_tax_only": 0.0,
                "adjustments": p_adjustments,
                "reinvest_interest": p_reinvest_int,
                "reinvest_dividends": p_reinvest_div,
                "reinvest_cap_gains": p_reinvest_cg,
                "filing_status": _yr_filing_status,
                "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                "dependents": p_dependents,
                "retirement_deduction": p_retirement_deduction * inf_factor,
                "out_of_state_gain": p_out_of_state_gain,
                "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                "mortgage_payment": mtg_payment,
                "property_tax": p_property_tax * inf_factor,
                "medical_expenses": p_medical_expenses * inf_factor,
                "charitable": yr_charitable_deduction,
                "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
            }
            blend_balances = {
                "cash": curr_cash, "brokerage": curr_brokerage,
                "pretax": max(0.0, curr_pre_filer + curr_pre_spouse - qcd_reserve),
                "roth": curr_roth, "life": curr_life,
                "annuity_value": curr_ann, "annuity_basis": curr_ann_basis,
                "dyn_gain_pct": dyn_gain_pct,
            }
            (wd_cash, wd_brokerage, wd_pretax, wd_roth, wd_life, wd_annuity,
             ann_gains_withdrawn, cap_gains_realized, final_res) = _fill_shortfall_dynamic(
                total_spend_need, cash_received, blend_balances, base_year_inp,
                yr_cap_gain, bracket_inf, conversion_this_year, medicare_inf_factor=medicare_inf)
            yr_tax = final_res["total_tax"]
            yr_agi = final_res["agi"]
            yr_medicare = final_res["medicare_premiums"]

        elif _yr_pt_cap is not None:
            # Smart blend: targeted pre-tax + brokerage each year
            _pt_cap = _yr_pt_cap * inf_factor  # inflation-adjust the cap
            for iteration in range(20):
                dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
                cap_gains_realized = wd_brokerage * dyn_gain_pct
                trial_inp = {
                    "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                    "rmd_amount": taxable_rmd,
                    "taxable_ira": wd_pretax + conversion_this_year,
                    "total_ordinary_dividends": yr_ordinary_div,
                    "qualified_dividends": yr_qualified_div,
                    "tax_exempt_interest": p_tax_exempt_interest,
                    "interest_taxable": yr_interest,
                    "cap_gain_loss": yr_cap_gain + cap_gains_realized,
                    "other_income": ann_gains_withdrawn + yr_extra_taxable + p_other_income,
                    "ordinary_tax_only": 0.0,
                    "adjustments": p_adjustments,
                    "reinvest_dividends": p_reinvest_div,
                    "reinvest_cap_gains": p_reinvest_cg,
                    "filing_status": _yr_filing_status,
                    "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                    "dependents": p_dependents,
                    "retirement_deduction": p_retirement_deduction * inf_factor,
                    "out_of_state_gain": p_out_of_state_gain,
                    "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                    "mortgage_payment": mtg_payment,
                    "property_tax": p_property_tax * inf_factor,
                    "medical_expenses": p_medical_expenses * inf_factor,
                    "charitable": yr_charitable_deduction,
                    "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
                }
                trial_res = compute_case_cached(_serialize_inputs_for_cache(trial_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
                taxes = trial_res["total_tax"]
                medicare = trial_res["medicare_premiums"]

                cash_available = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
                cash_needed = total_spend_need + taxes + medicare
                shortfall = cash_needed - cash_available
                if shortfall <= 1.0:
                    break

                # 1. Cash first ($0 tax cost)
                avail = curr_cash - wd_cash
                if avail > 0:
                    pull = min(shortfall, avail)
                    wd_cash += pull; shortfall -= pull

                # 2. Pre-tax up to the annual cap (fills low brackets), reserving for QCD
                if shortfall > 0:
                    avail_pt = max(0.0, curr_pre_filer + curr_pre_spouse - wd_pretax - qcd_reserve)
                    pt_room = max(0.0, _pt_cap - wd_pretax)  # remaining cap this year
                    pull = min(shortfall, avail_pt, pt_room)
                    if pull > 0:
                        wd_pretax += pull; shortfall -= pull

                # 3. Brokerage for remaining (preferential cap gains rate)
                if shortfall > 0:
                    avail = max(0.0, curr_brokerage - wd_brokerage)
                    pull = min(shortfall, avail)
                    if pull > 0:
                        wd_brokerage += pull; shortfall -= pull

                # 4. Annuity
                if shortfall > 0:
                    avail = max(0.0, curr_ann - wd_annuity)
                    pull = min(shortfall, avail)
                    if pull > 0:
                        total_gains = max(0.0, curr_ann - curr_ann_basis)
                        remaining_gains = max(0.0, total_gains - ann_gains_withdrawn)
                        new_gains = min(pull, remaining_gains)
                        wd_annuity += pull; ann_gains_withdrawn += new_gains; shortfall -= pull

                # 5. Pre-tax overflow (if brokerage/annuity ran out, use more pre-tax beyond cap)
                if shortfall > 0:
                    avail = max(0.0, curr_pre_filer + curr_pre_spouse - wd_pretax)
                    pull = min(shortfall, avail)
                    if pull > 0:
                        wd_pretax += pull; shortfall -= pull

                # 6. Tax-free last resort (Roth, then Life)
                if shortfall > 0:
                    if conversion_this_year == 0:
                        avail = max(0.0, curr_roth - wd_roth)
                        pull = min(shortfall, avail)
                        if pull > 0:
                            wd_roth += pull; shortfall -= pull
                    if shortfall > 0:
                        avail = max(0.0, curr_life - wd_life)
                        pull = min(shortfall, avail)
                        if pull > 0:
                            wd_life += pull; shortfall -= pull

            # Final tax calc with settled withdrawals
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            cap_gains_realized = wd_brokerage * dyn_gain_pct
            final_inp = dict(trial_inp)
            final_inp["taxable_ira"] = wd_pretax + conversion_this_year
            final_inp["cap_gain_loss"] = yr_cap_gain + cap_gains_realized
            final_inp["other_income"] = ann_gains_withdrawn + yr_extra_taxable + p_other_income
            final_inp["cashflow_taxfree"] = yr_extra_taxfree
            final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
            yr_tax = final_res["total_tax"]
            yr_agi = final_res["agi"]
            yr_medicare = final_res["medicare_premiums"]

        elif _yr_prorata:
            # Pro-rata blend: pull from ALL accounts in proportion to balance (or custom weights)
            _pw = prorata_weights or {}  # optional weight overrides: {"pretax": 2.0, "roth": 0.5, ...}
            for iteration in range(20):
                dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
                cap_gains_realized = wd_brokerage * dyn_gain_pct
                trial_inp = {
                    "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                    "rmd_amount": taxable_rmd,
                    "taxable_ira": wd_pretax + conversion_this_year,
                    "total_ordinary_dividends": yr_ordinary_div,
                    "qualified_dividends": yr_qualified_div,
                    "tax_exempt_interest": p_tax_exempt_interest,
                    "interest_taxable": yr_interest,
                    "cap_gain_loss": yr_cap_gain + cap_gains_realized,
                    "other_income": ann_gains_withdrawn + yr_extra_taxable + p_other_income,
                    "ordinary_tax_only": 0.0,
                    "adjustments": p_adjustments,
                    "reinvest_dividends": p_reinvest_div,
                    "reinvest_cap_gains": p_reinvest_cg,
                    "filing_status": _yr_filing_status,
                    "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                    "dependents": p_dependents,
                    "retirement_deduction": p_retirement_deduction * inf_factor,
                    "out_of_state_gain": p_out_of_state_gain,
                    "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                    "mortgage_payment": mtg_payment,
                    "property_tax": p_property_tax * inf_factor,
                    "medical_expenses": p_medical_expenses * inf_factor,
                    "charitable": yr_charitable_deduction,
                    "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
                }
                trial_res = compute_case_cached(_serialize_inputs_for_cache(trial_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
                taxes = trial_res["total_tax"]
                medicare = trial_res["medicare_premiums"]

                cash_available = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
                cash_needed = total_spend_need + taxes + medicare
                shortfall = cash_needed - cash_available
                if shortfall <= 1.0:
                    break

                # Compute weighted balances for proportional allocation
                avail_cash = max(0.0, curr_cash - wd_cash)
                avail_brok = max(0.0, curr_brokerage - wd_brokerage)
                avail_pt = max(0.0, curr_pre_filer + curr_pre_spouse - wd_pretax - qcd_reserve)
                avail_roth = max(0.0, curr_roth - wd_roth) if conversion_this_year == 0 else 0.0
                avail_life = max(0.0, curr_life - wd_life)
                avail_ann = max(0.0, curr_ann - wd_annuity)

                w_cash = avail_cash * _pw.get("cash", 1.0)
                w_brok = avail_brok * _pw.get("brokerage", 1.0)
                w_pt = avail_pt * _pw.get("pretax", 1.0)
                w_roth = avail_roth * _pw.get("roth", 1.0)
                w_life = avail_life * _pw.get("life", 1.0)
                w_ann = avail_ann * _pw.get("annuity", 1.0)
                w_total = w_cash + w_brok + w_pt + w_roth + w_life + w_ann

                if w_total <= 0:
                    break

                # Pull from each account proportionally
                if w_cash > 0:
                    pull = min(shortfall * w_cash / w_total, avail_cash)
                    wd_cash += pull
                if w_brok > 0:
                    pull = min(shortfall * w_brok / w_total, avail_brok)
                    wd_brokerage += pull
                if w_pt > 0:
                    pull = min(shortfall * w_pt / w_total, avail_pt)
                    wd_pretax += pull
                if w_ann > 0:
                    pull = min(shortfall * w_ann / w_total, avail_ann)
                    rem_gains = max(0.0, (curr_ann - (wd_annuity - pull)) - curr_ann_basis)
                    ann_gains_withdrawn = min(wd_annuity, max(0.0, curr_ann - curr_ann_basis))
                    wd_annuity += pull
                if w_roth > 0:
                    pull = min(shortfall * w_roth / w_total, avail_roth)
                    wd_roth += pull
                if w_life > 0:
                    pull = min(shortfall * w_life / w_total, avail_life)
                    wd_life += pull

            # Final tax calc with settled withdrawals
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            cap_gains_realized = wd_brokerage * dyn_gain_pct
            ann_gains_withdrawn = min(wd_annuity, max(0.0, curr_ann - curr_ann_basis)) if wd_annuity > 0 else 0.0
            final_inp = dict(trial_inp)
            final_inp["taxable_ira"] = wd_pretax + conversion_this_year
            final_inp["cap_gain_loss"] = yr_cap_gain + cap_gains_realized
            final_inp["other_income"] = ann_gains_withdrawn + yr_extra_taxable + p_other_income
            final_inp["cashflow_taxfree"] = yr_extra_taxfree
            final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
            yr_tax = final_res["total_tax"]
            yr_agi = final_res["agi"]
            yr_medicare = final_res["medicare_premiums"]

        else:
            # Fixed waterfall ordering
            for iteration in range(20):
                # Dynamic brokerage gain % based on current basis
                dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
                # Recompute cap gains from brokerage withdrawals with dynamic basis
                cap_gains_realized = wd_brokerage * dyn_gain_pct

                # Build tax inputs for full compute_case
                trial_inp = {
                    "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                    "rmd_amount": taxable_rmd,
                    "taxable_ira": wd_pretax + conversion_this_year,
                    "total_ordinary_dividends": yr_ordinary_div,
                    "qualified_dividends": yr_qualified_div,
                    "tax_exempt_interest": p_tax_exempt_interest,
                    "interest_taxable": yr_interest,
                    "cap_gain_loss": yr_cap_gain + cap_gains_realized,
                    "other_income": ann_gains_withdrawn + yr_extra_taxable + p_other_income,
                    "ordinary_tax_only": 0.0,
                    "adjustments": p_adjustments,
                    "reinvest_dividends": p_reinvest_div,
                    "reinvest_cap_gains": p_reinvest_cg,
                    "filing_status": _yr_filing_status,
                    "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                    "dependents": p_dependents,
                    "retirement_deduction": p_retirement_deduction * inf_factor,
                    "out_of_state_gain": p_out_of_state_gain,
                    "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                    "mortgage_payment": mtg_payment,
                    "property_tax": p_property_tax * inf_factor,
                    "medical_expenses": p_medical_expenses * inf_factor,
                    "charitable": yr_charitable_deduction,
                    "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
                }
                trial_res = compute_case_cached(_serialize_inputs_for_cache(trial_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
                taxes = trial_res["total_tax"]
                medicare = trial_res["medicare_premiums"]

                # Cash flow: fixed sources (incl. investment income) + withdrawals
                cash_available = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
                cash_needed = total_spend_need + taxes + medicare
                shortfall = cash_needed - cash_available

                if shortfall <= 1.0:
                    break

                pulled = False
                for bucket in spending_order:
                    if shortfall <= 0:
                        break

                    if bucket == "Taxable":
                        avail_cash = curr_cash - wd_cash
                        pull = min(shortfall, max(0.0, avail_cash))
                        if pull > 0:
                            wd_cash += pull
                            shortfall -= pull
                            pulled = True
                        if shortfall > 0:
                            avail_brok = curr_brokerage - wd_brokerage
                            pull = min(shortfall, max(0.0, avail_brok))
                            if pull > 0:
                                wd_brokerage += pull
                                shortfall -= pull
                                pulled = True

                    elif bucket == "Tax-Free":
                        if conversion_this_year == 0:
                            avail = curr_roth - wd_roth
                            pull = min(shortfall, max(0.0, avail))
                            if pull > 0:
                                wd_roth += pull
                                shortfall -= pull
                                pulled = True
                        if shortfall > 0:
                            avail = curr_life - wd_life
                            pull = min(shortfall, max(0.0, avail))
                            if pull > 0:
                                wd_life += pull
                                shortfall -= pull
                                pulled = True

                    elif bucket == "Pre-Tax":
                        avail = curr_pre_filer + curr_pre_spouse - wd_pretax - qcd_reserve
                        pull = min(shortfall, max(0.0, avail))
                        if pull > 0:
                            wd_pretax += pull
                            shortfall -= pull
                            pulled = True

                    elif bucket == "Tax-Deferred":
                        avail = curr_ann - wd_annuity
                        pull = min(shortfall, max(0.0, avail))
                        if pull > 0:
                            total_gains = max(0.0, curr_ann - curr_ann_basis)
                            remaining_gains = max(0.0, total_gains - ann_gains_withdrawn)
                            new_gains = min(pull, remaining_gains)
                            wd_annuity += pull
                            ann_gains_withdrawn += new_gains
                            shortfall -= pull
                            pulled = True

                if not pulled:
                    break

            # Final tax calc with settled withdrawals
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            cap_gains_realized = wd_brokerage * dyn_gain_pct
            final_inp = dict(trial_inp)
            final_inp["taxable_ira"] = wd_pretax + conversion_this_year
            final_inp["cap_gain_loss"] = yr_cap_gain + cap_gains_realized
            final_inp["other_income"] = ann_gains_withdrawn + yr_extra_taxable + p_other_income
            final_inp["cashflow_taxfree"] = yr_extra_taxfree
            final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
            yr_tax = final_res["total_tax"]
            yr_agi = final_res["agi"]
            yr_medicare = final_res["medicare_premiums"]

        # Accelerated pre-tax bracket fill: pull extra pre-tax beyond spending needs
        _accel_pt_extra = 0.0
        if extra_pretax_bracket is not None:
            _avail_pt_accel = max(0.0, curr_pre_filer + curr_pre_spouse - wd_pretax - qcd_reserve)
            if _avail_pt_accel > 0:
                # Build base non-SS income from already-determined components
                _base_non_ss = (pen_now + taxable_rmd + wd_pretax + conversion_this_year +
                                yr_interest + yr_ordinary_div + yr_cap_gain +
                                cap_gains_realized + ann_gains_withdrawn +
                                yr_extra_taxable + p_other_income + p_wages)
                _pref_amt = max(0.0, yr_cap_gain + cap_gains_realized) + max(0.0, yr_qualified_div)
                if extra_pretax_bracket == "irmaa":
                    _fill_amt = compute_irmaa_safe_amount(
                        0, _base_non_ss, ss_now, _yr_filing_status, inf_factor=bracket_inf)
                else:
                    _fill_amt = compute_bracket_fill_amount(
                        extra_pretax_bracket, _base_non_ss, ss_now, _yr_filing_status,
                        _yr_filer_65, _yr_spouse_65, p_retirement_deduction,
                        preferential_amount=_pref_amt, inf_factor=bracket_inf, tax_year=yr)
                _accel_pt_extra = min(max(0.0, _fill_amt), _avail_pt_accel)
                if _accel_pt_extra > 1.0:
                    wd_pretax += _accel_pt_extra
                    # Recompute taxes with the additional pre-tax withdrawal
                    dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
                    cap_gains_realized = wd_brokerage * dyn_gain_pct
                    _accel_inp = {
                        "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                        "rmd_amount": taxable_rmd,
                        "taxable_ira": wd_pretax + conversion_this_year,
                        "total_ordinary_dividends": yr_ordinary_div,
                        "qualified_dividends": yr_qualified_div,
                        "tax_exempt_interest": p_tax_exempt_interest,
                        "interest_taxable": yr_interest,
                        "cap_gain_loss": yr_cap_gain + cap_gains_realized,
                        "other_income": ann_gains_withdrawn + yr_extra_taxable + p_other_income,
                        "ordinary_tax_only": 0.0,
                        "adjustments": p_adjustments,
                        "reinvest_dividends": p_reinvest_div,
                        "reinvest_cap_gains": p_reinvest_cg,
                        "filing_status": _yr_filing_status,
                        "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                        "dependents": p_dependents,
                        "retirement_deduction": p_retirement_deduction * inf_factor,
                        "out_of_state_gain": p_out_of_state_gain,
                        "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                        "mortgage_payment": mtg_payment,
                        "property_tax": p_property_tax * inf_factor,
                        "medical_expenses": p_medical_expenses * inf_factor,
                        "charitable": yr_charitable_deduction,
                        "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
                    }
                    _accel_res = compute_case_cached(_serialize_inputs_for_cache(_accel_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
                    yr_tax = _accel_res["total_tax"]
                    yr_agi = _accel_res["agi"]
                    yr_medicare = _accel_res["medicare_premiums"]
                else:
                    _accel_pt_extra = 0.0

        # 0% capital gains harvesting: sell-and-rebuy to step up basis at zero tax cost
        _harvest_gains = 0.0
        if harvest_gains_bracket is not None:
            _remaining_brok = curr_brokerage - wd_brokerage
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            _unrealized_gains = max(0.0, _remaining_brok * dyn_gain_pct)
            if dyn_gain_pct > 0.01 and _unrealized_gains > 1.0:
                _harv_base_non_ss = (pen_now + taxable_rmd + wd_pretax + conversion_this_year +
                                     yr_interest + yr_ordinary_div + yr_cap_gain +
                                     cap_gains_realized + ann_gains_withdrawn +
                                     yr_extra_taxable + p_other_income + p_wages)
                _harv_existing_pref = max(0.0, yr_cap_gain + cap_gains_realized) + max(0.0, yr_qualified_div)
                _harv_room = compute_gains_harvest_amount(
                    _harv_base_non_ss, _harv_existing_pref, ss_now, _yr_filing_status,
                    _yr_filer_65, _yr_spouse_65, p_retirement_deduction,
                    inf_factor=bracket_inf, tax_year=yr)
                _harvest_gains = min(max(0.0, _harv_room), _unrealized_gains)
                if _harvest_gains > 1.0:
                    cap_gains_realized += _harvest_gains
                    brokerage_basis += _harvest_gains
                    # Recompute taxes with harvest gains on return (net cash = 0)
                    _harv_inp = {
                        "wages": p_wages, "gross_ss": ss_now, "taxable_pensions": pen_now,
                        "rmd_amount": taxable_rmd,
                        "taxable_ira": wd_pretax + conversion_this_year,
                        "total_ordinary_dividends": yr_ordinary_div,
                        "qualified_dividends": yr_qualified_div,
                        "tax_exempt_interest": p_tax_exempt_interest,
                        "interest_taxable": yr_interest,
                        "cap_gain_loss": yr_cap_gain + cap_gains_realized,
                        "other_income": ann_gains_withdrawn + yr_extra_taxable + p_other_income,
                        "ordinary_tax_only": 0.0,
                        "adjustments": p_adjustments,
                        "reinvest_dividends": p_reinvest_div,
                        "reinvest_cap_gains": p_reinvest_cg,
                        "filing_status": _yr_filing_status,
                        "filer_65_plus": _yr_filer_65, "spouse_65_plus": _yr_spouse_65,
                        "dependents": p_dependents,
                        "retirement_deduction": p_retirement_deduction * inf_factor,
                        "out_of_state_gain": p_out_of_state_gain,
                        "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                        "mortgage_payment": mtg_payment,
                        "property_tax": p_property_tax * inf_factor,
                        "medical_expenses": p_medical_expenses * inf_factor,
                        "charitable": yr_charitable_deduction,
                        "cashflow_taxfree": yr_extra_taxfree, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0, "tax_year": yr,
                    }
                    _harv_res = compute_case_cached(_serialize_inputs_for_cache(_harv_inp), bracket_inf, medicare_inflation_factor=medicare_inf)
                    yr_tax = _harv_res["total_tax"]
                    yr_agi = _harv_res["agi"]
                    yr_medicare = _harv_res["medicare_premiums"]
                else:
                    _harvest_gains = 0.0

        # Apply withdrawals to balances
        # Reduce brokerage basis proportionally (from Pre-Ret)
        if wd_brokerage > 0 and curr_brokerage > 0:
            basis_reduction = brokerage_basis * (wd_brokerage / curr_brokerage)
            brokerage_basis = max(0.0, brokerage_basis - basis_reduction)
        curr_cash -= wd_cash
        curr_brokerage -= wd_brokerage

        if wd_pretax > 0:
            avail = curr_pre_filer + curr_pre_spouse
            if avail > 0:
                filer_share = curr_pre_filer / avail
                curr_pre_filer -= wd_pretax * filer_share
                curr_pre_spouse -= wd_pretax * (1 - filer_share)

        curr_roth -= wd_roth
        curr_life -= wd_life
        curr_ann -= wd_annuity

        if wd_annuity > 0:
            basis_withdrawn = max(0.0, wd_annuity - ann_gains_withdrawn)
            curr_ann_basis = max(0.0, curr_ann_basis - basis_withdrawn)

        total_taxes_paid += yr_tax + yr_medicare

        # Surplus: income exceeds spending + taxes -> reinvest to cash (with basis tracking)
        cash_available_final = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
        cash_needed_final = total_spend_need + yr_tax + yr_medicare
        yr_surplus = max(0.0, cash_available_final - cash_needed_final)
        if yr_surplus > 0 and surplus_dest != "none":
            if surplus_dest == "cash":
                curr_cash += yr_surplus
            else:
                curr_brokerage += yr_surplus
                brokerage_basis += yr_surplus

        # Reinvested income increases cost basis (already taxed as income)
        if reinvested_base > 0:
            brokerage_basis += reinvested_base

        # Track mortgage paydown
        if curr_mtg_bal > 0 and mtg_payment > 0:
            _, curr_mtg_bal = calc_mortgage_interest_for_year(curr_mtg_bal, mtg_rate, mtg_payment)

        # Growth with negative balance protection
        _cash_before_growth = max(0.0, curr_cash)
        curr_cash = _cash_before_growth * (1 + r_cash)
        if not p_reinvest_int:
            curr_cash -= _cash_before_growth * r_cash  # interest already counted as income
        curr_ef = max(0.0, curr_ef) * (1 + r_cash)
        curr_brokerage = max(0.0, curr_brokerage) * (1 + r_taxable - _div_drag)
        curr_pre_filer = max(0.0, curr_pre_filer) * (1 + r_pretax)
        curr_pre_spouse = max(0.0, curr_pre_spouse) * (1 + r_pretax)
        curr_roth = max(0.0, curr_roth) * (1 + r_roth)
        curr_ann = max(0.0, curr_ann) * (1 + r_annuity)
        curr_life = max(0.0, curr_life) * (1 + r_life)

        # Home appreciation
        curr_home_val *= (1 + home_appr)
        home_equity = max(0.0, curr_home_val - curr_mtg_bal)

        # Estate calculation — present-value after-tax (what heirs receive today)
        total_wealth_yr = curr_cash + curr_ef + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
        _pretax_yr = curr_pre_filer + curr_pre_spouse
        _heir_pretax_net = _pretax_yr * (1.0 - heir_tax_rate)
        _heir_roth_net = curr_roth  # tax-free, no haircut
        _heir_ann_net = curr_ann_basis + max(0.0, curr_ann - curr_ann_basis) * (1.0 - heir_tax_rate)
        at_wealth_yr = curr_cash + curr_ef + curr_brokerage + curr_life + _heir_pretax_net + _heir_roth_net + _heir_ann_net
        gross_estate_yr = total_wealth_yr + home_equity
        # Estate tax: unlimited marital deduction → $0 while both spouses alive.
        # Only applies after first death (survivor's estate) or for single filers.
        _estate_tax_yr = 0.0
        _both_alive = _estate_is_joint and (first_death_idx is None or i < first_death_idx)
        if _estate_tax_enabled and not _both_alive:
            _estate_tax_yr = compute_estate_tax(
                gross_estate_yr, _estate_fed_exemption, _estate_exemption_growth, i,
                _estate_is_joint, _estate_use_portability,
                _estate_state_rate, _estate_state_exemption)
        net_estate_yr = at_wealth_yr + home_equity - _estate_tax_yr

        # Total income for display
        total_income_disp = ss_now + pen_now + spendable_inv + rmd_total + yr_extra_income + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity

        _status_label = ""
        if first_death_idx is not None:
            if i == first_death_idx:
                _status_label = "1st Death"
            elif _survivor_mode:
                _status_label = "Survivor"
        row = {
            "Year": yr, "Age": age_f, "Sp Age": age_s if age_s else "",
        }
        if first_death_idx is not None:
            row["Status"] = _status_label
        row.update({
            "Spending": round(total_spend_need, 0),
            "Addl Expense": round(yr_addl_expense, 0),
            "Extra Income": round(yr_extra_income, 0),
            "SS": round(ss_now, 0), "Pension": round(pen_now, 0),
            "Fixed Inc": round(ss_now + pen_now, 0),
            "Inv Inc": round(spendable_inv, 0),
            "IRA Dist": round(rmd_total + qcd_beyond_rmd, 0),
            "QCD": round(yr_qcd, 0),
            "Conversion": round(conversion_this_year, 0),
            "W/D Cash": round(wd_cash, 0), "W/D Taxable": round(wd_brokerage, 0),
            "W/D Pre-Tax": round(wd_pretax + qcd_beyond_rmd, 0),
            "W/D Roth": round(wd_roth, 0), "W/D Life": round(wd_life, 0),
            "W/D Tax-Free": round(wd_roth + wd_life, 0),
            "W/D Annuity": round(wd_annuity, 0),
            "Accel PT": round(_accel_pt_extra, 0),
            "Harvest Gains": round(_harvest_gains, 0),
            "Cap Gains": round(cap_gains_realized, 0),
            "Total Income": round(total_income_disp, 0),
            "AGI": round(yr_agi, 0),
            "Taxes": round(yr_tax, 0), "Medicare": round(yr_medicare, 0),
            "Surplus": round(yr_surplus, 0),
            "Bal Cash": round(curr_cash, 0), "Bal EF": round(curr_ef, 0), "Bal Taxable": round(curr_brokerage, 0),
            "Bal Pre-Tax": round(curr_pre_filer + curr_pre_spouse, 0),
            "Bal Roth": round(curr_roth, 0), "Bal Annuity": round(curr_ann, 0),
            "Bal Life": round(curr_life, 0),
            "Bal Tax-Free": round(curr_roth + curr_life, 0),
            "Portfolio": round(total_wealth_yr, 0),
            "Total Wealth": round(total_wealth_yr, 0),
        })
        if curr_home_val > 0 or home_val > 0:
            row["Home Value"] = round(curr_home_val, 0)
            row["Home Equity"] = round(home_equity, 0)
        row["Gross Estate"] = round(gross_estate_yr, 0)
        if _estate_tax_enabled:
            row["Estate Tax"] = round(_estate_tax_yr, 0)
        row["Estate (Net)"] = round(net_estate_yr, 0)
        row["_net_draw"] = (wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity) - yr_surplus
        year_details.append(row)

    total_wealth = curr_cash + curr_ef + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
    # Present-value after-tax estate (what heirs receive today)
    _final_pretax = curr_pre_filer + curr_pre_spouse
    _final_heir_pretax = _final_pretax * (1.0 - heir_tax_rate)
    _final_heir_roth = curr_roth  # tax-free
    _final_heir_ann = curr_ann_basis + max(0.0, curr_ann - curr_ann_basis) * (1.0 - heir_tax_rate)
    after_tax_estate = curr_cash + curr_ef + curr_brokerage + curr_life + _final_heir_pretax + _final_heir_roth + _final_heir_ann
    home_equity_final = max(0.0, curr_home_val - curr_mtg_bal)
    _gross_final = total_wealth + home_equity_final
    _final_estate_tax = 0.0
    _both_alive_final = _estate_is_joint and (first_death_idx is None or years < first_death_idx)
    if _estate_tax_enabled and not _both_alive_final:
        _final_estate_tax = compute_estate_tax(
            _gross_final, _estate_fed_exemption, _estate_exemption_growth, years,
            _estate_is_joint, _estate_use_portability,
            _estate_state_rate, _estate_state_exemption)
    _net_after_estate_tax = after_tax_estate + home_equity_final - _final_estate_tax

    return {
        "after_tax_estate": _net_after_estate_tax,
        "total_wealth": total_wealth,
        "gross_estate": _gross_final,
        "estate_tax": _final_estate_tax,
        "total_taxes": total_taxes_paid, "total_converted": total_converted,
        "final_cash": curr_cash + curr_ef, "final_brokerage": curr_brokerage,
        "final_pretax": curr_pre_filer + curr_pre_spouse,
        "final_roth": curr_roth, "final_annuity": curr_ann, "final_life": curr_life,
        "year_details": year_details,
    }


# ═══════════════════════════════════════════════════════════════════════
# 14. INHERITED IRA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def calc_inherited_ira_distribution(balance, rule, years_remaining, beneficiary_age,
                                     owner_was_taking_rmds=False, additional_distribution=0.0):
    """Calculate annual inherited IRA distribution.
    Returns dict: {"total_distribution": float, "minimum_rmd": float}

    Three scenarios:
    - 10-year flexible (owner NOT taking RMDs): no annual minimum, user chooses amount
    - 10-year with RMDs (owner WAS taking RMDs): annual minimum via Single Life Table + must empty by year 10
    - Lifetime (eligible designated beneficiary): annual RMD via Single Life Table

    owner_was_taking_rmds: only relevant for 10_year rule
    additional_distribution: extra amount above minimum (or planned amount for flexible)
    """
    if balance <= 0:
        return {"total_distribution": 0.0, "minimum_rmd": 0.0}

    # Final year of 10-year window: force full balance
    if rule == "10_year" and years_remaining <= 1:
        return {"total_distribution": balance, "minimum_rmd": balance}

    if rule == "10_year":
        if owner_was_taking_rmds:
            # Must take annual RMDs based on Single Life Table AND empty by year 10
            divisor = SINGLE_LIFE_TABLE.get(beneficiary_age, 1.0)
            if divisor <= 0:
                divisor = 1.0
            minimum_rmd = balance / divisor
            dist = minimum_rmd + additional_distribution
        else:
            # Flexible: no annual minimum, user chooses
            minimum_rmd = 0.0
            if additional_distribution > 0:
                dist = additional_distribution
            else:
                dist = balance / years_remaining
    else:  # lifetime
        divisor = SINGLE_LIFE_TABLE.get(beneficiary_age, 1.0)
        if divisor <= 0:
            divisor = 1.0
        minimum_rmd = balance / divisor
        dist = minimum_rmd + additional_distribution

    # Cap at balance
    dist = min(dist, balance)
    minimum_rmd = min(minimum_rmd, balance)

    return {"total_distribution": dist, "minimum_rmd": minimum_rmd}

def calc_inherited_ira_recommendation(balance, years_remaining, beneficiary_age,
                                       owner_was_taking_rmds, other_ordinary_income,
                                       filing_status, state_rate=0.05, growth_rate=0.06, inflation=0.03,
                                       rule="10_year", horizon_years=None):
    """Recommend an annual distribution that fills the current tax bracket.

    Strategy:
    1. Calculate bracket space (top of current bracket - current taxable income).
    2. Recommend that amount each year so all distributions stay in the current bracket.
    3. If that's not enough to deplete by the deadline (10-year), increase just enough
       so the minimum number of years spill into the next bracket.
    4. For lifetime: recommend additional above min RMD to fill the current bracket,
       preventing balance growth that would push future RMDs into higher brackets.
    """
    if balance <= 0:
        return None

    is_lifetime = (rule == "lifetime")

    # Determine simulation horizon
    if is_lifetime:
        if horizon_years and horizon_years > 0:
            sim_years = horizon_years
        else:
            divisor = SINGLE_LIFE_TABLE.get(beneficiary_age, 20.0)
            sim_years = max(5, int(math.ceil(divisor)))
    else:
        if years_remaining <= 0:
            return None
        sim_years = years_remaining

    # Current bracket context
    std_ded = get_std_deduction(filing_status, False, False, 1.0)
    taxable_without = max(0, other_ordinary_income - std_ded)
    marginal_without = get_marginal_fed_rate(taxable_without, filing_status, 1.0)
    brackets = get_fed_brackets(filing_status, 1.0)

    # Find current bracket index and bracket space
    current_bracket_idx = 0
    for i, (low_b, high_b, rate) in enumerate(brackets):
        if taxable_without < high_b:
            current_bracket_idx = i
            break
    bracket_space_yr1 = brackets[current_bracket_idx][1] - taxable_without
    if brackets[current_bracket_idx][1] == float("inf"):
        bracket_space_yr1 = balance  # already in the top bracket

    def _simulate_bracket_fill(target_bracket_idx, extra_above_bracket=0.0):
        """Simulate taking distributions that fill up to the given bracket boundary.

        For 10-year: distribution = bracket_fill (or min_rmd if larger); last year takes all.
        For lifetime: additional above min RMD = bracket_fill - min_rmd (floored at 0).
        extra_above_bracket: flat amount added above the bracket top (for when bracket
        fill alone isn't enough to deplete).
        """
        bal = balance
        schedule = []
        total_tax_cost = 0.0
        years_in_higher_bracket = 0
        for yr in range(sim_years):
            if bal <= 0:
                break
            age = beneficiary_age + yr
            inf_f = (1 + inflation) ** yr
            yr_other = other_ordinary_income * inf_f
            yr_std_ded = get_std_deduction(filing_status, False, False, inf_f)
            yr_taxable_base = max(0, yr_other - yr_std_ded)

            # Bracket fill for this year (inflation-adjusted bracket boundary)
            yr_bracket_top = brackets[target_bracket_idx][1] * inf_f if brackets[target_bracket_idx][1] != float("inf") else float("inf")
            yr_bracket_fill = max(0, yr_bracket_top - yr_taxable_base) + extra_above_bracket

            # Minimum RMD for this year
            if is_lifetime or owner_was_taking_rmds:
                div = SINGLE_LIFE_TABLE.get(age, 1.0)
                if div <= 0:
                    div = 1.0
                min_rmd = bal / div
            else:
                min_rmd = 0.0

            if is_lifetime:
                # Additional above min to fill the bracket
                additional = max(0, yr_bracket_fill - min_rmd)
                dist = min_rmd + additional
            else:
                dist = max(yr_bracket_fill, min_rmd)

            # 10-year final year: take everything remaining
            if not is_lifetime and yr == sim_years - 1:
                dist = bal

            dist = min(dist, bal)

            taxable_with = max(0, yr_other + dist - yr_std_ded)
            yr_fed_tax = calc_federal_tax(taxable_with, filing_status, inf_f)
            base_fed_tax = calc_federal_tax(yr_taxable_base, filing_status, inf_f)
            incremental_tax = yr_fed_tax - base_fed_tax + dist * state_rate
            yr_marginal = get_marginal_fed_rate(taxable_with, filing_status, inf_f)

            # Track if this year exceeds the target bracket
            yr_target_rate = brackets[target_bracket_idx][2]
            if yr_marginal > yr_target_rate:
                years_in_higher_bracket += 1

            schedule.append({
                "year": yr + 1, "age": age,
                "balance_start": round(bal, 0), "min_rmd": round(min_rmd, 0),
                "distribution": round(dist, 0), "incremental_tax": round(incremental_tax, 0),
                "marginal_rate": yr_marginal,
            })
            total_tax_cost += incremental_tax
            bal -= dist
            bal = max(0, bal) * (1 + growth_rate)

        return bal, schedule, total_tax_cost, years_in_higher_bracket

    if is_lifetime:
        # --- Lifetime RMD path ---
        # Simulate min-only baseline
        _, schedule_min, tax_min_only, _ = _simulate_bracket_fill(current_bracket_idx)
        # We need a true min-only simulation:
        bal_min = balance
        schedule_min = []
        tax_min_only = 0.0
        for yr in range(sim_years):
            if bal_min <= 0:
                break
            age = beneficiary_age + yr
            inf_f = (1 + inflation) ** yr
            yr_other = other_ordinary_income * inf_f
            yr_std_ded = get_std_deduction(filing_status, False, False, inf_f)
            yr_taxable_base = max(0, yr_other - yr_std_ded)
            div = SINGLE_LIFE_TABLE.get(age, 1.0)
            if div <= 0: div = 1.0
            min_rmd = bal_min / div
            dist = min(min_rmd, bal_min)
            taxable_with = max(0, yr_other + dist - yr_std_ded)
            yr_fed_tax = calc_federal_tax(taxable_with, filing_status, inf_f)
            base_fed_tax = calc_federal_tax(yr_taxable_base, filing_status, inf_f)
            inc_tax = yr_fed_tax - base_fed_tax + dist * state_rate
            yr_marginal = get_marginal_fed_rate(taxable_with, filing_status, inf_f)
            schedule_min.append({
                "year": yr + 1, "age": age,
                "balance_start": round(bal_min, 0), "min_rmd": round(min_rmd, 0),
                "distribution": round(dist, 0), "incremental_tax": round(inc_tax, 0),
                "marginal_rate": yr_marginal,
            })
            tax_min_only += inc_tax
            bal_min -= dist
            bal_min = max(0, bal_min) * (1 + growth_rate)

        peak_row = max(schedule_min, key=lambda r: r["distribution"]) if schedule_min else None
        peak_dist = peak_row["distribution"] if peak_row else 0
        peak_marginal = peak_row["marginal_rate"] if peak_row else marginal_without

        # Now simulate bracket-fill strategy: fill current bracket each year
        _, schedule_rec, tax_recommended, _ = _simulate_bracket_fill(current_bracket_idx)

        # The recommended additional (year 1) = bracket_fill - min_rmd
        yr1_min = schedule_min[0]["min_rmd"] if schedule_min else 0
        recommended_additional = max(0, bracket_space_yr1 - yr1_min)
        recommended_total = yr1_min + recommended_additional

        taxable_with = max(0, other_ordinary_income + recommended_total - std_ded)
        marginal_with = get_marginal_fed_rate(taxable_with, filing_status, 1.0)
        target_bracket_rate = brackets[current_bracket_idx][2]
        tax_savings = tax_min_only - tax_recommended

        # Build description
        warning = None
        if recommended_additional < 1:
            # Min RMD already fills or exceeds current bracket
            if peak_marginal > marginal_without:
                warning = (f"Min-only RMDs grow to {money(peak_dist)} (age {peak_row['age']}), "
                           f"hitting the {peak_marginal:.0%} bracket. The minimum RMD already fills "
                           f"your current bracket \u2014 consider a Roth conversion or other strategies.")
        elif peak_marginal > marginal_without and tax_savings > 0:
            warning = (f"Without extra distributions, min-only RMDs grow to {money(peak_dist)} "
                       f"(age {peak_row['age']}), hitting the {peak_marginal:.0%} bracket. "
                       f"Taking {money(recommended_additional)} above the minimum fills your "
                       f"{target_bracket_rate:.0%} bracket and saves ~{money(tax_savings)} in total tax.")

        return {
            "recommended_annual": round(recommended_total, 0),
            "recommended_additional": round(recommended_additional, 0),
            "target_bracket_top": round(brackets[current_bracket_idx][1], 0),
            "target_bracket_rate": target_bracket_rate,
            "marginal_without": marginal_without,
            "marginal_with": marginal_with,
            "total_tax_recommended": round(tax_recommended, 0),
            "total_tax_min_only": round(tax_min_only, 0),
            "tax_savings": round(tax_savings, 0),
            "peak_min_only_dist": round(peak_dist, 0),
            "peak_min_only_age": peak_row["age"] if peak_row else None,
            "peak_min_only_rate": peak_marginal,
            "total_tax_lump": None,
            "total_tax_even": None,
            "schedule": schedule_rec,
            "schedule_min_only": schedule_min,
            "warning": warning,
            "is_lifetime": True,
        }

    # --- 10-year rule path ---

    def _simulate_flat(flat_annual):
        """Simulate a flat annual distribution (respecting min RMDs); last year takes all."""
        bal = balance
        schedule = []
        total_tax_cost = 0.0
        for yr in range(sim_years):
            if bal <= 0:
                break
            age = beneficiary_age + yr
            inf_f = (1 + inflation) ** yr
            yr_other = other_ordinary_income * inf_f
            yr_std_ded = get_std_deduction(filing_status, False, False, inf_f)
            yr_taxable_base = max(0, yr_other - yr_std_ded)

            if owner_was_taking_rmds:
                div = SINGLE_LIFE_TABLE.get(age, 1.0)
                if div <= 0: div = 1.0
                min_rmd = bal / div
            else:
                min_rmd = 0.0

            if yr == sim_years - 1:
                dist = bal  # final year: take everything
            else:
                dist = max(flat_annual, min_rmd)

            dist = min(dist, bal)
            taxable_with = max(0, yr_other + dist - yr_std_ded)
            yr_fed_tax = calc_federal_tax(taxable_with, filing_status, inf_f)
            base_fed_tax = calc_federal_tax(yr_taxable_base, filing_status, inf_f)
            incremental_tax = yr_fed_tax - base_fed_tax + dist * state_rate
            yr_marginal = get_marginal_fed_rate(taxable_with, filing_status, inf_f)

            schedule.append({
                "year": yr + 1, "age": age,
                "balance_start": round(bal, 0), "min_rmd": round(min_rmd, 0),
                "distribution": round(dist, 0), "incremental_tax": round(incremental_tax, 0),
                "marginal_rate": yr_marginal,
            })
            total_tax_cost += incremental_tax
            bal -= dist
            bal = max(0, bal) * (1 + growth_rate)

        return bal, schedule, total_tax_cost

    # Binary search for the flat annual amount (PMT-equivalent) where the final year's
    # forced distribution ~ the annual amount -- i.e., evenly spread across all years.
    def _final_year_dist(flat_annual):
        """Return the final year's distribution for a given flat annual amount."""
        _, sched, _ = _simulate_flat(flat_annual)
        return sched[-1]["distribution"] if sched else 0

    lo, hi = 0.0, balance
    best_d = balance / sim_years  # fallback
    for _ in range(60):
        mid = (lo + hi) / 2
        fyd = _final_year_dist(mid)
        if fyd <= mid * 1.005:
            # Final year fits within the annual amount -- try taking less to stretch
            hi = mid
            best_d = mid
        else:
            # Final year is too big -- need to take more each year
            lo = mid

    # Check: does this amount stay within the current bracket?
    can_stay_in_bracket = (best_d <= bracket_space_yr1)

    if can_stay_in_bracket:
        _, schedule_rec, tax_recommended = _simulate_flat(best_d)
        recommended_annual = best_d
        recommended_bracket_idx = current_bracket_idx
        spill_desc = None
    else:
        # The PMT amount exceeds the current bracket. Use it anyway (it's the
        # minimum needed), but tell the user how much spills into the next bracket.
        _, schedule_rec, tax_recommended = _simulate_flat(best_d)
        recommended_annual = best_d

        # Determine which bracket this lands in
        test_taxable = max(0, other_ordinary_income + best_d - std_ded)
        recommended_bracket_idx = current_bracket_idx
        for i, (low_b, high_b, rate) in enumerate(brackets):
            if test_taxable <= high_b:
                recommended_bracket_idx = i
                break

        next_rate = brackets[min(current_bracket_idx + 1, len(brackets) - 1)][2]
        spill_above = best_d - bracket_space_yr1
        spill_desc = (f"Filling the {brackets[current_bracket_idx][2]:.0%} bracket alone won't deplete by year {sim_years}. "
                      f"Recommend {money(spill_above)}/yr into the {next_rate:.0%} bracket to fully distribute on time.")

    # Lump-sum-in-final-year comparison (take only min RMDs, then everything in last year)
    _, schedule_lump, tax_lump = _simulate_flat(0.0)
    final_lump_row = schedule_lump[-1] if schedule_lump else None
    final_yr_dist = final_lump_row["distribution"] if final_lump_row else 0
    final_yr_marginal = final_lump_row["marginal_rate"] if final_lump_row else marginal_without

    taxable_with_rec = max(0, other_ordinary_income + recommended_annual - std_ded)
    marginal_with = get_marginal_fed_rate(taxable_with_rec, filing_status, 1.0)
    target_bracket_rate = brackets[min(recommended_bracket_idx, len(brackets) - 1)][2]

    # Warning
    warning = None
    if final_yr_marginal > marginal_with and final_yr_dist > recommended_annual * 1.5:
        warning = (f"Taking less than recommended risks a {money(final_yr_dist)} lump sum in the final year "
                   f"at the {final_yr_marginal:.0%} bracket. Spreading {money(recommended_annual)}/yr "
                   f"keeps you in the {target_bracket_rate:.0%} bracket for all {sim_years} years.")
    elif marginal_with >= 0.32:
        warning = "Balance is large relative to available bracket space \u2014 distributions will push into higher brackets regardless of strategy."

    return {
        "recommended_annual": round(recommended_annual, 0),
        "recommended_additional": None,
        "target_bracket_rate": target_bracket_rate,
        "marginal_without": marginal_without,
        "marginal_with": marginal_with,
        "total_tax_recommended": round(tax_recommended, 0),
        "total_tax_lump": round(tax_lump, 0),
        "total_tax_even": None,
        "final_yr_lump": round(final_yr_dist, 0),
        "final_yr_rate": final_yr_marginal,
        "spill_desc": spill_desc,
        "schedule": schedule_rec,
        "warning": warning,
        "is_lifetime": False,
    }


# ── 15. ACCUMULATION ENGINE'S RETIREMENT PROJECTION ──────────────────────────

def run_retirement_projection(balances, params, spending_order):
    bal_pt = balances["pretax"]
    bal_brokerage = balances.get("brokerage", balances["taxable"])
    bal_cash = balances.get("cash", 0.0)
    brokerage_basis = balances.get("brokerage_basis", bal_brokerage)
    bal_ro = balances["roth"]
    bal_hs = balances["hsa"]

    retire_age = params["retire_age"]
    life_exp = params.get("filer_life_expectancy", params.get("life_expectancy", 95))
    spouse_life_exp = params.get("spouse_life_expectancy")
    survivor_spending_pct = params.get("survivor_spending_pct", 100) / 100.0
    pension_survivor_pct = params.get("pension_survivor_pct", 100) / 100.0
    retire_year = params["retire_year"]
    inflation = params["inflation"]
    bracket_growth = params.get("bracket_growth", inflation)
    medicare_growth = params.get("medicare_growth", inflation)
    post_return = params["post_retire_return"]
    filing_status = params["filing_status"]
    state_rate = params["state_tax_rate"]
    _sc_dependents = params.get("dependents", 0)
    _sc_out_of_state = params.get("out_of_state_gain", 0.0)
    _sc_base_tax_year = params.get("base_tax_year", 2026)
    living_expenses_yr1 = params["expenses_at_retirement"]
    ss_filer_fra = params["ss_filer_fra"]
    ss_spouse_fra = params["ss_spouse_fra"]
    ss_filer_claim_age = params["ss_filer_claim_age"]
    ss_spouse_claim_age = params["ss_spouse_claim_age"]
    ssdi_filer_flag = params.get("ssdi_filer", False)
    ssdi_spouse_flag = params.get("ssdi_spouse", False)
    ss_filer_already = params.get("ss_filer_already", False)
    ss_filer_current_ret = params.get("ss_filer_current_benefit", 0)
    ss_spouse_already = params.get("ss_spouse_already", False)
    ss_spouse_current_ret = params.get("ss_spouse_current_benefit", 0)
    filer_dob_ret = params.get("filer_dob")
    spouse_dob_ret = params.get("spouse_dob")
    _ss_filer_bday_frac = ss_first_year_fraction(filer_dob_ret)
    _ss_spouse_bday_frac = ss_first_year_fraction(spouse_dob_ret)
    pen_filer = params.get("pension_filer_at_retire", 0.0)
    pen_filer_start = params.get("pension_filer_start_age", 65)
    pen_filer_cola = params.get("pension_filer_cola", 0.0)
    pen_spouse = params.get("pension_spouse_at_retire", 0.0)
    pen_spouse_start = params.get("pension_spouse_start_age", 65)
    pen_spouse_cola = params.get("pension_spouse_cola", 0.0)
    mortgage_pmt = params.get("mortgage_payment", 0.0)
    mortgage_yrs = params.get("mortgage_years_at_retire", 0)
    home_val = params.get("home_value_at_retire", 0.0)
    home_appr = params.get("home_appreciation", 0.0)
    inv_re_val = params.get("inv_re_value_at_retire", 0.0)
    inv_re_appr = params.get("inv_re_appr", 0.0)
    fut_expenses = params.get("future_expenses", [])
    div_yield = params.get("dividend_yield", 0.015)
    cash_int_rate = params.get("cash_interest_rate", 0.04)
    brok_int_yield = params.get("brok_interest_yield", 0.0)
    surplus_dest = params.get("surplus_destination", "brokerage")
    heir_bracket_option = params.get("heir_bracket_option", "same")  # "same", "lower", "higher"
    ret_life = params.get("life_cash_value", 0.0)
    ret_life_return = params.get("r_life", 0.04)
    ret_annuity = params.get("annuity_value", 0.0)
    ret_annuity_basis = params.get("annuity_basis", 0.0)
    ret_annuity_return = params.get("r_annuity", 0.04)
    ret_other_income = params.get("other_income", 0.0)
    ret_other_income_tax_free = params.get("other_income_tax_free", False)
    ret_other_income_inflation = params.get("other_income_inflation", False)
    ret_other_income_years = params.get("other_income_years", 0)
    ret_re_props = params.get("inv_re_properties", [])
    ret_inherited_iras = params.get("inherited_iras", [])
    ret_iira_bals = [ira["balance"] for ira in ret_inherited_iras]
    ret_iira_rules = [ira["rule"] for ira in ret_inherited_iras]
    ret_iira_yrs = [ira["years_remaining"] for ira in ret_inherited_iras]
    ret_iira_ages = [ira["owner_age"] for ira in ret_inherited_iras]
    ret_iira_rmd_required = [ira.get("owner_was_taking_rmds", False) for ira in ret_inherited_iras]
    ret_iira_additional = [ira.get("additional_distribution", 0.0) for ira in ret_inherited_iras]

    base_charitable = params.get("charitable", 0.0)
    base_qcd = params.get("qcd_annual", 0.0)

    return_sequence = params.get("return_sequence", None)

    # Roth conversion params
    conv_strategy = params.get("roth_conversion_strategy", "none")  # "none", numeric, or "fill_to_target"
    conv_target_agi = params.get("roth_conversion_target_agi", 0)
    conv_stop_age = params.get("roth_conversion_stop_age", 100)
    defer_first_rmd = params.get("defer_first_rmd", False)
    do_conversions = conv_strategy != "none" and conv_strategy != 0
    total_converted = 0.0
    deferred_rmd = 0.0

    # Estate tax params
    _ret_estate_enabled = params.get("estate_tax_enabled", False)
    _ret_estate_exemption = params.get("federal_estate_exemption", 15000000.0)
    _ret_estate_exemption_growth = params.get("exemption_inflation", 2.5)
    _ret_estate_portability = params.get("use_portability", True)
    _ret_estate_state_rate = params.get("state_estate_tax_rate", 0.0)
    _ret_estate_state_exemption = params.get("state_estate_exemption", 0.0)
    _ret_estate_is_joint = "joint" in filing_status.lower()

    total_taxes_paid = 0.0
    rows = []
    curr_home_val = home_val
    curr_inv_re_val = inv_re_val
    curr_life = ret_life
    curr_annuity = ret_annuity

    spouse_age_at_retire = params.get("spouse_age_at_retire")

    # Derive is_joint locally (was module-level in accum engine)
    is_joint = "joint" in filing_status.lower()

    # Compute projection length and first-death year for survivor logic
    retire_years_filer = life_exp - retire_age
    if spouse_life_exp and spouse_age_at_retire:
        retire_years_spouse = spouse_life_exp - spouse_age_at_retire
        total_retire_years = max(retire_years_filer, retire_years_spouse)
        filer_death_year_idx = retire_years_filer
        spouse_death_year_idx = retire_years_spouse
        first_death_idx = min(filer_death_year_idx, spouse_death_year_idx)
        filer_dies_first = filer_death_year_idx <= spouse_death_year_idx
    else:
        total_retire_years = retire_years_filer
        first_death_idx = None
        filer_dies_first = None

    for i in range(total_retire_years + 1):
        age = retire_age + i
        spouse_age_now = (spouse_age_at_retire + i) if spouse_age_at_retire else age
        year = retire_year + i
        inf_factor = (1 + inflation) ** i
        bracket_inf = (1 + bracket_growth) ** i
        medicare_inf = (1 + medicare_growth) ** i
        filer_65 = age >= 65
        spouse_65 = (spouse_age_now >= 65) if spouse_age_at_retire else False

        # Living expenses inflate; mortgage payment is fixed until paid off
        living_exp = living_expenses_yr1 * inf_factor
        mtg_pmt = mortgage_pmt if i < mortgage_yrs else 0.0

        # Future expense changes
        yr_future_exp = 0.0
        for fe in fut_expenses:
            if fe["start_age"] <= age < fe["end_age"]:
                yrs_from_start = age - fe["start_age"]
                amt = fe["amount"] * ((1 + inflation) ** yrs_from_start) if fe["inflates"] else fe["amount"]
                yr_future_exp += amt

        expenses = living_exp + mtg_pmt + yr_future_exp

        # Charitable giving as real spending
        yr_charitable = base_charitable * inf_factor
        expenses += yr_charitable

        # Survivor mode: reduce spending after first spouse death
        _survivor_mode = first_death_idx is not None and i >= first_death_idx
        # Filing status: year of death can still file joint; year AFTER switches to single
        if first_death_idx is not None and i > first_death_idx and is_joint:
            _yr_filing_status = "Single"
            _yr_filer_65 = (age >= 65) if not filer_dies_first else (spouse_age_now >= 65 if spouse_age_at_retire else False)
            _yr_spouse_65 = False
        else:
            _yr_filing_status = filing_status
            _yr_filer_65 = filer_65
            _yr_spouse_65 = spouse_65
        if _survivor_mode:
            expenses *= survivor_spending_pct

        # SS: fra values are in retirement-year dollars; inf_factor continues COLA from there
        # SSDI: already receiving at 100% PIA, just apply inflation
        ss_filer_calc = 0.0
        if ss_filer_already and ss_filer_current_ret > 0:
            ss_filer_calc = ss_filer_current_ret * inf_factor
        elif ss_filer_fra > 0:
            if ssdi_filer_flag:
                ss_filer_calc = ss_filer_fra * inf_factor
            elif age >= ss_filer_claim_age:
                ss_filer_calc = ss_filer_fra * ss_claim_factor(ss_filer_claim_age) * inf_factor
                if age == ss_filer_claim_age:
                    ss_filer_calc *= _ss_filer_bday_frac

        ss_spouse_calc = 0.0
        if ss_spouse_already and ss_spouse_current_ret > 0:
            ss_spouse_calc = ss_spouse_current_ret * inf_factor
        elif ss_spouse_fra > 0:
            if ssdi_spouse_flag:
                ss_spouse_calc = ss_spouse_fra * inf_factor
            elif spouse_age_now >= ss_spouse_claim_age:
                ss_spouse_calc = ss_spouse_fra * ss_claim_factor(ss_spouse_claim_age) * inf_factor
                if spouse_age_now == ss_spouse_claim_age:
                    ss_spouse_calc *= _ss_spouse_bday_frac

        # Survivor SS: survivor gets the higher of the two benefits
        if _survivor_mode:
            if filer_dies_first:
                ss_filer_now = 0.0
                ss_spouse_now = max(ss_spouse_calc, ss_filer_calc)
            else:
                ss_spouse_now = 0.0
                ss_filer_now = max(ss_filer_calc, ss_spouse_calc)
        else:
            ss_filer_now = ss_filer_calc
            ss_spouse_now = ss_spouse_calc

        gross_ss = max(0, ss_filer_now) + max(0, ss_spouse_now)

        # Pensions: each has its own start age and COLA
        pen_filer_income = 0.0
        if pen_filer > 0 and age >= pen_filer_start:
            yrs_receiving_f = age - pen_filer_start
            pen_filer_income = pen_filer * ((1 + pen_filer_cola) ** yrs_receiving_f) if pen_filer_cola > 0 else pen_filer

        pen_spouse_income = 0.0
        if pen_spouse > 0 and spouse_age_now >= pen_spouse_start:
            yrs_receiving_s = spouse_age_now - pen_spouse_start
            pen_spouse_income = pen_spouse * ((1 + pen_spouse_cola) ** yrs_receiving_s) if pen_spouse_cola > 0 else pen_spouse

        # Survivor pension: apply survivor benefit percentage
        if _survivor_mode:
            if filer_dies_first:
                pen_filer_income *= pension_survivor_pct
            else:
                pen_spouse_income *= pension_survivor_pct

        pen_income = pen_filer_income + pen_spouse_income

        # Investment income from taxable accounts (generated before withdrawals)
        yr_dividends = bal_brokerage * div_yield if bal_brokerage > 0 else 0.0
        yr_cash_interest = bal_cash * cash_int_rate if bal_cash > 0 else 0.0
        yr_brok_interest = bal_brokerage * brok_int_yield if bal_brokerage > 0 else 0.0
        yr_interest = yr_cash_interest + yr_brok_interest
        # Dividends are qualified -> cap gains rate; all interest -> ordinary
        inv_cap_gains_income = yr_dividends
        inv_ordinary_income = yr_interest

        # Other income (alimony, disability, etc.) — ends after N years from start
        yr_other_income = 0.0
        if ret_other_income > 0:
            if ret_other_income_years == 0 or i < ret_other_income_years:
                yr_other_income = ret_other_income * inf_factor if ret_other_income_inflation else ret_other_income

        # Investment RE rental income (per-property with mortgage payoff boost)
        yr_rental_income = 0.0
        for _rep in ret_re_props:
            # Income grows from its retirement-year base value
            _r_inc = _rep["net_income"] * ((1 + _rep["income_growth"]) ** i)
            if _rep["mortgage_pmt"] > 0 and i >= _rep["mortgage_years"]:
                _r_inc += _rep["mortgage_pmt"]
            yr_rental_income += _r_inc
        yr_other_income += yr_rental_income

        fixed_cash = gross_ss + pen_income + yr_dividends + yr_interest + yr_other_income

        # Inherited IRA distributions (mandatory, ordinary income)
        yr_inherited_dist = 0.0
        yr_inherited_min_rmd = 0.0
        for idx in range(len(ret_inherited_iras)):
            if ret_iira_bals[idx] <= 0:
                continue
            dist_info = calc_inherited_ira_distribution(
                ret_iira_bals[idx], ret_iira_rules[idx], ret_iira_yrs[idx], ret_iira_ages[idx] + i,
                owner_was_taking_rmds=ret_iira_rmd_required[idx],
                additional_distribution=ret_iira_additional[idx])
            dist = dist_info["total_distribution"]
            yr_inherited_min_rmd += dist_info["minimum_rmd"]
            dist = min(dist, ret_iira_bals[idx])
            yr_inherited_dist += dist
            ret_iira_bals[idx] -= dist
            if ret_iira_rules[idx] == "10_year":
                ret_iira_yrs[idx] = max(0, ret_iira_yrs[idx] - 1)

        fixed_cash += yr_inherited_dist  # inherited distributions are cash received

        # RMD: forced minimum pre-tax withdrawal at age 73+
        rmd = calc_rmd(age, bal_pt)
        rmd_amount = min(rmd, bal_pt)

        # First-year RMD deferral: skip age 73 RMD, double up at age 74
        if defer_first_rmd and age == 73:
            deferred_rmd = rmd_amount
            rmd_amount = 0.0
        elif defer_first_rmd and age == 74:
            rmd_amount += deferred_rmd
            rmd_amount = min(rmd_amount, bal_pt)
            deferred_rmd = 0.0

        wd_pretax = rmd_amount  # start with at least the RMD
        wd_cash = 0.0
        wd_brokerage = 0.0
        wd_roth = 0.0
        wd_hsa = 0.0
        yr_cap_gains = 0.0  # capital gains from brokerage sales

        # QCD: direct IRA-to-charity transfer
        yr_qcd = 0.0
        qcd_beyond_rmd = 0.0
        if base_qcd > 0 and age >= 70:
            is_joint_yr = "joint" in _yr_filing_status.lower()
            qcd_cap = (210000.0 if is_joint_yr else 105000.0) * inf_factor
            yr_qcd_requested = min(base_qcd * inf_factor, qcd_cap)
            qcd_within_rmd = min(yr_qcd_requested, rmd_amount)
            qcd_beyond_rmd = max(0.0, yr_qcd_requested - rmd_amount)
            avail_pt = bal_pt - wd_pretax  # after RMD
            qcd_beyond_rmd = min(qcd_beyond_rmd, max(0.0, avail_pt))
            yr_qcd = qcd_within_rmd + qcd_beyond_rmd

        # QCD adjustments
        yr_qcd_cash_offset = min(yr_qcd, rmd_amount)  # QCD within RMD reduces spendable cash
        expenses -= yr_qcd  # QCD satisfies this portion of charitable spending directly from IRA

        # SC tax params for this year (retirement_income updated before each calc_year_taxes call)
        _yr_sc_params = {
            "dependents": _sc_dependents,
            "out_of_state_gain": _sc_out_of_state,
            "tax_year": _sc_base_tax_year + i,
        }

        # Roth conversion (after RMD, before waterfall)
        conversion_this_year = 0.0
        conv_tax_withheld = 0.0  # portion of conversion tax paid from the conversion itself
        conv_tax_total = 0.0  # total incremental tax from conversion
        if do_conversions and age < conv_stop_age:
            avail_pretax = max(0, bal_pt - wd_pretax - qcd_beyond_rmd)  # what's left after RMD + QCD
            if avail_pretax > 0:
                yr_taxable_other = yr_other_income if not ret_other_income_tax_free else 0.0
                if conv_strategy == "fill_to_target":
                    # Estimate current taxable income before conversion (QCD portion excluded)
                    base_income = (wd_pretax - yr_qcd_cash_offset) + pen_income + yr_inherited_dist + inv_ordinary_income + inv_cap_gains_income + yr_taxable_other
                    est_taxable_ss = gross_ss * 0.85  # conservative: assume 85% taxable
                    room = max(0, conv_target_agi * bracket_inf - base_income - est_taxable_ss)
                    conversion_this_year = min(room, avail_pretax)
                else:
                    conversion_this_year = min(float(conv_strategy), avail_pretax)

                # Tax on conversion: withhold from conversion if no other source to pay
                if conversion_this_year > 0:
                    _no_conv_pretax_inc = (wd_pretax - yr_qcd_cash_offset) + pen_income + yr_inherited_dist + yr_taxable_other
                    _yr_sc_params["retirement_income"] = wd_pretax + pen_income + yr_inherited_dist
                    _no_conv_tax = calc_year_taxes(gross_ss, _no_conv_pretax_inc,
                                                    inv_cap_gains_income, inv_ordinary_income,
                                                    _yr_filing_status, _yr_filer_65, _yr_spouse_65,
                                                    bracket_inf, state_rate, sc_params=_yr_sc_params)["total_tax"]
                    _yr_sc_params["retirement_income"] = wd_pretax + pen_income + yr_inherited_dist + conversion_this_year
                    _with_conv_tax = calc_year_taxes(gross_ss, _no_conv_pretax_inc + conversion_this_year,
                                                     inv_cap_gains_income, inv_ordinary_income,
                                                     _yr_filing_status, _yr_filer_65, _yr_spouse_65,
                                                     bracket_inf, state_rate, sc_params=_yr_sc_params)["total_tax"]
                    _conv_tax_cost = _with_conv_tax - _no_conv_tax
                    conv_tax_total = _conv_tax_cost
                    # Available external funding: surplus income + taxable accounts
                    _base_cash_needed = expenses + _no_conv_tax
                    _surplus = max(0, fixed_cash - _base_cash_needed)
                    _external_funding = _surplus + bal_brokerage + bal_cash
                    if _conv_tax_cost > _external_funding:
                        # Withhold unfunded tax from the conversion (reduces net to Roth)
                        conv_tax_withheld = min(_conv_tax_cost - max(0, _external_funding), conversion_this_year)

            if conversion_this_year > 0:
                bal_pt -= conversion_this_year
                bal_ro += conversion_this_year - conv_tax_withheld  # net to Roth after tax withholding
                total_converted += conversion_this_year

        bal_tx = bal_brokerage + bal_cash  # for withdrawal limit tracking

        yr_medicare = 0.0
        for iteration in range(20):
            # Compute cap gains: gains from brokerage sales + dividend income
            total_cap_gains = yr_cap_gains + inv_cap_gains_income
            yr_taxable_other = yr_other_income if not ret_other_income_tax_free else 0.0
            pretax_income = (wd_pretax - yr_qcd_cash_offset) + pen_income + yr_inherited_dist + yr_taxable_other + conversion_this_year
            _yr_sc_params["retirement_income"] = wd_pretax + pen_income + yr_inherited_dist + conversion_this_year
            tax_result = calc_year_taxes(gross_ss, pretax_income, total_cap_gains,
                                         inv_ordinary_income, _yr_filing_status,
                                         _yr_filer_65, _yr_spouse_65, bracket_inf, state_rate,
                                         sc_params=_yr_sc_params)
            taxes = tax_result["total_tax"]
            # Medicare premiums (based on AGI, applies when 65+)
            if _yr_filer_65 or _yr_spouse_65:
                yr_medicare, _ = estimate_medicare_premiums(tax_result["agi"], _yr_filing_status, bracket_inf, medicare_inf,
                                                            filer_65=_yr_filer_65, spouse_65=_yr_spouse_65)
            else:
                yr_medicare = 0.0
            cash_needed = expenses + taxes + yr_medicare - conv_tax_withheld  # withheld portion already paid from conversion
            wd_taxable = wd_cash + wd_brokerage
            cash_available = fixed_cash + (wd_pretax - yr_qcd_cash_offset) + wd_taxable + wd_roth + wd_hsa
            shortfall = cash_needed - cash_available

            if shortfall <= 1.0:
                break

            pulled = False
            for bucket in spending_order:
                if shortfall <= 0: break
                if bucket == "Pre-Tax":
                    avail = bal_pt - wd_pretax
                    pull = min(shortfall, avail)
                    if pull > 0:
                        wd_pretax += pull
                        shortfall -= pull
                        pulled = True
                elif bucket == "Taxable":
                    # Pull from cash first (no capital gains), then brokerage
                    avail_cash = bal_cash - wd_cash
                    if avail_cash > 0 and shortfall > 0:
                        pull_cash = min(shortfall, avail_cash)
                        wd_cash += pull_cash
                        shortfall -= pull_cash
                        pulled = True
                    if shortfall > 0:
                        avail_brok = bal_brokerage - wd_brokerage
                        if avail_brok > 0:
                            pull_brok = min(shortfall, avail_brok)
                            # Capital gains = pull * gain percentage
                            gain_pct = max(0.0, 1.0 - brokerage_basis / bal_brokerage) if bal_brokerage > 0 else 0.0
                            yr_cap_gains = (wd_brokerage + pull_brok) * gain_pct
                            wd_brokerage += pull_brok
                            shortfall -= pull_brok
                            pulled = True
                elif bucket == "Tax-Free":
                    avail_roth = bal_ro - wd_roth
                    pull = min(shortfall, avail_roth)
                    if pull > 0:
                        wd_roth += pull
                        shortfall -= pull
                        pulled = True
                    if shortfall > 0:
                        avail_hsa = bal_hs - wd_hsa
                        pull = min(shortfall, avail_hsa)
                        if pull > 0:
                            wd_hsa += pull
                            shortfall -= pull
                            pulled = True
            if not pulled:
                break

        # Final tax calc with settled withdrawals
        total_cap_gains = yr_cap_gains + inv_cap_gains_income
        yr_taxable_other = yr_other_income if not ret_other_income_tax_free else 0.0
        pretax_income = (wd_pretax - yr_qcd_cash_offset) + pen_income + yr_inherited_dist + yr_taxable_other + conversion_this_year
        _yr_sc_params["retirement_income"] = wd_pretax + pen_income + yr_inherited_dist + conversion_this_year
        tax_result = calc_year_taxes(gross_ss, pretax_income, total_cap_gains,
                                     inv_ordinary_income, _yr_filing_status,
                                     _yr_filer_65, _yr_spouse_65, bracket_inf, state_rate,
                                     sc_params=_yr_sc_params)
        taxes = tax_result["total_tax"]
        if _yr_filer_65 or _yr_spouse_65:
            yr_medicare, _ = estimate_medicare_premiums(tax_result["agi"], _yr_filing_status, bracket_inf, medicare_inf,
                                                        filer_65=_yr_filer_65, spouse_65=_yr_spouse_65)
        else:
            yr_medicare = 0.0
        wd_taxable = wd_cash + wd_brokerage

        # Surplus: income exceeds expenses + taxes + medicare -> reinvest
        cash_available_final = fixed_cash + (wd_pretax - yr_qcd_cash_offset) + wd_cash + wd_brokerage + wd_roth + wd_hsa
        cash_needed_final = expenses + taxes + yr_medicare - conv_tax_withheld
        yr_surplus = max(0.0, cash_available_final - cash_needed_final)

        # Update balances
        bal_pt -= wd_pretax
        bal_pt -= qcd_beyond_rmd  # additional IRA withdrawal for QCD beyond RMD
        # Reduce brokerage basis proportionally to withdrawal
        if wd_brokerage > 0 and bal_brokerage > 0:
            basis_reduction = brokerage_basis * (wd_brokerage / bal_brokerage)
            brokerage_basis = max(0.0, brokerage_basis - basis_reduction)
        bal_brokerage -= wd_brokerage
        bal_cash -= wd_cash
        bal_ro -= wd_roth
        bal_hs -= wd_hsa

        # Reinvest surplus (after-tax money -> 100% cost basis)
        if yr_surplus > 0 and surplus_dest != "none":
            if surplus_dest == "cash":
                bal_cash += yr_surplus
            else:
                bal_brokerage += yr_surplus
                brokerage_basis += yr_surplus

        total_taxes_paid += taxes + yr_medicare

        # Growth
        yr_return = return_sequence[i] if return_sequence else post_return
        bal_pt = max(0.0, bal_pt) * (1 + yr_return)
        bal_ro = max(0.0, bal_ro) * (1 + yr_return)
        # Brokerage: reduce by div + interest yield since those flow to spending income
        bal_brokerage = max(0.0, bal_brokerage) * (1 + yr_return - div_yield - brok_int_yield)
        bal_cash = max(0.0, bal_cash) * (1 + cash_int_rate)
        bal_hs = max(0.0, bal_hs) * (1 + yr_return)
        bal_tx = bal_brokerage + bal_cash
        # Grow remaining inherited IRA balances
        for idx in range(len(ret_inherited_iras)):
            if ret_iira_bals[idx] > 0:
                ret_iira_bals[idx] *= (1 + yr_return)

        # Appreciate home, investment RE, life insurance, annuity
        if i > 0:
            curr_home_val *= (1 + home_appr)
            if curr_inv_re_val > 0:
                curr_inv_re_val *= (1 + inv_re_appr)
            if curr_life > 0:
                curr_life *= (1 + ret_life_return)
            if curr_annuity > 0:
                curr_annuity *= (1 + ret_annuity_return)

        bal_inherited_total = sum(ret_iira_bals)
        total_bal = bal_pt + bal_ro + bal_tx + bal_hs + bal_inherited_total
        gross_estate = total_bal + curr_home_val + curr_inv_re_val + curr_life + curr_annuity

        # After-tax estate: what heirs actually receive
        my_marginal = tax_result["marginal_rate"]
        if heir_bracket_option == "lower":
            heir_fed = _heir_rate_from_offset(my_marginal, -1)
        elif heir_bracket_option == "higher":
            heir_fed = _heir_rate_from_offset(my_marginal, +1)
        else:
            heir_fed = my_marginal
        heir_total_rate = heir_fed + state_rate
        # Pre-tax & HSA & inherited IRA -> fully taxable as ordinary income to heirs
        # Roth -> tax-free; Brokerage -> stepped-up basis; Cash -> no tax; Home/RE -> stepped-up basis
        # Life insurance -> tax-free; Annuity gains -> taxed as ordinary income
        _ret_ann_gain = max(0, curr_annuity - ret_annuity_basis)
        _ret_ann_net = ret_annuity_basis + _ret_ann_gain * (1 - heir_total_rate)
        _heir_estate = (
            bal_pt * (1 - heir_total_rate) +
            bal_ro +
            bal_tx +  # stepped-up basis
            bal_hs * (1 - heir_total_rate) +
            bal_inherited_total * (1 - heir_total_rate) +
            curr_home_val + curr_inv_re_val +
            curr_life + _ret_ann_net
        )
        # Estate tax: unlimited marital deduction → $0 while both spouses alive.
        _ret_yr_estate_tax = 0.0
        _ret_both_alive = _ret_estate_is_joint and (first_death_idx is None or i < first_death_idx)
        if _ret_estate_enabled and not _ret_both_alive:
            _ret_yr_estate_tax = compute_estate_tax(
                gross_estate, _ret_estate_exemption, _ret_estate_exemption_growth, i,
                _ret_estate_is_joint, _ret_estate_portability,
                _ret_estate_state_rate, _ret_estate_state_exemption)
        after_tax_estate = _heir_estate - _ret_yr_estate_tax

        row = {
            "Year": year, "Age": age,
        }
        if spouse_age_at_retire is not None:
            row["Spouse Age"] = spouse_age_now
        if first_death_idx is not None:
            if i == first_death_idx:
                row["Status"] = "1st Death"
            elif _survivor_mode:
                row["Status"] = "Survivor"
            else:
                row["Status"] = ""
        row.update({
            "SS Income": round(gross_ss, 0), "Pension": round(pen_income, 0),
            "Other Income": round(yr_other_income, 0),
            "Dividends": round(yr_dividends, 0),
            "Interest": round(yr_interest, 0),
        })
        if balances["pretax"] > 0:
            row["IRA Dist"] = round(rmd_amount + qcd_beyond_rmd, 0)
            row["QCD"] = round(yr_qcd, 0)
            row["W/D Pre-Tax"] = round(max(0, wd_pretax - rmd_amount) + qcd_beyond_rmd, 0)
        row.update({
            "W/D Taxable": round(wd_taxable, 0),
            "W/D Roth": round(wd_roth, 0), "W/D HSA": round(wd_hsa, 0),
        })
        if any(ira["balance"] > 0 for ira in ret_inherited_iras):
            row["Inherited Dist"] = round(yr_inherited_dist, 0)
        row.update({
            "Realized CG": round(yr_cap_gains, 0),
            "Conv Gross": round(conversion_this_year, 0),
            "Conv Tax": round(conv_tax_total, 0),
            "Conv to Roth": round(conversion_this_year - conv_tax_withheld, 0),
            "Surplus Reinv": round(yr_surplus, 0),
            "Living Exp": round(living_exp, 0), "Mortgage": round(mtg_pmt, 0),
            "Total Exp": round(expenses, 0),
            "Fed Tax": round(tax_result["fed_tax"], 0),
            "State Tax": round(tax_result["state_tax"], 0),
            "Medicare": round(yr_medicare, 0),
            "Total Tax": round(taxes + yr_medicare, 0),
        })
        if balances["pretax"] > 0:
            row["Bal Pre-Tax"] = round(bal_pt, 0)
        row.update({
            "Bal Taxable": round(bal_tx, 0),
            "Bal Roth": round(bal_ro, 0), "Bal HSA": round(bal_hs, 0),
        })
        if any(ira["balance"] > 0 for ira in ret_inherited_iras):
            row["Bal Inherited"] = round(bal_inherited_total, 0)
        row["Portfolio"] = round(total_bal, 0)
        if curr_life > 0 or ret_life > 0:
            row["Life Ins"] = round(curr_life, 0)
        if curr_annuity > 0 or ret_annuity > 0:
            row["Annuity"] = round(curr_annuity, 0)
        row["Home Value"] = round(curr_home_val, 0)
        if curr_inv_re_val > 0 or inv_re_val > 0:
            row["Inv RE"] = round(curr_inv_re_val, 0)
        row["Gross Estate"] = round(gross_estate, 0)
        if _ret_estate_enabled:
            row["Estate Tax"] = round(_ret_yr_estate_tax, 0)
        row["Estate (Net)"] = round(after_tax_estate, 0)
        rows.append(row)

    bal_inherited_final = sum(ret_iira_bals)
    final_total = bal_pt + bal_ro + bal_tx + bal_hs + bal_inherited_final
    gross_estate_final = final_total + curr_home_val + curr_inv_re_val + curr_life + curr_annuity
    # Use last row's after-tax estate if available
    net_estate_final = rows[-1]["Estate (Net)"] if rows else gross_estate_final
    return {
        "rows": rows, "total_taxes": total_taxes_paid,
        "final_pretax": bal_pt, "final_roth": bal_ro, "final_taxable": bal_tx, "final_hsa": bal_hs,
        "final_inherited": bal_inherited_final,
        "final_total": final_total,
        "estate": net_estate_final,  # after-tax estate (used by SS optimizer)
        "gross_estate": gross_estate_final,
        "final_home_value": curr_home_val,
        "total_converted": total_converted,
    }


# ── 16. ACCUMULATION ENGINE ──────────────────────────────────────────────────

def run_accumulation(current_age, years_to_ret, start_balances, contributions, salary_growth_rate, pre_ret_return, income_info=None):
    """Run accumulation projection and return list of year rows."""
    bal_pretax = start_balances["pretax"]
    bal_roth = start_balances["roth"]
    bal_brokerage = start_balances.get("brokerage", 0)
    bal_cash = start_balances.get("cash", 0)
    bal_taxable = bal_brokerage + bal_cash  # combined for display
    bal_hsa = start_balances["hsa"]
    brokerage_basis = start_balances.get("brokerage_basis", bal_brokerage)
    base_income = income_info.get("total_income", 0) if income_info else 0
    base_other_income = income_info.get("other_income", 0) if income_info else 0
    other_income_tax_free = income_info.get("other_income_tax_free", False) if income_info else False
    other_income_inflation = income_info.get("other_income_inflation", False) if income_info else False
    other_income_years = income_info.get("other_income_years", 0) if income_info else 0
    other_income_recipient = income_info.get("other_income_recipient", "Household") if income_info else "Household"
    base_expenses = income_info.get("living_expenses", 0) if income_info else 0
    base_mortgage = income_info.get("mortgage_annual", 0) if income_info else 0
    mortgage_yrs_left = income_info.get("mortgage_years", 0) if income_info else 0
    # If mortgage exists but years not set, show for entire projection
    if base_mortgage > 0 and mortgage_yrs_left <= 0:
        mortgage_yrs_left = years_to_ret + 1
    filing = income_info.get("filing_status", "Single") if income_info else "Single"
    st_rate = income_info.get("state_rate", 0.05) if income_info else 0.05
    _accum_sc_dependents = income_info.get("dependents", 0) if income_info else 0
    _accum_sc_base_tax_year = income_info.get("base_tax_year", 2026) if income_info else 2026
    pretax_deductions = income_info.get("pretax_deductions", 0) if income_info else 0
    inf_rate = income_info.get("inflation", 0.03) if income_info else 0.03
    bracket_growth_rate = income_info.get("bracket_growth", inf_rate) if income_info else inf_rate
    medicare_growth_rate = income_info.get("medicare_growth", inf_rate) if income_info else inf_rate
    fut_expenses = income_info.get("future_expenses", []) if income_info else []
    # Per-account return rates (fall back to pre_ret_return for backward compat)
    r_pretax = income_info.get("r_pretax", pre_ret_return) if income_info else pre_ret_return
    r_roth = income_info.get("r_roth", pre_ret_return) if income_info else pre_ret_return
    r_taxable = income_info.get("r_taxable", pre_ret_return) if income_info else pre_ret_return
    r_hsa = income_info.get("r_hsa", pre_ret_return) if income_info else pre_ret_return
    div_yield = income_info.get("dividend_yield", 0.015) if income_info else 0.015
    ann_cg_pct = income_info.get("annual_cap_gain_pct", 0.0) if income_info else 0.0
    cash_int_rate = income_info.get("cash_interest_rate", 0.04) if income_info else 0.04
    reinvest_inv_income = income_info.get("reinvest_inv_income", True) if income_info else True

    # Derive effective yields from user's entered actual income values
    # so projections match what the user actually receives
    _yr0_interest = income_info.get("interest_taxable", 0) if income_info else 0
    _yr0_dividends = income_info.get("total_ordinary_dividends", 0) if income_info else 0

    # Override div_yield if user entered actual dividends
    if _yr0_dividends > 0 and bal_brokerage > 0:
        div_yield = _yr0_dividends / bal_brokerage

    # Brokerage interest yield (bonds in brokerage)
    # User's total interest = cash interest + brokerage bond interest
    _cash_int_at_start = bal_cash * cash_int_rate if bal_cash > 0 else 0.0
    _brok_int_yield = (max(0, _yr0_interest - _cash_int_at_start) / bal_brokerage
                       if bal_brokerage > 0 and _yr0_interest > _cash_int_at_start else 0.0)
    deficit_mode = income_info.get("deficit_action", "reduce_savings") if income_info else "reduce_savings"
    base_salary_filer = income_info.get("salary_filer", 0) if income_info else 0
    base_salary_spouse = income_info.get("salary_spouse", 0) if income_info else 0
    use_itemize = income_info.get("itemize", False) if income_info else False
    mtg_bal_track = income_info.get("mortgage_balance", 0) if income_info else 0
    mtg_rate = income_info.get("mortgage_rate", 0) if income_info else 0
    base_property_tax = income_info.get("property_tax", 0) if income_info else 0
    base_medical = income_info.get("medical_expenses", 0) if income_info else 0
    base_charitable = income_info.get("charitable", 0) if income_info else 0
    ss_filer_pia = income_info.get("ss_filer_pia", 0) if income_info else 0
    ss_spouse_pia = income_info.get("ss_spouse_pia", 0) if income_info else 0
    ss_filer_claim = income_info.get("ss_filer_claim_age", 67) if income_info else 67
    ss_spouse_claim = income_info.get("ss_spouse_claim_age", 67) if income_info else 67
    ssdi_filer_flag = income_info.get("ssdi_filer", False) if income_info else False
    ssdi_spouse_flag = income_info.get("ssdi_spouse", False) if income_info else False
    ss_filer_already = income_info.get("ss_filer_already", False) if income_info else False
    ss_filer_current = income_info.get("ss_filer_current_benefit", 0) if income_info else 0
    ss_spouse_already = income_info.get("ss_spouse_already", False) if income_info else False
    ss_spouse_current = income_info.get("ss_spouse_current_benefit", 0) if income_info else 0
    se_filer_flag = income_info.get("self_employed_filer", False) if income_info else False
    se_spouse_flag = income_info.get("self_employed_spouse", False) if income_info else False
    accum_surplus_dest = income_info.get("surplus_destination", "brokerage") if income_info else "brokerage"
    accum_spouse_age = income_info.get("spouse_age", None) if income_info else None
    filer_dob_accum = income_info.get("filer_dob") if income_info else None
    spouse_dob_accum = income_info.get("spouse_dob") if income_info else None
    accum_retire_age = income_info.get("retire_age", current_age + years_to_ret) if income_info else current_age + years_to_ret
    accum_spouse_retire_age = income_info.get("spouse_retire_age", accum_retire_age) if income_info else accum_retire_age
    accum_return_sequence = income_info.get("return_sequence", None) if income_info else None
    inherited_iras = income_info.get("inherited_iras", []) if income_info else []
    iira_bals = [ira["balance"] for ira in inherited_iras]
    iira_rules = [ira["rule"] for ira in inherited_iras]
    iira_yrs = [ira["years_remaining"] for ira in inherited_iras]
    iira_ages = [ira["owner_age"] for ira in inherited_iras]
    iira_rmd_required = [ira.get("owner_was_taking_rmds", False) for ira in inherited_iras]
    iira_additional = [ira.get("additional_distribution", 0.0) for ira in inherited_iras]

    # Roth conversion params (pre-retirement conversions)
    accum_conv_strategy = income_info.get("roth_conversion_strategy", "none") if income_info else "none"
    accum_conv_target_agi = income_info.get("roth_conversion_target_agi", 0) if income_info else 0
    accum_conv_start_age = income_info.get("roth_conversion_start_age", 999) if income_info else 999
    accum_conv_stop_age = income_info.get("roth_conversion_stop_age", 100) if income_info else 100
    accum_do_conversions = accum_conv_strategy != "none" and accum_conv_strategy != 0
    accum_total_converted = 0.0

    # Home value + estate tracking
    accum_home_value = income_info.get("home_value", 0) if income_info else 0
    accum_home_appr = income_info.get("home_appreciation", 0) if income_info else 0
    accum_inv_re_value = income_info.get("inv_re_value", 0) if income_info else 0
    accum_inv_re_appr = income_info.get("inv_re_appr", 0) if income_info else 0
    accum_heir_bracket = income_info.get("heir_bracket_option", "same") if income_info else "same"
    _accum_estate_enabled = income_info.get("estate_tax_enabled", False) if income_info else False
    _accum_estate_exemption = income_info.get("federal_estate_exemption", 15000000.0) if income_info else 15000000.0
    _accum_estate_growth = income_info.get("exemption_inflation", 2.5) if income_info else 2.5
    _accum_estate_portability = income_info.get("use_portability", True) if income_info else True
    _accum_estate_state_rate = income_info.get("state_estate_tax_rate", 0.0) if income_info else 0.0
    _accum_estate_state_exemption = income_info.get("state_estate_exemption", 0.0) if income_info else 0.0
    _accum_estate_is_joint = "joint" in filing.lower()

    # Life insurance & annuity (display-only during accumulation — no withdrawals)
    accum_life = income_info.get("life_cash_value", 0) if income_info else 0
    accum_life_return = income_info.get("r_life", 0.04) if income_info else 0.04
    accum_annuity = income_info.get("annuity_value", 0) if income_info else 0
    accum_annuity_return = income_info.get("r_annuity", 0.04) if income_info else 0.04

    # Investment RE per-property income tracking
    accum_re_props = income_info.get("inv_re_properties", []) if income_info else []

    # Compute retirement year proration: fraction of year worked before birthday
    def _birthday_fraction(dob, retire_age):
        """Fraction of the retirement year spent working (before birthday)."""
        if dob is None:
            return 0.5  # default: mid-year
        retire_year = dob.year + retire_age
        bday_in_retire_year = dt.date(retire_year, dob.month, dob.day)
        jan1 = dt.date(retire_year, 1, 1)
        days_worked = (bday_in_retire_year - jan1).days
        return max(0.0, min(1.0, days_worked / 365.0))

    filer_retire_frac = _birthday_fraction(filer_dob_accum, accum_retire_age)
    spouse_retire_frac = _birthday_fraction(spouse_dob_accum, accum_spouse_retire_age) if accum_spouse_age else filer_retire_frac

    _ss_filer_bday_frac = ss_first_year_fraction(filer_dob_accum)
    _ss_spouse_bday_frac = ss_first_year_fraction(spouse_dob_accum)

    rows = []
    retire_age_accum = current_age + years_to_ret
    for yr in range(years_to_ret + 1):
        age = current_age + yr
        year = dt.date.today().year + yr
        is_retire_year = (age == retire_age_accum) and years_to_ret > 0
        sf = (1 + salary_growth_rate) ** yr if yr > 0 else 1.0
        inf_f = (1 + inf_rate) ** yr
        bracket_inf_f = (1 + bracket_growth_rate) ** yr
        medicare_inf_f = (1 + medicare_growth_rate) ** yr

        # Spouse age for this year
        spouse_age_yr = (accum_spouse_age + yr) if accum_spouse_age else age

        # Retirement year proration factor (fraction of year still working)
        work_frac = filer_retire_frac if is_retire_year else (1.0 if age < retire_age_accum else 0.0)
        retire_frac = 1.0 - work_frac  # fraction of year in retirement

        # Spouse may retire at a different age
        spouse_retire_age_yr = accum_spouse_retire_age if accum_spouse_age else retire_age_accum
        is_spouse_retire_year = accum_spouse_age and (spouse_age_yr == spouse_retire_age_yr) and spouse_retire_age_yr > current_age
        spouse_work_frac = spouse_retire_frac if is_spouse_retire_year else (1.0 if accum_spouse_age and spouse_age_yr < spouse_retire_age_yr else 0.0)
        if not accum_spouse_age:
            spouse_work_frac = work_frac  # single filer, mirror filer

        # Planned contributions (prorated in retirement year, zero after)
        # Split filer/spouse contributions so each uses their own work fraction
        c_pretax_f = contributions.get("pretax_filer", contributions["pretax"]) * sf * work_frac
        c_pretax_s = contributions.get("pretax_spouse", 0) * sf * spouse_work_frac
        c_roth_f = contributions.get("roth_filer", contributions["roth"]) * sf * work_frac
        c_roth_s = contributions.get("roth_spouse", 0) * sf * spouse_work_frac
        c_pretax_plan = c_pretax_f + c_pretax_s
        c_roth_plan = c_roth_f + c_roth_s
        # Taxable and HSA: assume filer controls these (use average of both fracs if married)
        avg_work_frac = (work_frac + spouse_work_frac) / 2.0 if accum_spouse_age else work_frac
        c_taxable_plan = contributions["taxable"] * sf * avg_work_frac
        c_cash_plan = contributions.get("cash", 0) * sf * avg_work_frac
        c_hsa_plan = contributions["hsa"] * sf * avg_work_frac

        # Investment income from taxable accounts (before growth)
        yr_dividends = bal_brokerage * div_yield if bal_brokerage > 0 else 0.0
        yr_cap_gain_dist = bal_brokerage * ann_cg_pct if bal_brokerage > 0 else 0.0
        yr_cash_interest = bal_cash * cash_int_rate if bal_cash > 0 else 0.0
        yr_brok_interest = bal_brokerage * _brok_int_yield if bal_brokerage > 0 else 0.0
        yr_interest = yr_cash_interest + yr_brok_interest
        yr_inv_income = yr_dividends + yr_cap_gain_dist + yr_interest

        # Cap gain distributions add to basis (already taxed)
        brokerage_basis += yr_cap_gain_dist

        # Social Security (prorated if starting mid-year in retirement year)
        # SSDI: pays 100% PIA from current age, converts to regular SS at FRA
        yr_ss_filer = 0.0
        if ss_filer_already and ss_filer_current > 0:
            yr_ss_filer = ss_filer_current * inf_f
        elif ss_filer_pia > 0:
            if ssdi_filer_flag:
                # SSDI pays 100% PIA immediately, growing with inflation
                yr_ss_filer = ss_filer_pia * inf_f
            elif age >= ss_filer_claim:
                ss_annual = ss_filer_pia * ss_claim_factor(ss_filer_claim) * inf_f
                if age == ss_filer_claim:
                    yr_ss_filer = ss_annual * _ss_filer_bday_frac
                else:
                    yr_ss_filer = ss_annual
        yr_ss_spouse = 0.0
        if ss_spouse_already and ss_spouse_current > 0:
            yr_ss_spouse = ss_spouse_current * inf_f
        elif ss_spouse_pia > 0:
            if ssdi_spouse_flag:
                yr_ss_spouse = ss_spouse_pia * inf_f
            elif spouse_age_yr >= ss_spouse_claim:
                ss_sp_annual = ss_spouse_pia * ss_claim_factor(ss_spouse_claim) * inf_f
                if spouse_age_yr == ss_spouse_claim:
                    yr_ss_spouse = ss_sp_annual * _ss_spouse_bday_frac
                else:
                    yr_ss_spouse = ss_sp_annual
        yr_ss = yr_ss_filer + yr_ss_spouse

        # Calculate income and non-savings expenses
        yr_wages = base_salary_filer * sf * work_frac + base_salary_spouse * sf * spouse_work_frac
        yr_spendable_inv_income = 0.0 if reinvest_inv_income else yr_inv_income
        # Other income (disability, inheritance, etc.) -- ends after N years if specified
        if base_other_income > 0 and (other_income_years == 0 or yr < other_income_years):
            yr_other_income = base_other_income * inf_f if other_income_inflation else base_other_income
        else:
            yr_other_income = 0.0

        # Investment RE rental income (per-property with mortgage payoff boost)
        yr_rental_income = 0.0
        for _rep in accum_re_props:
            _r_inc = _rep["net_income"] * ((1 + _rep["income_growth"]) ** yr)
            if _rep["mortgage_pmt"] > 0 and yr >= _rep["mortgage_years"]:
                _r_inc += _rep["mortgage_pmt"]
            yr_rental_income += _r_inc
        yr_other_income += yr_rental_income

        yr_income = yr_wages + yr_ss + yr_spendable_inv_income + yr_other_income
        yr_living = base_expenses * inf_f
        yr_mortgage = base_mortgage if yr < mortgage_yrs_left else 0.0

        yr_future_exp = 0.0
        for fe in fut_expenses:
            if fe["start_age"] <= age < fe["end_age"]:
                yrs_from_start = age - fe["start_age"]
                amt = fe["amount"] * ((1 + inf_rate) ** yrs_from_start) if fe["inflates"] else fe["amount"]
                yr_future_exp += amt

        # Inherited IRA distributions (before tax calc -- they are ordinary income)
        yr_inherited_dist = 0.0
        yr_inherited_min_rmd = 0.0
        for idx in range(len(inherited_iras)):
            if iira_bals[idx] <= 0:
                continue
            dist_info = calc_inherited_ira_distribution(
                iira_bals[idx], iira_rules[idx], iira_yrs[idx], iira_ages[idx] + yr,
                owner_was_taking_rmds=iira_rmd_required[idx],
                additional_distribution=iira_additional[idx])
            dist = dist_info["total_distribution"]
            yr_inherited_min_rmd += dist_info["minimum_rmd"]
            dist = min(dist, iira_bals[idx])
            yr_inherited_dist += dist
            iira_bals[idx] -= dist
            if iira_rules[idx] == "10_year":
                iira_yrs[idx] = max(0, iira_yrs[idx] - 1)

        # Roth conversion during accumulation
        accum_conversion_this_year = 0.0
        accum_conv_tax_withheld = 0.0
        accum_conv_tax_total = 0.0
        if accum_do_conversions and accum_conv_start_age <= age < accum_conv_stop_age and bal_pretax > 0:
            _est_pretax_ded = pretax_deductions * sf * work_frac
            _est_taxable_wages = max(0, yr_wages - _est_pretax_ded)
            _est_other_taxable = yr_other_income if not other_income_tax_free else 0.0
            if accum_conv_strategy == "fill_to_target":
                # Estimate AGI without conversion to find room
                _est_base = _est_taxable_wages + yr_interest + yr_inherited_dist + yr_dividends + yr_cap_gain_dist + _est_other_taxable
                _est_taxable_ss = yr_ss * 0.85
                _est_agi = _est_base + _est_taxable_ss
                room = max(0, accum_conv_target_agi * bracket_inf_f - _est_agi)
                accum_conversion_this_year = min(room, bal_pretax)
            else:
                accum_conversion_this_year = min(float(accum_conv_strategy), bal_pretax)

            # Tax on conversion: withhold from conversion if no other source to pay
            accum_conv_tax_withheld = 0.0
            if accum_conversion_this_year > 0:
                _ac_pretax_inc = _est_taxable_wages + yr_inherited_dist + _est_other_taxable
                _ac_cg_inc = yr_dividends + yr_cap_gain_dist
                _ac_filer65 = age >= 65
                _ac_sp65 = spouse_age_yr >= 65 if accum_spouse_age else False
                _ac_sc_params = {"dependents": _accum_sc_dependents, "out_of_state_gain": 0.0,
                                 "tax_year": _accum_sc_base_tax_year + yr,
                                 "retirement_income": yr_inherited_dist}
                _ac_no_conv_tax = calc_year_taxes(yr_ss, _ac_pretax_inc, _ac_cg_inc,
                                                   yr_interest, filing, _ac_filer65, _ac_sp65,
                                                   bracket_inf_f, st_rate, sc_params=_ac_sc_params)["total_tax"]
                _ac_sc_params["retirement_income"] = yr_inherited_dist + accum_conversion_this_year
                _ac_with_conv_tax = calc_year_taxes(yr_ss, _ac_pretax_inc + accum_conversion_this_year,
                                                     _ac_cg_inc, yr_interest, filing,
                                                     _ac_filer65, _ac_sp65, bracket_inf_f, st_rate,
                                                     sc_params=_ac_sc_params)["total_tax"]
                _ac_conv_tax = _ac_with_conv_tax - _ac_no_conv_tax
                accum_conv_tax_total = _ac_conv_tax
                # Available external funding: surplus income + taxable accounts
                _ac_base_expenses = yr_living + yr_mortgage + yr_future_exp + _ac_no_conv_tax
                _ac_planned = c_pretax_plan + c_roth_plan + c_taxable_plan + c_cash_plan + c_hsa_plan
                _ac_surplus = max(0, yr_income + yr_inherited_dist - _ac_base_expenses - _ac_planned)
                _ac_external = _ac_surplus + bal_brokerage + bal_cash
                if _ac_conv_tax > _ac_external:
                    accum_conv_tax_withheld = min(_ac_conv_tax - max(0, _ac_external), accum_conversion_this_year)

            if accum_conversion_this_year > 0:
                bal_pretax -= accum_conversion_this_year
                bal_roth += accum_conversion_this_year - accum_conv_tax_withheld  # net after tax withholding
                accum_total_converted += accum_conversion_this_year

        # Estimate taxes on wages + SS + investment income (with planned pre-tax deductions)
        yr_pretax_ded = pretax_deductions * sf * work_frac
        filer_65 = age >= 65
        spouse_65_yr = spouse_age_yr >= 65 if accum_spouse_age else False
        taxable_wages = max(0, yr_wages - yr_pretax_ded)
        yr_ordinary_inv_income = yr_interest
        yr_cap_gain_income = yr_dividends + yr_cap_gain_dist
        yr_taxable_other = yr_other_income if not other_income_tax_free else 0.0
        # SS taxability: up to 85% based on provisional income
        pretax_for_ss = taxable_wages + yr_ordinary_inv_income + yr_inherited_dist + yr_cap_gain_income + yr_taxable_other + accum_conversion_this_year
        taxable_ss = calc_taxable_ss(pretax_for_ss, yr_ss, filing)
        total_ordinary = taxable_wages + yr_ordinary_inv_income + yr_inherited_dist + yr_taxable_other + taxable_ss + accum_conversion_this_year

        # FICA / SE tax (compute early so SE deduction can reduce AGI)
        yr_wages_filer = base_salary_filer * sf * work_frac
        yr_wages_spouse = base_salary_spouse * sf * spouse_work_frac
        yr_fica, yr_se_ded = calc_fica(yr_wages_filer, yr_wages_spouse, filing, inf_f,
                                        se_filer=se_filer_flag, se_spouse=se_spouse_flag)

        yr_agi = total_ordinary + yr_cap_gain_income - yr_se_ded
        std_ded = get_std_deduction(filing, filer_65, spouse_65_yr, bracket_inf_f)

        # Charitable spending (always a real expense, even if not itemizing)
        yr_charitable = base_charitable * inf_f

        # SC params for this accumulation year
        _accum_yr_ret_income = yr_inherited_dist + accum_conversion_this_year
        _accum_yr_sc_params = {"dependents": _accum_sc_dependents, "out_of_state_gain": 0.0,
                               "tax_year": _accum_sc_base_tax_year + yr,
                               "retirement_income": _accum_yr_ret_income}

        # Determine deduction: itemized vs standard
        deduction = std_ded
        if use_itemize and yr < mortgage_yrs_left + 5:  # itemize while it helps
            yr_mtg_interest, mtg_bal_end = calc_mortgage_interest_for_year(mtg_bal_track, mtg_rate, base_mortgage)
            _enh_for_salt = get_federal_enhanced_extra(yr_agi, filer_65, spouse_65_yr, filing, bracket_inf_f,
                                                       tax_year=_accum_sc_base_tax_year + yr)
            est_sc_result = calculate_sc_tax(max(0, yr_agi - std_ded), _accum_sc_dependents, taxable_ss,
                                             0.0, filer_65, spouse_65_yr, _accum_yr_ret_income,
                                             yr_cap_gain_income, enhanced_elderly_addback=_enh_for_salt)
            est_state_tax = est_sc_result["sc_tax"]
            yr_property_tax = base_property_tax * inf_f
            salt = min(10000.0 * bracket_inf_f, est_state_tax + yr_property_tax)
            yr_medical = base_medical * inf_f
            medical_ded = max(0.0, yr_medical - yr_agi * 0.075)
            itemized = salt + yr_mtg_interest + medical_ded + yr_charitable
            if itemized > std_ded:
                deduction = itemized
            mtg_bal_track = mtg_bal_end
        elif mtg_bal_track > 0 and yr_mortgage > 0:
            # Still track mortgage balance even if not itemizing
            _, mtg_bal_track = calc_mortgage_interest_for_year(mtg_bal_track, mtg_rate, base_mortgage)

        # Income tax (SE deduction already subtracted from yr_agi)
        ordinary_taxable = max(0, total_ordinary - yr_se_ded - deduction)
        fed_tax = calc_federal_tax(ordinary_taxable, filing, bracket_inf_f)
        fed_tax += calc_cg_tax(yr_cap_gain_income, ordinary_taxable, filing, bracket_inf_f)
        _accum_fed_taxable = max(0, yr_agi - deduction)
        _accum_enh = get_federal_enhanced_extra(yr_agi, filer_65, spouse_65_yr, filing, bracket_inf_f,
                                                 tax_year=_accum_sc_base_tax_year + yr)
        _accum_sc = calculate_sc_tax(_accum_fed_taxable, _accum_sc_dependents, taxable_ss, 0.0,
                                      filer_65, spouse_65_yr, _accum_yr_ret_income, yr_cap_gain_income,
                                      enhanced_elderly_addback=_accum_enh)
        state_tax = _accum_sc["sc_tax"]
        yr_taxes = fed_tax + state_tax + yr_fica

        # Non-savings expenses (subtract any conversion tax already withheld from the conversion)
        yr_fixed_expenses = yr_living + yr_mortgage + yr_future_exp + yr_charitable + yr_taxes - accum_conv_tax_withheld
        total_planned_contrib = c_pretax_plan + c_roth_plan + c_taxable_plan + c_cash_plan + c_hsa_plan
        yr_surplus = yr_income + yr_inherited_dist - yr_fixed_expenses - total_planned_contrib

        # Handle deficit
        c_pretax = c_pretax_plan
        c_roth = c_roth_plan
        c_taxable = c_taxable_plan
        c_cash_contrib = c_cash_plan
        c_hsa = c_hsa_plan
        yr_liquidation = 0.0
        yr_liquidation_cg_tax = 0.0

        if yr_surplus < 0:
            deficit = abs(yr_surplus)
            if deficit_mode == "reduce_savings":
                # Reduce contributions: cash first, then taxable, HSA, Roth, pre-tax
                for bucket in ["cash", "taxable", "hsa", "roth", "pretax"]:
                    if deficit <= 0:
                        break
                    if bucket == "cash":
                        cut = min(c_cash_contrib, deficit); c_cash_contrib -= cut; deficit -= cut
                    elif bucket == "taxable":
                        cut = min(c_taxable, deficit); c_taxable -= cut; deficit -= cut
                    elif bucket == "hsa":
                        cut = min(c_hsa, deficit); c_hsa -= cut; deficit -= cut
                    elif bucket == "roth":
                        cut = min(c_roth, deficit); c_roth -= cut; deficit -= cut
                    elif bucket == "pretax":
                        cut = min(c_pretax, deficit); c_pretax -= cut; deficit -= cut
                # If still a deficit after zeroing contributions, liquidate
                if deficit > 0:
                    # Sell cash first (no capital gains), then brokerage
                    sell_cash = min(bal_cash, deficit)
                    bal_cash -= sell_cash
                    deficit -= sell_cash
                    yr_liquidation += sell_cash
                    if deficit > 0 and bal_brokerage > 0:
                        sell_brok = min(bal_brokerage, deficit)
                        gain_pct_at_sale = max(0, (bal_brokerage - brokerage_basis) / bal_brokerage) if bal_brokerage > 0 else 0
                        realized_gain = sell_brok * gain_pct_at_sale
                        yr_liquidation_cg_tax = realized_gain * 0.15
                        brokerage_basis -= sell_brok * (1 - gain_pct_at_sale)  # reduce basis proportionally
                        bal_brokerage -= sell_brok
                        deficit -= sell_brok
                        yr_liquidation += sell_brok
            else:  # liquidate investments, keep savings
                # Sell cash first, then brokerage
                sell_cash = min(bal_cash, deficit)
                bal_cash -= sell_cash
                deficit -= sell_cash
                yr_liquidation += sell_cash
                if deficit > 0 and bal_brokerage > 0:
                    sell_brok = min(bal_brokerage, deficit)
                    gain_pct_at_sale = max(0, (bal_brokerage - brokerage_basis) / bal_brokerage) if bal_brokerage > 0 else 0
                    realized_gain = sell_brok * gain_pct_at_sale
                    yr_liquidation_cg_tax = realized_gain * 0.15
                    brokerage_basis -= sell_brok * (1 - gain_pct_at_sale)
                    bal_brokerage -= sell_brok
                    deficit -= sell_brok
                    yr_liquidation += sell_brok

        yr_taxes += yr_liquidation_cg_tax
        total_contrib = c_pretax + c_roth + c_taxable + c_cash_contrib + c_hsa
        yr_total_expenses = yr_fixed_expenses + yr_liquidation_cg_tax + total_contrib

        # Reinvest surplus (after-tax income exceeding expenses + contributions)
        yr_actual_surplus = max(0.0, yr_surplus) if yr_surplus > 0 else 0.0
        if yr > 0 and yr_actual_surplus > 0 and accum_surplus_dest != "none":
            if accum_surplus_dest == "cash":
                bal_cash += yr_actual_surplus
            else:  # brokerage
                bal_brokerage += yr_actual_surplus
                brokerage_basis += yr_actual_surplus  # after-tax money = 100% basis

        # Apply actual contributions and growth
        if yr > 0:
            bal_pretax += c_pretax
            bal_roth += c_roth
            bal_brokerage += c_taxable
            brokerage_basis += c_taxable
            bal_cash += c_cash_contrib
            bal_hsa += c_hsa
            # Per-account growth (MC return sequence overrides all rates)
            if accum_return_sequence:
                yr_return = accum_return_sequence[yr]
                bal_pretax *= (1 + yr_return)
                bal_roth *= (1 + yr_return)
                _div_drag = (div_yield + ann_cg_pct + _brok_int_yield) if not reinvest_inv_income else 0.0
                bal_brokerage *= (1 + yr_return - _div_drag)
                bal_hsa *= (1 + yr_return)
            else:
                bal_pretax *= (1 + r_pretax)
                bal_roth *= (1 + r_roth)
                _div_drag = (div_yield + ann_cg_pct + _brok_int_yield) if not reinvest_inv_income else 0.0
                bal_brokerage *= (1 + r_taxable - _div_drag)
                bal_hsa *= (1 + r_hsa)
            bal_cash *= (1 + cash_int_rate)
            if not reinvest_inv_income:
                bal_cash -= yr_cash_interest
            # Grow remaining inherited IRA balances
            for idx in range(len(inherited_iras)):
                if iira_bals[idx] > 0:
                    iira_bals[idx] *= (1 + (accum_return_sequence[yr] if accum_return_sequence else r_pretax))

        bal_taxable = bal_brokerage + bal_cash
        bal_inherited_total = sum(iira_bals)
        total = bal_pretax + bal_roth + bal_taxable + bal_hsa + bal_inherited_total
        unrealized_gain = max(0, bal_brokerage - brokerage_basis)

        # Appreciate home value, investment RE, life insurance, annuity
        if yr > 0 and accum_home_value > 0:
            accum_home_value *= (1 + accum_home_appr)
        if yr > 0 and accum_inv_re_value > 0:
            accum_inv_re_value *= (1 + accum_inv_re_appr)
        if yr > 0 and accum_life > 0:
            accum_life *= (1 + accum_life_return)
        if yr > 0 and accum_annuity > 0:
            accum_annuity *= (1 + accum_annuity_return)

        # Gross & net estate (matching retirement projection layout)
        gross_estate = total + accum_home_value + accum_inv_re_value + accum_life + accum_annuity
        _acc_marginal = get_marginal_fed_rate(ordinary_taxable, filing, bracket_inf_f)
        if accum_heir_bracket == "lower":
            _acc_heir_fed = _heir_rate_from_offset(_acc_marginal, -1)
        elif accum_heir_bracket == "higher":
            _acc_heir_fed = _heir_rate_from_offset(_acc_marginal, +1)
        else:
            _acc_heir_fed = _acc_marginal
        _acc_heir_rate = _acc_heir_fed + st_rate
        _acc_ann_basis = income_info.get("annuity_basis", 0) if income_info else 0
        _acc_ann_gain = max(0, accum_annuity - _acc_ann_basis)
        _acc_ann_net = _acc_ann_basis + _acc_ann_gain * (1 - _acc_heir_rate)
        _acc_heir_estate = (
            bal_pretax * (1 - _acc_heir_rate) +
            bal_roth +
            bal_taxable +  # stepped-up basis
            bal_hsa * (1 - _acc_heir_rate) +
            bal_inherited_total * (1 - _acc_heir_rate) +
            accum_home_value + accum_inv_re_value +
            accum_life +  # life insurance: tax-free to beneficiaries
            _acc_ann_net  # annuity: gains taxed as ordinary income to heirs
        )
        # Estate tax: unlimited marital deduction → $0 while both spouses alive (always both alive during accumulation).
        _acc_estate_tax = 0.0
        if _accum_estate_enabled and not _accum_estate_is_joint:
            _acc_estate_tax = compute_estate_tax(
                gross_estate, _accum_estate_exemption, _accum_estate_growth, yr,
                False, False,
                _accum_estate_state_rate, _accum_estate_state_exemption)
        after_tax_estate = _acc_heir_estate - _acc_estate_tax

        row = {
            "Year": year, "Age": age,
        }
        if accum_spouse_age:
            row["Spouse Age"] = spouse_age_yr
        row.update({
            "Income": round(yr_wages, 0),
            "Other Income": round(yr_other_income, 0),
            "SS Income": round(yr_ss, 0),
            "Pension": 0,
            "Dividends": round(yr_dividends, 0),
            "Interest": round(yr_interest, 0),
            "Living Exp": round(yr_living, 0),
            "Mortgage": round(yr_mortgage, 0),
            "Other Exp": round(yr_future_exp, 0),
            "Fed Tax": round(fed_tax, 0), "State Tax": round(state_tax, 0),
            "FICA": round(yr_fica, 0), "Total Tax": round(yr_taxes, 0),
            "Save Pre-Tax": round(c_pretax, 0), "Save Roth": round(c_roth, 0),
            "Save Taxable": round(c_taxable, 0), "Save Cash": round(c_cash_contrib, 0),
            "Save HSA": round(c_hsa, 0),
            "Total Saved": round(total_contrib, 0),
            "Conv Gross": round(accum_conversion_this_year, 0),
            "Conv Tax": round(accum_conv_tax_total, 0),
            "Conv to Roth": round(accum_conversion_this_year - accum_conv_tax_withheld, 0),
            "Liquidated": round(yr_liquidation, 0),
            "Total Outflow": round(yr_total_expenses, 0),
            "Surplus": round(yr_actual_surplus, 0),
            "Bal Pre-Tax": round(bal_pretax, 0), "Bal Roth": round(bal_roth, 0),
            "Bal Taxable": round(bal_taxable, 0), "Basis": round(brokerage_basis, 0),
            "Unreal Gain": round(unrealized_gain, 0),
            "Bal HSA": round(bal_hsa, 0),
        })
        if any(ira["balance"] > 0 for ira in inherited_iras):
            row["Inherited Dist"] = round(yr_inherited_dist, 0)
            row["Bal Inherited"] = round(bal_inherited_total, 0)
        row["Portfolio"] = round(total, 0)
        if accum_life > 0 or (income_info and income_info.get("life_cash_value", 0) > 0):
            row["Life Ins"] = round(accum_life, 0)
        if accum_annuity > 0 or (income_info and income_info.get("annuity_value", 0) > 0):
            row["Annuity"] = round(accum_annuity, 0)
        row["Home Value"] = round(accum_home_value, 0)
        if accum_inv_re_value > 0 or (income_info and income_info.get("inv_re_value", 0) > 0):
            row["Inv RE"] = round(accum_inv_re_value, 0)
        row["Gross Estate"] = round(gross_estate, 0)
        if _accum_estate_enabled:
            row["Estate Tax"] = round(_acc_estate_tax, 0)
        row["Estate (Net)"] = round(after_tax_estate, 0)
        rows.append(row)
    return {"rows": rows, "final_brokerage": bal_brokerage, "final_cash": bal_cash, "final_basis": brokerage_basis,
            "final_inherited": sum(iira_bals),
            "total_converted": accum_total_converted,
            "derived_div_yield": div_yield, "derived_brok_int_yield": _brok_int_yield,
            "inherited_iras_state": [{"balance": iira_bals[i], "rule": iira_rules[i],
                                      "years_remaining": iira_yrs[i], "owner_age": iira_ages[i] + years_to_ret,
                                      "owner_was_taking_rmds": iira_rmd_required[i],
                                      "additional_distribution": iira_additional[i]}
                                     for i in range(len(inherited_iras))]}


# ── 17. OPTIMIZER FUNCTIONS ──────────────────────────────────────────────────

def optimize_ss_claiming(balances_at_retire, base_params, spending_order,
                         filer_current_age, spouse_current_age=None):
    """Evaluate all claiming-age combos using full retirement projections.

    Instead of isolated SS NPV, runs the complete retirement simulation for each
    combo and compares ending estate values. This captures portfolio drawdown,
    tax bracket interactions, RMDs, pensions, and the full withdrawal waterfall.

    Returns dict with keys:
        best: dict with filer_claim, spouse_claim, estate, total_taxes
        rankings: list of all combos sorted by estate desc
        schedule: year-by-year rows from best combo's full projection
        explanation: text explaining why the recommended strategy wins
    """
    import copy

    is_married = spouse_current_age is not None and base_params.get("ss_spouse_fra", 0) > 0

    retire_age = base_params.get("retire_age", 65)
    filer_min_claim = max(62, filer_current_age, retire_age)
    spouse_retire = base_params.get("spouse_retire_age", retire_age)
    spouse_min_claim = max(62, spouse_current_age, spouse_retire) if spouse_current_age else 62

    if is_married:
        claim_combos = [(fc, sc) for fc in range(filer_min_claim, 71)
                        for sc in range(spouse_min_claim, 71)]
    else:
        claim_combos = [(fc, None) for fc in range(filer_min_claim, 71)]

    results = []
    for fc, sc in claim_combos:
        params = copy.deepcopy(base_params)
        params["ss_filer_claim_age"] = fc
        if sc is not None:
            params["ss_spouse_claim_age"] = sc

        bals = copy.deepcopy(balances_at_retire)
        res = run_retirement_projection(bals, params, spending_order)

        results.append({
            "filer_claim": fc,
            "spouse_claim": sc if sc is not None else "\u2014",
            "estate": res["estate"],
            "total_taxes": res["total_taxes"],
            "final_total": res["final_total"],
            "rows": res["rows"],
        })

    results.sort(key=lambda x: x["estate"], reverse=True)

    best = results[0]
    best_estate = best["estate"]

    rankings = []
    for rank, r in enumerate(results, 1):
        rankings.append({
            "rank": rank,
            "filer_claim": r["filer_claim"],
            "spouse_claim": r["spouse_claim"],
            "estate": r["estate"],
            "total_taxes": r["total_taxes"],
            "diff_vs_best": r["estate"] - best_estate,
        })

    explanation = _generate_projection_explanation(
        best, results, is_married, filer_current_age, spouse_current_age)

    return {
        "best": {
            "filer_claim": best["filer_claim"],
            "spouse_claim": best["spouse_claim"],
            "estate": best_estate,
            "total_taxes": best["total_taxes"],
        },
        "rankings": rankings,
        "schedule": best["rows"],
        "explanation": explanation,
    }


def _generate_projection_explanation(best, all_results, is_married,
                                     filer_current_age, spouse_current_age):
    """Generate explanation referencing portfolio impact."""
    lines = []
    fc = best["filer_claim"]
    sc = best["spouse_claim"]

    # Find early claiming result for comparison
    early_fc = max(62, filer_current_age)
    early_sc = max(62, spouse_current_age) if spouse_current_age else None
    early_result = None
    for r in all_results:
        if r["filer_claim"] == early_fc and r["spouse_claim"] == early_sc:
            early_result = r
            break

    if early_result and (early_result["filer_claim"] != fc or early_result["spouse_claim"] != sc):
        estate_diff = best["estate"] - early_result["estate"]
        tax_diff = best["total_taxes"] - early_result["total_taxes"]
        best_rows = best["rows"]
        early_rows = early_result["rows"]
        delayed = fc > early_fc or (is_married and isinstance(sc, int) and sc > max(62, spouse_current_age))

        if delayed:
            early_label = f"you at {early_fc}" + (f", spouse at {early_sc}" if is_married and early_sc else "")
            claim_age = max(fc, sc) if is_married and isinstance(sc, int) else fc
            gap_draw_best = sum(
                r.get("W/D Pre-Tax", 0) + r.get("W/D Taxable", 0) + r.get("W/D Roth", 0) + r.get("W/D HSA", 0)
                for r in best_rows if r["Age"] < claim_age)
            gap_draw_early = sum(
                r.get("W/D Pre-Tax", 0) + r.get("W/D Taxable", 0) + r.get("W/D Roth", 0) + r.get("W/D HSA", 0)
                for r in early_rows if r["Age"] < claim_age)
            extra_draw = gap_draw_best - gap_draw_early

            if extra_draw > 0:
                lines.append(f"- Delaying SS draws **{money(extra_draw)}** more from your portfolio before benefits begin")
            if estate_diff > 0:
                lines.append(f"- Produces **{money(estate_diff)}** more in ending estate vs. earliest claiming ({early_label})")
            elif estate_diff < 0:
                lines.append(f"- Earlier claiming preserves your portfolio during the early retirement years")

            # Recovery age
            best_by_age = {r["Age"]: r["Estate (Net)"] for r in best_rows}
            early_by_age = {r["Age"]: r["Estate (Net)"] for r in early_rows}
            recovery = next((a for a in sorted(best_by_age) if a in early_by_age and best_by_age[a] >= early_by_age[a]), None)
            if recovery:
                lines.append(f"- Delayed strategy recovers by age **{recovery}**")
        else:
            if estate_diff > 0:
                lines.append(f"- Earlier claiming produces **{money(estate_diff)}** more in ending estate")

        if abs(tax_diff) > 1000:
            if tax_diff < 0:
                lines.append(f"- Saves **{money(abs(tax_diff))}** in total lifetime taxes")
            else:
                lines.append(f"- Pays **{money(tax_diff)}** more in taxes, but portfolio growth outweighs it")

    if len(all_results) > 1:
        worst = all_results[-1]
        diff = best["estate"] - worst["estate"]
        if diff > 0:
            lines.append(f"- Best vs. worst strategy spread: **{money(diff)}**")

    return "\n".join(lines) if lines else "All claiming strategies produce similar results."


def optimize_roth_conversions(balances_at_retire, base_params, spending_order,
                               start_age=None, accum_inputs=None):
    """Evaluate auto-generated Roth conversion strategies using full retirement projections.

    Auto-generates bracket-aware (fill-to-target) and fixed-duration strategies,
    tests each, and compares after-tax estate values.

    If start_age < retire_age and accum_inputs is provided, re-runs the
    accumulation phase with conversions to get new retirement balances.

    accum_inputs: dict with keys current_age, years_to_ret, start_balances,
                  contributions, salary_growth, pre_ret_return, income_info

    Returns dict with keys:
        best: dict with strategy name, estate, total_taxes, total_converted
        baseline: estate with no conversions
        rankings: list of all strategies sorted by estate desc
        schedule: year-by-year rows from best strategy's full projection
    """
    import copy

    retire_age = base_params.get("retire_age", 65)
    filing_status = base_params.get("filing_status", "Single")
    is_joint = "joint" in filing_status.lower()
    eff_start = start_age if start_age is not None else retire_age
    pre_retire = (start_age is not None and start_age < retire_age and accum_inputs is not None)

    # Cache baseline accumulation (no conversions) for retire-only strategies
    _baseline_accum = None

    def _get_baseline_accum():
        nonlocal _baseline_accum
        if _baseline_accum is None and pre_retire:
            ai = accum_inputs
            ii = copy.deepcopy(ai["income_info"])
            # No conversion params -> runs accumulation without conversions
            _baseline_accum = run_accumulation(ai["current_age"], ai["years_to_ret"],
                                               copy.deepcopy(ai["start_balances"]),
                                               ai["contributions"], ai["salary_growth"],
                                               ai["pre_ret_return"], ii)
        return _baseline_accum

    def _run_strategy(strategy_val, target, stop, defer_rmd, retire_only=False):
        """Run accumulation (if pre-retire) + retirement projection for one strategy.

        If retire_only=True and pre_retire, run accumulation WITHOUT conversions
        (same balances as Tab 4) but apply the conversion strategy only to the
        retirement phase. This tests whether waiting for a lower bracket is better.
        """
        if pre_retire:
            if retire_only:
                # Use baseline accumulation (no conversions) -- same as Tab 4
                accum_res = _get_baseline_accum()
            else:
                ai = accum_inputs
                ii = copy.deepcopy(ai["income_info"])
                ii["roth_conversion_strategy"] = strategy_val
                ii["roth_conversion_target_agi"] = target
                ii["roth_conversion_start_age"] = start_age
                ii["roth_conversion_stop_age"] = stop
                accum_res = run_accumulation(ai["current_age"], ai["years_to_ret"],
                                             copy.deepcopy(ai["start_balances"]),
                                             ai["contributions"], ai["salary_growth"],
                                             ai["pre_ret_return"], ii)
            accum_rows = accum_res["rows"]
            last = accum_rows[-1] if accum_rows else {}
            bals = {
                "pretax": float(last.get("Bal Pre-Tax", 0)),
                "roth": float(last.get("Bal Roth", 0)),
                "taxable": float(last.get("Bal Taxable", 0)),
                "brokerage": float(accum_res.get("final_brokerage", last.get("Bal Taxable", 0))),
                "cash": float(accum_res.get("final_cash", 0)),
                "brokerage_basis": float(accum_res.get("final_basis", last.get("Bal Taxable", 0))),
                "hsa": float(last.get("Bal HSA", 0)),
            }
            accum_converted = 0 if retire_only else accum_res.get("total_converted", 0)
        else:
            bals = copy.deepcopy(balances_at_retire)
            accum_converted = 0
            accum_rows = []

        params = copy.deepcopy(base_params)
        params["roth_conversion_strategy"] = strategy_val
        params["roth_conversion_target_agi"] = target
        params["roth_conversion_stop_age"] = stop
        params["defer_first_rmd"] = defer_rmd
        if pre_retire:
            params["inherited_iras"] = accum_res.get("inherited_iras_state", [])
        res = run_retirement_projection(bals, params, spending_order)
        retire_converted = res.get("total_converted", 0)
        res["total_converted"] = retire_converted + accum_converted
        res["accum_converted"] = accum_converted
        res["retire_converted"] = retire_converted
        res["starting_pretax"] = bals["pretax"]
        res["starting_roth"] = bals["roth"]
        res["starting_taxable"] = bals.get("brokerage", 0) + bals.get("cash", 0)
        res["starting_hsa"] = bals["hsa"]
        res["accum_rows"] = accum_rows if not retire_only else []
        return res

    # --- Auto-generate strategies ---
    pretax_bal = balances_at_retire.get("pretax", 0)

    # A) Fill-to-bracket targets (AGI thresholds = bracket top + approx std deduction)
    if is_joint:
        fill_targets = [
            ("Fill to 22% bracket", 238000),
            ("Fill to IRMAA", 212000),
            ("Fill to 24% bracket", 426000),
            ("Fill to 32% bracket", 533000),
        ]
    else:
        fill_targets = [
            ("Fill to 22% bracket", 119000),
            ("Fill to IRMAA", 106000),
            ("Fill to 24% bracket", 213000),
            ("Fill to 32% bracket", 266000),
        ]

    # B) Fixed amount x duration grid -- covers small totals to large
    #    amounts: $10k, $25k, $50k, $100k, $200k/yr
    #    windows: 5, 10, 15 years from start age
    #    gives total conversions ranging from $50k to $3M+
    fixed_amounts = [a for a in [10000, 25000, 50000, 100000, 200000] if a <= pretax_bal]
    fixed_windows = [5, 10, 15]
    fixed_grid = []  # (label, annual_amt, stop_age)
    for amt in fixed_amounts:
        for yrs in fixed_windows:
            stop = eff_start + yrs
            total_approx = amt * yrs
            if total_approx <= pretax_bal * 1.1:  # skip if would far exceed balance
                fixed_grid.append((f"${amt:,.0f}/yr x {yrs}yrs", amt, stop))

    results = []

    # Always test baseline (no conversions)
    res = _run_strategy("none", 0, 100, False)
    baseline_estate = res["estate"]
    baseline_start_bals = {
        "pretax": res.get("starting_pretax", 0),
        "roth": res.get("starting_roth", 0),
        "taxable": res.get("starting_taxable", 0),
    }
    results.append({
        "strategy": "No Conversion",
        "strategy_val": "none",
        "estate": res["estate"],
        "total_taxes": res["total_taxes"],
        "total_converted": res.get("total_converted", 0),
        "accum_converted": res.get("accum_converted", 0),
        "retire_converted": res.get("retire_converted", 0),
        "starting_pretax": res.get("starting_pretax", 0),
        "starting_roth": res.get("starting_roth", 0),
        "starting_taxable": res.get("starting_taxable", 0),
        "final_pretax": res["final_pretax"],
        "final_roth": res["final_roth"],
        "final_taxable": res["final_taxable"],
        "rows": res["rows"],
        "accum_rows": res.get("accum_rows", []),
        "defer": False,
        "conv_strategy_val": "none",
        "conv_target_agi": 0,
        "conv_stop_age": 100,
    })

    def _append_result(label, res, defer, conv_strategy_val="none", conv_target_agi=0, conv_stop_age=100):
        results.append({
            "strategy": label,
            "strategy_val": label,
            "estate": res["estate"],
            "total_taxes": res["total_taxes"],
            "total_converted": res.get("total_converted", 0),
            "accum_converted": res.get("accum_converted", 0),
            "retire_converted": res.get("retire_converted", 0),
            "starting_pretax": res.get("starting_pretax", 0),
            "starting_roth": res.get("starting_roth", 0),
            "starting_taxable": res.get("starting_taxable", 0),
            "final_pretax": res["final_pretax"],
            "final_roth": res["final_roth"],
            "final_taxable": res["final_taxable"],
            "rows": res["rows"],
            "accum_rows": res.get("accum_rows", []),
            "defer": defer,
            "conv_strategy_val": conv_strategy_val,
            "conv_target_agi": conv_target_agi,
            "conv_stop_age": conv_stop_age,
        })

    # Test each strategy with both defer=False and defer=True
    for defer_rmd in [False, True]:
        defer_label = " (defer 1st RMD)" if defer_rmd else ""

        # Fill-to-bracket strategies (high stop age -- they self-taper when no room)
        for name, target in fill_targets:
            res = _run_strategy("fill_to_target", target, 85, defer_rmd)
            _append_result(f"{name}{defer_label}", res, defer_rmd,
                           conv_strategy_val="fill_to_target", conv_target_agi=target, conv_stop_age=85)

        # Fixed amount x duration strategies
        for name, amt, stop in fixed_grid:
            res = _run_strategy(amt, 0, stop, defer_rmd)
            _append_result(f"{name}{defer_label}", res, defer_rmd,
                           conv_strategy_val=amt, conv_target_agi=0, conv_stop_age=stop)

        # -- Retire-only variants: same strategies but NO pre-retirement conversions --
        # Tests the philosophy: wait for a lower bracket in retirement instead of
        # converting during higher-income working years.
        if pre_retire:
            for name, target in fill_targets:
                res = _run_strategy("fill_to_target", target, 85, defer_rmd, retire_only=True)
                _append_result(f"{name} [retire only]{defer_label}", res, defer_rmd,
                               conv_strategy_val="fill_to_target", conv_target_agi=target, conv_stop_age=85)

            for name, amt, stop in fixed_grid:
                # For retire-only, shift the window to start at retirement
                retire_stop = retire_age + (stop - eff_start)
                res = _run_strategy(amt, 0, retire_stop, defer_rmd, retire_only=True)
                _append_result(f"{name} [retire only]{defer_label}", res, defer_rmd,
                               conv_strategy_val=amt, conv_target_agi=0, conv_stop_age=retire_stop)

    results.sort(key=lambda x: x["estate"], reverse=True)

    best = results[0]
    rankings = []
    for rank, r in enumerate(results, 1):
        row = {
            "Rank": rank,
            "Strategy": r["strategy"],
            "Estate (Net)": r["estate"],
            "vs Baseline": r["estate"] - baseline_estate,
            "Total Taxes": r["total_taxes"],
            "Total Converted": r["total_converted"],
        }
        if pre_retire:
            row["Accum Conv"] = r.get("accum_converted", 0)
            row["Retire Conv"] = r.get("retire_converted", 0)
        row.update({
            "Start Pre-Tax": r.get("starting_pretax", 0),
            "Start Roth": r.get("starting_roth", 0),
            "Start Taxable": r.get("starting_taxable", 0),
            "Final Pre-Tax": r["final_pretax"],
            "Final Roth": r["final_roth"],
            "Final Taxable": r["final_taxable"],
        })
        rankings.append(row)

    return {
        "best": best,
        "baseline": baseline_estate,
        "baseline_start_bals": baseline_start_bals,
        "pre_retire": pre_retire,
        "rankings": rankings,
        "schedule": best["rows"],
        "all_results": results,
    }


# ── 18. MONTE CARLO ─────────────────────────────────────────────────────────

def run_monte_carlo(run_fn, n_sims=500, mean_return=0.07, return_std=0.12,
                    n_years=30, seed=None, mean_return_post=None, n_years_pre=0):
    """Run Monte Carlo simulation by randomizing year-by-year returns.

    Args:
        run_fn: callable(return_sequence) -> dict with "estate" and "final_total" keys
        n_sims: number of simulations
        mean_return: mean annual return (used for all years, or just pre-retirement if mean_return_post is set)
        return_std: standard deviation of annual returns
        n_years: number of years to generate returns for
        seed: optional RNG seed for reproducibility
        mean_return_post: if set, use this mean for retirement years (after n_years_pre)
        n_years_pre: number of pre-retirement years (only used when mean_return_post is set)

    Returns:
        dict with median_estate, p10, p25, p75, p90, mean_estate, success_rate, all_estates
    """
    rng = np.random.default_rng(seed)
    if mean_return_post is not None and n_years_pre > 0 and n_years_pre < n_years:
        n_years_post = n_years - n_years_pre
        pre_returns = rng.normal(mean_return, return_std, (n_sims, n_years_pre))
        post_returns = rng.normal(mean_return_post, return_std, (n_sims, n_years_post))
        all_returns = np.concatenate([pre_returns, post_returns], axis=1)
    else:
        all_returns = rng.normal(mean_return, return_std, (n_sims, n_years))
    all_returns = np.maximum(all_returns, -0.50)

    all_estates = []
    all_finals = []
    all_retire_portfolios = []
    for sim in range(n_sims):
        result = run_fn(all_returns[sim].tolist())
        all_estates.append(result["estate"])
        all_finals.append(result.get("final_total", result["estate"]))
        all_retire_portfolios.append(result.get("retire_portfolio", 0))

    all_estates = np.array(all_estates)
    all_finals = np.array(all_finals)
    all_retire_portfolios = np.array(all_retire_portfolios)
    # Success = portfolio (excluding home) never depleted
    success_rate = float(np.mean(all_finals > 0))
    out = {
        "median_estate": float(np.median(all_estates)),
        "p10": float(np.percentile(all_estates, 10)),
        "p25": float(np.percentile(all_estates, 25)),
        "p75": float(np.percentile(all_estates, 75)),
        "p90": float(np.percentile(all_estates, 90)),
        "mean_estate": float(np.mean(all_estates)),
        "success_rate": success_rate,
        "all_estates": all_estates.tolist(),
        # Portfolio (no home) end-of-plan stats
        "median_portfolio": float(np.median(all_finals)),
        "portfolio_p10": float(np.percentile(all_finals, 10)),
        "portfolio_p90": float(np.percentile(all_finals, 90)),
    }
    if np.any(all_retire_portfolios > 0):
        out["retire_median"] = float(np.median(all_retire_portfolios))
        out["retire_p10"] = float(np.percentile(all_retire_portfolios, 10))
        out["retire_p90"] = float(np.percentile(all_retire_portfolios, 90))
        out["retire_mean"] = float(np.mean(all_retire_portfolios))
    return out


# ── 19. PDF HELPERS ──────────────────────────────────────────────────────────

def _pdf_safe(text):
    """Replace Unicode characters that Helvetica can't render with ASCII equivalents."""
    s = str(text)
    s = s.replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
    s = s.replace("\u2018", "'").replace("\u2019", "'")   # smart single quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"')   # smart double quotes
    s = s.replace("\u2026", "...").replace("\u2022", "*") # ellipsis, bullet
    s = s.replace("\u00b7", "*")                           # middle dot
    s = s.replace("\u2192", "->")                          # right arrow
    return s

def pdf_money(x):
    """Format number as $1,234 for PDF (no HTML)."""
    try:
        v = float(x)
        if v < 0:
            return f"(${abs(v):,.0f})"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "$0"

def _pdf_section_header(pdf, title):
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, _pdf_safe(f"  {title}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

def _pdf_table(pdf, headers, rows, col_widths=None, header_fill=(220, 230, 241)):
    """Render a table in the PDF with header and data rows."""
    if col_widths is None:
        total = pdf.w - pdf.l_margin - pdf.r_margin
        col_widths = [total / len(headers)] * len(headers)
    # Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*header_fill)
    for i, h in enumerate(headers):
        align = "R" if i > 0 else "L"
        pdf.cell(col_widths[i], 6, _pdf_safe(h), border=1, align=align, fill=True)
    pdf.ln()
    # Rows
    pdf.set_font("Helvetica", "", 8)
    for row_idx, row in enumerate(rows):
        fill = row_idx % 2 == 1
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for i, val in enumerate(row):
            align = "R" if i > 0 else "L"
            pdf.cell(col_widths[i], 5, _pdf_safe(val), border=1, align=align, fill=fill)
        pdf.ln()

def _pdf_kv_line(pdf, label, value, bold_value=False):
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(60, 5, _pdf_safe(label), new_x="END")
    pdf.set_font("Helvetica", "B" if bold_value else "", 9)
    pdf.cell(0, 5, _pdf_safe(value), new_x="LMARGIN", new_y="NEXT")

def generate_pdf_report(data):
    """Generate a PDF report from a data dict. Returns bytes.

    The caller must build `data` from st.session_state (or any other source)
    and pass it in.  This function has zero Streamlit dependency.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # Gather state
    base_res = data.get("base_results")
    base_inp = data.get("base_inputs")
    solved_res = data.get("last_solved_results")
    solved_inp = data.get("last_solved_inputs")
    tab3_rows = data.get("tab3_rows")
    tab3_mc = data.get("tab3_mc_results")
    tab3_params = data.get("tab3_params")
    p1_results = data.get("phase1_results")
    p1_best_order = data.get("phase1_best_order")
    p1_best_details = data.get("phase1_best_details")
    p2_results = data.get("phase2_results")
    p2_best_details = data.get("phase2_best_details")
    p2_best_name = data.get("phase2_best_name")
    tab5_conv_res = data.get("tab5_conv_res")
    tab5_actual_conversion = data.get("tab5_actual_conversion")
    tab5_conversion_room = data.get("tab5_conversion_room")
    tab5_total_additional_cost = data.get("tab5_total_additional_cost")

    # ==================== PAGE 1: COVER ====================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(40)
    pdf.cell(0, 15, _pdf_safe("OptiPlan"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, _pdf_safe("Wealth Optimization Report"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(50, pdf.get_y(), pdf.w - 50, pdf.get_y())
    pdf.ln(10)

    # Client name
    c_name = data.get("client_name", "")
    if c_name:
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _pdf_safe(c_name), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Filing info
    pdf.set_font("Helvetica", "", 11)
    if base_inp:
        pdf.cell(0, 7, _pdf_safe(f"Filing Status: {base_inp.get('filing_status', 'N/A')}"), align="C", new_x="LMARGIN", new_y="NEXT")
    filer_dob_val = data.get("filer_dob")
    spouse_dob_val = data.get("spouse_dob")
    if filer_dob_val:
        age_f = age_at_date(filer_dob_val, dt.date.today())
        age_line = f"Filer Age: {age_f}"
        if spouse_dob_val and base_inp and "joint" in base_inp.get("filing_status", "").lower():
            age_s = age_at_date(spouse_dob_val, dt.date.today())
            age_line += f"  |  Spouse Age: {age_s}"
        pdf.cell(0, 7, age_line, align="C", new_x="LMARGIN", new_y="NEXT")
    tax_yr = data.get("tax_year", "")
    if tax_yr:
        pdf.cell(0, 7, f"Tax Year: {tax_yr}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {dt.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ==================== PAGE 2: CURRENT TAX POSITION (Tab 1) ====================
    if base_res:
        pdf.add_page()
        _pdf_section_header(pdf, "Current Tax Position")

        # Income summary
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Income Summary", new_x="LMARGIN", new_y="NEXT")
        r = base_res
        ira_gross = r.get("taxable_ira", 0) + r.get("rmd_amount", 0)
        ira_taxable = r.get("taxable_ira", 0) + r.get("taxable_rmd", r.get("rmd_amount", 0))
        other = r.get("other_income", 0) + r.get("ordinary_tax_only", 0)
        income_rows = [
            ("Wages", r.get("wages", 0)),
            ("Social Security (gross)", r.get("gross_ss", 0)),
            ("Social Security (taxable)", r.get("taxable_ss", 0)),
            ("Pensions", r.get("taxable_pensions", 0)),
            ("IRA distributions (gross)", ira_gross),
            ("IRA distributions (taxable)", ira_taxable),
            ("Ordinary dividends", r.get("total_ordinary_dividends", 0)),
            ("Qualified dividends", r.get("qualified_dividends", 0)),
            ("Capital gain/(loss)", r.get("cap_gain_loss", 0)),
            ("Taxable interest", r.get("interest_taxable", 0)),
            ("Other income", other),
        ]
        income_rows = [(k, v) for k, v in income_rows if v != 0]
        income_rows.append(("Total Income", r.get("total_income_for_tax", 0)))
        _pdf_table(pdf,
                   ["Description", "Amount"],
                   [[desc, pdf_money(val)] for desc, val in income_rows],
                   col_widths=[100, 70])
        pdf.ln(3)

        # AGI & Deductions
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "AGI & Deductions", new_x="LMARGIN", new_y="NEXT")
        ded_type = "Itemized" if r.get("is_itemizing") else "Standard"
        agi_rows = [
            ["Adjustments", pdf_money(r.get("adjustments", 0))],
            ["Adjusted Gross Income (AGI)", pdf_money(r.get("agi", 0))],
            [f"{ded_type} Deduction", pdf_money(r.get("deduction_used", 0))],
        ]
        if r.get("is_itemizing"):
            agi_rows.append(["  - Mortgage interest", pdf_money(r.get("mortgage_interest", 0))])
            agi_rows.append(["  - SALT (capped $10k)", pdf_money(r.get("salt", 0))])
            if r.get("medical_deduction", 0) > 0:
                agi_rows.append(["  - Medical (less 7.5% AGI)", pdf_money(r.get("medical_deduction", 0))])
            agi_rows.append(["  - Charitable", pdf_money(r.get("charitable", 0))])
        agi_rows.append(["Federal Taxable Income", pdf_money(r.get("fed_taxable", 0))])
        _pdf_table(pdf, ["Description", "Amount"], agi_rows, col_widths=[100, 70])
        pdf.ln(3)

        # Tax summary
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Tax Summary", new_x="LMARGIN", new_y="NEXT")
        tax_rows = [
            ["Federal Tax", pdf_money(r.get("fed_tax", 0)), f"{r.get('effective_fed', 0)}%"],
            ["SC Tax", pdf_money(r.get("sc_tax", 0)), f"{r.get('effective_sc', 0)}%"],
            ["Total Tax", pdf_money(r.get("total_tax", 0)), ""],
            ["Medicare Premiums", pdf_money(r.get("medicare_premiums", 0)),
             "IRMAA" if r.get("has_irmaa") else "No IRMAA"],
            ["Total Tax + Medicare", pdf_money(r.get("total_tax", 0) + r.get("medicare_premiums", 0)), ""],
        ]
        _pdf_table(pdf, ["Tax", "Amount", "Rate/Status"], tax_rows, col_widths=[80, 50, 40])
        pdf.ln(3)

        # Cashflow
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Cashflow Summary", new_x="LMARGIN", new_y="NEXT")
        reinv = r.get("reinvested_amount", 0)
        cf_rows = [["Gross income received", pdf_money(r["spendable_gross"] + reinv)]]
        if reinv > 0:
            cf_rows.append(["Less: Reinvested (div/gains)", f"({pdf_money(reinv)})"])
            cf_rows.append(["Spendable income", pdf_money(r["spendable_gross"])])
        cf_rows.append(["Less: Federal tax", f"({pdf_money(r['fed_tax'])})"])
        cf_rows.append(["Less: SC tax", f"({pdf_money(r['sc_tax'])})"])
        cf_rows.append(["Less: Medicare", f"({pdf_money(r['medicare_premiums'])})"])
        mtg_pmt = float(base_inp.get("mortgage_payment", 0))
        if mtg_pmt > 0:
            cf_rows.append(["Less: Mortgage payment", f"({pdf_money(mtg_pmt)})"])
            cf_rows.append(["Net after tax & mortgage", pdf_money(r["net_after_tax"] - mtg_pmt)])
        else:
            cf_rows.append(["Net after tax", pdf_money(r["net_after_tax"])])
        _pdf_table(pdf, ["", "Amount"], cf_rows, col_widths=[100, 70])

    # ==================== PAGE 3: INCOME NEEDS (Tab 2) ====================
    if solved_res and solved_inp:
        pdf.add_page()
        _pdf_section_header(pdf, "Income Needs Analysis")

        net_needed_val = float(data.get("last_net_needed", 0))
        source_used = data.get("last_source", "N/A")

        withdrawal_amt = float(data.get("last_withdrawal_proceeds", 0))

        pdf.set_font("Helvetica", "", 10)
        _pdf_kv_line(pdf, "Net Income Needed:", pdf_money(net_needed_val), bold_value=True)
        _pdf_kv_line(pdf, "Funding Source:", source_used, bold_value=True)
        _pdf_kv_line(pdf, "Withdrawal Needed:", pdf_money(withdrawal_amt), bold_value=True)
        pdf.ln(3)

        if base_res:
            # Before/After comparison
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Tax Impact Comparison", new_x="LMARGIN", new_y="NEXT")
            delta_fed = solved_res["fed_tax"] - base_res["fed_tax"]
            delta_sc = solved_res["sc_tax"] - base_res["sc_tax"]
            delta_med = solved_res["medicare_premiums"] - base_res["medicare_premiums"]
            delta_total = delta_fed + delta_sc + delta_med
            cmp_rows = [
                ["Federal Tax", pdf_money(base_res["fed_tax"]), pdf_money(solved_res["fed_tax"]), pdf_money(delta_fed)],
                ["SC Tax", pdf_money(base_res["sc_tax"]), pdf_money(solved_res["sc_tax"]), pdf_money(delta_sc)],
                ["Medicare", pdf_money(base_res["medicare_premiums"]), pdf_money(solved_res["medicare_premiums"]), pdf_money(delta_med)],
                ["Total", pdf_money(base_res["total_tax"] + base_res["medicare_premiums"]),
                 pdf_money(solved_res["total_tax"] + solved_res["medicare_premiums"]), pdf_money(delta_total)],
            ]
            _pdf_table(pdf, ["", "Before", "After", "Change"], cmp_rows, col_widths=[50, 40, 40, 40])
            pdf.ln(3)

        # After-withdrawal tax return summary
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "After-Withdrawal Tax Summary", new_x="LMARGIN", new_y="NEXT")
        sr = solved_res
        summary_rows = [
            ["AGI", pdf_money(sr.get("agi", 0))],
            ["Federal Taxable Income", pdf_money(sr.get("fed_taxable", 0))],
            ["Federal Tax", pdf_money(sr.get("fed_tax", 0))],
            ["SC Tax", pdf_money(sr.get("sc_tax", 0))],
            ["Total Tax", pdf_money(sr.get("total_tax", 0))],
            ["Medicare Premiums", pdf_money(sr.get("medicare_premiums", 0))],
        ]
        _pdf_table(pdf, ["Metric", "Amount"], summary_rows, col_widths=[100, 70])
        pdf.ln(3)

        # Cashflow
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Cashflow After Withdrawal", new_x="LMARGIN", new_y="NEXT")
        cf2_rows = [
            ["Spendable gross", pdf_money(sr["spendable_gross"])],
            ["Less: Total tax", f"({pdf_money(sr['total_tax'])})"],
            ["Less: Medicare", f"({pdf_money(sr['medicare_premiums'])})"],
        ]
        mtg_pmt2 = float(solved_inp.get("mortgage_payment", 0))
        if mtg_pmt2 > 0:
            cf2_rows.append(["Less: Mortgage", f"({pdf_money(mtg_pmt2)})"])
            cf2_rows.append(["Net after tax & mortgage", pdf_money(sr["net_after_tax"] - mtg_pmt2)])
        else:
            cf2_rows.append(["Net after tax", pdf_money(sr["net_after_tax"])])
        _pdf_table(pdf, ["", "Amount"], cf2_rows, col_widths=[100, 70])

    # ==================== PAGES 4-5: WEALTH PROJECTION (Tab 3) ====================
    if tab3_rows and len(tab3_rows) > 0:
        pdf.add_page("L")  # Landscape for wide table
        _pdf_section_header(pdf, "Wealth Projection")

        # Assumptions
        if tab3_params:
            pdf.set_font("Helvetica", "", 8)
            tp = tab3_params
            assumptions = (
                f"Spending: {pdf_money(tp.get('spending_goal', 0))}  |  "
                f"Inflation: {tp.get('inflation', 0):.1%}  |  "
                f"Years: {tp.get('years', 0)}  |  "
                f"Heir Tax: {tp.get('heir_tax_rate', 0):.0%}  |  "
                f"Order: {tp.get('spending_order', 'N/A')}"
            )
            pdf.cell(0, 5, _pdf_safe(assumptions), new_x="LMARGIN", new_y="NEXT")
            growth = (
                f"Growth - Cash: {tp.get('r_cash', 0.02):.1%}  |  "
                f"Brokerage: {tp.get('r_taxable', 0):.1%}  |  "
                f"Pre-Tax: {tp.get('r_pretax', 0):.1%}  |  "
                f"Roth: {tp.get('r_roth', 0):.1%}  |  "
                f"Annuity: {tp.get('r_annuity', 0):.1%}  |  "
                f"Life: {tp.get('r_life', 0):.1%}"
            )
            pdf.cell(0, 5, _pdf_safe(growth), new_x="LMARGIN", new_y="NEXT")
            _pdf_ef = data.get("emergency_fund", 0.0)
            if _pdf_ef > 0:
                pdf.cell(0, 5, _pdf_safe(f"Emergency Fund Reserve: {pdf_money(_pdf_ef)} (excluded from available cash)"), new_x="LMARGIN", new_y="NEXT")
            # Additional expenses and future income in assumptions
            _pdf_ae = tp.get("additional_expenses", [])
            _pdf_fi = tp.get("future_income", [])
            if _pdf_ae:
                ae_lines = []
                for ae in _pdf_ae:
                    _eff_end = ae["end_age"] if ae["end_age"] > ae["start_age"] else ae["start_age"] + 1
                    _dur = f"age {ae['start_age']} (one-time)" if _eff_end <= ae["start_age"] + 1 else f"ages {ae['start_age']}-{_eff_end}"
                    ae_lines.append(f"{ae.get('name', 'Expense')}: {pdf_money(ae['net_amount'])} @ {_dur} from {ae['source']}")
                pdf.cell(0, 5, _pdf_safe("Addl Expenses: " + " | ".join(ae_lines)), new_x="LMARGIN", new_y="NEXT")
            if _pdf_fi:
                fi_lines = []
                for fi in _pdf_fi:
                    _eff_end = fi["end_age"] if fi["end_age"] > fi["start_age"] else "ongoing"
                    _tax_lbl = "taxable" if fi["taxable"] else "non-taxable"
                    fi_lines.append(f"{fi.get('name', 'Income')}: {pdf_money(fi['amount'])}/yr @ ages {fi['start_age']}-{_eff_end} ({_tax_lbl})")
                pdf.cell(0, 5, _pdf_safe("Future Income: " + " | ".join(fi_lines)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        # Select key columns for the table -- include Addl Expense and Extra Income if present
        display_cols = ["Year", "Age", "Spending", "Addl Expense", "Extra Income", "Fixed Inc", "IRA Dist",
                        "W/D Cash", "W/D Taxable", "W/D Pre-Tax", "W/D Tax-Free", "W/D Annuity",
                        "Taxes", "Medicare", "Portfolio", "Gross Estate", "Estate (Net)"]
        # Filter to columns that exist in the data
        available_cols = [c for c in display_cols if c in tab3_rows[0]]
        n_cols = len(available_cols)
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        # Year/Age get smaller widths, financial columns get more
        col_widths = []
        for c in available_cols:
            if c in ("Year", "Age"):
                col_widths.append(14)
            else:
                col_widths.append(max(16, (page_w - 28) / max(1, n_cols - 2)))
        # Normalize to fit page
        total_w = sum(col_widths)
        if total_w > page_w:
            scale = page_w / total_w
            col_widths = [w * scale for w in col_widths]

        header_labels = []
        for c in available_cols:
            short = c.replace("W/D ", "").replace("Gross Estate", "Gross Est").replace("Estate (Net)", "Net Est")
            header_labels.append(short)

        rows_data = []
        for row in tab3_rows:
            row_vals = []
            for c in available_cols:
                v = row.get(c, 0)
                if c in ("Year", "Age"):
                    row_vals.append(str(int(v)))
                else:
                    row_vals.append(pdf_money(v))
            rows_data.append(row_vals)

        _pdf_table(pdf, header_labels, rows_data, col_widths)

        # Summary metrics
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Wealth Projection Summary", new_x="LMARGIN", new_y="NEXT")
        last_row = tab3_rows[-1]
        total_taxes = sum(r.get("Taxes", 0) + r.get("Medicare", 0) for r in tab3_rows)
        summ_rows = [
            ["Final Portfolio", pdf_money(last_row.get("Portfolio", 0))],
            ["Final Estate (Net)", pdf_money(last_row.get("Estate (Net)", 0))],
            ["Total Taxes + Medicare", pdf_money(total_taxes)],
            ["Final Year Spending", pdf_money(last_row.get("Spending", 0))],
        ]
        _pdf_table(pdf, ["Metric", "Value"], summ_rows, col_widths=[80, 60])

        # Monte Carlo results
        if tab3_mc:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Monte Carlo Analysis", new_x="LMARGIN", new_y="NEXT")
            mc_rows = [
                ["Success Rate", f"{tab3_mc.get('success_rate', 0):.1f}%"],
                ["Median Ending Portfolio", pdf_money(tab3_mc.get("median", 0))],
                ["10th Percentile", pdf_money(tab3_mc.get("p10", 0))],
                ["90th Percentile", pdf_money(tab3_mc.get("p90", 0))],
            ]
            if tab3_mc.get("median_fail_year"):
                mc_rows.append(["Median Failure Year", str(tab3_mc["median_fail_year"])])
            _pdf_table(pdf, ["Metric", "Value"], mc_rows, col_widths=[80, 60])

    # ==================== ROTH CONVERSION OPPORTUNITY (Tab 5) ====================
    if tab5_conv_res and solved_res:
        pdf.add_page()
        _pdf_section_header(pdf, "Roth Conversion Opportunity")

        actual_conv = float(tab5_actual_conversion or 0)
        conv_room = float(tab5_conversion_room or 0)
        total_add_cost = float(tab5_total_additional_cost or 0)
        net_to_roth = actual_conv - total_add_cost
        eff_rate = (total_add_cost / actual_conv * 100) if actual_conv > 0 else 0

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Conversion Summary", new_x="LMARGIN", new_y="NEXT")
        conv_summ = [
            ["Conversion Room", pdf_money(conv_room)],
            ["Actual Conversion", pdf_money(actual_conv)],
            ["Additional Tax Cost", pdf_money(total_add_cost)],
            ["Net to Roth", pdf_money(net_to_roth)],
            ["Effective Tax Rate", f"{eff_rate:.1f}%"],
        ]
        _pdf_table(pdf, ["Metric", "Value"], conv_summ, col_widths=[80, 60])
        pdf.ln(3)

        # Before/After tax comparison
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Tax Comparison (Before vs After Conversion)", new_x="LMARGIN", new_y="NEXT")
        tax_cmp = [
            ["AGI", pdf_money(solved_res["agi"]), pdf_money(tab5_conv_res["agi"])],
            ["Federal Taxable", pdf_money(solved_res["fed_taxable"]), pdf_money(tab5_conv_res["fed_taxable"])],
            ["Federal Tax", pdf_money(solved_res["fed_tax"]), pdf_money(tab5_conv_res["fed_tax"])],
            ["SC Tax", pdf_money(solved_res["sc_tax"]), pdf_money(tab5_conv_res["sc_tax"])],
            ["Total Tax", pdf_money(solved_res["total_tax"]), pdf_money(tab5_conv_res["total_tax"])],
            ["Medicare", pdf_money(solved_res["medicare_premiums"]), pdf_money(tab5_conv_res["medicare_premiums"])],
        ]
        _pdf_table(pdf, ["Metric", "Before", "After"], tax_cmp, col_widths=[60, 50, 50])
        pdf.ln(3)

        # IRMAA status
        pdf.set_font("Helvetica", "", 9)
        before_irmaa = "IRMAA applies" if solved_res.get("has_irmaa") else "No IRMAA"
        after_irmaa = "IRMAA applies" if tab5_conv_res.get("has_irmaa") else "No IRMAA"
        pdf.cell(0, 5, _pdf_safe(f"IRMAA Status:  Before: {before_irmaa}  |  After: {after_irmaa}"), new_x="LMARGIN", new_y="NEXT")

        # Roth Conversion Projection (estate impact over time)
        if tab3_rows and len(tab3_rows) > 0:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Wealth Projection with Roth Conversion", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)

            # Show a summary projection table: key years
            proj_cols = ["Year", "Age", "Portfolio", "Gross Estate", "Estate (Net)"]
            avail_proj = [c for c in proj_cols if c in tab3_rows[0]]
            if avail_proj:
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                cw = [page_w / len(avail_proj)] * len(avail_proj)
                proj_display = []
                step = max(1, len(tab3_rows) // 10)
                for idx in range(0, len(tab3_rows), step):
                    row = tab3_rows[idx]
                    rv = []
                    for c in avail_proj:
                        v = row.get(c, 0)
                        rv.append(str(int(v)) if c in ("Year", "Age") else pdf_money(v))
                    proj_display.append(rv)
                # Always include last row
                if len(tab3_rows) - 1 not in range(0, len(tab3_rows), step):
                    row = tab3_rows[-1]
                    rv = []
                    for c in avail_proj:
                        v = row.get(c, 0)
                        rv.append(str(int(v)) if c in ("Year", "Age") else pdf_money(v))
                    proj_display.append(rv)
                _pdf_table(pdf, avail_proj, proj_display, cw)

    # ==================== INCOME OPTIMIZER (Tab 4) ====================
    if p1_results:
        pdf.add_page()
        _pdf_section_header(pdf, "Income Optimizer")

        best = p1_results[0]
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Phase 1: Optimal Spending Strategy", new_x="LMARGIN", new_y="NEXT")
        _pdf_kv_line(pdf, "Best Strategy:", best["waterfall"], bold_value=True)
        pdf.ln(2)

        opt_summ = [
            ["Gross Estate", pdf_money(best.get("gross_estate", best["total_wealth"]))],
            ["Net Estate (After Heir Tax)", pdf_money(best["after_tax_estate"])],
            ["Total Taxes + Medicare", pdf_money(best["total_taxes"])],
        ]
        if len(p1_results) > 1:
            worst = p1_results[-1]
            diff = best["after_tax_estate"] - worst["after_tax_estate"]
            opt_summ.append(["Worst Net Estate", pdf_money(worst["after_tax_estate"])])
            opt_summ.append(["Best vs Worst", pdf_money(diff)])
        _pdf_table(pdf, ["Metric", "Value"], opt_summ, col_widths=[80, 60])
        pdf.ln(3)

        if len(p1_results) > 1:
            # Top strategies ranking
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Strategy Rankings", new_x="LMARGIN", new_y="NEXT")
            top5_rows = []
            for i, r in enumerate(p1_results[:5]):
                top5_rows.append([
                    str(i + 1), r["waterfall"],
                    pdf_money(r["after_tax_estate"]), pdf_money(r["total_taxes"]),
                ])
            _pdf_table(pdf, ["Rank", "Waterfall", "Net Estate", "Total Taxes"], top5_rows,
                       col_widths=[15, 80, 40, 35])

        # Phase 2 results
        if p2_results:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Phase 2: Roth Conversion Strategy", new_x="LMARGIN", new_y="NEXT")
            if p2_best_name:
                _pdf_kv_line(pdf, "Best Strategy:", p2_best_name, bold_value=True)
            p2_sorted = sorted(p2_results, key=lambda x: x["after_tax_estate"], reverse=True)
            p2_table_rows = []
            for r in p2_sorted:
                p2_table_rows.append([
                    r["strategy_name"],
                    pdf_money(r["after_tax_estate"]),
                    pdf_money(r["improvement"]),
                    pdf_money(r["total_converted"]),
                ])
            _pdf_table(pdf, ["Strategy", "Net Estate", "vs Baseline", "Total Converted"], p2_table_rows,
                       col_widths=[55, 40, 35, 40])

        # Year-by-year projection for best optimizer result
        best_details = p2_best_details if p2_best_details else p1_best_details
        if best_details and len(best_details) > 0:
            pdf.add_page("L")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Projection Summary", new_x="LMARGIN", new_y="NEXT")
            opt_proj_cols = ["Year", "Age", "Spending", "Fixed Inc", "IRA Dist",
                             "W/D Cash", "W/D Taxable", "W/D Pre-Tax", "W/D Tax-Free", "W/D Annuity",
                             "Taxes", "Medicare", "Portfolio", "Gross Estate", "Estate (Net)"]
            avail_opt = [c for c in opt_proj_cols if c in best_details[0]]
            n_oc = len(avail_opt)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            opt_cw = []
            for c in avail_opt:
                if c in ("Year", "Age"):
                    opt_cw.append(14)
                else:
                    opt_cw.append(max(16, (page_w - 28) / max(1, n_oc - 2)))
            total_ow = sum(opt_cw)
            if total_ow > page_w:
                scale = page_w / total_ow
                opt_cw = [w * scale for w in opt_cw]
            opt_headers = [c.replace("W/D ", "").replace("Gross Estate", "Gross Est").replace("Estate (Net)", "Net Est") for c in avail_opt]
            opt_rows = []
            for row in best_details:
                rv = []
                for c in avail_opt:
                    v = row.get(c, 0)
                    rv.append(str(int(v)) if c in ("Year", "Age") else pdf_money(v))
                opt_rows.append(rv)
            _pdf_table(pdf, opt_headers, opt_rows, opt_cw)

    # ==================== FOOTER ====================
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.set_y(-10)
    pdf.cell(0, 5, "RTF Financial Planning Report - For planning purposes only. Not tax advice.", align="C")

    return bytes(pdf.output())