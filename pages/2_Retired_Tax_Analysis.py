import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import json, os, io
from functools import lru_cache
from itertools import permutations
from fpdf import FPDF

st.set_page_config(page_title="Retired Tax Analysis", layout="wide")

DEFAULT_STATE = {
    "base_results": None, "base_inputs": None, "assets": None,
    "gross_from_needs": None, "last_net_needed": None, "last_taxes_paid_by_cash": None,
    "last_source": None, "last_withdrawal_proceeds": None,
    "last_solved_results": None, "last_solved_inputs": None, "last_solved_assets": None,
    "phase1_results": None, "phase1_best_order": None, "phase1_best_details": None,
    "phase1_params": None, "phase2_results": None, "phase2_best_details": None,
    "phase2_baseline_details": None, "phase2_best_name": None,
    "tab5_conv_res": None, "tab5_conv_inputs": None, "tab5_actual_conversion": None,
    "tab5_conversion_room": None, "tab5_total_additional_cost": None,
    "tab3_rows": None, "tab3_mc_results": None, "tab3_params": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Client Profile Save / Load ----------
_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tax_profiles")
os.makedirs(_PROFILE_DIR, exist_ok=True)

_PROFILE_KEYS = [
    "tax_year", "filing_status", "inflation",
    "enter_dobs", "filer_dob", "spouse_dob",
    "filer_ss_already", "filer_ss_current", "filer_ss_start_year", "filer_ss_fra", "filer_ss_claim",
    "spouse_ss_already", "spouse_ss_current", "spouse_ss_start_year", "spouse_ss_fra", "spouse_ss_claim",
    "pension_filer", "pension_spouse", "pension_cola",
    "auto_rmd", "pretax_bal_filer_prior", "pretax_bal_spouse_prior", "baseline_pretax_dist", "rmd_manual",
    "wages", "tax_exempt_interest", "interest_taxable",
    "total_ordinary_div", "qualified_div", "reinvest_dividends",
    "cap_gain_loss", "reinvest_cap_gains", "other_income",
    "filer_65_plus", "spouse_65_plus",
    "adjustments", "dependents", "retirement_deduction", "out_of_state_gain",
    "home_value", "home_appreciation",
    "mortgage_balance", "mortgage_rate", "mortgage_payment",
    "property_tax", "medical_expenses", "charitable",
    "taxable_cash_bal", "taxable_brokerage_bal", "brokerage_gain_pct",
    "pretax_bal_filer_current", "pretax_bal_spouse_current", "roth_bal", "life_cash_value", "annuity_value", "annuity_basis",
    "r_taxable_side", "r_pretax_side", "r_roth_side", "r_annuity_side", "r_life_side",
]
_DATE_KEYS = {"filer_dob", "spouse_dob"}

def _collect_profile():
    data = {}
    for k in _PROFILE_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            if isinstance(v, dt.date):
                data[k] = v.isoformat()
            elif v is not None:
                data[k] = v
    return data

def _apply_profile(data):
    for k, v in data.items():
        if k in _DATE_KEYS and v is not None:
            st.session_state[k] = dt.date.fromisoformat(v)
        else:
            st.session_state[k] = v

def age_at_date(dob, asof):
    if dob is None: return None
    years = asof.year - dob.year
    if (asof.month, asof.day) < (dob.month, dob.day): years -= 1
    return years

def money(x): return f"${float(x):,.2f}"

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

def ss_claim_factor(choice):
    c = (choice or "").strip().lower()
    if c == "62": return 0.70
    if c == "fra": return 1.00
    if c == "70": return 1.24
    return 1.00

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

UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
    88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
    104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4,
    112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
}

def compute_rmd_uniform_start73(balance, age):
    if balance <= 0 or age is None or age < 73: return 0.0
    factor = UNIFORM_LIFETIME.get(age, UNIFORM_LIFETIME[120])
    return float(balance) / float(factor)

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

def get_federal_enhanced_extra(agi, filer_65, spouse_65, status, inf=1.0):
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

def calculate_sc_tax(fed_taxable, dependents, taxable_ss, out_of_state_gain, filer_65_plus, spouse_65_plus, retirement_deduction, cap_gain_loss):
    sc_start = max(0.0, float(fed_taxable))
    sub = max(0.0, float(taxable_ss)) + max(0.0, float(retirement_deduction))
    if filer_65_plus: sub += 5000.0
    if spouse_65_plus: sub += 5000.0
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

def estimate_medicare_premiums(agi, filing_status, inf=1.0):
    is_joint = "joint" in filing_status.lower()
    threshold_1 = (212000 if is_joint else 106000) * inf
    threshold_2 = (266000 if is_joint else 133000) * inf
    base = (195 + 35) * 12 * inf
    irmaa = 0.0
    if float(agi) > threshold_2: irmaa = (185 + 33) * 12 * inf
    elif float(agi) > threshold_1: irmaa = (75 + 13) * 12 * inf
    people = 2 if is_joint else 1
    return (base + irmaa) * people, irmaa > 0

def compute_taxes_only(gross_ss, taxable_pensions, rmd_amount, taxable_ira, conversion_amount,
                       ordinary_income, cap_gains, filing_status, filer_65, spouse_65,
                       retirement_deduction, inf_factor=1.0):
    base_non_ss = taxable_pensions + rmd_amount + taxable_ira + conversion_amount + ordinary_income + cap_gains
    taxable_ss = calculate_taxable_ss(base_non_ss, 0.0, gross_ss, filing_status)
    agi = base_non_ss + taxable_ss
    base_std = get_federal_base_std(filing_status, inf_factor)
    trad_extra = get_federal_traditional_extra(filing_status, filer_65, spouse_65, inf_factor)
    enh_extra = get_federal_enhanced_extra(agi, filer_65, spouse_65, filing_status, inf_factor)
    deduction = base_std + trad_extra + enh_extra
    fed_taxable = max(0.0, agi - deduction)
    preferential = max(0.0, cap_gains)
    fed = calculate_federal_tax(fed_taxable, preferential, filing_status, inf_factor)
    sc = calculate_sc_tax(fed_taxable, 0, taxable_ss, 0.0, filer_65, spouse_65, 
                          retirement_deduction * inf_factor, cap_gains)
    medicare, has_irmaa = estimate_medicare_premiums(agi, filing_status, inf_factor)
    total_tax = fed["federal_tax"] + sc["sc_tax"]
    return {
        "agi": agi, "fed_taxable": fed_taxable, "fed_tax": fed["federal_tax"],
        "sc_tax": sc["sc_tax"], "total_tax": total_tax, "medicare": medicare,
        "has_irmaa": has_irmaa, "total_outflow": total_tax + medicare
    }

def compute_case(inputs, inflation_factor=1.0):
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

    base_non_ss = wages + interest_taxable + total_ordinary_dividends + taxable_ira + rmd_amount + taxable_pensions + ordinary_tax_only + cap_gain_loss + other_income
    taxable_ss = calculate_taxable_ss(base_non_ss, tax_exempt_interest, gross_ss, filing_status)
    total_income_for_tax = base_non_ss + taxable_ss
    agi = max(0.0, total_income_for_tax - adjustments)
    base_std = get_federal_base_std(filing_status, inflation_factor)
    traditional_extra = get_federal_traditional_extra(filing_status, filer_65_plus, spouse_65_plus, inflation_factor)
    enhanced_extra = get_federal_enhanced_extra(agi, filer_65_plus, spouse_65_plus, filing_status, inflation_factor)
    fed_std = base_std + traditional_extra + enhanced_extra

    # Calculate itemized deductions from components
    mortgage_interest = calc_mortgage_interest_for_year(mtg_balance, mtg_rate, mtg_payment)[0] if mtg_balance > 0 else 0.0
    # SALT: estimate state tax using standard deduction, add property tax, cap at $10k
    est_sc = calculate_sc_tax(max(0.0, agi - fed_std), dependents, taxable_ss, out_of_state_gain,
                              filer_65_plus, spouse_65_plus, retirement_deduction, cap_gain_loss)
    salt = min(10000.0 * inflation_factor, est_sc["sc_tax"] + prop_tax)
    # Medical: amount exceeding 7.5% of AGI
    medical_deduction = max(0.0, medical_exp - agi * 0.075)
    itemized_total = mortgage_interest + salt + medical_deduction + charitable_amt
    is_itemizing = itemized_total > fed_std
    deduction_used = itemized_total if is_itemizing else fed_std

    fed_taxable = max(0.0, agi - deduction_used)
    preferential_amount = qualified_dividends + max(0.0, cap_gain_loss)
    fed = calculate_federal_tax(fed_taxable, preferential_amount, filing_status, inflation_factor)
    sc = calculate_sc_tax(fed_taxable, dependents, taxable_ss, out_of_state_gain, filer_65_plus, spouse_65_plus, retirement_deduction, cap_gain_loss)
    total_tax = fed["federal_tax"] + sc["sc_tax"]
    medicare_premiums, has_irmaa = estimate_medicare_premiums(agi, filing_status, inflation_factor)
    reinvest_div = bool(inputs.get("reinvest_dividends", False))
    reinvest_cg = bool(inputs.get("reinvest_cap_gains", False))
    reinvested_amount = 0.0
    if reinvest_div:
        reinvested_amount += total_ordinary_dividends
    if reinvest_cg and cap_gain_loss > 0:
        reinvested_amount += cap_gain_loss
    spendable_gross = wages + gross_ss + taxable_pensions + total_ordinary_dividends + taxable_ira + rmd_amount + other_income + cashflow_taxfree + brokerage_proceeds + annuity_proceeds
    spendable_gross -= reinvested_amount
    net_before_tax = spendable_gross - medicare_premiums
    net_after_tax = spendable_gross - medicare_premiums - total_tax
    return {
        # Income components
        "wages": wages, "interest_taxable": interest_taxable, "tax_exempt_interest": tax_exempt_interest,
        "total_ordinary_dividends": total_ordinary_dividends, "qualified_dividends": qualified_dividends,
        "taxable_ira": taxable_ira, "rmd_amount": rmd_amount, "taxable_pensions": taxable_pensions,
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
def compute_case_cached(serialized_inputs, inflation_factor=1.0):
    inputs = {k: v for k, v in serialized_inputs}
    return compute_case(inputs, inflation_factor)

def annuity_gains(val, basis): return max(0.0, float(val) - float(basis))

def apply_withdrawal(base_inputs, base_assets, source, amount, gain_pct):
    inp = dict(base_inputs); assets = {k: dict(v) for k, v in base_assets.items()}
    for k in ["cashflow_taxfree", "brokerage_proceeds", "annuity_proceeds", "ordinary_tax_only"]:
        inp[k] = float(inp.get(k, 0.0))
    amt = max(0.0, float(amount))
    if amt == 0.0: return inp, assets
    if source == "Taxable – Cash":
        amt = min(amt, assets["taxable"]["cash"]); assets["taxable"]["cash"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Taxable – Brokerage":
        amt = min(amt, assets["taxable"]["brokerage"]); assets["taxable"]["brokerage"] -= amt
        inp["cap_gain_loss"] = float(inp.get("cap_gain_loss", 0.0)) + amt * max(0.0, min(1.0, float(gain_pct))); inp["brokerage_proceeds"] += amt
    elif source == "Pre-Tax – IRA/401k":
        amt = min(amt, assets["pretax"]["balance"]); assets["pretax"]["balance"] -= amt
        inp["taxable_ira"] = float(inp.get("taxable_ira", 0.0)) + amt
    elif source == "Roth":
        amt = min(amt, assets["taxfree"]["roth"]); assets["taxfree"]["roth"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Life Insurance (loan)":
        amt = min(amt, assets["taxfree"]["life_cash_value"]); assets["taxfree"]["life_cash_value"] -= amt; inp["cashflow_taxfree"] += amt
    elif source == "Annuity":
        value = assets["annuity"]["value"]; basis = assets["annuity"]["basis"]; amt = min(amt, value)
        if amt > 0:
            gains = annuity_gains(value, basis); taxable_ordinary = min(amt, gains); nontax_basis = amt - taxable_ordinary
            value -= amt; basis = max(0.0, basis - nontax_basis)
            assets["annuity"]["value"] = value; assets["annuity"]["basis"] = basis
            inp["ordinary_tax_only"] += taxable_ordinary; inp["annuity_proceeds"] += amt
    return inp, assets

def max_withdrawable(assets, source):
    if source == "Taxable – Cash": return assets["taxable"]["cash"]
    if source == "Taxable – Brokerage": return assets["taxable"]["brokerage"]
    if source == "Pre-Tax – IRA/401k": return assets["pretax"]["balance"]
    if source == "Roth": return assets["taxfree"]["roth"]
    if source == "Life Insurance (loan)": return assets["taxfree"]["life_cash_value"]
    if source == "Annuity": return assets["annuity"]["value"]
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

def display_tax_return(r, mortgage_pmt=0.0, filer_65=False, spouse_65=False):
    """Display the full 1040-style tax return for a compute_case result."""
    def _amt(val):
        return money(val) if val != 0 else "\u2014"

    # --- Federal 1040-Style Summary ---
    st.markdown("### Federal Return Summary (Form 1040)")
    _ira_dist = r.get("taxable_ira", 0) + r.get("rmd_amount", 0)
    _other = r.get("other_income", 0) + r.get("ordinary_tax_only", 0)
    _inc_table = "| Line | Description | Amount |\n|-----:|:------------|-------:|\n"
    _inc_table += f"| 1 | Wages, salaries, tips | {_amt(r.get('wages', 0))} |\n"
    _inc_table += f"| 2a | Tax-exempt interest | {_amt(r.get('tax_exempt_interest', 0))} |\n"
    _inc_table += f"| 2b | Taxable interest | {_amt(r.get('interest_taxable', 0))} |\n"
    _inc_table += f"| 3a | Qualified dividends | {_amt(r.get('qualified_dividends', 0))} |\n"
    _inc_table += f"| 3b | Ordinary dividends | {_amt(r.get('total_ordinary_dividends', 0))} |\n"
    _inc_table += f"| 4a | IRA distributions (gross) | {_amt(_ira_dist)} |\n"
    _inc_table += f"| 4b | IRA distributions (taxable) | {_amt(_ira_dist)} |\n"
    _inc_table += f"| 5a | Pensions and annuities (gross) | {_amt(r.get('taxable_pensions', 0))} |\n"
    _inc_table += f"| 5b | Pensions and annuities (taxable) | {_amt(r.get('taxable_pensions', 0))} |\n"
    _inc_table += f"| 6a | Social Security benefits (gross) | {_amt(r.get('gross_ss', 0))} |\n"
    _inc_table += f"| 6b | Social Security benefits (taxable) | {_amt(r.get('taxable_ss', 0))} |\n"
    _inc_table += f"| 7 | Capital gain / (loss) | {_amt(r.get('cap_gain_loss', 0))} |\n"
    _inc_table += f"| 8 | Other income | {_amt(_other)} |\n"
    _inc_table += f"| **9** | **Total income** | **{_amt(r.get('total_income_for_tax', 0))}** |\n"
    st.markdown(_inc_table)

    # Adjustments & AGI
    _agi_table = "| Line | Description | Amount |\n|-----:|:------------|-------:|\n"
    _agi_table += f"| 10 | Adjustments to income | {_amt(r.get('adjustments', 0))} |\n"
    _agi_table += f"| **11** | **Adjusted gross income (AGI)** | **{_amt(r.get('agi', 0))}** |\n"
    st.markdown(_agi_table)

    # Deductions
    _ded_type = "Itemized" if r.get("is_itemizing") else "Standard"
    _ded_table = "| Line | Description | Amount |\n|-----:|:------------|-------:|\n"
    _ded_table += f"| 12 | {_ded_type} deduction | {_amt(r.get('deduction_used', 0))} |\n"
    if r.get("is_itemizing"):
        _ded_table += f"| | *\u2014 Mortgage interest* | *{_amt(r.get('mortgage_interest', 0))}* |\n"
        _ded_table += f"| | *\u2014 SALT (capped at $10k)* | *{_amt(r.get('salt', 0))}* |\n"
        _med_exp = r.get("medical_expenses", 0)
        _med_ded = r.get("medical_deduction", 0)
        if _med_exp > 0:
            _ded_table += f"| | *\u2014 Medical (less 7.5% AGI floor)* | *{_amt(_med_ded)}* |\n"
        _ded_table += f"| | *\u2014 Charitable contributions* | *{_amt(r.get('charitable', 0))}* |\n"
        _ded_table += f"| | *Standard deduction would be* | *{_amt(r.get('fed_std', 0))}* |\n"
    else:
        _item_total = r.get("itemized_total", 0)
        if _item_total > 0:
            _ded_table += f"| | *Itemized total (did not exceed standard)* | *{_amt(_item_total)}* |\n"
    _ded_table += f"| 13 | Qualified business income deduction | \u2014 |\n"
    _ded_table += f"| 14 | Total deductions | {_amt(r.get('deduction_used', 0))} |\n"
    _ded_table += f"| **15** | **Taxable income** | **{_amt(r.get('fed_taxable', 0))}** |\n"
    st.markdown(_ded_table)

    # Tax
    _tax_table = "| Line | Description | Amount |\n|-----:|:------------|-------:|\n"
    _tax_table += f"| 16 | Tax | {_amt(r.get('fed_tax', 0))} |\n"
    _tax_table += f"| **24** | **Total federal tax** | **{_amt(r.get('fed_tax', 0))}** |\n"
    st.markdown(_tax_table)
    st.markdown(f"Effective federal rate: **{r.get('effective_fed', 0)}%**")

    # --- SC Return Summary ---
    st.divider()
    st.markdown("### South Carolina Return Summary")
    _sc_table = "| Description | Amount |\n|:------------|-------:|\n"
    _sc_table += f"| Federal taxable income | {_amt(r.get('fed_taxable', 0))} |\n"
    _sc_dep = r.get("dependents", 0)
    if _sc_dep > 0:
        _sc_table += f"| Less: dependent exemption ({_sc_dep} x $4,930) | ({_amt(_sc_dep * 4930)}) |\n"
    _sc_table += f"| Less: taxable SS (SC exempt) | ({_amt(r.get('taxable_ss', 0))}) |\n"
    _ret_ded = r.get("retirement_deduction", 0)
    if _ret_ded > 0:
        _sc_table += f"| Less: retirement deduction | ({_amt(_ret_ded)}) |\n"
    _65_count = int(bool(filer_65)) + int(bool(spouse_65))
    if _65_count > 0:
        _sc_table += f"| Less: 65+ deduction | ({_amt(5000 * _65_count)}) |\n"
    _cg = r.get("cap_gain_loss", 0)
    if _cg > 0:
        _sc_table += f"| Less: 44% capital gains exclusion | ({_amt(_cg * 0.44)}) |\n"
    _sc_table += f"| **SC taxable income** | **{_amt(r.get('sc_taxable', 0))}** |\n"
    _sc_table += f"| **SC income tax** | **{_amt(r.get('sc_tax', 0))}** |\n"
    st.markdown(_sc_table)
    st.markdown(f"Effective SC rate: **{r.get('effective_sc', 0)}%**")

    # --- Combined Summary ---
    st.divider()
    st.markdown("### Combined Tax Summary")
    _sum_c1, _sum_c2, _sum_c3 = st.columns(3)
    with _sum_c1:
        st.metric("Federal Tax", money(r["fed_tax"]))
        st.metric("SC Tax", money(r["sc_tax"]))
    with _sum_c2:
        st.metric("Total Tax", money(r["total_tax"]))
        st.metric("Medicare Premiums", money(r["medicare_premiums"]))
        if r.get("has_irmaa"):
            st.warning("IRMAA surcharge applies")
    with _sum_c3:
        st.metric("Total Tax + Medicare", money(r["total_tax"] + r["medicare_premiums"]))

    # --- Cashflow View ---
    st.divider()
    st.markdown("### Cashflow Summary")
    _cf_table = "| | Amount |\n|:---|-------:|\n"
    _reinv = r.get("reinvested_amount", 0)
    _cf_table += f"| Gross income received | {_amt(r['spendable_gross'] + _reinv)} |\n"
    if _reinv > 0:
        _cf_table += f"| Less: Reinvested (dividends/gains) | ({_amt(_reinv)}) |\n"
        _cf_table += f"| **Spendable income** | **{_amt(r['spendable_gross'])}** |\n"
    _cf_table += f"| Less: Federal tax | ({_amt(r['fed_tax'])}) |\n"
    _cf_table += f"| Less: SC tax | ({_amt(r['sc_tax'])}) |\n"
    _cf_table += f"| Less: Medicare premiums | ({_amt(r['medicare_premiums'])}) |\n"
    if mortgage_pmt > 0:
        _cf_table += f"| Less: Mortgage payment | ({_amt(mortgage_pmt)}) |\n"
        _cf_table += f"| **Net after tax & mortgage** | **{_amt(r['net_after_tax'] - mortgage_pmt)}** |\n"
    else:
        _cf_table += f"| **Net after tax** | **{_amt(r['net_after_tax'])}** |\n"
    st.markdown(_cf_table)

def display_cashflow_comparison(before_inp, before_res, after_inp, after_res, net_needed, roth_conversion=0.0, title_before="Before", title_after="After", mortgage_payment=0.0):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {title_before}")
        st.markdown("**Income (Cash Received)**")
        pretax_dist_before = float(before_inp.get("taxable_ira", 0.0)) + float(before_inp.get("rmd_amount", 0.0))
        income_before = [
            ("Wages", float(before_inp.get("wages", 0.0))),
            ("Social Security (gross)", float(before_inp.get("gross_ss", 0.0))),
            ("Pensions", float(before_inp.get("taxable_pensions", 0.0))),
            ("Dividends", float(before_inp.get("total_ordinary_dividends", 0.0))),
            ("Pre-tax dist", pretax_dist_before),
            ("Other taxable", float(before_inp.get("other_income", 0.0))),
            ("Tax-free cash", float(before_inp.get("cashflow_taxfree", 0.0))),
            ("Brokerage proceeds", float(before_inp.get("brokerage_proceeds", 0.0))),
            ("Annuity proceeds", float(before_inp.get("annuity_proceeds", 0.0))),
        ]
        income_before = [(k, v) for k, v in income_before if v > 0]
        total_inc_before = sum(x[1] for x in income_before)
        st.dataframe([{"Income": k, "Amount": money(v)} for k, v in income_before], use_container_width=True, hide_index=True)
        st.metric("Total Income", money(total_inc_before))
        st.markdown("**Expenses & Outflows**")
        out_before = [("Living expenses", net_needed), ("Mortgage payment", mortgage_payment), ("Medicare", float(before_res.get("medicare_premiums", 0.0))), ("Taxes", float(before_res.get("total_tax", 0.0)))]
        out_before = [(k, v) for k, v in out_before if v > 0]
        total_out_before = sum(x[1] for x in out_before)
        st.dataframe([{"Expense": k, "Amount": money(v)} for k, v in out_before], use_container_width=True, hide_index=True)
        st.metric("Total Outflows", money(total_out_before))
    with col2:
        st.markdown(f"### {title_after}")
        st.markdown("**Income (Cash Received)**")
        total_taxable_ira_after = float(after_inp.get("taxable_ira", 0.0))
        rmd_after = float(after_inp.get("rmd_amount", 0.0))
        regular_pretax_after = total_taxable_ira_after - roth_conversion + rmd_after
        income_after = [
            ("Wages", float(after_inp.get("wages", 0.0))),
            ("Social Security (gross)", float(after_inp.get("gross_ss", 0.0))),
            ("Pensions", float(after_inp.get("taxable_pensions", 0.0))),
            ("Dividends", float(after_inp.get("total_ordinary_dividends", 0.0))),
            ("Pre-tax dist", regular_pretax_after),
            ("Roth conv dist", roth_conversion),
            ("Other taxable", float(after_inp.get("other_income", 0.0))),
            ("Tax-free cash", float(after_inp.get("cashflow_taxfree", 0.0))),
            ("Brokerage proceeds", float(after_inp.get("brokerage_proceeds", 0.0))),
            ("Annuity proceeds", float(after_inp.get("annuity_proceeds", 0.0))),
        ]
        income_after = [(k, v) for k, v in income_after if v > 0]
        total_inc_after = sum(x[1] for x in income_after)
        st.dataframe([{"Income": k, "Amount": money(v)} for k, v in income_after], use_container_width=True, hide_index=True)
        st.metric("Total Income", money(total_inc_after))
        st.markdown("**Expenses & Outflows**")
        out_after = [("Living expenses", net_needed), ("Mortgage payment", mortgage_payment), ("Medicare", float(after_res.get("medicare_premiums", 0.0))), ("Taxes", float(after_res.get("total_tax", 0.0)))]
        out_after = [(k, v) for k, v in out_after if v > 0]
        total_out_after = sum(x[1] for x in out_after)
        st.dataframe([{"Expense": k, "Amount": money(v)} for k, v in out_after], use_container_width=True, hide_index=True)
        st.metric("Total Outflows", money(total_out_after))

def _fill_shortfall_dynamic(total_spend_need, cash_received, balances,
                             base_year_inp, p_base_cap_gain, inf_factor,
                             conversion_this_year):
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
        curr_res = compute_case_cached(_serialize_inputs_for_cache(curr_inp), inf_factor)
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
        for fill_round in range(6):
            if shortfall <= 1.0:
                break

            PROBE = min(1000.0, shortfall)
            candidates = []

            avail_brok = balances["brokerage"] - wd_brokerage
            if avail_brok > 0:
                p = min(PROBE, avail_brok)
                test_cg = (wd_brokerage + p) * dyn_gain_pct
                test_inp = _build_inp(wd_pretax, test_cg, ann_gains_withdrawn)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor)
                cost = (_tax_total(test_res) - curr_tax) / p
                candidates.append(("brokerage", cost, avail_brok))

            avail_pre = balances["pretax"] - wd_pretax
            if avail_pre > 0:
                p = min(PROBE, avail_pre)
                test_inp = _build_inp(wd_pretax + p, cap_gains_realized, ann_gains_withdrawn)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor)
                cost = (_tax_total(test_res) - curr_tax) / p
                candidates.append(("pretax", cost, avail_pre))

            avail_ann = balances["annuity_value"] - wd_annuity
            if avail_ann > 0:
                p = min(PROBE, avail_ann)
                rem_gains = max(0.0, ann_total_gains - ann_gains_withdrawn)
                test_ag = ann_gains_withdrawn + min(p, rem_gains)
                test_inp = _build_inp(wd_pretax, cap_gains_realized, test_ag)
                test_res = compute_case_cached(_serialize_inputs_for_cache(test_inp), inf_factor)
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
                test_res_full = compute_case_cached(_serialize_inputs_for_cache(test_inp_full), inf_factor)
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
                        test_res_mid = compute_case_cached(_serialize_inputs_for_cache(test_inp_mid), inf_factor)
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
            curr_res = compute_case_cached(_serialize_inputs_for_cache(curr_inp), inf_factor)
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
    final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), inf_factor)

    return (wd_cash, wd_brokerage, wd_pretax, wd_roth, wd_life, wd_annuity,
            ann_gains_withdrawn, cap_gains_realized, final_res)


def run_wealth_projection(initial_assets, params, spending_order, conversion_strategy="none",
                           target_agi=0, stop_conversion_age=100, conversion_years_limit=0,
                           blend_mode=False):
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
    pension_cola = params["pension_cola"]
    heir_tax_rate = params["heir_tax_rate"]
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

    # Investment income detail (for full tax calc)
    p_interest_taxable = params.get("interest_taxable", 0.0)
    p_total_ordinary_div = params.get("total_ordinary_dividends", 0.0)
    p_qualified_div = params.get("qualified_dividends", 0.0)
    p_base_cap_gain = params.get("cap_gain_loss", 0.0)
    p_reinvest_div = params.get("reinvest_dividends", False)
    p_reinvest_cg = params.get("reinvest_cap_gains", False)

    # Deduction inputs
    p_retirement_deduction = params.get("retirement_deduction", 0.0)
    p_out_of_state_gain = params.get("out_of_state_gain", 0.0)
    p_dependents = params.get("dependents", 0)
    p_property_tax = params.get("property_tax", 0.0)
    p_medical_expenses = params.get("medical_expenses", 0.0)
    p_charitable = params.get("charitable", 0.0)

    # Mortgage
    mtg_balance = params.get("mortgage_balance", 0.0)
    mtg_rate = params.get("mortgage_rate", 0.0)
    mtg_payment = params.get("mortgage_payment", 0.0)

    # Home
    home_val = params.get("home_value", 0.0)
    home_appr = params.get("home_appreciation", 0.0)

    # Initialize balances
    curr_cash = initial_assets["taxable"]["cash"]
    curr_brokerage = initial_assets["taxable"]["brokerage"]
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

    # Investment income: compute reinvested vs spendable (fixed, not inflated)
    reinvested_base = 0.0
    if p_reinvest_div:
        reinvested_base += p_total_ordinary_div
    if p_reinvest_cg and p_base_cap_gain > 0:
        reinvested_base += p_base_cap_gain
    spendable_inv_base = (p_interest_taxable + p_total_ordinary_div + p_base_cap_gain) - reinvested_base

    total_taxes_paid = 0.0
    total_converted = 0.0
    year_details = []

    do_conversions = conversion_strategy != "none" and conversion_strategy != 0 and conversion_strategy != 0.0

    for i in range(years):
        yr = start_year + i
        age_f = current_age_filer + i
        age_s = (current_age_spouse + i) if current_age_spouse else None
        inf_factor = (1 + inflation) ** i
        filer_65 = age_f >= 65
        spouse_65 = age_s >= 65 if age_s else False

        # Spending
        yr_mtg_payment = mtg_payment if curr_mtg_bal > 0 else 0.0
        total_spend_need = base_non_mtg * inf_factor + yr_mtg_payment

        # Fixed income
        ss_now = gross_ss_total * inf_factor
        pen_now = taxable_pensions_total * ((1 + pension_cola) ** i)

        # Investment income: spendable portion is cash received
        spendable_inv = spendable_inv_base
        curr_cash += spendable_inv

        # Pre-tax: RMD
        boy_pretax = curr_pre_filer + curr_pre_spouse
        rmd_f = compute_rmd_uniform_start73(curr_pre_filer, age_f)
        rmd_s = compute_rmd_uniform_start73(curr_pre_spouse, age_s)
        rmd_total = rmd_f + rmd_s
        curr_pre_filer -= rmd_f
        curr_pre_spouse -= rmd_s

        # Roth conversion (after RMD, before waterfall)
        conversion_this_year = 0.0
        if do_conversions and i < conversion_years_limit and age_f < stop_conversion_age:
            avail_pretax = curr_pre_filer + curr_pre_spouse
            if avail_pretax > 0:
                base_taxable = pen_now + rmd_total + p_interest_taxable + p_total_ordinary_div + p_base_cap_gain
                if conversion_strategy == "fill_to_target":
                    room = max(0.0, target_agi * inf_factor - base_taxable - ss_now * 0.85)
                    conversion_this_year = min(room, avail_pretax)
                else:
                    conversion_this_year = min(float(conversion_strategy), avail_pretax)

        if conversion_this_year > 0:
            avail_pretax = curr_pre_filer + curr_pre_spouse
            if avail_pretax > 0:
                filer_share = curr_pre_filer / avail_pretax
                curr_pre_filer -= conversion_this_year * filer_share
                curr_pre_spouse -= conversion_this_year * (1 - filer_share)
            curr_roth += conversion_this_year
            total_converted += conversion_this_year

        # Cash received from fixed sources (for cash flow tracking)
        cash_received = ss_now + pen_now + rmd_total

        # Withdrawal loop
        wd_cash = 0.0
        wd_brokerage = 0.0
        wd_pretax = 0.0
        wd_roth = 0.0
        wd_life = 0.0
        wd_annuity = 0.0
        ann_gains_withdrawn = 0.0
        cap_gains_realized = 0.0

        if blend_mode:
            # Dynamic blend: tax-optimal source selection
            dyn_gain_pct = max(0.0, 1.0 - brokerage_basis / curr_brokerage) if curr_brokerage > 0 else 0.0
            base_year_inp = {
                "wages": 0.0, "gross_ss": ss_now, "taxable_pensions": pen_now,
                "rmd_amount": rmd_total,
                "taxable_ira": conversion_this_year,
                "total_ordinary_dividends": p_total_ordinary_div,
                "qualified_dividends": p_qualified_div,
                "tax_exempt_interest": 0.0,
                "interest_taxable": p_interest_taxable,
                "cap_gain_loss": p_base_cap_gain,
                "other_income": 0.0,
                "ordinary_tax_only": 0.0,
                "adjustments": 0.0,
                "reinvest_dividends": p_reinvest_div,
                "reinvest_cap_gains": p_reinvest_cg,
                "filing_status": filing_status,
                "filer_65_plus": filer_65, "spouse_65_plus": spouse_65,
                "dependents": p_dependents,
                "retirement_deduction": p_retirement_deduction * inf_factor,
                "out_of_state_gain": p_out_of_state_gain,
                "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                "mortgage_payment": mtg_payment,
                "property_tax": p_property_tax * inf_factor,
                "medical_expenses": p_medical_expenses * inf_factor,
                "charitable": p_charitable * inf_factor,
                "cashflow_taxfree": 0.0, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0,
            }
            blend_balances = {
                "cash": curr_cash, "brokerage": curr_brokerage,
                "pretax": curr_pre_filer + curr_pre_spouse,
                "roth": curr_roth, "life": curr_life,
                "annuity_value": curr_ann, "annuity_basis": curr_ann_basis,
                "dyn_gain_pct": dyn_gain_pct,
            }
            (wd_cash, wd_brokerage, wd_pretax, wd_roth, wd_life, wd_annuity,
             ann_gains_withdrawn, cap_gains_realized, final_res) = _fill_shortfall_dynamic(
                total_spend_need, cash_received, blend_balances, base_year_inp,
                p_base_cap_gain, inf_factor, conversion_this_year)
            yr_tax = final_res["total_tax"]
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
                    "wages": 0.0, "gross_ss": ss_now, "taxable_pensions": pen_now,
                    "rmd_amount": rmd_total,
                    "taxable_ira": wd_pretax + conversion_this_year,
                    "total_ordinary_dividends": p_total_ordinary_div,
                    "qualified_dividends": p_qualified_div,
                    "tax_exempt_interest": 0.0,
                    "interest_taxable": p_interest_taxable,
                    "cap_gain_loss": p_base_cap_gain + cap_gains_realized,
                    "other_income": ann_gains_withdrawn,
                    "ordinary_tax_only": 0.0,
                    "adjustments": 0.0,
                    "reinvest_dividends": p_reinvest_div,
                    "reinvest_cap_gains": p_reinvest_cg,
                    "filing_status": filing_status,
                    "filer_65_plus": filer_65, "spouse_65_plus": spouse_65,
                    "dependents": p_dependents,
                    "retirement_deduction": p_retirement_deduction * inf_factor,
                    "out_of_state_gain": p_out_of_state_gain,
                    "mortgage_balance": curr_mtg_bal, "mortgage_rate": mtg_rate,
                    "mortgage_payment": mtg_payment,
                    "property_tax": p_property_tax * inf_factor,
                    "medical_expenses": p_medical_expenses * inf_factor,
                    "charitable": p_charitable * inf_factor,
                    "cashflow_taxfree": 0.0, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0,
                }
                trial_res = compute_case_cached(_serialize_inputs_for_cache(trial_inp), inf_factor)
                taxes = trial_res["total_tax"]
                medicare = trial_res["medicare_premiums"]

                # Cash flow: fixed sources + investment income (already in curr_cash) + withdrawals
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
                        avail = curr_pre_filer + curr_pre_spouse - wd_pretax
                        pull = min(shortfall, max(0.0, avail))
                        if pull > 0:
                            wd_pretax += pull
                            shortfall -= pull
                            pulled = True

                    elif bucket == "Tax-Deferred":
                        avail = curr_ann - wd_annuity
                        pull = min(shortfall, max(0.0, avail))
                        if pull > 0:
                            current_gains = max(0.0, (curr_ann - wd_annuity + pull) - curr_ann_basis)
                            new_gains = min(pull, max(0.0, current_gains))
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
            final_inp["cap_gain_loss"] = p_base_cap_gain + cap_gains_realized
            final_inp["other_income"] = ann_gains_withdrawn
            final_res = compute_case_cached(_serialize_inputs_for_cache(final_inp), inf_factor)
            yr_tax = final_res["total_tax"]
            yr_medicare = final_res["medicare_premiums"]

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

        # Surplus: income exceeds spending + taxes → reinvest to cash (with basis tracking)
        cash_available_final = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
        cash_needed_final = total_spend_need + yr_tax + yr_medicare
        yr_surplus = max(0.0, cash_available_final - cash_needed_final)
        curr_cash += yr_surplus

        # Track mortgage paydown
        if curr_mtg_bal > 0 and mtg_payment > 0:
            _, curr_mtg_bal = calc_mortgage_interest_for_year(curr_mtg_bal, mtg_rate, mtg_payment)

        # Growth with negative balance protection
        curr_cash = max(0.0, curr_cash) * (1 + r_taxable)
        curr_brokerage = max(0.0, curr_brokerage) * (1 + r_taxable)
        curr_pre_filer = max(0.0, curr_pre_filer) * (1 + r_pretax)
        curr_pre_spouse = max(0.0, curr_pre_spouse) * (1 + r_pretax)
        curr_roth = max(0.0, curr_roth) * (1 + r_roth)
        curr_ann = max(0.0, curr_ann) * (1 + r_annuity)
        curr_life = max(0.0, curr_life) * (1 + r_life)

        # Home appreciation
        curr_home_val *= (1 + home_appr)
        home_equity = max(0.0, curr_home_val - curr_mtg_bal)

        # Estate calculation
        total_wealth_yr = curr_cash + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
        ann_gains_yr = max(0.0, curr_ann - curr_ann_basis)
        heir_tax_yr = (curr_pre_filer + curr_pre_spouse) * heir_tax_rate + ann_gains_yr * heir_tax_rate
        at_wealth_yr = total_wealth_yr - heir_tax_yr
        gross_estate_yr = total_wealth_yr + home_equity
        net_estate_yr = at_wealth_yr + home_equity

        # Total income for display
        total_income_disp = ss_now + pen_now + spendable_inv + rmd_total + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity

        row = {
            "Year": yr, "Age": age_f, "Spending": round(total_spend_need, 0),
            "SS": round(ss_now, 0), "Pension": round(pen_now, 0),
            "Fixed Inc": round(ss_now + pen_now, 0),
            "Inv Inc": round(spendable_inv, 0),
            "RMD": round(rmd_total, 0),
            "Conversion": round(conversion_this_year, 0),
            "W/D Cash": round(wd_cash, 0), "W/D Taxable": round(wd_brokerage, 0),
            "W/D Pre-Tax": round(wd_pretax, 0),
            "W/D Roth": round(wd_roth, 0), "W/D Life": round(wd_life, 0),
            "W/D Tax-Free": round(wd_roth + wd_life, 0),
            "W/D Annuity": round(wd_annuity, 0),
            "Cap Gains": round(cap_gains_realized, 0),
            "Total Income": round(total_income_disp, 0),
            "Taxes": round(yr_tax, 0), "Medicare": round(yr_medicare, 0),
            "Bal Cash": round(curr_cash, 0), "Bal Taxable": round(curr_brokerage, 0),
            "Bal Pre-Tax": round(curr_pre_filer + curr_pre_spouse, 0),
            "Bal Roth": round(curr_roth, 0), "Bal Annuity": round(curr_ann, 0),
            "Bal Life": round(curr_life, 0),
            "Bal Tax-Free": round(curr_roth + curr_life, 0),
            "Portfolio": round(total_wealth_yr, 0),
            "Total Wealth": round(total_wealth_yr, 0),
        }
        if curr_home_val > 0 or home_val > 0:
            row["Home Value"] = round(curr_home_val, 0)
            row["Home Equity"] = round(home_equity, 0)
            row["Gross Estate"] = round(gross_estate_yr, 0)
        row["Estate (Net)"] = round(net_estate_yr, 0)
        row["_net_draw"] = (wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity) - yr_surplus
        year_details.append(row)

    total_wealth = curr_cash + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
    ann_gains_final = max(0.0, curr_ann - curr_ann_basis)
    heir_tax = (curr_pre_filer + curr_pre_spouse) * heir_tax_rate + ann_gains_final * heir_tax_rate
    after_tax_estate = total_wealth - heir_tax
    home_equity_final = max(0.0, curr_home_val - curr_mtg_bal)

    return {
        "after_tax_estate": after_tax_estate + home_equity_final,
        "total_wealth": total_wealth,
        "total_taxes": total_taxes_paid, "total_converted": total_converted,
        "final_cash": curr_cash, "final_brokerage": curr_brokerage,
        "final_pretax": curr_pre_filer + curr_pre_spouse,
        "final_roth": curr_roth, "final_annuity": curr_ann, "final_life": curr_life,
        "year_details": year_details,
    }


# ---------- PDF Report Generator ----------
def _pdf_safe(text):
    """Replace Unicode characters that Helvetica can't render with ASCII equivalents."""
    s = str(text)
    s = s.replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
    s = s.replace("\u2018", "'").replace("\u2019", "'")   # smart single quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"')   # smart double quotes
    s = s.replace("\u2026", "...").replace("\u2022", "*") # ellipsis, bullet
    s = s.replace("\u00b7", "*")                           # middle dot
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

def generate_pdf_report():
    """Generate a PDF report from session state data. Returns bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # Gather state
    base_res = st.session_state.get("base_results")
    base_inp = st.session_state.get("base_inputs")
    solved_res = st.session_state.get("last_solved_results")
    solved_inp = st.session_state.get("last_solved_inputs")
    tab3_rows = st.session_state.get("tab3_rows")
    tab3_mc = st.session_state.get("tab3_mc_results")
    tab3_params = st.session_state.get("tab3_params")
    p1_results = st.session_state.get("phase1_results")
    p1_best_order = st.session_state.get("phase1_best_order")
    p1_best_details = st.session_state.get("phase1_best_details")
    p2_results = st.session_state.get("phase2_results")
    p2_best_details = st.session_state.get("phase2_best_details")
    p2_best_name = st.session_state.get("phase2_best_name")
    tab5_conv_res = st.session_state.get("tab5_conv_res")
    tab5_actual_conversion = st.session_state.get("tab5_actual_conversion")
    tab5_conversion_room = st.session_state.get("tab5_conversion_room")
    tab5_total_additional_cost = st.session_state.get("tab5_total_additional_cost")

    # ==================== PAGE 1: COVER ====================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(40)
    pdf.cell(0, 15, _pdf_safe("Retirement Tax Free"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, _pdf_safe("Wealth Optimization Report"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(50, pdf.get_y(), pdf.w - 50, pdf.get_y())
    pdf.ln(10)

    # Client name
    c_name = st.session_state.get("client_name", "")
    if c_name:
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _pdf_safe(c_name), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Filing info
    pdf.set_font("Helvetica", "", 11)
    if base_inp:
        pdf.cell(0, 7, _pdf_safe(f"Filing Status: {base_inp.get('filing_status', 'N/A')}"), align="C", new_x="LMARGIN", new_y="NEXT")
    filer_dob_val = st.session_state.get("filer_dob")
    spouse_dob_val = st.session_state.get("spouse_dob")
    if filer_dob_val:
        age_f = age_at_date(filer_dob_val, dt.date.today())
        age_line = f"Filer Age: {age_f}"
        if spouse_dob_val and base_inp and "joint" in base_inp.get("filing_status", "").lower():
            age_s = age_at_date(spouse_dob_val, dt.date.today())
            age_line += f"  |  Spouse Age: {age_s}"
        pdf.cell(0, 7, age_line, align="C", new_x="LMARGIN", new_y="NEXT")
    tax_yr = st.session_state.get("tax_year", "")
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
        ira_dist = r.get("taxable_ira", 0) + r.get("rmd_amount", 0)
        other = r.get("other_income", 0) + r.get("ordinary_tax_only", 0)
        income_rows = [
            ("Wages", r.get("wages", 0)),
            ("Social Security (gross)", r.get("gross_ss", 0)),
            ("Social Security (taxable)", r.get("taxable_ss", 0)),
            ("Pensions", r.get("taxable_pensions", 0)),
            ("IRA distributions", ira_dist),
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

        net_needed_val = float(st.session_state.get("last_net_needed", 0))
        source_used = st.session_state.get("last_source", "N/A")

        withdrawal_amt = float(st.session_state.get("last_withdrawal_proceeds", 0))

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
                f"Growth - Taxable: {tp.get('r_taxable', 0):.1%}  |  "
                f"Pre-Tax: {tp.get('r_pretax', 0):.1%}  |  "
                f"Roth: {tp.get('r_roth', 0):.1%}  |  "
                f"Annuity: {tp.get('r_annuity', 0):.1%}  |  "
                f"Life: {tp.get('r_life', 0):.1%}"
            )
            pdf.cell(0, 5, _pdf_safe(growth), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        # Select key columns for the table
        display_cols = ["Year", "Age", "Spending", "Fixed Inc", "RMD",
                        "W/D Cash", "W/D Taxable", "W/D Pre-Tax", "W/D Tax-Free", "W/D Annuity",
                        "Taxes", "Medicare", "Portfolio", "Estate (Net)"]
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
            short = c.replace("W/D ", "").replace("Estate (Net)", "Estate")
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
            proj_cols = ["Year", "Age", "Portfolio", "Estate (Net)"]
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
        is_dynamic = best.get("waterfall") == "Dynamic Blend"
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Phase 1: Optimal Spending Strategy", new_x="LMARGIN", new_y="NEXT")
        _label = "Strategy:" if is_dynamic else "Best Spending Order:"
        _pdf_kv_line(pdf, _label, best["waterfall"], bold_value=True)
        pdf.ln(2)

        opt_summ = [
            ["Gross Estate", pdf_money(best["total_wealth"])],
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

        if not is_dynamic and len(p1_results) > 1:
            # Top 5 waterfall rankings (only for legacy waterfall mode)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Top 5 Waterfall Rankings", new_x="LMARGIN", new_y="NEXT")
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
            opt_proj_cols = ["Year", "Age", "Spending", "Fixed Inc", "RMD",
                             "W/D Cash", "W/D Taxable", "W/D Pre-Tax", "W/D Tax-Free", "W/D Annuity",
                             "Taxes", "Medicare", "Portfolio", "Estate (Net)"]
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
            opt_headers = [c.replace("W/D ", "").replace("Estate (Net)", "Estate") for c in avail_opt]
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

try:
    _needs_auth = bool(st.secrets.get("APP_PASSWORD", ""))
except FileNotFoundError:
    _needs_auth = False
if _needs_auth and not st.session_state.get("authenticated"):
    st.warning("Please log in from the home page.")
    st.stop()

st.title("RTF Tax + Income Needs + LT Projection + Optimizer")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Base Tax Estimator", "Income Needs", "Wealth Projection", "Income Optimizer", "Roth Conversion Opportunity"])

with st.sidebar:
    with st.expander("Save / Load Client Profile"):
        _profiles = sorted([f[:-5] for f in os.listdir(_PROFILE_DIR) if f.endswith(".json")])
        if _profiles:
            _sel = st.selectbox("Select Profile", _profiles, key="_prof_sel")
            if st.button("Load Profile", key="_prof_load"):
                with open(os.path.join(_PROFILE_DIR, f"{_sel}.json")) as _f:
                    _apply_profile(json.load(_f))
                st.rerun()
        else:
            st.caption("No saved profiles yet.")
        st.divider()
        _save_name = st.text_input("Profile Name", key="_prof_name")
        _c1, _c2 = st.columns(2)
        with _c1:
            if st.button("Save", key="_prof_save") and _save_name:
                with open(os.path.join(_PROFILE_DIR, f"{_save_name}.json"), "w") as _f:
                    json.dump(_collect_profile(), _f, indent=2)
                st.success(f"Saved: {_save_name}")
        with _c2:
            if _profiles and st.button("Delete", key="_prof_del"):
                os.remove(os.path.join(_PROFILE_DIR, f"{_sel}.json"))
                st.rerun()

    st.header("Client Info")
    client_name = st.text_input("Client Name", key="client_name")
    current_year = dt.date.today().year
    tax_year = st.number_input("Tax year", min_value=2020, max_value=2100, value=2025, step=1, key="tax_year")
    filing_status = st.selectbox("Filing Status", ["Single", "Married Filing Jointly", "Head of Household"], key="filing_status")
    st.divider()
    st.header("Inflation / COLA")
    inflation = st.number_input("Inflation (also SS COLA)", value=0.030, step=0.005, format="%.3f", key="inflation")
    st.divider()
    st.header("DOBs (for RMD + SS not-yet)")
    enter_dobs = st.checkbox("Enter DOBs", value=True, key="enter_dobs")
    if enter_dobs:
        filer_dob = st.date_input("Filer DOB", value=dt.date(1955, 1, 1), min_value=dt.date(1930, 1, 1), max_value=dt.date(2000, 12, 31), key="filer_dob")
        spouse_dob = st.date_input("Spouse DOB", value=dt.date(1955, 1, 1), min_value=dt.date(1930, 1, 1), max_value=dt.date(2000, 12, 31), key="spouse_dob") if "joint" in filing_status.lower() else None
    else: filer_dob = None; spouse_dob = None
    st.divider()
    st.header("Social Security")
    filer_ss_already = st.checkbox("Filer already receiving SS?", value=False, key="filer_ss_already")
    if filer_ss_already:
        filer_ss_current = st.number_input("Filer current annual SS", value=0.0, step=1000.0, key="filer_ss_current")
        filer_ss_start_year = st.number_input("Filer SS start year", value=int(tax_year), step=1, key="filer_ss_start_year")
        filer_ss_fra = 0.0; filer_ss_claim = "FRA"
    else:
        filer_ss_current = 0.0; filer_ss_start_year = 9999
        filer_ss_fra = st.number_input("Filer SS at FRA", value=0.0, step=1000.0, key="filer_ss_fra")
        filer_ss_claim = st.selectbox("Filer claim choice", ["62", "FRA", "70"], index=1, key="filer_ss_claim")
    if "joint" in filing_status.lower():
        spouse_ss_already = st.checkbox("Spouse already receiving SS?", value=False, key="spouse_ss_already")
        if spouse_ss_already:
            spouse_ss_current = st.number_input("Spouse current annual SS", value=0.0, step=1000.0, key="spouse_ss_current")
            spouse_ss_start_year = st.number_input("Spouse SS start year", value=int(tax_year), step=1, key="spouse_ss_start_year")
            spouse_ss_fra = 0.0; spouse_ss_claim = "FRA"
        else:
            spouse_ss_current = 0.0; spouse_ss_start_year = 9999
            spouse_ss_fra = st.number_input("Spouse SS at FRA", value=0.0, step=1000.0, key="spouse_ss_fra")
            spouse_ss_claim = st.selectbox("Spouse claim choice", ["62", "FRA", "70"], index=1, key="spouse_ss_claim")
    else: spouse_ss_already = False; spouse_ss_current = 0.0; spouse_ss_start_year = 9999; spouse_ss_fra = 0.0; spouse_ss_claim = "FRA"
    st.divider()
    st.header("Pensions")
    pension_filer = st.number_input("Filer pension", value=0.0, step=1000.0, key="pension_filer")
    pension_spouse = st.number_input("Spouse pension", value=0.0, step=1000.0, key="pension_spouse") if "joint" in filing_status.lower() else 0.0
    pension_cola = st.number_input("Pension COLA (%)", value=0.00, step=0.005, format="%.3f", key="pension_cola")
    st.divider()
    st.header("RMD Inputs")
    auto_rmd = st.checkbox("Auto-calculate RMD", value=True, key="auto_rmd")
    pretax_balance_filer_prior = st.number_input("Filer prior-year 12/31 pre-tax balance", value=0.0, step=1000.0, key="pretax_bal_filer_prior")
    pretax_balance_spouse_prior = st.number_input("Spouse prior-year 12/31 pre-tax balance", value=0.0, step=1000.0, key="pretax_bal_spouse_prior") if "joint" in filing_status.lower() else 0.0
    baseline_pretax_distributions = st.number_input("Baseline pre-tax distributions", value=0.0, step=1000.0, key="baseline_pretax_dist")
    rmd_manual = st.number_input("RMD manual override", value=0.0, step=1000.0, key="rmd_manual") if not auto_rmd else 0.0
    st.divider()
    st.header("Other Income / Deductions / Assets")
    wages = st.number_input("Wages", value=0.0, step=1000.0, key="wages")
    tax_exempt_interest = st.number_input("Tax-exempt interest", value=0.0, step=100.0, key="tax_exempt_interest")
    interest_taxable = st.number_input("Taxable interest", value=0.0, step=100.0, key="interest_taxable")
    total_ordinary_dividends = st.number_input("Total ordinary dividends", value=0.0, step=100.0, key="total_ordinary_div")
    qualified_dividends = st.number_input("Qualified dividends", value=0.0, max_value=total_ordinary_dividends, step=100.0, key="qualified_div")
    reinvest_dividends = st.checkbox("Reinvest dividends (taxable but not spendable)", key="reinvest_dividends")
    cap_gain_loss = st.number_input("Baseline net cap gain/(loss)", value=0.0, step=1000.0, key="cap_gain_loss")
    reinvest_cap_gains = st.checkbox("Reinvest capital gains (taxable but not spendable)", key="reinvest_cap_gains")
    other_income = st.number_input("Other taxable income", value=0.0, step=500.0, key="other_income")
    filer_65_plus = st.checkbox("Filer age 65+", key="filer_65_plus")
    spouse_65_plus = st.checkbox("Spouse age 65+", key="spouse_65_plus") if "joint" in filing_status.lower() else False
    adjustments = st.number_input("Adjustments to income", value=0.0, step=500.0, key="adjustments")
    dependents = st.number_input("Dependents", value=0, step=1, key="dependents")
    retirement_deduction = st.number_input("SC retirement deduction", value=0.0, step=1000.0, key="retirement_deduction")
    out_of_state_gain = st.number_input("Out-of-state gain (SC)", value=0.0, step=1000.0, key="out_of_state_gain")
    st.divider()
    st.header("Real Estate")
    home_value = st.number_input("Home value", value=0.0, step=10000.0, key="home_value")
    home_appreciation = st.number_input("Home appreciation (%)", value=3.0, step=0.5, format="%.1f", key="home_appreciation") / 100
    mortgage_balance = st.number_input("Mortgage balance", value=0.0, step=10000.0, key="mortgage_balance")
    mortgage_rate = st.number_input("Mortgage rate (%)", value=0.0, step=0.125, format="%.3f", key="mortgage_rate") / 100
    mortgage_payment_monthly = st.number_input("Monthly mortgage payment", value=0.0, step=100.0, key="mortgage_payment")
    mortgage_payment = mortgage_payment_monthly * 12
    property_tax = st.number_input("Property tax (annual)", value=0.0, step=500.0, key="property_tax")
    st.divider()
    st.header("Itemized Deductions")
    st.caption("Auto-compared with standard deduction (mortgage interest & SALT from above)")
    medical_expenses = st.number_input("Medical/health care expenses", value=0.0, step=500.0, key="medical_expenses")
    charitable = st.number_input("Charitable contributions", value=0.0, step=500.0, key="charitable")
    taxable_cash_bal = st.number_input("After-tax cash", value=0.0, step=1000.0, key="taxable_cash_bal")
    taxable_brokerage_bal = st.number_input("After-tax brokerage", value=0.0, step=1000.0, key="taxable_brokerage_bal")
    brokerage_gain_pct = st.slider("Brokerage sale gain %", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="brokerage_gain_pct")
    pretax_bal_filer_current = st.number_input("Pre-tax (IRA/401k) — Filer", value=0.0, step=1000.0, key="pretax_bal_filer_current")
    pretax_bal_spouse_current = st.number_input("Pre-tax (IRA/401k) — Spouse", value=0.0, step=1000.0, key="pretax_bal_spouse_current") if "joint" in filing_status.lower() else 0.0
    pretax_bal = pretax_bal_filer_current + pretax_bal_spouse_current
    roth_bal = st.number_input("Roth balance", value=0.0, step=1000.0, key="roth_bal")
    life_cash_value = st.number_input("Life insurance CV", value=0.0, step=1000.0, key="life_cash_value")
    annuity_value = st.number_input("Annuity value", value=0.0, step=1000.0, key="annuity_value")
    annuity_basis = st.number_input("Annuity cost basis", value=0.0, step=1000.0, key="annuity_basis")
    st.divider()
    st.header("Growth Rate Assumptions (%)")
    r_taxable = st.number_input("Taxable return %", value=6.0, step=0.5, format="%.1f", key="r_taxable_side") / 100
    r_pretax = st.number_input("Pre-tax return %", value=6.0, step=0.5, format="%.1f", key="r_pretax_side") / 100
    r_roth = st.number_input("Roth return %", value=6.0, step=0.5, format="%.1f", key="r_roth_side") / 100
    r_annuity = st.number_input("Annuity return %", value=4.0, step=0.5, format="%.1f", key="r_annuity_side") / 100
    r_life = st.number_input("Life Ins return %", value=4.0, step=0.5, format="%.1f", key="r_life_side") / 100

gross_ss_filer = annual_ss_in_year(dob=filer_dob, tax_year=int(tax_year), cola=float(inflation), already_receiving=filer_ss_already, current_annual=float(filer_ss_current), start_year=int(filer_ss_start_year), fra_annual=float(filer_ss_fra), claim_choice=filer_ss_claim, current_year=int(current_year))
gross_ss_spouse = 0.0
if "joint" in filing_status.lower():
    gross_ss_spouse = annual_ss_in_year(dob=spouse_dob, tax_year=int(tax_year), cola=float(inflation), already_receiving=spouse_ss_already, current_annual=float(spouse_ss_current), start_year=int(spouse_ss_start_year), fra_annual=float(spouse_ss_fra), claim_choice=spouse_ss_claim, current_year=int(current_year))
gross_ss_total = gross_ss_filer + gross_ss_spouse
taxable_pensions_total = float(pension_filer) + float(pension_spouse)
tax_year_end = dt.date(int(tax_year), 12, 31)
age_filer_eoy = age_at_date(filer_dob, tax_year_end)
age_spouse_eoy = age_at_date(spouse_dob, tax_year_end) if "joint" in filing_status.lower() else None
rmd_filer = compute_rmd_uniform_start73(float(pretax_balance_filer_prior), age_filer_eoy) if auto_rmd else 0.0
rmd_spouse = compute_rmd_uniform_start73(float(pretax_balance_spouse_prior), age_spouse_eoy) if (auto_rmd and "joint" in filing_status.lower()) else 0.0
rmd_total = (rmd_filer + rmd_spouse) if auto_rmd else float(rmd_manual)

def current_inputs():
    return {
        "wages": float(wages), "tax_exempt_interest": float(tax_exempt_interest), "interest_taxable": float(interest_taxable),
        "total_ordinary_dividends": float(total_ordinary_dividends), "qualified_dividends": float(qualified_dividends),
        "taxable_ira": float(baseline_pretax_distributions), "rmd_amount": float(rmd_total),
        "taxable_pensions": float(taxable_pensions_total), "gross_ss": float(gross_ss_total),
        "reinvest_dividends": bool(reinvest_dividends), "reinvest_cap_gains": bool(reinvest_cap_gains),
        "cap_gain_loss": float(cap_gain_loss), "other_income": float(other_income),
        "adjustments": float(adjustments), "dependents": int(dependents),
        "filing_status": filing_status, "filer_65_plus": bool(filer_65_plus), "spouse_65_plus": bool(spouse_65_plus),
        "retirement_deduction": float(retirement_deduction), "out_of_state_gain": float(out_of_state_gain),
        "mortgage_balance": float(mortgage_balance), "mortgage_rate": float(mortgage_rate),
        "mortgage_payment": float(mortgage_payment), "property_tax": float(property_tax),
        "medical_expenses": float(medical_expenses), "charitable": float(charitable),
        "ordinary_tax_only": 0.0, "cashflow_taxfree": 0.0, "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0,
    }

def current_assets():
    basis = min(float(annuity_basis), float(annuity_value)) if float(annuity_value) > 0 else 0.0
    return {
        "taxable": {"cash": float(taxable_cash_bal), "brokerage": float(taxable_brokerage_bal)},
        "pretax": {"balance": float(pretax_bal)},
        "taxfree": {"roth": float(roth_bal), "life_cash_value": float(life_cash_value)},
        "annuity": {"value": float(annuity_value), "basis": float(basis)},
    }

with tab1:
    st.subheader("Estimated Tax Analysis")
    if st.button("Calculate Base Taxes", type="primary"):
        base_inputs = current_inputs(); base_assets = current_assets()
        res = compute_case(base_inputs)
        st.session_state.base_inputs = base_inputs; st.session_state.base_results = res; st.session_state.assets = base_assets
        st.session_state.gross_from_needs = None; st.session_state.last_solved_results = None
    if st.session_state.base_results:
        r = st.session_state.base_results
        display_tax_return(r, mortgage_pmt=float(mortgage_payment), filer_65=filer_65_plus, spouse_65=spouse_65_plus)

with tab2:
    st.subheader("Income Needs Analysis")
    if not st.session_state.base_results:
        st.warning("Run the Estimated Tax Analysis (Tab 1) first.")
    else:
        net_needed = st.number_input("Net income needed", min_value=0.0, step=1000.0)
        taxes_from_cash = st.checkbox("Pay taxes from cash (not from withdrawal)?", value=False)
        source = st.radio("Withdrawal source", ["Taxable – Cash", "Taxable – Brokerage", "Pre-Tax – IRA/401k", "Annuity", "Roth", "Life Insurance (loan)"])
        avail = max_withdrawable(st.session_state.assets, source)
        st.caption(f"Available in {source}: {money(avail)}")
        if st.button("Calculate Income Needs", type="primary"):
            extra, base_case, solved, solved_assets, solved_inputs = solve_gross_up_with_assets(
                st.session_state.base_inputs, st.session_state.assets, source, float(brokerage_gain_pct), float(net_needed), taxes_from_cash)
            if extra is None:
                st.error(f"Insufficient assets in {source} ({money(avail)}) to meet the income need of {money(net_needed)}.")
            elif extra == 0.0:
                st.info(f"Your base income already meets or exceeds the {money(net_needed)} target — no additional withdrawal needed.")
                st.session_state.last_solved_results = base_case; st.session_state.last_solved_inputs = st.session_state.base_inputs
                st.session_state.last_solved_assets = st.session_state.assets; st.session_state.last_net_needed = net_needed
                st.session_state.last_source = source; st.session_state.gross_from_needs = base_case["spendable_gross"]
                st.session_state.last_withdrawal_proceeds = 0
            else:
                st.success(f"**Withdrawal needed: {money(extra)}** from {source}")
                st.session_state.last_solved_results = solved; st.session_state.last_solved_inputs = solved_inputs
                st.session_state.last_solved_assets = solved_assets; st.session_state.last_net_needed = net_needed
                st.session_state.last_source = source; st.session_state.gross_from_needs = solved["spendable_gross"]
                st.session_state.last_withdrawal_proceeds = extra

                # Full tax return after withdrawal
                st.divider()
                st.markdown("## Estimated Tax Analysis — After Withdrawal")
                display_tax_return(solved, mortgage_pmt=float(mortgage_payment), filer_65=filer_65_plus, spouse_65=spouse_65_plus)

                # Additional tax liability from generating income needs
                st.divider()
                st.markdown("### Additional Tax Liability from Generating Income Needs")
                _delta_fed = solved["fed_tax"] - base_case["fed_tax"]
                _delta_sc = solved["sc_tax"] - base_case["sc_tax"]
                _delta_med = solved["medicare_premiums"] - base_case["medicare_premiums"]
                _delta_total = _delta_fed + _delta_sc + _delta_med
                _impact_table = "| | Before | After | Change |\n|:---|-------:|-------:|-------:|\n"
                _impact_table += f"| Federal tax | {money(base_case['fed_tax'])} | {money(solved['fed_tax'])} | {money(_delta_fed)} |\n"
                _impact_table += f"| SC tax | {money(base_case['sc_tax'])} | {money(solved['sc_tax'])} | {money(_delta_sc)} |\n"
                _impact_table += f"| Medicare premiums | {money(base_case['medicare_premiums'])} | {money(solved['medicare_premiums'])} | {money(_delta_med)} |\n"
                _impact_table += f"| **Total** | **{money(base_case['total_tax'] + base_case['medicare_premiums'])}** | **{money(solved['total_tax'] + solved['medicare_premiums'])}** | **{money(_delta_total)}** |\n"
                st.markdown(_impact_table)
                if _delta_total > 0:
                    st.caption(f"Generating {money(net_needed)} in income needs from {source} adds {money(_delta_total)} in total tax liability.")
                else:
                    st.caption(f"This withdrawal source does not increase your tax liability.")

with tab3:
    st.subheader("Wealth Projection")
    if st.session_state.assets is None: st.warning("Run Base Tax Estimator first.")
    else:
        a0 = st.session_state.assets
        # Pull spending goal from Tab 2 if available
        _default_spend = st.session_state.last_net_needed if st.session_state.last_net_needed else 125000.0
        colL, colR = st.columns(2)
        with colL:
            spending_goal = st.number_input("Annual Spending Goal (After-Tax)", value=_default_spend, step=5000.0)
            if st.session_state.last_net_needed:
                st.caption("(Pulled from Income Needs tab)")
            heir_tax_rate = st.number_input("Heir Tax Rate (%)", value=25.0, step=1.0, key="heir_tab3") / 100
            start_year = st.number_input("Start year", min_value=2020, max_value=2100, value=max(2026, int(tax_year) + 1), step=1)
            years = st.number_input("Years to Project", min_value=1, max_value=60, value=30, step=1)
            st.markdown("#### Liquidation Order (Waterfall)")
            c1, c2, c3, c4 = st.columns(4)
            opts = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
            so1 = c1.selectbox("1st", opts, index=0); so2 = c2.selectbox("2nd", opts, index=1)
            so3 = c3.selectbox("3rd", opts, index=2); so4 = c4.selectbox("4th", opts, index=3)
            spending_order = [so1, so2, so3, so4]
        with colR:
            st.markdown("#### Growth Rates (from sidebar)")
            st.write(f"Taxable: {r_taxable:.1%} | Pre-tax: {r_pretax:.1%} | Roth: {r_roth:.1%} | Annuity: {r_annuity:.1%} | Life: {r_life:.1%}")
            current_age_filer = age_at_date(filer_dob, dt.date.today()) if filer_dob else 70
            current_age_spouse = age_at_date(spouse_dob, dt.date.today()) if (spouse_dob and "joint" in filing_status.lower()) else None
            st.markdown("#### Monte Carlo Settings")
            mc_simulations = st.number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100, key="mc_sims")
            mc_volatility = st.number_input("Annual volatility (%)", value=12.0, step=1.0, key="mc_vol") / 100

        def _build_tab3_params():
            pf = float(pretax_bal_filer_current)
            ps = float(pretax_bal_spouse_current) if "joint" in filing_status.lower() else 0.0
            ratio = pf / (pf + ps) if (pf + ps) > 0 else 1.0
            return {
                "spending_goal": spending_goal, "start_year": int(start_year),
                "years": int(years), "inflation": float(inflation),
                "pension_cola": float(pension_cola), "heir_tax_rate": heir_tax_rate,
                "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
                "r_annuity": r_annuity, "r_life": r_life,
                "gross_ss_total": gross_ss_total, "taxable_pensions_total": taxable_pensions_total,
                "filing_status": filing_status,
                "current_age_filer": current_age_filer, "current_age_spouse": current_age_spouse,
                "pretax_filer_ratio": ratio, "brokerage_gain_pct": float(brokerage_gain_pct),
                "interest_taxable": float(interest_taxable),
                "total_ordinary_dividends": float(total_ordinary_dividends),
                "qualified_dividends": float(qualified_dividends),
                "cap_gain_loss": float(cap_gain_loss),
                "reinvest_dividends": bool(reinvest_dividends),
                "reinvest_cap_gains": bool(reinvest_cap_gains),
                "retirement_deduction": float(retirement_deduction),
                "out_of_state_gain": float(out_of_state_gain),
                "dependents": int(dependents),
                "property_tax": float(property_tax),
                "medical_expenses": float(medical_expenses),
                "charitable": float(charitable),
                "mortgage_balance": float(mortgage_balance), "mortgage_rate": float(mortgage_rate),
                "mortgage_payment": float(mortgage_payment),
                "home_value": float(home_value), "home_appreciation": float(home_appreciation),
            }

        if st.button("Run Wealth Projection", type="primary"):
            tab3_params = _build_tab3_params()
            result = run_wealth_projection(a0, tab3_params, spending_order)
            rows = result["year_details"]
            st.session_state.tab3_rows = rows
            st.session_state.tab3_params = {
                "spending_goal": spending_goal, "inflation": inflation,
                "years": int(years), "heir_tax_rate": heir_tax_rate,
                "spending_order": " -> ".join(spending_order),
                "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
                "r_annuity": r_annuity, "r_life": r_life,
            }
            # Hide detail sub-columns that are rolled up into summary columns
            _hide_cols = ["W/D Roth", "W/D Life",
                          "Bal Life",
                          "Total Wealth", "Gross Estate", "_net_draw"]
            _df_display = pd.DataFrame(rows).drop(columns=[c for c in _hide_cols if c in rows[0]], errors="ignore")
            st.dataframe(_df_display, use_container_width=True, hide_index=True)
            if len(rows) > 0:
                st.markdown("### Projection Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Portfolio", f"${rows[-1]['Portfolio']:,.0f}")
                with col2: st.metric("Final Estate (Net)", f"${rows[-1]['Estate (Net)']:,.0f}")
                with col3: st.metric("Total Taxes Paid", f"${sum(r['Taxes'] + r['Medicare'] for r in rows):,.0f}")
                with col4: st.metric("Final Year Spending", f"${rows[-1]['Spending']:,.0f}")

                # --- Monte Carlo Simulation ---
                st.divider()
                st.markdown("### Monte Carlo Analysis")
                n_sims = int(mc_simulations)
                vol = mc_volatility
                n_years = int(years)
                # Weighted average return across all asset classes
                init_total = a0["taxable"]["cash"] + a0["taxable"]["brokerage"] + float(pretax_bal_filer_current) + float(pretax_bal_spouse_current) + a0["taxfree"]["roth"] + a0["taxfree"]["life_cash_value"] + a0["annuity"]["value"]
                if init_total > 0:
                    w_avg_return = (
                        (a0["taxable"]["cash"] + a0["taxable"]["brokerage"]) * r_taxable +
                        (float(pretax_bal_filer_current) + float(pretax_bal_spouse_current)) * r_pretax +
                        a0["taxfree"]["roth"] * r_roth +
                        a0["taxfree"]["life_cash_value"] * r_life +
                        a0["annuity"]["value"] * r_annuity
                    ) / init_total
                else:
                    w_avg_return = r_taxable

                rng = np.random.default_rng(42)
                ending_portfolios = np.zeros(n_sims)
                ran_out_year = np.full(n_sims, n_years)  # year money ran out (n_years = never)
                # Store portfolio paths for percentile chart
                all_paths = np.zeros((n_sims, n_years + 1))
                all_paths[:, 0] = init_total

                # Use deterministic projection's net portfolio draw each year
                # This correctly reflects SS, pensions, RMDs, taxes, and surplus
                mc_net_draws = [r["_net_draw"] for r in rows]

                # Deterministic BOY portfolio for scaling surplus in MC
                det_boy = [init_total] + [rows[i]["Portfolio"] for i in range(len(rows) - 1)]
                while len(det_boy) < n_years:
                    det_boy.append(det_boy[-1] if det_boy else 1.0)

                for sim in range(n_sims):
                    port = init_total
                    for yr_i in range(n_years):
                        net_draw = mc_net_draws[yr_i] if yr_i < len(mc_net_draws) else mc_net_draws[-1]
                        # Scale negative net_draws (surplus from large RMDs) by portfolio ratio.
                        # Surplus depends on portfolio size — smaller portfolios have smaller RMDs
                        # and therefore less surplus. Without scaling, every MC path gets the
                        # same fixed surplus, compressing the distribution.
                        if net_draw < 0 and det_boy[yr_i] > 0:
                            ratio = port / det_boy[yr_i]
                            net_draw = net_draw * ratio
                        # Random return: normal distribution around mean with given volatility
                        yr_return = rng.normal(w_avg_return, vol)
                        port = port * (1 + yr_return) - net_draw
                        if port < 0:
                            ran_out_year[sim] = yr_i
                            port = 0.0
                        all_paths[sim, yr_i + 1] = port
                    ending_portfolios[sim] = port

                success_count = np.sum(ending_portfolios > 0)
                success_rate = success_count / n_sims * 100

                mc_data = {
                    "success_rate": success_rate,
                    "median": float(np.median(ending_portfolios)),
                    "p10": float(np.percentile(ending_portfolios, 10)),
                    "p90": float(np.percentile(ending_portfolios, 90)),
                }
                if success_rate < 100:
                    failed = ran_out_year[ran_out_year < n_years]
                    if len(failed) > 0:
                        mc_data["median_fail_year"] = int(start_year) + int(np.median(failed))
                st.session_state.tab3_mc_results = mc_data

                mc_c1, mc_c2, mc_c3 = st.columns(3)
                with mc_c1:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                    st.caption(f"({int(success_count):,} of {n_sims:,} simulations)")
                with mc_c2:
                    st.metric("Median Ending Portfolio", f"${np.median(ending_portfolios):,.0f}")
                    st.metric("10th Percentile", f"${np.percentile(ending_portfolios, 10):,.0f}")
                with mc_c3:
                    st.metric("90th Percentile", f"${np.percentile(ending_portfolios, 90):,.0f}")
                    if success_rate < 100:
                        failed = ran_out_year[ran_out_year < n_years]
                        if len(failed) > 0:
                            avg_fail_yr = int(start_year) + int(np.median(failed))
                            st.metric("Median Failure Year", str(avg_fail_yr))

                # Percentile band chart
                years_axis = list(range(int(start_year), int(start_year) + n_years + 1))
                p10 = np.percentile(all_paths, 10, axis=0)
                p25 = np.percentile(all_paths, 25, axis=0)
                p50 = np.percentile(all_paths, 50, axis=0)
                p75 = np.percentile(all_paths, 75, axis=0)
                p90 = np.percentile(all_paths, 90, axis=0)
                det_line = [init_total] + [rows[i]["Portfolio"] for i in range(len(rows))]
                # Pad if needed
                while len(det_line) < n_years + 1:
                    det_line.append(0)
                chart_df = pd.DataFrame({
                    "Year": years_axis,
                    "10th Pctl": p10, "25th Pctl": p25,
                    "Median": p50,
                    "75th Pctl": p75, "90th Pctl": p90,
                    "Deterministic": det_line[:n_years + 1],
                })
                chart_df = chart_df.set_index("Year")
                st.line_chart(chart_df, use_container_width=True)
                st.caption("Shaded bands: 10th–90th percentile range. Bold line: deterministic projection.")

with tab4:
    st.subheader("Income Optimizer")
    st.write("Finds the optimal spending strategy to maximize after-tax wealth to heirs, then tests Roth conversions for further improvement.")

    if st.session_state.assets is None:
        st.warning("Run Base Tax Estimator first.")
    else:
        a0 = st.session_state.assets
        is_joint = "joint" in filing_status.lower()

        col1, col2 = st.columns(2)
        with col1:
            opt_spending_goal = st.number_input("Annual Spending Goal (After-Tax)", value=125000.0, step=5000.0, key="opt_spend")
            opt_years = st.number_input("Years to Project", min_value=5, max_value=50, value=30, step=1, key="opt_years")
        with col2:
            opt_start_year = st.number_input("Start Year", min_value=2024, max_value=2100, value=max(2025, int(tax_year)), step=1, key="opt_start")
            opt_heir_rate = st.number_input("Heir Tax Rate (%)", value=25.0, step=1.0, key="opt_heir") / 100

        def _build_opt_params():
            ca_filer = age_at_date(filer_dob, dt.date.today()) if filer_dob else 70
            ca_spouse = age_at_date(spouse_dob, dt.date.today()) if (spouse_dob and "joint" in filing_status.lower()) else None
            pf = float(pretax_bal_filer_current)
            ps = float(pretax_bal_spouse_current) if "joint" in filing_status.lower() else 0.0
            ratio = pf / (pf + ps) if (pf + ps) > 0 else 1.0
            return {
                "spending_goal": opt_spending_goal, "start_year": int(opt_start_year),
                "years": int(opt_years), "inflation": float(inflation),
                "pension_cola": float(pension_cola), "heir_tax_rate": opt_heir_rate,
                "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
                "r_annuity": r_annuity, "r_life": r_life,
                "gross_ss_total": gross_ss_total, "taxable_pensions_total": taxable_pensions_total,
                "filing_status": filing_status,
                "current_age_filer": ca_filer, "current_age_spouse": ca_spouse,
                "pretax_filer_ratio": ratio, "brokerage_gain_pct": float(brokerage_gain_pct),
                "interest_taxable": float(interest_taxable),
                "total_ordinary_dividends": float(total_ordinary_dividends),
                "qualified_dividends": float(qualified_dividends),
                "cap_gain_loss": float(cap_gain_loss),
                "reinvest_dividends": bool(reinvest_dividends),
                "reinvest_cap_gains": bool(reinvest_cap_gains),
                "retirement_deduction": float(retirement_deduction),
                "out_of_state_gain": float(out_of_state_gain),
                "dependents": int(dependents),
                "property_tax": float(property_tax),
                "medical_expenses": float(medical_expenses),
                "charitable": float(charitable),
                "mortgage_balance": float(mortgage_balance), "mortgage_rate": float(mortgage_rate),
                "mortgage_payment": float(mortgage_payment),
                "home_value": float(home_value), "home_appreciation": float(home_appreciation),
            }

        st.divider()
        st.markdown("### Phase 1: Optimal Spending Strategy")
        st.info("Dynamic Blend determines the tax-optimal split across all withdrawal sources each year, respecting bracket boundaries, SS taxation hinges, and IRMAA cliffs.")

        if st.button("Run Dynamic Income Plan", type="primary", key="run_phase1"):
            params = _build_opt_params()
            with st.spinner("Running Dynamic Blend optimization..."):
                result = run_wealth_projection(a0, params, [], blend_mode=True)
            estate = result["after_tax_estate"]
            results = [{
                "order": "dynamic",
                "waterfall": "Dynamic Blend",
                "after_tax_estate": estate,
                "total_wealth": result["total_wealth"],
                "total_taxes": result["total_taxes"],
                "final_cash": result["final_cash"],
                "final_brokerage": result["final_brokerage"],
                "final_pretax": result["final_pretax"],
                "final_roth": result["final_roth"],
                "final_annuity": result["final_annuity"],
                "final_life": result["final_life"],
            }]
            st.session_state.phase1_results = results
            st.session_state.phase1_best_order = "dynamic"
            st.session_state.phase1_best_details = result["year_details"]
            st.session_state.phase1_params = params
            st.session_state.phase2_results = None
            st.session_state.phase2_best_details = None

        if st.session_state.phase1_results:
            p1 = st.session_state.phase1_results
            best = p1[0]

            st.success(f"Strategy: **{best['waterfall']}**")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Gross Estate", f"${best['total_wealth']:,.0f}")
            with col2: st.metric("Net Estate (After Heir Tax)", f"${best['after_tax_estate']:,.0f}")
            with col3: st.metric("Total Taxes + Medicare", f"${best['total_taxes']:,.0f}")

            if st.session_state.phase1_best_details:
                with st.expander("Year-by-Year Detail"):
                    st.dataframe(pd.DataFrame(st.session_state.phase1_best_details), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Phase 2: Roth Conversion Layering")

        if st.session_state.phase1_best_order is None:
            st.warning("Run Phase 1 first to establish the optimal spending strategy.")
        else:
            winning_order = st.session_state.phase1_best_order
            if winning_order == "dynamic":
                st.write("**Locked Strategy:** Dynamic Blend")
            else:
                st.write(f"**Locked Spending Order:** {' -> '.join(winning_order)}")

            col1, col2 = st.columns(2)
            with col1:
                opt_stop_age = st.number_input("Stop Conversions at Age", min_value=60, max_value=100, value=75, step=1, key="opt_stop")
                opt_conv_years = st.number_input("Max Years of Conversions", min_value=1, max_value=30, value=15, step=1, key="opt_conv_years")
            with col2:
                st.markdown("**Common Thresholds**")
                if is_joint:
                    st.write("22% bracket: $206,700 | 24% bracket: $394,600 | IRMAA: $212,000")
                else:
                    st.write("22% bracket: $103,350 | 24% bracket: $197,300 | IRMAA: $106,000")
                target_agi_input = st.number_input("Target AGI (for 'fill to bracket')",
                    value=212000.0 if is_joint else 106000.0, step=10000.0, key="opt_target")

            conv_amounts_str = st.text_input("Conversion amounts to test (comma separated)",
                value="25000, 50000, 75000, 100000, 150000, 200000", key="opt_amounts")
            include_fill = st.checkbox("Also test 'Fill to Target AGI' strategy", value=True, key="opt_fill")

            if st.button("Run Phase 2 - Test Roth Conversions", type="primary", key="run_phase2"):
                params = st.session_state.phase1_params or _build_opt_params()
                try:
                    conv_amounts = [float(x.strip()) for x in conv_amounts_str.split(",")]
                except Exception:
                    conv_amounts = [25000, 50000, 75000, 100000]

                strategies = [("none", "No Conversion (baseline)")]
                for amt in conv_amounts:
                    strategies.append((amt, f"${amt:,.0f}/yr"))
                if include_fill:
                    strategies.append(("fill_to_target", f"Fill to ${target_agi_input:,.0f}"))

                results = []
                best_details = None
                baseline_details = None
                best_estate = -float("inf")
                best_name = ""
                baseline_estate = 0.0
                progress_bar = st.progress(0)
                status_text = st.empty()

                _use_blend = (winning_order == "dynamic")
                _p2_order = [] if _use_blend else winning_order
                for idx, (strategy, strategy_name) in enumerate(strategies):
                    result = run_wealth_projection(
                        a0, params, _p2_order,
                        conversion_strategy=strategy,
                        target_agi=target_agi_input,
                        stop_conversion_age=opt_stop_age,
                        conversion_years_limit=opt_conv_years,
                        blend_mode=_use_blend,
                    )
                    estate = result["after_tax_estate"]
                    if strategy == "none":
                        baseline_estate = estate
                        baseline_details = result["year_details"]
                    results.append({
                        "strategy_name": strategy_name,
                        "after_tax_estate": estate,
                        "improvement": estate - baseline_estate,
                        "total_wealth": result["total_wealth"],
                        "total_taxes": result["total_taxes"],
                        "total_converted": result["total_converted"],
                        "final_pretax": result["final_pretax"],
                        "final_roth": result["final_roth"],
                    })
                    if estate > best_estate:
                        best_estate = estate
                        best_details = result["year_details"]
                        best_name = strategy_name
                    progress_bar.progress((idx + 1) / len(strategies))
                    status_text.text(f"Testing strategy {idx + 1} of {len(strategies)}...")

                progress_bar.empty()
                status_text.empty()

                st.session_state.phase2_results = results
                st.session_state.phase2_best_details = best_details
                st.session_state.phase2_baseline_details = baseline_details
                st.session_state.phase2_best_name = best_name

            if st.session_state.phase2_results:
                p2 = st.session_state.phase2_results
                p2_sorted = sorted(p2, key=lambda x: x["after_tax_estate"], reverse=True)
                best_conv = p2_sorted[0]
                baseline = next((r for r in p2 if "baseline" in r["strategy_name"].lower()), p2[0])

                if best_conv["strategy_name"] != baseline["strategy_name"]:
                    improvement = best_conv["after_tax_estate"] - baseline["after_tax_estate"]
                    st.success(f"Best Strategy: **{best_conv['strategy_name']}** adds **${improvement:,.0f}** to after-tax estate vs no conversion")
                else:
                    st.info("No Roth conversion strategy improves upon the baseline (no conversions).")

                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Baseline Estate (No Conv)", f"${baseline['after_tax_estate']:,.0f}")
                with col2: st.metric("Best Estate (With Conv)", f"${best_conv['after_tax_estate']:,.0f}")
                with col3: st.metric("Improvement", f"${best_conv['after_tax_estate'] - baseline['after_tax_estate']:,.0f}")

                with st.expander("All Conversion Strategies Compared"):
                    df_p2 = pd.DataFrame(p2_sorted)
                    df_p2.index = range(1, len(df_p2) + 1)
                    df_p2.index.name = "Rank"
                    display_cols2 = ["strategy_name", "after_tax_estate", "improvement", "total_wealth",
                                     "total_taxes", "total_converted", "final_pretax", "final_roth"]
                    df_show2 = df_p2[display_cols2].copy()
                    for c in display_cols2[1:]:
                        df_show2[c] = df_show2[c].apply(lambda x: f"${x:,.0f}")
                    df_show2.columns = ["Strategy", "After-Tax Estate", "vs Baseline", "Total Wealth",
                                        "Total Taxes", "Total Converted", "Final Pre-Tax", "Final Roth"]
                    st.dataframe(df_show2, use_container_width=True)

                if st.session_state.phase2_best_details:
                    with st.expander("Year-by-Year Detail (Best Strategy)"):
                        st.dataframe(pd.DataFrame(st.session_state.phase2_best_details), use_container_width=True, hide_index=True)

                if st.session_state.phase2_baseline_details and st.session_state.phase2_best_details:
                    st.divider()
                    bl_details = st.session_state.phase2_baseline_details
                    bc_details = st.session_state.phase2_best_details
                    best_strat_name = st.session_state.phase2_best_name or "Best Strategy"

                    yr1_bl = bl_details[0] if bl_details else {}
                    yr1_bc = bc_details[0] if bc_details else {}

                    st.markdown(f"### Year 1 Cash Flow: No Conversion vs {best_strat_name}")
                    col1, col2 = st.columns(2)

                    for col, yr, label in [(col1, yr1_bl, "No Conversion"), (col2, yr1_bc, best_strat_name)]:
                        with col:
                            st.markdown(f"#### {label}")
                            st.markdown("**Income (Cash Received)**")
                            income_rows = [
                                ("Social Security", yr.get("SS", 0)),
                                ("Pensions", yr.get("Pension", 0)),
                                ("RMD", yr.get("RMD", 0)),
                                ("Investment Income", yr.get("Inv Inc", 0)),
                                ("W/D Cash", yr.get("W/D Cash", 0)),
                                ("W/D Taxable", yr.get("W/D Taxable", 0)),
                                ("W/D Pre-Tax", yr.get("W/D Pre-Tax", 0)),
                                ("Roth Conversion", yr.get("Conversion", 0)),
                                ("W/D Roth", yr.get("W/D Roth", 0)),
                                ("W/D Life Ins", yr.get("W/D Life", 0)),
                                ("W/D Annuity", yr.get("W/D Annuity", 0)),
                            ]
                            income_rows = [(k, v) for k, v in income_rows if v > 0]
                            total_inc = sum(v for _, v in income_rows)
                            st.dataframe([{"Income": k, "Amount": money(v)} for k, v in income_rows],
                                         use_container_width=True, hide_index=True)
                            st.metric("Total Income", money(total_inc))

                            st.markdown("**Expenses & Outflows**")
                            out_rows = [
                                ("Living Expenses", yr.get("Spending", 0)),
                                ("Taxes", yr.get("Taxes", 0)),
                                ("Medicare", yr.get("Medicare", 0)),
                            ]
                            out_rows = [(k, v) for k, v in out_rows if v > 0]
                            total_out = sum(v for _, v in out_rows)
                            st.dataframe([{"Expense": k, "Amount": money(v)} for k, v in out_rows],
                                         use_container_width=True, hide_index=True)
                            st.metric("Total Outflows", money(total_out))

                    st.divider()
                    st.markdown("### End of Projection Comparison")
                    col1, col2 = st.columns(2)
                    final_bl = bl_details[-1] if bl_details else {}
                    final_bc = bc_details[-1] if bc_details else {}
                    with col1:
                        st.markdown("#### No Conversion")
                        st.metric("Final Pre-Tax", money(final_bl.get("Bal Pre-Tax", 0)))
                        st.metric("Final Roth", money(final_bl.get("Bal Roth", 0)))
                        st.metric("After-Tax Estate", money(final_bl.get("Estate (Net)", 0)))
                    with col2:
                        st.markdown(f"#### {best_strat_name}")
                        st.metric("Final Pre-Tax", money(final_bc.get("Bal Pre-Tax", 0)))
                        st.metric("Final Roth", money(final_bc.get("Bal Roth", 0)))
                        st.metric("After-Tax Estate", money(final_bc.get("Estate (Net)", 0)))

with tab5:
    st.subheader("Roth Conversion Opportunity")
    st.write("Calculate how much you can convert to fill up to a target income level based on your Income Needs scenario.")
    if st.session_state.last_solved_results is None or st.session_state.last_solved_inputs is None:
        st.warning("Run Income Needs (Tab 2) first to establish your baseline spending scenario.")
    else:
        solved_res = st.session_state.last_solved_results; solved_inp = st.session_state.last_solved_inputs
        net_needed_val = float(st.session_state.last_net_needed or 0.0); source_used = st.session_state.last_source or "Unknown"
        st.markdown("### Current Scenario (from Income Needs)")
        st.write(f"**Funding Source:** {source_used}"); st.write(f"**Net Income Needed:** {money(net_needed_val)}")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Current AGI", money(solved_res["agi"]))
        with col2: st.metric("Current Federal Tax", money(solved_res["fed_tax"]))
        with col3: st.metric("Current Total Tax", money(solved_res["total_tax"]))
        st.divider()
        is_joint = "joint" in filing_status.lower()
        st.markdown("### Common Target Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tax Bracket Tops (Taxable Income)**")
            if is_joint: st.write("12% bracket: $96,950"); st.write("22% bracket: $206,700"); st.write("24% bracket: $394,600")
            else: st.write("12% bracket: $48,475"); st.write("22% bracket: $103,350"); st.write("24% bracket: $197,300")
        with col2:
            st.markdown("**IRMAA Thresholds (AGI)**")
            if is_joint: st.write("No IRMAA: $212,000"); st.write("Tier 1: $266,000")
            else: st.write("No IRMAA: $106,000"); st.write("Tier 1: $133,000")
        st.divider()
        target_type = st.radio("Target Type", ["AGI Target", "Taxable Income Target"], horizontal=True, key="tab5_target_type")
        if target_type == "AGI Target":
            default_target = 212000.0 if is_joint else 106000.0
            target_amount = st.number_input("Target AGI", value=default_target, step=1000.0, key="tab5_target_amt"); current_level = solved_res["agi"]
        else:
            default_target = 206700.0 if is_joint else 103350.0
            target_amount = st.number_input("Target Taxable Income", value=default_target, step=1000.0, key="tab5_target_amt"); current_level = solved_res["fed_taxable"]
        conversion_room = max(0.0, target_amount - current_level)
        available_pretax = st.session_state.assets["pretax"]["balance"] if st.session_state.assets else 0.0
        actual_conversion = min(conversion_room, available_pretax)
        if st.button("Calculate Conversion Opportunity", type="primary", key="tab5_calc"):
            if conversion_room <= 0:
                st.session_state.tab5_conv_res = None
                st.warning(f"You are already at or above the target. Current: {money(current_level)}, Target: {money(target_amount)}")
            else:
                conv_inputs = dict(solved_inp); conv_inputs["taxable_ira"] = float(conv_inputs.get("taxable_ira", 0.0)) + actual_conversion
                conv_res = compute_case(conv_inputs)
                additional_tax = conv_res["total_tax"] - solved_res["total_tax"]
                additional_medicare = conv_res["medicare_premiums"] - solved_res["medicare_premiums"]
                total_additional_cost = additional_tax + additional_medicare
                st.session_state.tab5_conv_res = conv_res
                st.session_state.tab5_conv_inputs = conv_inputs
                st.session_state.tab5_actual_conversion = actual_conversion
                st.session_state.tab5_conversion_room = conversion_room
                st.session_state.tab5_total_additional_cost = total_additional_cost

        # Display results from session state so they persist across reruns
        if st.session_state.tab5_conv_res is not None:
            conv_res = st.session_state.tab5_conv_res
            conv_inputs = st.session_state.tab5_conv_inputs
            actual_conversion = st.session_state.tab5_actual_conversion
            conversion_room = st.session_state.tab5_conversion_room
            total_additional_cost = st.session_state.tab5_total_additional_cost
            additional_tax = conv_res["total_tax"] - solved_res["total_tax"]
            net_to_roth = actual_conversion - total_additional_cost
            effective_rate = (total_additional_cost / actual_conversion * 100) if actual_conversion > 0 else 0
            st.markdown("### Conversion Opportunity")
            st.success(f"**Roth Conversion Room: {money(conversion_room)}** &nbsp;&nbsp;|&nbsp;&nbsp; **Additional Tax: {money(total_additional_cost)}** &nbsp;&nbsp;|&nbsp;&nbsp; **Net to Roth: {money(net_to_roth)}**")
            if actual_conversion < conversion_room: st.info(f"Note: You only have {money(available_pretax)} available in pre-tax accounts.")
            st.markdown("### Tax Impact Analysis")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Conversion Amount", money(actual_conversion)); st.metric("Net to Roth", money(net_to_roth))
            with col2: st.metric("Tax Before", money(solved_res["total_tax"])); st.metric("Tax After", money(conv_res["total_tax"]))
            with col3: st.metric("Additional Tax", money(additional_tax)); st.metric("Effective Tax Rate", f"{effective_rate:.1f}%")
            st.markdown("### IRMAA Status")
            col1, col2 = st.columns(2)
            with col1: st.write(f"**Before Conversion:** {'IRMAA applies' if solved_res['has_irmaa'] else 'No IRMAA'}"); st.write(f"Medicare Premiums: {money(solved_res['medicare_premiums'])}")
            with col2: st.write(f"**After Conversion:** {'IRMAA applies' if conv_res['has_irmaa'] else 'No IRMAA'}"); st.write(f"Medicare Premiums: {money(conv_res['medicare_premiums'])}")
            if conv_res["has_irmaa"] and not solved_res["has_irmaa"]:
                irmaa_cost = conv_res["medicare_premiums"] - solved_res["medicare_premiums"]
                st.warning(f"Warning: This conversion would trigger IRMAA, adding {money(irmaa_cost)} in Medicare premiums!")
            st.markdown("### Detailed Comparison")
            comparison_data = [
                {"Metric": "AGI", "Before": money(solved_res["agi"]), "After": money(conv_res["agi"])},
                {"Metric": "Federal Taxable", "Before": money(solved_res["fed_taxable"]), "After": money(conv_res["fed_taxable"])},
                {"Metric": "Federal Tax", "Before": money(solved_res["fed_tax"]), "After": money(conv_res["fed_tax"])},
                {"Metric": "SC Tax", "Before": money(solved_res["sc_tax"]), "After": money(conv_res["sc_tax"])},
                {"Metric": "Total Tax", "Before": money(solved_res["total_tax"]), "After": money(conv_res["total_tax"])},
                {"Metric": "Medicare Premiums", "Before": money(solved_res["medicare_premiums"]), "After": money(conv_res["medicare_premiums"])},
            ]
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            st.divider(); st.markdown("### Cash Flow Comparison")
            display_cashflow_comparison(solved_inp, solved_res, conv_inputs, conv_res, net_needed=net_needed_val, roth_conversion=actual_conversion, mortgage_payment=float(mortgage_payment), title_before="Before Conversion", title_after="After Conversion")

            # --- Before/After Estate Impact ---
            st.divider()
            st.markdown("### Estate Impact — Before & After Conversion")
            _ec1, _ec2 = st.columns(2)
            with _ec1:
                _est_heir_rate = st.number_input("Heir Tax Rate (%)", value=25.0, step=1.0, key="heir_tab5") / 100
            with _ec2:
                _conv_years = st.number_input("Years of Conversion", min_value=1, max_value=30, value=1, step=1, key="conv_years_tab5")

            _est_assets = st.session_state.assets
            _est_taxable = _est_assets["taxable"]["cash"] + _est_assets["taxable"]["brokerage"]
            _est_pretax = _est_assets["pretax"]["balance"]
            _est_roth = _est_assets["taxfree"]["roth"]
            _est_life = _est_assets["taxfree"]["life_cash_value"]
            _est_ann = _est_assets["annuity"]["value"]
            _est_ann_basis = _est_assets["annuity"]["basis"]

            # --- No-conversion path: just grow for N years ---
            _nc_taxable = _est_taxable; _nc_pretax = _est_pretax; _nc_roth = _est_roth
            _nc_life = _est_life; _nc_ann = _est_ann
            for _yr in range(_conv_years):
                _nc_taxable *= (1 + r_taxable); _nc_pretax *= (1 + r_pretax)
                _nc_roth *= (1 + r_roth); _nc_life *= (1 + r_life); _nc_ann *= (1 + r_annuity)
            _nc_ann_gains = max(0.0, _nc_ann - _est_ann_basis)
            _nc_total = _nc_taxable + _nc_pretax + _nc_roth + _nc_life + _nc_ann
            _nc_heir_tax = (_nc_pretax * _est_heir_rate) + (_nc_ann_gains * _est_heir_rate)
            _nc_net = _nc_total - _nc_heir_tax

            # --- With-conversion path: convert each year, tax paid from conversion (net to Roth), then grow ---
            _wc_taxable = _est_taxable; _wc_pretax = _est_pretax; _wc_roth = _est_roth
            _wc_life = _est_life; _wc_ann = _est_ann
            _total_converted = 0.0; _total_tax_cost = 0.0
            _conv_detail_rows = []
            for _yr in range(_conv_years):
                # Each year: convert the same room amount (capped by remaining pre-tax)
                _yr_conv = min(actual_conversion, max(0.0, _wc_pretax))
                # Estimate tax cost using same effective rate as year 1
                _yr_tax_cost = _yr_conv * (total_additional_cost / actual_conversion) if actual_conversion > 0 else 0.0
                _yr_net_to_roth = _yr_conv - _yr_tax_cost
                _wc_pretax -= _yr_conv
                _wc_roth += _yr_net_to_roth  # net to Roth after tax
                _total_converted += _yr_conv
                _total_tax_cost += _yr_tax_cost
                _conv_detail_rows.append({
                    "Year": _yr + 1, "Converted": round(_yr_conv, 0), "Tax Cost": round(_yr_tax_cost, 0),
                    "Net to Roth": round(_yr_net_to_roth, 0),
                    "Pre-Tax": round(_wc_pretax, 0), "Roth": round(_wc_roth, 0),
                })
                # Grow all accounts
                _wc_taxable *= (1 + r_taxable); _wc_pretax *= (1 + r_pretax)
                _wc_roth *= (1 + r_roth); _wc_life *= (1 + r_life); _wc_ann *= (1 + r_annuity)
            _wc_ann_gains = max(0.0, _wc_ann - _est_ann_basis)
            _wc_total = _wc_taxable + _wc_pretax + _wc_roth + _wc_life + _wc_ann
            _wc_heir_tax = (_wc_pretax * _est_heir_rate) + (_wc_ann_gains * _est_heir_rate)
            _wc_net = _wc_total - _wc_heir_tax

            _yr_label = "Year" if _conv_years == 1 else f"{_conv_years} Years"
            _estate_table = f"| | No Conversion ({_yr_label}) | With Conversion ({_yr_label}) | Change |\n|:---|-------:|-------:|-------:|\n"
            _estate_table += f"| Pre-Tax (IRA/401k) | {money(_nc_pretax)} | {money(_wc_pretax)} | {money(_wc_pretax - _nc_pretax)} |\n"
            _estate_table += f"| Roth | {money(_nc_roth)} | {money(_wc_roth)} | {money(_wc_roth - _nc_roth)} |\n"
            _estate_table += f"| Taxable (less tax paid) | {money(_nc_taxable)} | {money(_wc_taxable)} | {money(_wc_taxable - _nc_taxable)} |\n"
            if _est_life > 0:
                _estate_table += f"| Life Insurance CV | {money(_nc_life)} | {money(_wc_life)} | — |\n"
            if _est_ann > 0:
                _estate_table += f"| Annuity | {money(_nc_ann)} | {money(_wc_ann)} | — |\n"
            _estate_table += f"| **Gross Estate** | **{money(_nc_total)}** | **{money(_wc_total)}** | **{money(_wc_total - _nc_total)}** |\n"
            _estate_table += f"| Heir tax on pre-tax @ {_est_heir_rate:.0%} | ({money(_nc_heir_tax)}) | ({money(_wc_heir_tax)}) | {money(_wc_heir_tax - _nc_heir_tax)} |\n"
            if _nc_ann_gains > 0 or _wc_ann_gains > 0:
                _estate_table += f"| Heir tax on annuity gains @ {_est_heir_rate:.0%} | ({money(_nc_ann_gains * _est_heir_rate)}) | ({money(_wc_ann_gains * _est_heir_rate)}) | — |\n"
            _estate_table += f"| **Net Estate** | **{money(_nc_net)}** | **{money(_wc_net)}** | **{money(_wc_net - _nc_net)}** |\n"
            st.markdown(_estate_table)

            if _conv_years > 1:
                st.caption(f"Total converted over {_conv_years} years: {money(_total_converted)} | Total tax cost: {money(_total_tax_cost)} | Net to Roth: {money(_total_converted - _total_tax_cost)}")
                with st.expander("Year-by-Year Conversion Detail"):
                    st.dataframe(pd.DataFrame(_conv_detail_rows), use_container_width=True, hide_index=True)

            _net_benefit = _wc_net - _nc_net
            if _net_benefit > 0:
                st.success(f"Converting {money(actual_conversion)}/yr for {_conv_years} year(s) improves the net estate by **{money(_net_benefit)}** (heirs @ {_est_heir_rate:.0%}).")
            elif _net_benefit < 0:
                st.warning(f"Converting {money(actual_conversion)}/yr for {_conv_years} year(s) reduces the net estate by **{money(abs(_net_benefit))}** (heirs @ {_est_heir_rate:.0%}).")
            else:
                st.info("The conversion strategy is estate-neutral at this heir tax rate.")

# ---------- PDF Download Button ----------
st.divider()
_has_any_data = any([
    st.session_state.get("base_results"),
    st.session_state.get("last_solved_results"),
    st.session_state.get("tab3_rows"),
    st.session_state.get("phase1_results"),
    st.session_state.get("tab5_conv_res"),
])
if _has_any_data:
    _pdf_bytes = generate_pdf_report()
    st.download_button(
        label="Download PDF Report",
        data=_pdf_bytes,
        file_name=f"RTF_Report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.caption("Run at least one analysis tab to generate a PDF report.")