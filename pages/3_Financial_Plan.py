import streamlit as st
import pandas as pd
import datetime as dt
import json, os, sys, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tax_estimator_advanced as TEA
import engine

st.set_page_config(page_title="OptiPlan Wealth Optimization Planner", layout="wide")

# ---------- Auth ----------
try:
    _needs_auth = bool(st.secrets.get("APP_PASSWORD", ""))
except FileNotFoundError:
    _needs_auth = False
if _needs_auth and not st.session_state.get("authenticated"):
    st.warning("Please log in from the home page.")
    st.stop()

# ---------- CSS: larger sidebar nav + hide multipage nav ----------
st.markdown("""<style>
/* Hide default Streamlit multipage navigation */
[data-testid="stSidebarNav"] { display: none; }
/* Larger sidebar radio nav items */
section[data-testid="stSidebar"] label {
    font-size: 1.4rem !important;
    padding: 0.4rem 0 !important;
}
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] label span,
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 1.4rem !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.3rem !important;
}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PERSISTENT DATA STORE
# Widget keys (_w_*) are ephemeral; fp_data persists across nav changes.
# ══════════════════════════════════════════════════════════════════════
_FP_DEFAULTS = {
    # --- Planning ---
    "client_name": "",
    "filing_status": "Married Filing Jointly",
    "tax_year": 2026,
    "filer_dob": dt.date(1965, 1, 1),
    "spouse_dob": dt.date(1965, 1, 1),
    "is_working": True,
    "ret_age": 65, "spouse_ret_age": 65,
    "filer_plan_age": 95, "spouse_plan_age": 95,
    "inflation": 3.0, "bracket_growth": 2.0,
    "medicare_growth": 5.0, "salary_growth": 3.0,
    "state_tax_rate": 7.0,

    # --- Growing: Employment ---
    "salary_filer": 100000.0, "salary_spouse": 0.0,
    "self_employed_f": False, "self_employed_s": False,
    "plan_type_f": "401(k)", "plan_type_s": "401(k)",
    "max_401k_f": True, "c401k_f": 0.0, "roth_pct_f": 0,
    "ematch_rate_f": 6.0, "ematch_upto_f": 6.0,
    "backdoor_roth_f": False, "max_ira_f": False,
    "contrib_trad_ira_f": 0.0, "contrib_roth_ira_f": 0.0,
    "max_401k_s": True, "c401k_s": 0.0, "roth_pct_s": 0,
    "ematch_rate_s": 6.0, "ematch_upto_s": 6.0,
    "backdoor_roth_s": False, "max_ira_s": False,
    "contrib_trad_ira_s": 0.0, "contrib_roth_ira_s": 0.0,
    "hsa_eligible": False, "max_hsa": True, "contrib_hsa": 0.0,
    "contrib_taxable": 0.0, "pre_ret_return": 7.0,

    # --- Growing: Balances ---
    "curr_401k_f": 0.0, "curr_401k_s": 0.0,
    "curr_trad_ira_f": 0.0, "curr_trad_ira_s": 0.0,
    "curr_roth_ira_f": 0.0, "curr_roth_ira_s": 0.0,
    "curr_roth_401k": 0.0,
    "curr_taxable": 0.0, "taxable_basis": 0.0,
    "curr_taxable_s": 0.0, "taxable_basis_s": 0.0,
    "curr_cash": 0.0, "emergency_fund": 0.0,
    "curr_hsa": 0.0,

    # --- Growing: Insurance Products ---
    "annuity_value_f": 0.0, "annuity_basis_f": 0.0, "annuity_monthly_f": 0.0,
    "annuity_value_s": 0.0, "annuity_basis_s": 0.0, "annuity_monthly_s": 0.0,
    "life_cash_value": 0.0, "life_basis": 0.0, "life_monthly": 0.0,

    # --- Growing: Growth Rates ---
    "r_cash": 2.0, "r_taxable": 6.0, "r_pretax": 6.0,
    "r_roth": 6.0, "r_annuity": 4.0, "r_life": 4.0,

    # --- Growing: Asset Allocation ---
    "use_asset_alloc": False,
    "cma_us_equity": 10.0, "cma_intl_equity": 8.0,
    "cma_fixed_income": 4.5, "cma_real_assets": 6.0, "cma_cash": 2.5,
    # CMA yield decomposition (components of total return)
    "cma_us_equity_div": 1.5, "cma_us_equity_qual": 85, "cma_us_equity_int": 0.0, "cma_us_equity_cg": 1.0,
    "cma_intl_equity_div": 2.5, "cma_intl_equity_qual": 70, "cma_intl_equity_int": 0.0, "cma_intl_equity_cg": 0.5,
    "cma_fixed_income_div": 0.0, "cma_fixed_income_qual": 0, "cma_fixed_income_int": 4.0, "cma_fixed_income_cg": 0.0,
    "cma_real_assets_div": 3.0, "cma_real_assets_qual": 15, "cma_real_assets_int": 0.0, "cma_real_assets_cg": 0.5,
    "cma_cash_div": 0.0, "cma_cash_qual": 0, "cma_cash_int": 2.5, "cma_cash_cg": 0.0,
    "aa_pretax_f_eq": 60, "aa_pretax_f_intl": 15, "aa_pretax_f_fi": 20, "aa_pretax_f_re": 5,
    "aa_pretax_s_eq": 60, "aa_pretax_s_intl": 15, "aa_pretax_s_fi": 20, "aa_pretax_s_re": 5,
    "aa_roth_f_eq": 70, "aa_roth_f_intl": 15, "aa_roth_f_fi": 10, "aa_roth_f_re": 5,
    "aa_roth_s_eq": 70, "aa_roth_s_intl": 15, "aa_roth_s_fi": 10, "aa_roth_s_re": 5,
    "aa_taxable_eq": 50, "aa_taxable_intl": 10, "aa_taxable_fi": 30, "aa_taxable_re": 10,
    "aa_annuity_eq": 40, "aa_annuity_intl": 10, "aa_annuity_fi": 40, "aa_annuity_re": 10,

    # --- Growing: Investment Assumptions for Projections ---
    "use_invest_assumptions": False,
    "proj_div_yield": 1.5, "proj_cg_pct": 1.0, "proj_int_rate": 2.0,
    "reinvest_dividends": False, "reinvest_cap_gains": False, "reinvest_interest": False,

    # --- Growing: Real Estate ---
    "home_value": 0.0, "home_appr": 3.0,
    # --- Growing: Investment Real Estate ---
    "inv_re_value": 0.0, "inv_re_basis": 0.0, "inv_re_appr": 3.0,
    "inv_re_net_income": 0.0, "inv_re_liquidate": False,
    "inv_re_mortgage_pmt": 0.0, "inv_re_mortgage_years": 0, "inv_re_income_growth": 2.0,
    "inv_re_1_include": False, "inv_re_1_name": "Property 1",
    "inv_re_1_value": 0.0, "inv_re_1_basis": 0.0, "inv_re_1_appr": 3.0,
    "inv_re_1_mortgage": 0.0, "inv_re_1_rate": 6.0, "inv_re_1_years": 30,
    "inv_re_1_net_income": 0.0, "inv_re_1_income_growth": 2.0,
    "inv_re_1_liquidate": False,
    "inv_re_2_include": False, "inv_re_2_name": "Property 2",
    "inv_re_2_value": 0.0, "inv_re_2_basis": 0.0, "inv_re_2_appr": 3.0,
    "inv_re_2_mortgage": 0.0, "inv_re_2_rate": 6.0, "inv_re_2_years": 30,
    "inv_re_2_net_income": 0.0, "inv_re_2_income_growth": 2.0,
    "inv_re_2_liquidate": False,
    "inv_re_3_include": False, "inv_re_3_name": "Property 3",
    "inv_re_3_value": 0.0, "inv_re_3_basis": 0.0, "inv_re_3_appr": 3.0,
    "inv_re_3_mortgage": 0.0, "inv_re_3_rate": 6.0, "inv_re_3_years": 30,
    "inv_re_3_net_income": 0.0, "inv_re_3_income_growth": 2.0,
    "inv_re_3_liquidate": False,

    # --- Receiving: SS ---
    "ssdi_filer": False,
    "filer_ss_already": False, "filer_ss_current": 0.0,
    "filer_ss_start_year": 2025, "filer_ss_fra": 0.0, "filer_ss_claim": "FRA",
    "ssdi_spouse": False,
    "spouse_ss_already": False, "spouse_ss_current": 0.0,
    "spouse_ss_start_year": 2025, "spouse_ss_fra": 0.0, "spouse_ss_claim": "FRA",

    # --- Receiving: Pensions (per-spouse COLA) ---
    "pension_filer": 0.0, "pension_filer_age": 65, "pension_cola_filer": 0.0,
    "pension_spouse": 0.0, "pension_spouse_age": 65, "pension_cola_spouse": 0.0,

    # --- Receiving: Investment Income (current year) ---
    "interest_taxable": 0.0,
    "total_ordinary_dividends": 0.0,
    "qualified_dividends": 0.0,
    "cap_gain_loss": 0.0,

    # --- Receiving: Other Income ---
    "wages": 0.0, "tax_exempt_interest": 0.0, "other_income": 0.0,
    "wages_start_age": 0, "wages_end_age": 0,
    "other_income_start_age": 0, "other_income_end_age": 0,
    "tax_exempt_start_age": 0, "tax_exempt_end_age": 0,

    # --- Receiving: RMD ---
    "has_rmd": False, "auto_rmd": True,
    "pretax_bal_filer_prior": 0.0, "pretax_bal_spouse_prior": 0.0,
    "baseline_pretax_dist": 0.0, "rmd_manual": 0.0,

    # --- Receiving: Tax Details ---
    "adjustments": 0.0, "dependents": 0,
    "filer_65_plus": False, "spouse_65_plus": False,
    "retirement_deduction": 0.0, "out_of_state_gain": 0.0,
    "medical_expenses": 0.0,
    "deduction_method": "Auto (higher of standard/itemized)",
    "custom_itemized_amount": 0.0,

    # --- Income Needs ---
    "net_income_needed": 0.0,

    # --- Spending ---
    "living_expenses": 100000.0, "ret_pct": 100.0, "heir_tax_rate": 25.0,
    "mtg_balance": 0.0, "mtg_rate": 0.0, "mtg_payment_monthly": 0.0,
    "mtg_years": 0, "property_tax": 0.0,
    "so1": "Taxable", "so2": "Pre-Tax", "so3": "Tax-Free", "so4": "Tax-Deferred",
    "surplus_dest": "Taxable Brokerage",

    # --- Giving ---
    "charitable": 0.0, "qcd_annual": 0.0,

    # --- Protecting ---
    "survivor_spend_pct": 80, "pension_survivor_pct": 50,

    # --- Estate Tax ---
    "estate_tax_enabled": True,
    "federal_estate_exemption": 15000000.0,
    "exemption_inflation": 2.5,
    "use_portability": True,
    "state_estate_tax_rate": 0.0,
    "state_estate_exemption": 0.0,
}

if "fp_data" not in st.session_state:
    st.session_state.fp_data = dict(_FP_DEFAULTS)
# Ensure new keys are merged into existing data
for _k, _v in _FP_DEFAULTS.items():
    if _k not in st.session_state.fp_data:
        st.session_state.fp_data[_k] = _v

D = st.session_state.fp_data

# Computation state (not persisted in profiles)
_COMP_DEFAULTS = {
    "base_results": None, "base_inputs": None, "assets": None,
    "last_solved_results": None, "last_solved_inputs": None, "last_solved_assets": None,
    "last_net_needed": None, "last_source": None, "last_withdrawal_proceeds": 0,
    "gross_from_needs": 0, "additional_expenses": [], "future_income": [],
    "tab3_rows": None, "tab3_params": None, "tab3_mc": None,
    # Unified optimizer
    "opt_quick_results": None, "opt_deep_results": None, "opt_all_results": None,
    "opt_best_combo": None, "opt_best_details": None, "opt_baseline_details": None,
    "opt_params": None, "opt_selected_combo": None,
    # Backward compat for PDF
    "phase1_results": None, "phase1_best_order": None, "phase1_best_details": None,
    "phase1_all_details": None, "phase1_selected_strategy": None, "phase1_params": None,
    "phase2_results": None, "phase2_best_details": None, "phase2_baseline_details": None,
    "phase2_best_name": None,
    "tab5_conv_res": None, "tab5_conv_inputs": None, "tab5_actual_conversion": None,
    "tab5_conversion_room": None, "tab5_total_additional_cost": None,
    "projection_results": None, "retire_projection": None,
    "_preret_mc": None, "_preret_opt": None,
}
for _k, _v in _COMP_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════
# WIDGET HELPERS — sync widget ↔ fp_data
# ══════════════════════════════════════════════════════════════════════
def _sync(key):
    D[key] = st.session_state[f"_w_{key}"]

def w_num(label, key, **kw):
    return st.number_input(label, value=D[key], key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_check(label, key, **kw):
    return st.checkbox(label, value=D[key], key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_select(label, opts, key, **kw):
    idx = opts.index(D[key]) if D[key] in opts else 0
    return st.selectbox(label, opts, index=idx, key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_slider(label, key, **kw):
    return st.slider(label, value=D[key], key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_text(label, key, **kw):
    return st.text_input(label, value=D[key], key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_date(label, key, **kw):
    val = D[key]
    if isinstance(val, str):
        val = dt.date.fromisoformat(val)
        D[key] = val
    return st.date_input(label, value=val, key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

def w_radio(label, opts, key, **kw):
    idx = opts.index(D[key]) if D[key] in opts else 0
    return st.radio(label, opts, index=idx, key=f"_w_{key}", on_change=_sync, args=(key,), **kw)

# ══════════════════════════════════════════════════════════════════════
# PROFILE SYSTEM
# ══════════════════════════════════════════════════════════════════════
_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plan_profiles")
os.makedirs(_PROFILE_DIR, exist_ok=True)
_TAX_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tax_profiles")

def _save_profile(name):
    data = {}
    for k, v in D.items():
        if isinstance(v, dt.date):
            data[k] = v.isoformat()
        else:
            data[k] = v
    data["_additional_expenses"] = st.session_state.additional_expenses
    data["_future_income"] = st.session_state.future_income
    with open(os.path.join(_PROFILE_DIR, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=2)

def _load_profile(name):
    path = os.path.join(_PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    # Migrate old taxable_basis_pct → taxable_basis (dollar amount)
    if "taxable_basis_pct" in data and "taxable_basis" not in data:
        data["taxable_basis"] = data.get("curr_taxable", 0.0) * (1.0 - data["taxable_basis_pct"])
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k in D:
            if isinstance(_FP_DEFAULTS.get(k), dt.date) and isinstance(v, str):
                D[k] = dt.date.fromisoformat(v)
            else:
                D[k] = v
    if "_additional_expenses" in data:
        st.session_state.additional_expenses = data["_additional_expenses"]
    if "_future_income" in data:
        st.session_state.future_income = data["_future_income"]
    # Update cached widget keys to match loaded D values
    _keep = {"_btn_load", "_sb_load_sel", "_btn_save"}
    for wk in list(st.session_state.keys()):
        if wk.startswith("_w_"):
            _dkey = wk[3:]  # strip "_w_" prefix
            if _dkey in D:
                st.session_state[wk] = D[_dkey]
            else:
                del st.session_state[wk]
        elif wk.startswith("_sb_") and wk not in _keep:
            del st.session_state[wk]
        elif wk.startswith("fp_rr_") or wk.startswith("fp_preret_"):
            del st.session_state[wk]
    # Sync working/retired radio (uses special key, not _w_is_working)
    st.session_state["_w_status_toggle"] = "Currently Working" if D["is_working"] else "Retired"
    # Set retirement readiness withdrawal order from D
    _so_keys = {"fp_rr_o1": "so1", "fp_rr_o2": "so2", "fp_rr_o3": "so3", "fp_rr_o4": "so4"}
    for wk, dk in _so_keys.items():
        st.session_state[wk] = D[dk]
    # Clear cached computation results
    for ck in list(_COMP_DEFAULTS.keys()):
        if ck in st.session_state:
            st.session_state[ck] = _COMP_DEFAULTS[ck]
    return True

def _load_tax_profile(name):
    """Load a retired client profile from tax_profiles/ into FP data dict."""
    path = os.path.join(_TAX_PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)

    # Always set retired mode
    D["is_working"] = False

    # --- Direct mappings (same key name in both schemas) ---
    _direct = [
        "client_name", "filing_status", "tax_year",
        "filer_ss_already", "filer_ss_current", "filer_ss_start_year",
        "filer_ss_fra", "filer_ss_claim",
        "spouse_ss_already", "spouse_ss_current", "spouse_ss_start_year",
        "spouse_ss_fra", "spouse_ss_claim",
        "ssdi_filer", "ssdi_spouse",
        "pension_filer", "pension_spouse",
        "pension_filer_age", "pension_spouse_age",
        "pension_cola_filer", "pension_cola_spouse",
        "auto_rmd", "pretax_bal_filer_prior", "pretax_bal_spouse_prior",
        "baseline_pretax_dist",
        "wages", "tax_exempt_interest", "interest_taxable",
        "wages_start_age", "wages_end_age",
        "other_income_start_age", "other_income_end_age",
        "tax_exempt_start_age", "tax_exempt_end_age",
        "cap_gain_loss", "other_income",
        "filer_65_plus", "spouse_65_plus",
        "adjustments", "dependents", "retirement_deduction", "out_of_state_gain",
        "property_tax", "medical_expenses", "charitable", "qcd_annual",
        "emergency_fund", "life_cash_value",
        "annuity_value_f", "annuity_basis_f", "annuity_value_s", "annuity_basis_s",
        "survivor_spend_pct", "pension_survivor_pct",
        "reinvest_dividends", "reinvest_cap_gains", "reinvest_interest",
    ]
    for k in _direct:
        if k in data and k in D:
            D[k] = data[k]

    # --- Renamed fields (tax profile key → FP key) ---
    _renamed = {
        "total_ordinary_div": "total_ordinary_dividends",
        "qualified_div": "qualified_dividends",
        "r_cash_side": "r_cash",
        "r_taxable_side": "r_taxable",
        "r_pretax_side": "r_pretax",
        "r_roth_side": "r_roth",
        "r_annuity_side": "r_annuity",
        "r_life_side": "r_life",
        "heir_tab3": "heir_tax_rate",
        "ret_surplus_dest": "surplus_dest",
        "home_appreciation": "home_appr",
        "mortgage_balance": "mtg_balance",
        "mortgage_rate": "mtg_rate",
        "mortgage_payment": "mtg_payment_monthly",
    }
    for src, dst in _renamed.items():
        if src in data and dst in D:
            D[dst] = data[src]

    # --- Date fields ---
    if "filer_dob" in data:
        D["filer_dob"] = dt.date.fromisoformat(data["filer_dob"]) if isinstance(data["filer_dob"], str) else data["filer_dob"]
    if "spouse_dob" in data:
        D["spouse_dob"] = dt.date.fromisoformat(data["spouse_dob"]) if isinstance(data["spouse_dob"], str) else data["spouse_dob"]

    # --- Home value ---
    if "home_value" in data:
        D["home_value"] = data["home_value"]

    # --- Balance fields: retired profiles don't have 401k, put all in IRA ---
    if "pretax_bal_filer_current" in data:
        D["curr_trad_ira_f"] = data["pretax_bal_filer_current"]
        D["curr_401k_f"] = 0.0
    if "pretax_bal_spouse_current" in data:
        D["curr_trad_ira_s"] = data["pretax_bal_spouse_current"]
        D["curr_401k_s"] = 0.0

    # Roth — may be per-spouse or combined
    if "roth_bal_filer" in data:
        D["curr_roth_ira_f"] = data["roth_bal_filer"]
    if "roth_bal_spouse" in data:
        D["curr_roth_ira_s"] = data["roth_bal_spouse"]
    if "roth_bal" in data and "roth_bal_filer" not in data:
        D["curr_roth_ira_f"] = data["roth_bal"]
        D["curr_roth_ira_s"] = 0.0

    # Taxable brokerage
    if "taxable_brokerage_bal" in data:
        D["curr_taxable"] = data["taxable_brokerage_bal"]
    if "taxable_cash_bal" in data:
        D["curr_cash"] = data["taxable_cash_bal"]

    # Brokerage basis: convert gain_pct → dollar basis
    if "brokerage_gain_pct" in data:
        _brok = data.get("taxable_brokerage_bal", 0.0)
        D["taxable_basis"] = _brok * (1.0 - data["brokerage_gain_pct"])
    D["curr_taxable_s"] = 0.0
    D["taxable_basis_s"] = 0.0

    # Annuity — may be per-spouse or combined
    if "annuity_value_filer" in data:
        D["annuity_value_f"] = data["annuity_value_filer"]
    if "annuity_basis_filer" in data:
        D["annuity_basis_f"] = data["annuity_basis_filer"]
    if "annuity_value_spouse" in data:
        D["annuity_value_s"] = data["annuity_value_spouse"]
    if "annuity_basis_spouse" in data:
        D["annuity_basis_s"] = data["annuity_basis_spouse"]
    if "annuity_value" in data and "annuity_value_filer" not in data:
        D["annuity_value_f"] = data["annuity_value"]
        D["annuity_basis_f"] = data.get("annuity_basis", 0.0)

    # Pension COLA — tax profile has single value, FP has per-spouse
    if "pension_cola" in data:
        D["pension_cola_filer"] = data["pension_cola"]
        D["pension_cola_spouse"] = data["pension_cola"]

    # Inflation/growth rates: tax profiles store as decimal, FP stores as percentage
    if "inflation" in data:
        val = data["inflation"]
        D["inflation"] = val * 100 if val < 1 else val
    if "bracket_growth" in data:
        val = data["bracket_growth"]
        D["bracket_growth"] = val * 100 if val < 1 else val
    if "medicare_growth" in data:
        val = data["medicare_growth"]
        D["medicare_growth"] = val * 100 if val < 1 else val

    # Additional expenses & future income (no underscore prefix in tax profiles)
    if "additional_expenses" in data:
        st.session_state.additional_expenses = data["additional_expenses"]
    if "future_income" in data:
        st.session_state.future_income = data["future_income"]

    # Clear 401k and working-mode fields
    D["curr_roth_401k"] = 0.0
    D["curr_hsa"] = 0.0

    # Sync widget keys
    _keep = {"_btn_load", "_sb_load_sel", "_btn_save"}
    for wk in list(st.session_state.keys()):
        if wk.startswith("_w_"):
            _dkey = wk[3:]
            if _dkey in D:
                st.session_state[wk] = D[_dkey]
            else:
                del st.session_state[wk]
        elif wk.startswith("_sb_") and wk not in _keep:
            del st.session_state[wk]
        elif wk.startswith("fp_rr_") or wk.startswith("fp_preret_"):
            del st.session_state[wk]
    # Sync working/retired radio — tax profiles are always retired
    st.session_state["_w_status_toggle"] = "Retired"
    _so_keys = {"fp_rr_o1": "so1", "fp_rr_o2": "so2", "fp_rr_o3": "so3", "fp_rr_o4": "so4"}
    for wk, dk in _so_keys.items():
        st.session_state[wk] = D[dk]
    for ck in list(_COMP_DEFAULTS.keys()):
        if ck in st.session_state:
            st.session_state[ck] = _COMP_DEFAULTS[ck]
    return True

_profiles = sorted([f[:-5] for f in os.listdir(_PROFILE_DIR) if f.endswith(".json")])
_tax_profiles = sorted([f[:-5] for f in os.listdir(_TAX_PROFILE_DIR) if f.endswith(".json")]) if os.path.isdir(_TAX_PROFILE_DIR) else []

# One-time migration: clear stale widget keys so they pick up new defaults
if "_fp_v3_migrated" not in st.session_state:
    for _k in list(st.session_state.keys()):
        if _k.startswith("fp_rr_") or _k.startswith("fp_preret_"):
            del st.session_state[_k]
    st.session_state["_fp_v3_migrated"] = True

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.subheader("Profile")
    _pname = st.text_input("Profile name", value=D.get("client_name", ""), key="_sb_pname")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save", key="_btn_save") and _pname:
            _save_profile(_pname)
            st.toast(f"Saved: {_pname}")
    with c2:
        if st.button("Load", key="_btn_load"):
            pass  # handled below
    _all_profiles = list(_profiles)
    for _tp in _tax_profiles:
        if _tp not in _profiles:
            _all_profiles.append(f"{_tp} (Retired)")
    if _all_profiles:
        _load_sel = st.selectbox("Existing profiles", [""] + _all_profiles, key="_sb_load_sel")
        if st.session_state.get("_btn_load") and _load_sel:
            if _load_sel.endswith(" (Retired)"):
                _tp_name = _load_sel[:-10]
                _loaded = _load_tax_profile(_tp_name)
            else:
                _loaded = _load_profile(_load_sel)
            if _loaded:
                st.toast(f"Loaded: {_load_sel}")
                st.rerun()

    st.divider()
    nav = st.radio("Navigation",
                   ["Planning", "Growing", "Receiving", "Spending", "Giving", "Protecting", "Achieving"],
                   label_visibility="collapsed", key="fp_nav")

# ══════════════════════════════════════════════════════════════════════
# COMMON DERIVED
# ══════════════════════════════════════════════════════════════════════
filing_status = D["filing_status"]
is_joint = "joint" in filing_status.lower()
current_year = dt.date.today().year

# ══════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ══════════════════════════════════════════════════════════════════════
st.title("OptiPlan Wealth Optimization Planner")

# ── Prominent Working / Retired toggle ────────────────────────────────
_status_options = ["Currently Working", "Retired"]
_status_idx = 0 if D["is_working"] else 1

def _sync_working_status():
    D["is_working"] = st.session_state["_w_status_toggle"] == "Currently Working"

st.markdown("#### Status")
st.radio("", _status_options, index=_status_idx,
         key="_w_status_toggle", on_change=_sync_working_status,
         horizontal=True, label_visibility="collapsed")
st.divider()

# ══════════════════════════════════════════════════════════════════════
# PRE-RETIREMENT TABS HELPER
# ══════════════════════════════════════════════════════════════════════
def _fp_pre_ret_tabs(tab2, tab3, tab4, D, is_joint, filing_status,
                     filer_dob, spouse_dob, current_year, inflation, bracket_growth,
                     medicare_growth, pension_cola, spending_order, surplus_destination,
                     current_age_filer, current_age_spouse,
                     filer_plan_through_age, spouse_plan_through_age,
                     r_pretax, r_roth, r_taxable, r_cash,
                     r_annuity, r_life,
                     life_cash_value, annuity_value, annuity_basis,
                     filer_65_plus, spouse_65_plus,
                     wages, mortgage_balance, mortgage_rate, mortgage_payment,
                     property_tax, medical_expenses, charitable, qcd_annual,
                     home_value, home_appreciation, years,
                     pension_filer, pension_spouse,
                     filer_ss_already, filer_ss_current, filer_ss_start_year,
                     filer_ss_fra, filer_ss_claim,
                     spouse_ss_already, spouse_ss_current, spouse_ss_start_year,
                     spouse_ss_fra, spouse_ss_claim,
                     heir_tax_rate, survivor_spending_pct, pension_survivor_pct,
                     pretax_bal_filer_current, pretax_bal_spouse_current,
                     taxable_brokerage_bal, brokerage_gain_pct, taxable_cash_bal,
                     emergency_fund, interest_taxable, total_ordinary_dividends,
                     qualified_dividends, cap_gain_loss, other_income, adjustments,
                     dependents, retirement_deduction, out_of_state_gain,
                     tax_exempt_interest, start_year,
                     proj_interest, proj_dividends, proj_qual_div, proj_cap_gains,
                     reinvest_dividends, reinvest_cap_gains, reinvest_interest,
                     spending_goal):
    """Render pre-retirement Achieving tabs: Projection, Retirement Readiness, Savings Optimizer."""
    import copy

    # --- Derived values ---
    salary_filer = D["salary_filer"]
    salary_spouse = D["salary_spouse"] if is_joint else 0.0
    salary_growth = D["salary_growth"] / 100
    pre_ret_return = (r_pretax + r_roth + r_taxable) / 3  # blended fallback (per-account rates in income_info)
    post_ret_return = pre_ret_return
    ret_age = int(D["ret_age"])
    spouse_ret_age = int(D["spouse_ret_age"]) if is_joint else ret_age
    years_to_ret = max(0, ret_age - current_age_filer)
    state_rate = D["state_tax_rate"] / 100

    # SS PIA
    ss_filer_pia = filer_ss_fra
    ss_spouse_pia = spouse_ss_fra if is_joint else 0.0
    # Claim age (numeric)
    def _claim_age_num(claim_str):
        if claim_str == "FRA": return 67
        try: return int(claim_str)
        except: return 67
    ss_filer_claim_age = _claim_age_num(filer_ss_claim)
    ss_spouse_claim_age = _claim_age_num(spouse_ss_claim) if is_joint else 67

    # Contribution limits (2025)
    _401k_limit = 23500.0
    _401k_catch_up = 7500.0
    _simple_limit = 16500.0
    _simple_catch_up = 3500.0
    _sep_max_cap = 69000.0
    _ira_limit = 7000.0
    _ira_catch_up = 1000.0
    _hsa_family = 8550.0
    _hsa_single = 4300.0
    _hsa_catch_up = 1000.0

    filer_50_plus = current_age_filer >= 50
    spouse_50_plus = (current_age_spouse >= 50) if current_age_spouse else False

    # Compute contributions based on plan type — Filer
    _plan_f = D["plan_type_f"]
    if _plan_f == "401(k)":
        _lim_f = _401k_limit + (_401k_catch_up if filer_50_plus else 0)
        if D["max_401k_f"]:
            contrib_401k_filer = _lim_f
        else:
            contrib_401k_filer = min(D["c401k_f"], _lim_f)
        roth_pct_f = D["roth_pct_f"] / 100
        contrib_401k_pretax_filer = contrib_401k_filer * (1 - roth_pct_f)
        contrib_401k_roth_filer = contrib_401k_filer * roth_pct_f
        employer_match = min(salary_filer * D["ematch_rate_f"] / 100, salary_filer * D["ematch_upto_f"] / 100)
    elif _plan_f == "SEP-IRA":
        _sep_max_f = min(salary_filer * 0.25, _sep_max_cap)
        if D["max_401k_f"]:
            contrib_401k_filer = _sep_max_f
        else:
            contrib_401k_filer = min(D["c401k_f"], _sep_max_f)
        contrib_401k_pretax_filer = contrib_401k_filer  # All pre-tax
        contrib_401k_roth_filer = 0.0
        employer_match = 0.0  # SEP IS the employer contribution
    elif _plan_f == "SIMPLE":
        _lim_f = _simple_limit + (_simple_catch_up if filer_50_plus else 0)
        if D["max_401k_f"]:
            contrib_401k_filer = _lim_f
        else:
            contrib_401k_filer = min(D["c401k_f"], _lim_f)
        roth_pct_f = D["roth_pct_f"] / 100
        contrib_401k_pretax_filer = contrib_401k_filer * (1 - roth_pct_f)
        contrib_401k_roth_filer = contrib_401k_filer * roth_pct_f
        employer_match = salary_filer * 0.03  # 3% mandatory match
    else:  # "None"
        contrib_401k_filer = 0.0
        contrib_401k_pretax_filer = 0.0
        contrib_401k_roth_filer = 0.0
        employer_match = 0.0

    # Compute contributions based on plan type — Spouse
    if is_joint and salary_spouse > 0:
        _plan_s = D["plan_type_s"]
        if _plan_s == "401(k)":
            _lim_s = _401k_limit + (_401k_catch_up if spouse_50_plus else 0)
            if D["max_401k_s"]:
                contrib_401k_spouse = _lim_s
            else:
                contrib_401k_spouse = min(D["c401k_s"], _lim_s)
            roth_pct_s = D["roth_pct_s"] / 100
            contrib_401k_pretax_spouse = contrib_401k_spouse * (1 - roth_pct_s)
            contrib_401k_roth_spouse = contrib_401k_spouse * roth_pct_s
            employer_match_spouse = min(salary_spouse * D["ematch_rate_s"] / 100, salary_spouse * D["ematch_upto_s"] / 100)
        elif _plan_s == "SEP-IRA":
            _sep_max_s = min(salary_spouse * 0.25, _sep_max_cap)
            if D["max_401k_s"]:
                contrib_401k_spouse = _sep_max_s
            else:
                contrib_401k_spouse = min(D["c401k_s"], _sep_max_s)
            contrib_401k_pretax_spouse = contrib_401k_spouse
            contrib_401k_roth_spouse = 0.0
            employer_match_spouse = 0.0
        elif _plan_s == "SIMPLE":
            _lim_s = _simple_limit + (_simple_catch_up if spouse_50_plus else 0)
            if D["max_401k_s"]:
                contrib_401k_spouse = _lim_s
            else:
                contrib_401k_spouse = min(D["c401k_s"], _lim_s)
            roth_pct_s = D["roth_pct_s"] / 100
            contrib_401k_pretax_spouse = contrib_401k_spouse * (1 - roth_pct_s)
            contrib_401k_roth_spouse = contrib_401k_spouse * roth_pct_s
            employer_match_spouse = salary_spouse * 0.03
        else:  # "None"
            contrib_401k_pretax_spouse = contrib_401k_roth_spouse = employer_match_spouse = 0.0
    else:
        contrib_401k_pretax_spouse = contrib_401k_roth_spouse = employer_match_spouse = 0.0

    # IRA
    ira_max_f = _ira_limit + (_ira_catch_up if filer_50_plus else 0)
    ira_max_s = _ira_limit + (_ira_catch_up if spouse_50_plus else 0)
    if D["backdoor_roth_f"]:
        contrib_trad_ira = 0.0
        contrib_roth_ira = min(D["contrib_roth_ira_f"], ira_max_f)
    elif D["max_ira_f"]:
        contrib_trad_ira = ira_max_f
        contrib_roth_ira = 0.0
    else:
        contrib_trad_ira = min(D["contrib_trad_ira_f"], ira_max_f)
        contrib_roth_ira = min(D["contrib_roth_ira_f"], max(0, ira_max_f - contrib_trad_ira))

    if is_joint and salary_spouse > 0:
        if D["backdoor_roth_s"]:
            contrib_trad_ira_spouse = 0.0
            contrib_roth_ira_spouse = min(D["contrib_roth_ira_s"], ira_max_s)
        elif D["max_ira_s"]:
            contrib_trad_ira_spouse = ira_max_s
            contrib_roth_ira_spouse = 0.0
        else:
            contrib_trad_ira_spouse = min(D["contrib_trad_ira_s"], ira_max_s)
            contrib_roth_ira_spouse = min(D["contrib_roth_ira_s"], max(0, ira_max_s - contrib_trad_ira_spouse))
    else:
        contrib_trad_ira_spouse = contrib_roth_ira_spouse = 0.0

    # HSA
    if D["hsa_eligible"]:
        hsa_max = (_hsa_family if is_joint else _hsa_single) + (_hsa_catch_up if filer_50_plus else 0)
        contrib_hsa = hsa_max if D["max_hsa"] else min(D["contrib_hsa"], hsa_max)
    else:
        contrib_hsa = 0.0

    contrib_taxable = D["contrib_taxable"]

    # Build dicts
    _annual_pretax_filer = contrib_401k_pretax_filer + employer_match + contrib_trad_ira
    _annual_pretax_spouse = contrib_401k_pretax_spouse + employer_match_spouse + contrib_trad_ira_spouse
    _annual_roth_filer = contrib_401k_roth_filer + contrib_roth_ira
    _annual_roth_spouse = contrib_401k_roth_spouse + contrib_roth_ira_spouse

    contrib_dict = {
        "pretax": _annual_pretax_filer + _annual_pretax_spouse,
        "roth": _annual_roth_filer + _annual_roth_spouse,
        "taxable": contrib_taxable, "cash": 0.0, "hsa": contrib_hsa,
        "pretax_filer": _annual_pretax_filer, "pretax_spouse": _annual_pretax_spouse,
        "roth_filer": _annual_roth_filer, "roth_spouse": _annual_roth_spouse,
    }

    total_pretax = pretax_bal_filer_current + pretax_bal_spouse_current
    total_roth = D["curr_roth_ira_f"] + (D["curr_roth_ira_s"] if is_joint else 0) + D["curr_roth_401k"]
    _total_basis = min(D["taxable_basis"] + (D["taxable_basis_s"] if is_joint else 0), taxable_brokerage_bal)
    start_balances = {
        "pretax": total_pretax,
        "roth": total_roth,
        "brokerage": taxable_brokerage_bal,
        "cash": max(0, taxable_cash_bal - emergency_fund),
        "taxable": taxable_brokerage_bal + taxable_cash_bal,
        "brokerage_basis": _total_basis,
        "hsa": D["curr_hsa"],
    }

    pretax_deductions = (contrib_401k_pretax_filer + contrib_401k_pretax_spouse +
                         contrib_trad_ira + contrib_trad_ira_spouse + contrib_hsa)

    # Inherited IRAs from Growing tab
    inherited_iras = []  # TODO: wire if Growing tab gets inherited IRA inputs

    # Future expenses from Spending tab
    _ae_active = [ae for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0]
    future_expenses = []
    for ae in _ae_active:
        end_age = ae["end_age"] if ae["end_age"] > ae["start_age"] else ae["start_age"] + 1
        future_expenses.append({"name": ae.get("name", ""), "amount": ae["net_amount"],
                                "start_age": ae["start_age"], "end_age": end_age,
                                "inflates": ae.get("inflates", False)})

    # Dividend/cap gains yields for projections
    div_yield = D["proj_div_yield"] / 100 if D["use_invest_assumptions"] else 0.015
    ann_cg_pct = D["proj_cg_pct"] / 100 if D["use_invest_assumptions"] else 0.0
    cash_int_rate = D["proj_int_rate"] / 100 if D["use_invest_assumptions"] else r_cash
    reinvest_inv = reinvest_dividends and reinvest_cap_gains and reinvest_interest

    # Mortgage
    mtg_years = int(D["mtg_years"])
    mtg_payment_annual = D["mtg_payment_monthly"] * 12

    income_info = {
        "total_income": salary_filer + salary_spouse + D["other_income"],
        "salary_filer": salary_filer, "salary_spouse": salary_spouse,
        "other_income": D["other_income"], "other_income_tax_free": False,
        "other_income_inflation": True, "other_income_years": 0,
        "living_expenses": D["living_expenses"],
        "mortgage_annual": mtg_payment_annual, "mortgage_years": mtg_years,
        "filing_status": filing_status, "state_rate": state_rate,
        "pretax_deductions": pretax_deductions,
        "inflation": inflation, "bracket_growth": bracket_growth,
        "medicare_growth": medicare_growth,
        "r_pretax": r_pretax, "r_roth": r_roth, "r_taxable": r_taxable, "r_hsa": r_roth,
        "dividend_yield": div_yield, "annual_cap_gain_pct": ann_cg_pct,
        "cash_interest_rate": cash_int_rate, "reinvest_inv_income": reinvest_inv,
        "interest_taxable": D["interest_taxable"],
        "total_ordinary_dividends": D["total_ordinary_dividends"],
        "qualified_dividends": D["qualified_dividends"],
        "ss_filer_pia": ss_filer_pia, "ss_spouse_pia": ss_spouse_pia,
        "ss_filer_claim_age": ss_filer_claim_age, "ss_spouse_claim_age": ss_spouse_claim_age,
        "ssdi_filer": D["ssdi_filer"], "ssdi_spouse": D["ssdi_spouse"] if is_joint else False,
        "self_employed_filer": bool(D["self_employed_f"]),
        "self_employed_spouse": bool(D["self_employed_s"]) if is_joint else False,
        "ss_filer_already": filer_ss_already,
        "ss_filer_current_benefit": filer_ss_current if filer_ss_already else 0,
        "ss_spouse_already": spouse_ss_already if is_joint else False,
        "ss_spouse_current_benefit": spouse_ss_current if (is_joint and spouse_ss_already) else 0,
        "current_age": current_age_filer,
        "spouse_age": current_age_spouse,
        "retire_age": ret_age, "spouse_retire_age": spouse_ret_age,
        "filer_dob": filer_dob, "spouse_dob": spouse_dob,
        "deficit_action": "reduce_savings",
        "surplus_destination": "none" if D["surplus_dest"] == "Don't Reinvest" else ("cash" if D["surplus_dest"] == "Cash/Savings" else "brokerage"),
        "future_expenses": future_expenses,
        "inherited_iras": inherited_iras,
        "itemize": False,
        "mortgage_balance": mortgage_balance, "mortgage_rate": mortgage_rate,
        "property_tax": property_tax, "medical_expenses": medical_expenses,
        "charitable": charitable,
        "home_value": home_value, "home_appreciation": home_appreciation,
        "heir_bracket_option": "same",
        "life_cash_value": life_cash_value, "r_life": r_life,
        "annuity_value": annuity_value, "annuity_basis": annuity_basis, "r_annuity": r_annuity,
        # Estate tax
        "estate_tax_enabled": bool(D["estate_tax_enabled"]),
        "federal_estate_exemption": float(D["federal_estate_exemption"]),
        "exemption_inflation": float(D["exemption_inflation"]),
        "use_portability": bool(D["use_portability"]),
        "state_estate_tax_rate": float(D["state_estate_tax_rate"]),
        "state_estate_exemption": float(D["state_estate_exemption"]),
    }

    # --- Investment Real Estate at retirement ---
    _ire_value = D["inv_re_value"]
    _ire_basis = D["inv_re_basis"]
    _ire_appr = D["inv_re_appr"] / 100
    _ire_net_income = D["inv_re_net_income"]
    _ire_liquidate = D["inv_re_liquidate"]
    # If individual details, compute per-property liquidation
    _ire_liq_equity = 0.0  # net equity from liquidated properties (added to brokerage)
    _ire_liq_gain = 0.0    # taxable gain from liquidated properties
    _ire_keep_value = 0.0  # appreciated value of non-liquidated properties (for estate)
    _ire_keep_income = 0.0 # ongoing rental income from non-liquidated properties
    _ire_keep_value_today = 0.0  # current (today's) value of non-liquidated properties
    _ire_keep_appr = _ire_appr   # weighted avg appreciation for non-liquidated properties
    _ire_total_today = 0.0       # ALL inv RE (for accumulation display, regardless of liquidation)
    inv_re_properties = []  # per-property income tracking for engines

    # --- Process base (total-level) investment RE ---
    _wt_appr_num = 0.0; _wt_appr_den = 0.0
    if _ire_value > 0:
        _ire_total_today += _ire_value
        _wt_appr_num += _ire_value * _ire_appr
        _wt_appr_den += _ire_value
        _rv_at_ret = _ire_value * ((1 + _ire_appr) ** years_to_ret)
        _total_mtg_pmt = D["inv_re_mortgage_pmt"]
        _total_mtg_yrs = int(D["inv_re_mortgage_years"])
        _total_inc_growth = D["inv_re_income_growth"] / 100
        if _ire_liquidate:
            _ire_liq_equity += _rv_at_ret
            _ire_liq_gain += max(0, _rv_at_ret - _ire_basis)
        else:
            _ire_keep_value += _rv_at_ret
            _ire_keep_value_today += _ire_value
            _ire_keep_income += _ire_net_income * ((1 + _total_inc_growth) ** years_to_ret)
            if _ire_net_income > 0:
                inv_re_properties.append({
                    "name": "Investment RE",
                    "net_income": _ire_net_income,
                    "income_growth": _total_inc_growth,
                    "mortgage_pmt": _total_mtg_pmt,
                    "mortgage_years": _total_mtg_yrs,
                })

    # --- Process additional properties (additive to base) ---
    for _ri in range(1, 4):
        _rpfx = f"inv_re_{_ri}"
        if not D.get(f"{_rpfx}_include", False):
            continue
        _rv_today = D[f"{_rpfx}_value"]
        if _rv_today <= 0:
            continue
        _ire_total_today += _rv_today
        _rv_appr = D[f"{_rpfx}_appr"] / 100
        _wt_appr_num += _rv_today * _rv_appr
        _wt_appr_den += _rv_today
        _rv = _rv_today * ((1 + _rv_appr) ** years_to_ret)
        _rb = D[f"{_rpfx}_basis"]
        _rm = D[f"{_rpfx}_mortgage"]
        _rinc = D[f"{_rpfx}_net_income"]
        _rinc_growth = D[f"{_rpfx}_income_growth"] / 100
        _rmtg_pmt = engine.calc_mortgage_payment(_rm, D[f"{_rpfx}_rate"] / 100, D[f"{_rpfx}_years"])
        _rmtg_years = int(D[f"{_rpfx}_years"])
        if D[f"{_rpfx}_liquidate"]:
            _ire_liq_equity += max(0, _rv - _rm)
            _ire_liq_gain += max(0, _rv - _rb)
        else:
            _ire_keep_value += _rv
            _ire_keep_value_today += _rv_today
            _ire_keep_income += _rinc * ((1 + _rinc_growth) ** years_to_ret)
            inv_re_properties.append({
                "name": D[f"{_rpfx}_name"],
                "net_income": _rinc,
                "income_growth": _rinc_growth,
                "mortgage_pmt": _rmtg_pmt,
                "mortgage_years": _rmtg_years,
            })
    if _wt_appr_den > 0:
        _ire_keep_appr = _wt_appr_num / _wt_appr_den

    # Add investment RE to income_info for accumulation engine
    # Track ALL inv RE during accumulation (liquidation split happens at retirement)
    _ire_total_appr = (_wt_appr_num / _wt_appr_den) if _wt_appr_den > 0 else _ire_appr
    income_info["inv_re_properties"] = inv_re_properties
    income_info["inv_re_value"] = _ire_total_today
    income_info["inv_re_appr"] = _ire_total_appr

    def _build_retire_params(accum_result):
        """Build params dict for run_retirement_projection from accumulation results."""
        _final_row = accum_result["rows"][-1] if accum_result["rows"] else {}
        inf_factor = (1 + inflation) ** years_to_ret
        ret_expenses = D["living_expenses"] * (D["ret_pct"] / 100) * inf_factor
        # Pensions inflated to retirement
        pen_f_cola = D["pension_cola_filer"] / 100
        pen_s_cola = D["pension_cola_spouse"] / 100
        pen_f_at_ret = pension_filer * ((1 + pen_f_cola) ** years_to_ret) if pension_filer > 0 else 0.0
        pen_s_at_ret = pension_spouse * ((1 + pen_s_cola) ** years_to_ret) if pension_spouse > 0 else 0.0
        # SS inflated to retirement
        if filer_ss_already and filer_ss_current > 0:
            ss_f_fra_at_ret = filer_ss_current * inf_factor
        else:
            ss_f_fra_at_ret = ss_filer_pia * inf_factor
        if is_joint and spouse_ss_already and spouse_ss_current > 0:
            ss_s_fra_at_ret = spouse_ss_current * inf_factor
        else:
            ss_s_fra_at_ret = ss_spouse_pia * inf_factor
        # Home value (primary residence only)
        home_val_at_ret = home_value * ((1 + home_appreciation) ** years_to_ret)
        # Mortgage at retirement
        mtg_yrs_at_ret = max(0, mtg_years - years_to_ret)
        # Future expenses adjusted for retirement start
        ret_fut_exp = []
        for fe in future_expenses:
            if fe["end_age"] > ret_age:
                ret_fut_exp.append(dict(fe))
        # Inherited IRAs state from accumulation
        iira_state = accum_result.get("inherited_iras_state", [])
        # Surplus destination
        surp_dest = "none" if D["surplus_dest"] == "Don't Reinvest" else ("cash" if D["surplus_dest"] == "Cash/Savings" else "brokerage")
        return {
            "retire_age": ret_age,
            "filer_life_expectancy": filer_plan_through_age,
            "spouse_life_expectancy": spouse_plan_through_age,
            "survivor_spending_pct": survivor_spending_pct,
            "pension_survivor_pct": pension_survivor_pct,
            "retire_year": current_year + years_to_ret,
            "inflation": inflation, "bracket_growth": bracket_growth,
            "medicare_growth": medicare_growth,
            "post_retire_return": post_ret_return,
            "filing_status": filing_status,
            "state_tax_rate": state_rate,
            "expenses_at_retirement": ret_expenses,
            "ss_filer_fra": ss_f_fra_at_ret, "ss_spouse_fra": ss_s_fra_at_ret,
            "ss_filer_claim_age": ss_filer_claim_age,
            "ss_spouse_claim_age": ss_spouse_claim_age,
            "filer_dob": filer_dob,
            "spouse_dob": spouse_dob if is_joint else None,
            "ssdi_filer": D["ssdi_filer"],
            "ssdi_spouse": D["ssdi_spouse"] if is_joint else False,
            "ss_filer_already": filer_ss_already,
            "ss_filer_current_benefit": filer_ss_current if filer_ss_already else 0,
            "ss_spouse_already": spouse_ss_already if is_joint else False,
            "ss_spouse_current_benefit": spouse_ss_current if (is_joint and spouse_ss_already) else 0,
            "pension_filer_at_retire": pen_f_at_ret,
            "pension_filer_start_age": int(D["pension_filer_age"]),
            "pension_filer_cola": pen_f_cola,
            "pension_spouse_at_retire": pen_s_at_ret,
            "pension_spouse_start_age": int(D["pension_spouse_age"]) if is_joint else 65,
            "pension_spouse_cola": pen_s_cola,
            "spouse_age_at_retire": (current_age_spouse + years_to_ret) if current_age_spouse else None,
            "mortgage_payment": mtg_payment_annual,
            "mortgage_years_at_retire": mtg_yrs_at_ret,
            "home_value_at_retire": home_val_at_ret,
            "home_appreciation": home_appreciation,
            "inv_re_value_at_retire": _ire_keep_value,
            "inv_re_appr": _ire_keep_appr,
            "future_expenses": ret_fut_exp,
            "dividend_yield": accum_result.get("derived_div_yield", div_yield),
            "brok_interest_yield": accum_result.get("derived_brok_int_yield", 0.0),
            "cash_interest_rate": cash_int_rate,
            "charitable": charitable, "qcd_annual": qcd_annual,
            "inherited_iras": iira_state,
            "surplus_destination": surp_dest,
            "heir_bracket_option": "same",
            "other_income": D["other_income"] * inf_factor + (_ire_keep_income if not inv_re_properties else 0.0),
            "other_income_tax_free": False,
            "other_income_inflation": True,
            "other_income_years": 999,
            "inv_re_properties": [{
                "name": p["name"],
                "net_income": p["net_income"] * ((1 + p["income_growth"]) ** years_to_ret),
                "income_growth": p["income_growth"],
                "mortgage_pmt": p["mortgage_pmt"],
                "mortgage_years": max(0, p["mortgage_years"] - years_to_ret),
            } for p in inv_re_properties],
            "life_cash_value": life_cash_value * ((1 + r_life) ** years_to_ret),
            "r_life": r_life,
            "annuity_value": annuity_value * ((1 + r_annuity) ** years_to_ret),
            "annuity_basis": annuity_basis,
            "r_annuity": r_annuity,
            # Estate tax
            "estate_tax_enabled": bool(D["estate_tax_enabled"]),
            "federal_estate_exemption": float(D["federal_estate_exemption"]),
            "exemption_inflation": float(D["exemption_inflation"]),
            "use_portability": bool(D["use_portability"]),
            "state_estate_tax_rate": float(D["state_estate_tax_rate"]),
            "state_estate_exemption": float(D["state_estate_exemption"]),
        }

    def _run_accum():
        """Run accumulation and return result."""
        return engine.run_accumulation(
            current_age_filer, years_to_ret, copy.deepcopy(start_balances),
            dict(contrib_dict), salary_growth, pre_ret_return, income_info)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 (PRE-RET): Retirement Readiness
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Retirement Readiness")
        st.caption(f"Projects wealth growth from now (age {current_age_filer}) to retirement (age {ret_age}), {years_to_ret} years.")

        # Show current savings summary
        st.markdown("### Current Balances")
        _cur_portfolio = total_pretax + total_roth + taxable_brokerage_bal + taxable_cash_bal + D["curr_hsa"]
        cols = st.columns(6)
        with cols[0]: st.metric("Pre-Tax", TEA.money(total_pretax))
        with cols[1]: st.metric("Roth", TEA.money(total_roth))
        with cols[2]: st.metric("Taxable", TEA.money(taxable_brokerage_bal + taxable_cash_bal))
        with cols[3]: st.metric("HSA", TEA.money(D["curr_hsa"]))
        with cols[4]: st.metric("Portfolio", TEA.money(_cur_portfolio))

        # Non-portfolio assets
        _cur_home = home_value
        _cur_ire = _ire_total_today
        _cur_life = life_cash_value
        _cur_ann = annuity_value
        _cur_has_other = _cur_home > 0 or _cur_ire > 0 or _cur_life > 0 or _cur_ann > 0
        if _cur_has_other:
            _cur_items = []
            if _cur_home > 0: _cur_items.append(("Home", _cur_home))
            if _cur_ire > 0: _cur_items.append(("Inv RE", _cur_ire))
            if _cur_life > 0: _cur_items.append(("Life Ins", _cur_life))
            if _cur_ann > 0: _cur_items.append(("Annuity", _cur_ann))
            _cur_gross = _cur_portfolio + _cur_home + _cur_ire + _cur_life + _cur_ann
            _cur_items.append(("Gross Estate", _cur_gross))
            cols2 = st.columns(len(_cur_items))
            for _ci, (_lbl, _val) in enumerate(_cur_items):
                with cols2[_ci]: st.metric(_lbl, TEA.money(_val))

        st.caption(f"Annual savings: Pre-Tax {TEA.money(contrib_dict['pretax'])} | Roth {TEA.money(contrib_dict['roth'])} | "
                   f"Taxable {TEA.money(contrib_taxable)} | HSA {TEA.money(contrib_hsa)} | "
                   f"Total {TEA.money(contrib_dict['pretax'] + contrib_dict['roth'] + contrib_taxable + contrib_hsa)}")

        with st.expander("Monte Carlo Settings"):
            mc_c1, mc_c2, mc_c3 = st.columns(3)
            with mc_c1: mc_sims = st.number_input("Simulations", min_value=100, max_value=2000, value=500, step=100, key="fp_preret_mc_sims")
            with mc_c2: mc_vol = st.number_input("Return Std Dev (%)", value=12.0, step=1.0, key="fp_preret_mc_vol") / 100
            with mc_c3: mc_post_ret = st.number_input("Post-Retire Return (%)", value=post_ret_return * 100, step=0.5, key="fp_preret_mc_post") / 100

        if st.button("Run Projection", type="primary", key="fp_preret_run_proj"):
            accum_result = _run_accum()
            st.session_state.projection_results = accum_result
            # Run Monte Carlo automatically
            _ret_years = max(filer_plan_through_age, spouse_plan_through_age) - ret_age + 1
            n_total = years_to_ret + max(1, _ret_years)
            def _mc_run(return_seq):
                _ii = dict(income_info)
                _ii["return_sequence"] = return_seq[:years_to_ret + 1]
                _acc = engine.run_accumulation(
                    current_age_filer, years_to_ret, copy.deepcopy(start_balances),
                    dict(contrib_dict), salary_growth, pre_ret_return, _ii)
                _final = _acc["rows"][-1] if _acc["rows"] else {}
                _mc_brok = _acc.get("final_brokerage", _final.get("Bal Taxable", 0))
                _mc_cash = _acc.get("final_cash", 0)
                if _ire_liq_equity > 0:
                    _mc_brok += _ire_liq_equity
                _bals = {
                    "pretax": _final.get("Bal Pre-Tax", 0),
                    "roth": _final.get("Bal Roth", 0),
                    "brokerage": _mc_brok,
                    "cash": _mc_cash,
                    "taxable": _mc_brok + _mc_cash,
                    "brokerage_basis": _acc.get("final_basis", 0) + _ire_liq_equity,
                    "hsa": _final.get("Bal HSA", 0),
                }
                _rp = _build_retire_params(_acc)
                _rp["post_retire_return"] = mc_post_ret
                _ret_order = list(spending_order)
                _ret_seq = return_seq[years_to_ret + 1:] if len(return_seq) > years_to_ret + 1 else None
                if _ret_seq and len(_ret_seq) >= _ret_years:
                    _rp["return_sequence"] = _ret_seq
                _ret = engine.run_retirement_projection(_bals, _rp, _ret_order)
                return {"estate": _ret["estate"], "final_total": _ret["final_total"],
                        "retire_portfolio": _final.get("Portfolio", 0)}
            with st.spinner("Running projection and Monte Carlo..."):
                mc_result = engine.run_monte_carlo(
                    _mc_run, n_sims=int(mc_sims), mean_return=pre_ret_return,
                    return_std=mc_vol, n_years=n_total, seed=42,
                    mean_return_post=mc_post_ret, n_years_pre=years_to_ret)
            st.session_state._preret_mc = mc_result
        if st.session_state.projection_results:
            accum_result = st.session_state.projection_results
            rows = accum_result["rows"]
            if rows:
                # Display year-by-year table
                _hide = ["Basis", "Unreal Gain"]
                _df = pd.DataFrame(rows)
                # Hide zero columns
                for c in list(_df.columns):
                    if c in _hide:
                        _df = _df.drop(columns=[c], errors="ignore")
                    elif _df[c].dtype in ['float64', 'int64'] and (_df[c] == 0).all():
                        _df = _df.drop(columns=[c], errors="ignore")
                st.dataframe(_df, use_container_width=True, hide_index=True)

                # Summary at retirement
                final = rows[-1]
                st.markdown("### At Retirement")
                cols = st.columns(6)
                with cols[0]: st.metric("Pre-Tax", TEA.money(final.get("Bal Pre-Tax", 0)))
                with cols[1]: st.metric("Roth", TEA.money(final.get("Bal Roth", 0)))
                with cols[2]: st.metric("Taxable", TEA.money(final.get("Bal Taxable", 0)))
                with cols[3]: st.metric("HSA", TEA.money(final.get("Bal HSA", 0)))
                inh = final.get("Bal Inherited", 0)
                if inh > 0:
                    with cols[4]: st.metric("Inherited IRA", TEA.money(inh))
                with cols[4 if inh == 0 else 5]: st.metric("Portfolio", TEA.money(final.get("Portfolio", 0)))

                # Non-portfolio assets at retirement
                _ret_home = final.get("Home Value", 0)
                _ret_ire = final.get("Inv RE", 0)
                _ret_life = final.get("Life Ins", 0)
                _ret_ann = final.get("Annuity", 0)
                _has_other = _ret_home > 0 or _ret_ire > 0 or _ret_life > 0 or _ret_ann > 0
                if _has_other:
                    _other_items = []
                    if _ret_home > 0: _other_items.append(("Home", _ret_home))
                    if _ret_ire > 0: _other_items.append(("Inv RE", _ret_ire))
                    if _ret_life > 0: _other_items.append(("Life Ins", _ret_life))
                    if _ret_ann > 0: _other_items.append(("Annuity", _ret_ann))
                    _other_items.append(("Gross Estate", final.get("Gross Estate", 0)))
                    _other_items.append(("Estate (Net)", final.get("Estate (Net)", 0)))
                    cols2 = st.columns(len(_other_items))
                    for _ci, (_lbl, _val) in enumerate(_other_items):
                        with cols2[_ci]: st.metric(_lbl, TEA.money(_val))

                # Investment Real Estate at retirement
                if _ire_liq_equity > 0 or _ire_keep_value > 0 or _ire_keep_income > 0:
                    st.markdown("**Investment Real Estate**")
                    _re_parts = []
                    if _ire_liq_equity > 0:
                        _re_parts.append(f"Liquidation proceeds: {TEA.money(_ire_liq_equity)} "
                                         f"(taxable gain: {TEA.money(_ire_liq_gain)}) → added to brokerage")
                    if _ire_keep_value > 0:
                        _re_parts.append(f"Retained property value: {TEA.money(_ire_keep_value)} → estate")
                    if _ire_keep_income > 0:
                        _re_parts.append(f"Ongoing rental income: {TEA.money(_ire_keep_income)}/yr")
                    for _rep_prop in inv_re_properties:
                        if _rep_prop["mortgage_pmt"] > 0 and _rep_prop["mortgage_years"] > 0:
                            _yrs_left_at_ret = max(0, _rep_prop["mortgage_years"] - years_to_ret)
                            if _yrs_left_at_ret > 0:
                                _re_parts.append(f"{_rep_prop['name']}: mortgage paid off {_yrs_left_at_ret} yrs into retirement "
                                                 f"→ income increases by {TEA.money(_rep_prop['mortgage_pmt'])}/yr")
                            else:
                                _re_parts.append(f"{_rep_prop['name']}: mortgage paid off before retirement")
                    for _rp_line in _re_parts:
                        st.write(f"- {_rp_line}")

            # --- Monte Carlo ---
            st.info("Monte Carlo uses the same default withdrawal order: **Taxable → Pre-Tax → Tax-Free → Tax-Deferred**. "
                    "All spending orders are fully optimized once you switch to Retired mode!")

            if st.session_state.get("_preret_mc"):
                mc = st.session_state._preret_mc
                _mfmt = lambda v: f"${float(v):,.0f}"
                st.markdown("### At End of Plan")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("MC Success Rate", f"{mc['success_rate'] * 100:.1f}%")
                with c2: st.metric("Median Portfolio", _mfmt(mc.get("median_portfolio", 0)))
                with c3: st.metric("10th Pctile", _mfmt(mc.get("portfolio_p10", 0)))
                with c4: st.metric("90th Pctile", _mfmt(mc.get("portfolio_p90", 0)))
                st.caption("Monte Carlo portfolio at end of plan (excludes home equity)")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 (PRE-RET): Retirement Projection
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Retirement Projection")
        st.caption("Run accumulation through retirement to test if your savings last.")
        st.info("Default withdrawal order: **Taxable → Pre-Tax → Tax-Free → Tax-Deferred**. "
                "You can change it below. Once you are retired, use the **Income Optimizer** tab to find the best strategy!")

        # Show investment RE impact
        if _ire_liq_equity > 0 or _ire_keep_value > 0 or _ire_keep_income > 0:
            with st.expander("Investment Real Estate in Retirement", expanded=True):
                if _ire_liq_equity > 0:
                    st.write(f"Liquidation proceeds at retirement: **{TEA.money(_ire_liq_equity)}** "
                             f"(taxable gain: {TEA.money(_ire_liq_gain)}) → added to brokerage")
                if _ire_keep_value > 0:
                    st.write(f"Retained property value at retirement: **{TEA.money(_ire_keep_value)}** → included in estate")
                if _ire_keep_income > 0:
                    st.write(f"Ongoing rental income at retirement: **{TEA.money(_ire_keep_income)}**/yr → included as other income")

        # Run accumulation first
        if st.session_state.projection_results is None:
            st.info("Run Projection (Tab 2) first, or click below to run automatically.")
        accum_result = st.session_state.projection_results

        colL, colR = st.columns(2)
        with colL:
            rr_ret_pct = st.slider("Retirement Spending (% of current)", 50, 100, int(D["ret_pct"]), step=5, key="fp_rr_ret_pct")
            if is_joint:
                rr_survivor = st.slider("Survivor Spending %", 50, 100, int(D["survivor_spend_pct"]), step=5, key="fp_rr_survivor")
                rr_pension_surv = st.slider("Pension Survivor %", 0, 100, int(D["pension_survivor_pct"]), step=5, key="fp_rr_pen_surv")
            else:
                rr_survivor = 100; rr_pension_surv = 0
        with colR:
            st.markdown("**Withdrawal Order**")
            _rr_opts = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
            _rr_def = [D["so1"], D["so2"], D["so3"], D["so4"]]
            rr_o1 = st.selectbox("1st", _rr_opts, index=_rr_opts.index(_rr_def[0]) if _rr_def[0] in _rr_opts else 0, key="fp_rr_o1")
            rr_o2 = st.selectbox("2nd", _rr_opts, index=_rr_opts.index(_rr_def[1]) if _rr_def[1] in _rr_opts else 1, key="fp_rr_o2")
            rr_o3 = st.selectbox("3rd", _rr_opts, index=_rr_opts.index(_rr_def[2]) if _rr_def[2] in _rr_opts else 2, key="fp_rr_o3")
            rr_o4 = st.selectbox("4th", _rr_opts, index=_rr_opts.index(_rr_def[3]) if _rr_def[3] in _rr_opts else 3, key="fp_rr_o4")
            rr_order = [rr_o1, rr_o2, rr_o3, rr_o4]
            rr_heir = st.selectbox("Heir bracket", ["Same as mine", "One bracket lower", "One bracket higher"], key="fp_rr_heir")
            rr_heir_opt = {"Same as mine": "same", "One bracket lower": "lower", "One bracket higher": "higher"}[rr_heir]

        if st.button("Run Retirement Readiness", type="primary", key="fp_rr_run"):
            # Run accumulation if not already done
            if accum_result is None:
                accum_result = _run_accum()
                st.session_state.projection_results = accum_result
            _final = accum_result["rows"][-1] if accum_result["rows"] else {}
            _brok = accum_result.get("final_brokerage", _final.get("Bal Taxable", 0))
            _cash_bal = accum_result.get("final_cash", 0)
            # Add liquidation proceeds from investment RE
            if _ire_liq_equity > 0:
                _brok += _ire_liq_equity
            _bals = {
                "pretax": _final.get("Bal Pre-Tax", 0),
                "roth": _final.get("Bal Roth", 0),
                "brokerage": _brok,
                "cash": _cash_bal,
                "taxable": _brok + _cash_bal,
                "brokerage_basis": accum_result.get("final_basis", 0) + _ire_liq_equity,
                "hsa": _final.get("Bal HSA", 0),
            }
            _rp = _build_retire_params(accum_result)
            # Override with tab-specific inputs
            _rp["expenses_at_retirement"] = D["living_expenses"] * (rr_ret_pct / 100) * ((1 + inflation) ** years_to_ret)
            _rp["survivor_spending_pct"] = rr_survivor
            _rp["pension_survivor_pct"] = rr_pension_surv
            _rp["heir_bracket_option"] = rr_heir_opt
            ret_result = engine.run_retirement_projection(_bals, _rp, rr_order)
            st.session_state.retire_projection = ret_result

        if st.session_state.retire_projection:
            rr = st.session_state.retire_projection
            rr_rows = rr["rows"]
            if rr_rows:
                _rr_df = pd.DataFrame(rr_rows)
                for c in list(_rr_df.columns):
                    if _rr_df[c].dtype in ['float64', 'int64'] and (_rr_df[c] == 0).all():
                        _rr_df = _rr_df.drop(columns=[c], errors="ignore")
                st.dataframe(_rr_df, use_container_width=True, hide_index=True)
                st.markdown("### End of Plan Summary")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Final Portfolio", TEA.money(rr["final_total"]))
                with c2: st.metric("Net Estate", TEA.money(rr["estate"]))
                with c3: st.metric("Gross Estate", TEA.money(rr.get("gross_estate", rr["final_total"])))
                with c4: st.metric("Total Taxes", TEA.money(rr["total_taxes"]))
                if rr["final_total"] <= 0:
                    st.error("Your portfolio is depleted before end of plan.")
                else:
                    st.success("Your savings last through the plan horizon.")

            # Waterfall optimizer
            st.divider()
            if st.button("Test All Withdrawal Orders", key="fp_rr_waterfall"):
                from itertools import permutations
                if accum_result is None:
                    accum_result = _run_accum()
                    st.session_state.projection_results = accum_result
                _final = accum_result["rows"][-1] if accum_result["rows"] else {}
                _brok = accum_result.get("final_brokerage", _final.get("Bal Taxable", 0))
                _cash_bal = accum_result.get("final_cash", 0)
                if _ire_liq_equity > 0:
                    _brok += _ire_liq_equity
                _bals = {
                    "pretax": _final.get("Bal Pre-Tax", 0),
                    "roth": _final.get("Bal Roth", 0),
                    "brokerage": _brok,
                    "cash": _cash_bal,
                    "taxable": _brok + _cash_bal,
                    "brokerage_basis": accum_result.get("final_basis", 0) + _ire_liq_equity,
                    "hsa": _final.get("Bal HSA", 0),
                }
                _rp = _build_retire_params(accum_result)
                _rp["expenses_at_retirement"] = D["living_expenses"] * (rr_ret_pct / 100) * ((1 + inflation) ** years_to_ret)
                _rp["survivor_spending_pct"] = rr_survivor
                _rp["pension_survivor_pct"] = rr_pension_surv
                _rp["heir_bracket_option"] = rr_heir_opt
                all_orders = list(permutations(["Pre-Tax", "Taxable", "Tax-Free"]))
                wf_results = []
                for order in all_orders:
                    _r = engine.run_retirement_projection(copy.deepcopy(_bals), copy.deepcopy(_rp), list(order))
                    wf_results.append({
                        "Order": " → ".join(order), "Estate": _r["estate"],
                        "Total Taxes": _r["total_taxes"], "Final Total": _r["final_total"],
                    })
                wf_results.sort(key=lambda x: x["Estate"], reverse=True)
                st.success(f"Best order: **{wf_results[0]['Order']}** (Estate: {TEA.money(wf_results[0]['Estate'])})")
                st.dataframe(pd.DataFrame(wf_results), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 (PRE-RET): Savings Optimizer
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Savings Optimizer")
        st.caption("Tests different savings allocation strategies through your full retirement "
                   "and compares after-tax estate at end of life — the true measure of wealth.")

        # Show current allocation
        st.markdown("### Current Allocation")
        _curr_alloc = pd.DataFrame([{
            "Pre-Tax 401k": TEA.money(contrib_401k_pretax_filer + contrib_401k_pretax_spouse),
            "Roth 401k": TEA.money(contrib_401k_roth_filer + contrib_401k_roth_spouse),
            "Trad IRA": TEA.money(contrib_trad_ira + contrib_trad_ira_spouse),
            "Roth IRA": TEA.money(contrib_roth_ira + contrib_roth_ira_spouse),
            "HSA": TEA.money(contrib_hsa), "Taxable": TEA.money(contrib_taxable),
            "Employer Match": TEA.money(employer_match + employer_match_spouse),
        }])
        st.dataframe(_curr_alloc, use_container_width=True, hide_index=True)

        if st.button("Run Savings Optimizer", type="primary", key="fp_preret_run_opt"):
            # Total tax-advantaged savings pool (employee only, excludes employer match)
            employee_401k_f = contrib_401k_filer
            employee_401k_s = contrib_401k_pretax_spouse + contrib_401k_roth_spouse
            employee_ira_f = contrib_trad_ira + contrib_roth_ira
            employee_ira_s = contrib_trad_ira_spouse + contrib_roth_ira_spouse
            total_tax_adv = employee_401k_f + employee_401k_s + employee_ira_f + employee_ira_s + contrib_hsa

            # Define strategies
            strategies = []

            # Strategy helper: build contrib_dict variant
            def _make_contrib(pretax_401k_f, roth_401k_f, pretax_401k_s, roth_401k_s,
                              trad_ira_f, roth_ira_f, trad_ira_s, roth_ira_s,
                              hsa, taxable):
                pf = pretax_401k_f + employer_match + trad_ira_f
                ps = pretax_401k_s + employer_match_spouse + trad_ira_s
                rf = roth_401k_f + roth_ira_f
                rs = roth_401k_s + roth_ira_s
                return {
                    "pretax": pf + ps, "roth": rf + rs,
                    "taxable": taxable, "cash": 0.0, "hsa": hsa,
                    "pretax_filer": pf, "pretax_spouse": ps,
                    "roth_filer": rf, "roth_spouse": rs,
                }, pretax_401k_f + pretax_401k_s + trad_ira_f + trad_ira_s + hsa

            # Current plan
            strategies.append(("Current Plan", contrib_dict, pretax_deductions))

            # All Pre-Tax 401k
            _c, _d = _make_contrib(
                employee_401k_f, 0, employee_401k_s, 0,
                employee_ira_f, 0, employee_ira_s, 0,
                contrib_hsa, contrib_taxable)
            strategies.append(("All Pre-Tax 401k + Trad IRA", _c, _d))

            # All Roth 401k
            _c, _d = _make_contrib(
                0, employee_401k_f, 0, employee_401k_s,
                0, employee_ira_f, 0, employee_ira_s,
                contrib_hsa, contrib_taxable)
            strategies.append(("All Roth 401k + Roth IRA", _c, _d))

            # 50/50 split
            _c, _d = _make_contrib(
                employee_401k_f * 0.5, employee_401k_f * 0.5,
                employee_401k_s * 0.5, employee_401k_s * 0.5,
                employee_ira_f * 0.5, employee_ira_f * 0.5,
                employee_ira_s * 0.5, employee_ira_s * 0.5,
                contrib_hsa, contrib_taxable)
            strategies.append(("50/50 Pre-Tax/Roth", _c, _d))

            # 75% Roth
            _c, _d = _make_contrib(
                employee_401k_f * 0.25, employee_401k_f * 0.75,
                employee_401k_s * 0.25, employee_401k_s * 0.75,
                0, employee_ira_f, 0, employee_ira_s,
                contrib_hsa, contrib_taxable)
            strategies.append(("75% Roth / 25% Pre-Tax", _c, _d))

            # 25% Roth
            _c, _d = _make_contrib(
                employee_401k_f * 0.75, employee_401k_f * 0.25,
                employee_401k_s * 0.75, employee_401k_s * 0.25,
                employee_ira_f, 0, employee_ira_s, 0,
                contrib_hsa, contrib_taxable)
            strategies.append(("25% Roth / 75% Pre-Tax", _c, _d))

            # No HSA (redirect to taxable)
            if contrib_hsa > 0:
                _c, _d = _make_contrib(
                    contrib_401k_pretax_filer, contrib_401k_roth_filer,
                    contrib_401k_pretax_spouse, contrib_401k_roth_spouse,
                    contrib_trad_ira, contrib_roth_ira,
                    contrib_trad_ira_spouse, contrib_roth_ira_spouse,
                    0, contrib_taxable + contrib_hsa)
                strategies.append(("No HSA → Taxable", _c, _d))

            # Max taxable (minimal retirement accounts)
            _c, _d = _make_contrib(
                0, 0, 0, 0, 0, 0, 0, 0,
                0, contrib_taxable + total_tax_adv)
            strategies.append(("All to Taxable (no retirement accts)", _c, _d))

            # Default spending order for retirement phase
            _opt_spend_order = [D["so1"], D["so2"], D["so3"], D["so4"]]

            # Run each strategy: accumulation → retirement → after-tax estate
            opt_results = []
            progress = st.progress(0, text="Running full lifecycle simulations...")
            for idx, (name, c_dict, pt_ded) in enumerate(strategies):
                _ii = dict(income_info)
                _ii["pretax_deductions"] = pt_ded
                _acc = engine.run_accumulation(
                    current_age_filer, years_to_ret, copy.deepcopy(start_balances),
                    dict(c_dict), salary_growth, pre_ret_return, _ii)
                _final = _acc["rows"][-1] if _acc["rows"] else {}
                _yr1 = _acc["rows"][0] if _acc["rows"] else {}
                portfolio_at_ret = _final.get("Portfolio", 0)

                # Build retirement balances from accumulation result
                _opt_brok = _acc.get("final_brokerage", _final.get("Bal Taxable", 0))
                _opt_cash = _acc.get("final_cash", 0)
                if _ire_liq_equity > 0:
                    _opt_brok += _ire_liq_equity
                _opt_bals = {
                    "pretax": _final.get("Bal Pre-Tax", 0),
                    "roth": _final.get("Bal Roth", 0),
                    "brokerage": _opt_brok,
                    "cash": _opt_cash,
                    "taxable": _opt_brok + _opt_cash,
                    "brokerage_basis": _acc.get("final_basis", 0) + _ire_liq_equity,
                    "hsa": _final.get("Bal HSA", 0),
                }

                # Run retirement projection
                _opt_rp = _build_retire_params(_acc)
                _opt_rp["heir_bracket_option"] = "same"
                _opt_ret = engine.run_retirement_projection(_opt_bals, _opt_rp, _opt_spend_order)

                _ret_final = _opt_ret["rows"][-1] if _opt_ret["rows"] else {}
                net_estate = _opt_ret.get("estate", 0)
                gross_estate = _opt_ret.get("gross_estate", 0)
                total_taxes = _opt_ret.get("total_taxes", 0)
                final_total = _opt_ret.get("final_total", 0)
                depleted = final_total <= 0

                pretax_bal = _final.get("Bal Pre-Tax", 0)
                roth_bal = _final.get("Bal Roth", 0)
                taxable_bal = _final.get("Bal Taxable", 0)
                hsa_bal = _final.get("Bal HSA", 0)
                total_bal = pretax_bal + roth_bal + taxable_bal + hsa_bal

                opt_results.append({
                    "Strategy": name,
                    "Yr 1 Tax": TEA.money(_yr1.get("Total Tax", 0)),
                    "At Retirement": TEA.money(portfolio_at_ret),
                    "Net Estate": TEA.money(net_estate),
                    "Total Taxes": TEA.money(total_taxes),
                    "Pre-Tax %": f"{pretax_bal / total_bal * 100:.0f}%" if total_bal > 0 else "0%",
                    "Roth %": f"{roth_bal / total_bal * 100:.0f}%" if total_bal > 0 else "0%",
                    "Funded": "No" if depleted else "Yes",
                    "_estate": net_estate, "_name": name, "_depleted": depleted,
                    "_portfolio": portfolio_at_ret,
                })
                progress.progress((idx + 1) / len(strategies))
            progress.empty()

            # Sort by after-tax estate (highest = best)
            opt_results.sort(key=lambda x: x["_estate"], reverse=True)
            baseline = next((r for r in opt_results if r["_name"] == "Current Plan"), opt_results[0])
            for r in opt_results:
                r["vs Current"] = TEA.money(r["_estate"] - baseline["_estate"])

            best = opt_results[0]
            any_depleted = any(r["_depleted"] for r in opt_results)
            all_depleted = all(r["_depleted"] for r in opt_results)

            if all_depleted:
                st.error("**No savings strategy fully funds your retirement.** "
                         "You need to **save more**, **spend less in retirement**, or **both**.")
            elif any_depleted:
                _depleted_names = [r["Strategy"] for r in opt_results if r["_depleted"]]
                st.warning(f"Some strategies run out of money: {', '.join(_depleted_names)}. "
                           "Consider increasing savings or reducing retirement spending.")

            if best["_name"] != "Current Plan":
                st.success(f"Best strategy: **{best['Strategy']}** — "
                           f"{TEA.money(best['_estate'] - baseline['_estate'])} more after-tax estate than current plan")
            else:
                st.success("Your current savings plan produces the best after-tax estate!")

            _disp = [{k: v for k, v in r.items() if not k.startswith("_")} for r in opt_results]
            st.dataframe(pd.DataFrame(_disp), use_container_width=True, hide_index=True)
            st.session_state._preret_opt = opt_results

# ══════════════════════════════════════════════════════════════════════
# PLANNING
# ══════════════════════════════════════════════════════════════════════
if nav == "Planning":
    st.header("Planning")
    col1, col2 = st.columns(2)
    with col1:
        w_text("Client Name", "client_name")
        w_select("Filing Status", ["Single", "Married Filing Jointly", "Head of Household"], "filing_status")
        w_num("Tax Year", "tax_year", min_value=2020, max_value=2100, step=1)
    with col2:
        w_date("Filer DOB", "filer_dob", min_value=dt.date(1930, 1, 1), max_value=dt.date(2005, 12, 31))
        if is_joint:
            w_date("Spouse DOB", "spouse_dob", min_value=dt.date(1930, 1, 1), max_value=dt.date(2005, 12, 31))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if D["is_working"]:
            w_num("Filer Retirement Age", "ret_age", min_value=50, max_value=80, step=1)
            if is_joint:
                w_num("Spouse Retirement Age", "spouse_ret_age", min_value=50, max_value=80, step=1)
        else:
            st.text_input("Filer Retirement Age", value="N/A (Retired)", disabled=True)
            if is_joint:
                st.text_input("Spouse Retirement Age", value="N/A (Retired)", disabled=True)
    with col2:
        w_num("Filer Plan Through Age", "filer_plan_age", min_value=70, max_value=105, step=1)
        if is_joint:
            w_num("Spouse Plan Through Age", "spouse_plan_age", min_value=70, max_value=105, step=1)

    st.divider()
    st.subheader("Economic Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Inflation / SS COLA (%)", "inflation", step=0.5, format="%.1f")
        w_num("Tax Bracket Growth (%)", "bracket_growth", step=0.1, format="%.1f")
    with col2:
        w_num("Medicare Premium Growth (%)", "medicare_growth", step=0.5, format="%.1f")
        w_num("Salary Growth (%)", "salary_growth", step=0.5, format="%.1f")
    with col3:
        w_num("State Tax Rate (%)", "state_tax_rate", step=0.5, format="%.1f")

# ══════════════════════════════════════════════════════════════════════
# GROWING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Growing":
    st.header("Growing")
    is_working = D["is_working"]

    # --- Employment (if working) ---
    if is_working:
        st.subheader("Employment Income")
        col1, col2 = st.columns(2)
        with col1:
            w_num("Filer Salary", "salary_filer", step=5000.0)
            w_check("Self-employed?", "self_employed_f")
            if D["self_employed_f"]:
                st.caption("SE tax ≈ 15.3% (both employee + employer FICA)")
        with col2:
            if is_joint:
                w_num("Spouse Salary", "salary_spouse", step=5000.0)
                w_check("Self-employed?", "self_employed_s")
                if D["self_employed_s"]:
                    st.caption("SE tax ≈ 15.3% (both employee + employer FICA)")

        st.divider()
        st.subheader("Retirement Contributions")
        _plan_opts = ["401(k)", "SEP-IRA", "SIMPLE", "None"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Filer**")
            w_select("Retirement Plan", _plan_opts, "plan_type_f")
            _pt_f = D["plan_type_f"]
            if _pt_f == "401(k)":
                w_check("Max 401(k)?", "max_401k_f")
                if not D["max_401k_f"]:
                    w_num("401(k) contribution", "c401k_f", step=1000.0)
                w_slider("Roth % of 401(k)", "roth_pct_f", min_value=0, max_value=100, step=5)
                w_num("Employer match rate (%)", "ematch_rate_f", step=1.0, format="%.1f")
                w_num("Employer match up to (%)", "ematch_upto_f", step=1.0, format="%.1f")
            elif _pt_f == "SEP-IRA":
                _sep_max_f = min(D["salary_filer"] * 0.25, 69000.0)
                st.caption(f"Max SEP contribution: {_sep_max_f:,.0f} (25% of salary, up to $69k)")
                w_check("Max SEP?", "max_401k_f")
                if not D["max_401k_f"]:
                    w_num("SEP-IRA contribution", "c401k_f", step=1000.0)
            elif _pt_f == "SIMPLE":
                w_check("Max SIMPLE?", "max_401k_f")
                if not D["max_401k_f"]:
                    w_num("SIMPLE contribution", "c401k_f", step=1000.0)
                w_slider("Roth % of SIMPLE", "roth_pct_f", min_value=0, max_value=100, step=5)
                st.caption("Employer match: 3% of salary (mandatory)")
            if _pt_f != "None":
                w_check("Backdoor Roth?", "backdoor_roth_f")
                if D["backdoor_roth_f"]:
                    w_num("Backdoor Roth amount", "contrib_roth_ira_f", step=500.0)
                else:
                    w_check("Max IRA?", "max_ira_f")
                    if not D["max_ira_f"]:
                        w_num("Traditional IRA", "contrib_trad_ira_f", step=500.0)
                        w_num("Roth IRA", "contrib_roth_ira_f", step=500.0)
        with col2:
            if is_joint:
                st.markdown("**Spouse**")
                w_select("Retirement Plan", _plan_opts, "plan_type_s")
                _pt_s = D["plan_type_s"]
                if _pt_s == "401(k)":
                    w_check("Max 401(k)?", "max_401k_s")
                    if not D["max_401k_s"]:
                        w_num("401(k) contribution", "c401k_s", step=1000.0)
                    w_slider("Roth % of 401(k)", "roth_pct_s", min_value=0, max_value=100, step=5)
                    w_num("Employer match rate (%)", "ematch_rate_s", step=1.0, format="%.1f")
                    w_num("Employer match up to (%)", "ematch_upto_s", step=1.0, format="%.1f")
                elif _pt_s == "SEP-IRA":
                    _sep_max_s = min(D["salary_spouse"] * 0.25, 69000.0)
                    st.caption(f"Max SEP contribution: {_sep_max_s:,.0f} (25% of salary, up to $69k)")
                    w_check("Max SEP?", "max_401k_s")
                    if not D["max_401k_s"]:
                        w_num("SEP-IRA contribution", "c401k_s", step=1000.0)
                elif _pt_s == "SIMPLE":
                    w_check("Max SIMPLE?", "max_401k_s")
                    if not D["max_401k_s"]:
                        w_num("SIMPLE contribution", "c401k_s", step=1000.0)
                    w_slider("Roth % of SIMPLE", "roth_pct_s", min_value=0, max_value=100, step=5)
                    st.caption("Employer match: 3% of salary (mandatory)")
                if _pt_s != "None":
                    w_check("Backdoor Roth?", "backdoor_roth_s")
                    if D["backdoor_roth_s"]:
                        w_num("Backdoor Roth amount (Spouse)", "contrib_roth_ira_s", step=500.0)
                    else:
                        w_check("Max IRA?", "max_ira_s")
                        if not D["max_ira_s"]:
                            w_num("Traditional IRA (Spouse)", "contrib_trad_ira_s", step=500.0)
                            w_num("Roth IRA (Spouse)", "contrib_roth_ira_s", step=500.0)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            w_check("HSA eligible?", "hsa_eligible")
            if D["hsa_eligible"]:
                w_check("Max HSA?", "max_hsa")
                if not D["max_hsa"]:
                    w_num("HSA contribution", "contrib_hsa", step=500.0)
        with col2:
            w_num("Additional taxable savings", "contrib_taxable", step=1000.0)

    # --- Current Account Balances ---
    st.divider()
    st.subheader("Current Account Balances")
    col1, col2 = st.columns(2)
    with col1:
        w_num("401(k)/Pre-Tax — Filer", "curr_401k_f", step=10000.0)
        if is_joint:
            w_num("401(k)/Pre-Tax — Spouse", "curr_401k_s", step=10000.0)
        w_num("Traditional IRA — Filer", "curr_trad_ira_f", step=5000.0)
        if is_joint:
            w_num("Traditional IRA — Spouse", "curr_trad_ira_s", step=5000.0)
        w_num("Roth — Filer", "curr_roth_ira_f", step=5000.0)
        if is_joint:
            w_num("Roth — Spouse", "curr_roth_ira_s", step=5000.0)
        if is_working:
            w_num("Roth 401(k)", "curr_roth_401k", step=5000.0)
    with col2:
        w_num("Taxable Brokerage — Filer", "curr_taxable", step=10000.0)
        w_num("Cost Basis — Filer", "taxable_basis", step=10000.0,
              help="Original cost basis of filer's taxable brokerage holdings")
        if is_joint:
            w_num("Taxable Brokerage — Spouse", "curr_taxable_s", step=10000.0)
            w_num("Cost Basis — Spouse", "taxable_basis_s", step=10000.0,
                  help="Original cost basis of spouse's taxable brokerage holdings")
        w_num("Cash / Savings", "curr_cash", step=5000.0)
        w_num("Emergency Fund Reserve", "emergency_fund", step=1000.0,
              help="Fixed rainy day reserve excluded from available cash for withdrawals.")
        if is_working:
            w_num("HSA Balance", "curr_hsa", step=1000.0)

    # --- Insurance Products ---
    st.divider()
    st.subheader("Insurance Products")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Annuities**")
        w_num("Annuity Value — Filer", "annuity_value_f", step=5000.0)
        w_num("Annuity Basis — Filer", "annuity_basis_f", step=5000.0)
        w_num("Monthly Contribution — Filer", "annuity_monthly_f", step=50.0)
        if is_joint:
            w_num("Annuity Value — Spouse", "annuity_value_s", step=5000.0)
            w_num("Annuity Basis — Spouse", "annuity_basis_s", step=5000.0)
            w_num("Monthly Contribution — Spouse", "annuity_monthly_s", step=50.0)
    with col2:
        st.markdown("**Life Insurance**")
        w_num("Cash Value", "life_cash_value", step=1000.0)
        w_num("Cost Basis", "life_basis", step=1000.0)
        w_num("Monthly Premium", "life_monthly", step=50.0)

    # --- Growth Rate Assumptions ---
    st.divider()
    st.subheader("Growth Rate Assumptions (%)")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Cash/Savings return", "r_cash", step=0.5, format="%.1f")
        w_num("Taxable Brokerage return", "r_taxable", step=0.5, format="%.1f")
    with col2:
        w_num("Pre-Tax return", "r_pretax", step=0.5, format="%.1f")
        w_num("Roth return", "r_roth", step=0.5, format="%.1f")
    with col3:
        w_num("Annuity return", "r_annuity", step=0.5, format="%.1f")
        w_num("Life Insurance return", "r_life", step=0.5, format="%.1f")

    # --- Asset Allocation & CMA ---
    st.divider()
    w_check("Use Asset Allocation to compute growth rates", "use_asset_alloc")
    if D["use_asset_alloc"]:
        st.caption("Computed returns from CMA + allocation will override the simple growth rates above for projections.")
        st.subheader("Capital Market Assumptions (Expected Return %)")
        _cma_labels = ["US Equity", "Int'l Equity", "Fixed Income", "Real Assets", "Cash"]
        _cma_keys = ["cma_us_equity", "cma_intl_equity", "cma_fixed_income", "cma_real_assets", "cma_cash"]
        _cma_base = ["us_equity", "intl_equity", "fixed_income", "real_assets", "cash"]
        # Header row
        hdr_cma = st.columns([2, 1, 1, 1, 1, 1, 1])
        hdr_cma[0].markdown("**Asset Class**")
        hdr_cma[1].markdown("**Total**")
        hdr_cma[2].markdown("**Div%**")
        hdr_cma[3].markdown("**Qual%**")
        hdr_cma[4].markdown("**Int%**")
        hdr_cma[5].markdown("**CG%**")
        hdr_cma[6].markdown("**Deferred**")
        for _ci, (_clbl, _ck, _cb) in enumerate(zip(_cma_labels, _cma_keys, _cma_base)):
            _cr = st.columns([2, 1, 1, 1, 1, 1, 1])
            _cr[0].write(_clbl)
            with _cr[1]:
                w_num(_clbl[:3] + " Tot", _ck, step=0.5, format="%.1f", label_visibility="collapsed")
            with _cr[2]:
                w_num(_clbl[:3] + " Div", f"cma_{_cb}_div", step=0.1, format="%.1f", label_visibility="collapsed")
            with _cr[3]:
                w_num(_clbl[:3] + " Q%", f"cma_{_cb}_qual", step=5, min_value=0, max_value=100, label_visibility="collapsed")
            with _cr[4]:
                w_num(_clbl[:3] + " Int", f"cma_{_cb}_int", step=0.1, format="%.1f", label_visibility="collapsed")
            with _cr[5]:
                w_num(_clbl[:3] + " CG", f"cma_{_cb}_cg", step=0.1, format="%.1f", label_visibility="collapsed")
            _deferred = D[_ck] - D[f"cma_{_cb}_div"] - D[f"cma_{_cb}_int"] - D[f"cma_{_cb}_cg"]
            _cr[6].write(f"{_deferred:.1f}%")
            if _deferred < -0.01:
                _cr[6].caption("⚠️ yields > total")

        st.subheader("Account Allocations (%)")
        _AA_ACCOUNTS = [("Pre-Tax Filer", "pretax_f"), ("Pre-Tax Spouse", "pretax_s"),
                        ("Roth Filer", "roth_f"), ("Roth Spouse", "roth_s"),
                        ("Taxable", "taxable"), ("Annuity", "annuity")]
        _AA_FIELDS = [("US Eq", "eq"), ("Int'l", "intl"), ("Fixed", "fi"), ("Real", "re")]

        # Header row
        hdr = st.columns([2, 1, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7])
        hdr[0].markdown("**Account**")
        for i, (fl, _) in enumerate(_AA_FIELDS):
            hdr[i + 1].markdown(f"**{fl}**")
        hdr[5].markdown("**Cash**")
        hdr[6].markdown("**Return**")
        hdr[7].markdown("**Div**")
        hdr[8].markdown("**Int**")
        hdr[9].markdown("**CG**")
        hdr[10].markdown("**Def**")

        _cma_returns = [D[k] for k in _cma_keys]
        for acct_label, acct_key in _AA_ACCOUNTS:
            if not is_joint and "_s" in acct_key:
                continue
            cols = st.columns([2, 1, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7])
            cols[0].write(acct_label)
            total = 0
            for i, (_, fk) in enumerate(_AA_FIELDS):
                with cols[i + 1]:
                    k = f"aa_{acct_key}_{fk}"
                    st.number_input(acct_label[:3], value=D[k], min_value=0, max_value=100, step=5,
                                    key=f"_w_{k}", on_change=_sync, args=(k,), label_visibility="collapsed")
                    total += D[k]
            cash_pct = max(0, 100 - total)
            cols[5].write(f"{cash_pct}%")
            allocs = [D[f"aa_{acct_key}_{fk}"] for _, fk in _AA_FIELDS] + [cash_pct]
            ret = sum(a / 100 * r / 100 for a, r in zip(allocs, _cma_returns))
            cols[6].write(f"**{ret * 100:.1f}%**")
            # Yield breakdown from CMA decomposition
            _yd = _yc = _yi = 0.0
            for _ai, _cbk in enumerate(_cma_base):
                _aw = allocs[_ai] / 100
                _yd += _aw * D.get(f"cma_{_cbk}_div", 0.0) / 100
                _yi += _aw * D.get(f"cma_{_cbk}_int", 0.0) / 100
                _yc += _aw * D.get(f"cma_{_cbk}_cg", 0.0) / 100
            cols[7].write(f"{_yd * 100:.1f}%")
            cols[8].write(f"{_yi * 100:.1f}%")
            cols[9].write(f"{_yc * 100:.1f}%")
            cols[10].write(f"{(ret - _yd - _yi - _yc) * 100:.1f}%")

    # --- Investment Assumptions for Projections ---
    if not D["use_asset_alloc"]:
        st.divider()
        w_check("Investment assumptions for future projections", "use_invest_assumptions")
        if D["use_invest_assumptions"]:
            st.caption("These rates compute annual investment income in multi-year projections. "
                       "Current-year tax numbers are entered in Receiving.")
            col1, col2, col3 = st.columns(3)
            with col1:
                w_num("Dividend yield (%)", "proj_div_yield", step=0.1, format="%.1f")
            with col2:
                w_num("Annual cap gain (%)", "proj_cg_pct", step=0.1, format="%.1f")
            with col3:
                w_num("Cash interest rate (%)", "proj_int_rate", step=0.1, format="%.1f")

    # --- Real Estate ---
    st.divider()
    st.subheader("Real Estate")
    st.markdown("**Primary Residence**")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Home Value", "home_value", step=10000.0)
        w_num("Mortgage Balance", "mtg_balance", step=10000.0)
    with col2:
        w_num("Home Appreciation (%)", "home_appr", step=0.5, format="%.1f")
        w_num("Mortgage Rate (%)", "mtg_rate", step=0.125, format="%.3f")
    with col3:
        w_num("Monthly Payment", "mtg_payment_monthly", step=100.0)
        w_num("Years Remaining", "mtg_years", min_value=0, step=1)

    # --- Investment Real Estate ---
    st.markdown("**Investment Real Estate**")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Total Value", "inv_re_value", step=10000.0)
        w_num("Cost Basis", "inv_re_basis", step=10000.0,
              help="Original purchase price + improvements. Used to calculate capital gains tax on sale.")
    with col2:
        w_num("Appreciation (%)", "inv_re_appr", step=0.5, format="%.1f")
        w_num("Net Annual Income", "inv_re_net_income", step=1000.0,
              help="Rental income minus all expenses (taxes, insurance, maintenance, mgmt).")
    with col3:
        w_num("Mortgage Payment (annual)", "inv_re_mortgage_pmt", step=1000.0,
              help="Annual P&I payment on investment property mortgages. Net income increases when mortgage is paid off.")
        w_check("Liquidate to fund retirement", "inv_re_liquidate",
                help="Sell investment real estate at retirement to fund spending. "
                     "Capital gains tax will apply on the gain (value minus basis).")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Mortgage Years Left", "inv_re_mortgage_years", min_value=0, max_value=40, step=1)
    with col2:
        w_num("Income Growth (%)", "inv_re_income_growth", step=0.5, format="%.1f",
              help="Annual growth rate of rental income.")

    # --- Additional Properties (additive) ---
    for _pi in range(1, 4):
        _pfx = f"inv_re_{_pi}"
        w_check(f"Include Additional Property {_pi}", f"{_pfx}_include")
        if D[f"{_pfx}_include"]:
            w_text("Property Name", f"{_pfx}_name")
            c1, c2, c3 = st.columns(3)
            with c1:
                w_num("Market Value", f"{_pfx}_value", step=10000.0)
                w_num("Cost Basis", f"{_pfx}_basis", step=10000.0)
                w_num("Mortgage Balance", f"{_pfx}_mortgage", step=5000.0)
            with c2:
                w_num("Appreciation (%)", f"{_pfx}_appr", step=0.5, format="%.1f")
                w_num("Mortgage Rate (%)", f"{_pfx}_rate", step=0.25, format="%.2f")
                w_num("Mortgage Years Left", f"{_pfx}_years", min_value=0, max_value=40, step=1)
            with c3:
                w_num("Net Annual Income", f"{_pfx}_net_income", step=1000.0)
                w_num("Income Growth (%)", f"{_pfx}_income_growth", step=0.5, format="%.1f")
                w_check("Liquidate at retirement", f"{_pfx}_liquidate")
    # Auto-summary if additional properties exist
    _any_addl = any(D[f"inv_re_{i}_include"] and D[f"inv_re_{i}_value"] > 0 for i in range(1, 4))
    if _any_addl:
        _addl_val = sum(D[f"inv_re_{i}_value"] for i in range(1, 4) if D[f"inv_re_{i}_include"])
        _addl_inc = sum(D[f"inv_re_{i}_net_income"] for i in range(1, 4) if D[f"inv_re_{i}_include"])
        _combined = D["inv_re_value"] + _addl_val
        st.caption(f"Combined: {TEA.money(_combined)} value ({TEA.money(D['inv_re_value'])} base + {TEA.money(_addl_val)} additional) | {TEA.money(D['inv_re_net_income'] + _addl_inc)}/yr income")


# ══════════════════════════════════════════════════════════════════════
# RECEIVING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Receiving":
    st.header("Receiving")

    # --- Employment Income (if working) ---
    if D["is_working"]:
        st.subheader("Employment Income")
        col1, col2 = st.columns(2)
        with col1:
            w_num("Filer Salary", "salary_filer", step=5000.0)
        with col2:
            if is_joint:
                w_num("Spouse Salary", "salary_spouse", step=5000.0)
        st.divider()

    # --- Social Security ---
    st.subheader("Social Security")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Filer**")
        w_check("SSDI?", "ssdi_filer")
        w_check("Already receiving SS?", "filer_ss_already")
        if D["filer_ss_already"]:
            w_num("Current annual SS", "filer_ss_current", step=1000.0)
            w_num("SS start year", "filer_ss_start_year", min_value=2000, max_value=2060, step=1)
        else:
            w_num("SS at FRA", "filer_ss_fra", step=1000.0,
                  help="Leave at $0 to auto-estimate from salary.")
            w_select("Claim age", ["62", "63", "64", "65", "66", "FRA", "68", "69", "70"], "filer_ss_claim")
            # Show estimate if FRA is 0 and salary available
            if D["filer_ss_fra"] == 0 and D["is_working"] and D["salary_filer"] > 0:
                _filer_dob_ss = D["filer_dob"]
                _filer_age_ss = TEA.age_at_date(_filer_dob_ss, dt.date.today()) if isinstance(_filer_dob_ss, dt.date) else 60
                _est_pia = engine.estimate_ss_pia(D["salary_filer"], _filer_age_ss,
                                                  int(D["ret_age"]), D["salary_growth"] / 100)
                st.info(f"Will auto-estimate from salary: ~{TEA.money(_est_pia)}/yr at FRA")
    with col2:
        if is_joint:
            st.markdown("**Spouse**")
            w_check("SSDI?", "ssdi_spouse")
            w_check("Already receiving SS?", "spouse_ss_already")
            if D["spouse_ss_already"]:
                w_num("Current annual SS (Spouse)", "spouse_ss_current", step=1000.0)
                w_num("SS start year (Spouse)", "spouse_ss_start_year", min_value=2000, max_value=2060, step=1)
            else:
                w_num("SS at FRA (Spouse)", "spouse_ss_fra", step=1000.0,
                      help="Leave at $0 to auto-estimate. If spouse has no income, will use 50% of filer's PIA (spousal benefit).")
                w_select("Claim age (Spouse)", ["62", "63", "64", "65", "66", "FRA", "68", "69", "70"], "spouse_ss_claim")
                # Show estimate if FRA is 0
                if D["spouse_ss_fra"] == 0:
                    _spouse_dob_ss = D["spouse_dob"]
                    _spouse_age_ss = TEA.age_at_date(_spouse_dob_ss, dt.date.today()) if isinstance(_spouse_dob_ss, dt.date) else 60
                    _spouse_sal = D["salary_spouse"] if D["is_working"] else 0.0
                    if _spouse_sal > 0:
                        _est_sp_pia = engine.estimate_ss_pia(_spouse_sal, _spouse_age_ss,
                                                            int(D["spouse_ret_age"]), D["salary_growth"] / 100)
                        st.info(f"Will auto-estimate from salary: ~{TEA.money(_est_sp_pia)}/yr at FRA")
                    elif D["filer_ss_fra"] > 0 or (D["is_working"] and D["salary_filer"] > 0):
                        # Will use spousal benefit
                        _filer_pia_for_sp = D["filer_ss_fra"]
                        if _filer_pia_for_sp == 0 and D["salary_filer"] > 0:
                            _fa = TEA.age_at_date(D["filer_dob"], dt.date.today()) if isinstance(D["filer_dob"], dt.date) else 60
                            _filer_pia_for_sp = engine.estimate_ss_pia(D["salary_filer"], _fa,
                                                                       int(D["ret_age"]), D["salary_growth"] / 100)
                        st.info(f"No spouse income — will use spousal benefit: ~{TEA.money(_filer_pia_for_sp * 0.5)}/yr (50% of filer's PIA)")

    # --- Pensions (per-spouse COLA) ---
    st.divider()
    st.subheader("Pensions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Filer**")
        w_num("Annual Pension", "pension_filer", step=1000.0)
        w_num("Pension Start Age", "pension_filer_age", min_value=50, max_value=80, step=1)
        w_num("Pension COLA (%)", "pension_cola_filer", step=0.5, format="%.1f")
    with col2:
        if is_joint:
            st.markdown("**Spouse**")
            w_num("Annual Pension", "pension_spouse", step=1000.0)
            w_num("Pension Start Age", "pension_spouse_age", min_value=50, max_value=80, step=1)
            w_num("Pension COLA (%)", "pension_cola_spouse", step=0.5, format="%.1f")

    # --- Investment Income (current year) ---
    st.divider()
    st.subheader("Investment Income (Current Year)")
    st.caption("Enter actual current-year amounts for accurate tax calculation.")
    col1, col2 = st.columns(2)
    with col1:
        w_num("Taxable Interest", "interest_taxable", step=100.0)
        w_num("Ordinary Dividends", "total_ordinary_dividends", step=100.0)
    with col2:
        w_num("Qualified Dividends", "qualified_dividends", step=100.0)
        w_num("Capital Gains / (Losses)", "cap_gain_loss", step=500.0)
    w_check("Reinvest dividends (taxable but not spendable)", "reinvest_dividends")
    w_check("Reinvest capital gains", "reinvest_cap_gains")
    w_check("Reinvest interest", "reinvest_interest")

    # --- Other Income ---
    st.divider()
    st.subheader("Other Income")
    _oi_default_age = TEA.age_at_date(D["filer_dob"], dt.date.today()) if D["filer_dob"] else 70
    col1, col2 = st.columns(2)
    with col1:
        if not D["is_working"]:
            w_num("Wages (if any)", "wages", step=1000.0)
            if D["wages"] > 0:
                _wc1, _wc2 = st.columns(2)
                with _wc1:
                    if D["wages_start_age"] == 0:
                        D["wages_start_age"] = _oi_default_age
                    w_num("Start age", "wages_start_age", min_value=50, max_value=110, step=1, format="%d")
                with _wc2:
                    w_num("End age (0=ongoing)", "wages_end_age", min_value=0, max_value=110, step=1, format="%d")
        else:
            _total_wages = D['salary_filer'] + (D['salary_spouse'] if is_joint else 0.0)
            st.caption(f"Wages: {_total_wages:,.0f} (from Employment Income above)")
        w_num("Tax-exempt interest", "tax_exempt_interest", step=100.0)
        if D["tax_exempt_interest"] > 0:
            _tc1, _tc2 = st.columns(2)
            with _tc1:
                if D["tax_exempt_start_age"] == 0:
                    D["tax_exempt_start_age"] = _oi_default_age
                w_num("Start age", "tax_exempt_start_age", min_value=50, max_value=110, step=1, format="%d")
            with _tc2:
                w_num("End age (0=ongoing)", "tax_exempt_end_age", min_value=0, max_value=110, step=1, format="%d")
    with col2:
        w_num("Other taxable income", "other_income", step=500.0)
        if D["other_income"] > 0:
            _oc1, _oc2 = st.columns(2)
            with _oc1:
                if D["other_income_start_age"] == 0:
                    D["other_income_start_age"] = _oi_default_age
                w_num("Start age", "other_income_start_age", min_value=50, max_value=110, step=1, format="%d")
            with _oc2:
                w_num("End age (0=ongoing)", "other_income_end_age", min_value=0, max_value=110, step=1, format="%d")

    # --- Future Income ---
    st.divider()
    st.subheader("Future Income")
    st.caption("Add expected future income (e.g., inheritance, rental income, part-time work).")
    _filer_dob_recv = D["filer_dob"]
    _filer_age_recv = TEA.age_at_date(_filer_dob_recv, dt.date.today()) if _filer_dob_recv else 70
    if st.button("Add Income", key="_btn_add_income"):
        st.session_state.future_income.append({
            "name": "", "amount": 0.0,
            "start_age": _filer_age_recv, "end_age": 0,
            "inflates": False, "taxable": True,
        })
        st.rerun()
    _fi_to_remove = None
    for idx, fi in enumerate(st.session_state.future_income):
        st.markdown(f"**Income {idx + 1}**")
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1: fi["name"] = st.text_input("Description", value=fi["name"], key=f"_fi_name_{idx}")
        with c2: fi["amount"] = st.number_input("Annual $", value=float(fi["amount"]), min_value=0.0, step=1000.0, key=f"_fi_amt_{idx}")
        with c3: fi["start_age"] = st.number_input("Start Age", value=int(fi["start_age"]), min_value=50, max_value=110, step=1, key=f"_fi_start_{idx}")
        with c4: fi["end_age"] = st.number_input("End Age (0=ongoing)", value=int(fi["end_age"]), min_value=0, max_value=110, step=1, key=f"_fi_end_{idx}")
        c5, c6, c7 = st.columns([2, 2, 1])
        with c5: fi["inflates"] = st.checkbox("Inflates?", value=fi["inflates"], key=f"_fi_inf_{idx}")
        with c6: fi["taxable"] = st.checkbox("Taxable?", value=fi["taxable"], key=f"_fi_tax_{idx}")
        with c7:
            if st.button("Remove", key=f"_fi_rm_{idx}"):
                _fi_to_remove = idx
        st.markdown("---")
    if _fi_to_remove is not None:
        st.session_state.future_income.pop(_fi_to_remove)
        st.rerun()

    # --- RMD ---
    st.divider()
    w_check("Do you have an RMD? (e.g., inherited IRA)", "has_rmd")
    if D["has_rmd"]:
        st.subheader("RMD Inputs")
        col1, col2 = st.columns(2)
        with col1:
            w_check("Auto-calculate RMD", "auto_rmd")
            w_num("Filer prior-year 12/31 pre-tax balance", "pretax_bal_filer_prior", step=1000.0)
        with col2:
            if is_joint:
                w_num("Spouse prior-year 12/31 pre-tax balance", "pretax_bal_spouse_prior", step=1000.0)
            w_num("Baseline pre-tax distributions", "baseline_pretax_dist", step=1000.0)
            if not D["auto_rmd"]:
                w_num("RMD manual override", "rmd_manual", step=1000.0)

    # --- Tax Details ---
    st.divider()
    st.subheader("Tax Details")
    col1, col2 = st.columns(2)
    with col1:
        w_num("Adjustments to income", "adjustments", step=500.0)
        w_num("Dependents", "dependents", min_value=0, step=1)
        # Auto-compute 65+ from DOB (same as retired engine)
        _filer_dob_dt = D["filer_dob"] if isinstance(D["filer_dob"], dt.date) else dt.date.fromisoformat(D["filer_dob"])
        _ty_end = dt.date(int(D["tax_year"]), 12, 31)
        _filer_age_eoy = TEA.age_at_date(_filer_dob_dt, _ty_end)
        D["filer_65_plus"] = _filer_age_eoy is not None and _filer_age_eoy >= 65
        st.text(f"Filer age 65+: {'Yes' if D['filer_65_plus'] else 'No'} (age {_filer_age_eoy})")
        if is_joint:
            _spouse_dob_dt = D["spouse_dob"] if isinstance(D["spouse_dob"], dt.date) else dt.date.fromisoformat(D["spouse_dob"])
            _spouse_age_eoy = TEA.age_at_date(_spouse_dob_dt, _ty_end)
            D["spouse_65_plus"] = _spouse_age_eoy is not None and _spouse_age_eoy >= 65
            st.text(f"Spouse age 65+: {'Yes' if D['spouse_65_plus'] else 'No'} (age {_spouse_age_eoy})")
    with col2:
        w_num("SC retirement deduction", "retirement_deduction", step=1000.0)
        w_num("Out-of-state gain (SC)", "out_of_state_gain", step=1000.0)
        w_num("Medical / health expenses", "medical_expenses", step=500.0)

    # --- Deduction Method ---
    st.divider()
    st.subheader("Deduction Method")
    w_radio("Federal Deduction", ["Auto (higher of standard/itemized)", "Force Itemized (auto-calculate)",
            "Force Itemized (enter amount)", "Force Standard"], "deduction_method", horizontal=True)
    if D["deduction_method"] == "Force Itemized (enter amount)":
        w_num("Total Itemized Deduction Amount", "custom_itemized_amount", step=1000.0,
              help="Enter total itemized deductions (mortgage interest + SALT + medical + charitable + other).")
    elif D["deduction_method"] != "Force Standard":
        # Show itemized deduction components for reference
        _mtg_int_est = 0.0
        if D["mtg_balance"] > 0 and D["mtg_rate"] > 0:
            _mtg_int_est = D["mtg_balance"] * (D["mtg_rate"] / 100)  # rough estimate
        _salt_est = min(10000, D["property_tax"] + D["state_tax_rate"] / 100 * D.get("other_income", 0))
        _med_est = D["medical_expenses"]
        _char_est = D["charitable"]
        st.caption(f"Itemized components: Mortgage Interest ~{TEA.money(_mtg_int_est)} + "
                   f"SALT ~{TEA.money(_salt_est)} + Medical {TEA.money(_med_est)} + "
                   f"Charitable {TEA.money(_char_est)} = ~{TEA.money(_mtg_int_est + _salt_est + _med_est + _char_est)}")


# ══════════════════════════════════════════════════════════════════════
# SPENDING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Spending":
    st.header("Spending")
    col1, col2 = st.columns(2)
    with col1:
        w_num("Annual Living Expenses", "living_expenses", step=5000.0)
        if D["is_working"]:
            w_num("Retirement Spending % of Pre-Retirement", "ret_pct", step=5.0, format="%.0f")
    with col2:
        w_num("Heir Tax Rate (%)", "heir_tax_rate", step=1.0, format="%.0f",
              help="Heir's marginal tax rate on inherited IRA distributions (10-year SECURE Act).")

    st.divider()
    st.subheader("Housing Costs")
    _mtg_pmt_display = D["mtg_payment_monthly"]
    _mtg_yrs_display = D["mtg_years"]
    if _mtg_pmt_display > 0:
        st.caption(f"Mortgage: ${_mtg_pmt_display:,.0f}/mo, {_mtg_yrs_display} yrs remaining (edit on Growing → Real Estate)")
    else:
        st.caption("No mortgage entered (set on Growing → Real Estate)")
    w_num("Property Tax (annual)", "property_tax", step=500.0)

    st.divider()
    st.subheader("Withdrawal Strategy")
    st.markdown("**Liquidation Order (Waterfall)**")
    opts = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: w_select("1st", opts, "so1")
    with c2: w_select("2nd", opts, "so2")
    with c3: w_select("3rd", opts, "so3")
    with c4: w_select("4th", opts, "so4")
    # Validate: no duplicates — auto-fix if needed
    _so_list = [D["so1"], D["so2"], D["so3"], D["so4"]]
    if len(set(_so_list)) < 4:
        st.warning("Withdrawal order has duplicates — each bucket should appear exactly once. Auto-correcting.")
        _used = set()
        _fixed = []
        for s in _so_list:
            if s not in _used:
                _fixed.append(s)
                _used.add(s)
        for o in opts:
            if o not in _used:
                _fixed.append(o)
                _used.add(o)
        D["so1"], D["so2"], D["so3"], D["so4"] = _fixed[0], _fixed[1], _fixed[2], _fixed[3]
    w_radio("Reinvest surplus income in:", ["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"],
            "surplus_dest", horizontal=True,
            help="When income exceeds spending + taxes, where does the excess go?")

    st.divider()
    st.subheader("Additional Expenses")
    st.caption("Add one-time or recurring expenses beyond base spending (e.g., new car, home repair, long-term care).")
    _ae_sources = ["Taxable – Cash", "Taxable – Brokerage", "Pre-Tax – IRA/401k", "Annuity", "Roth", "Life Insurance (loan)"]
    if is_joint:
        _ae_sources.insert(_ae_sources.index("Roth") + 1, "Roth — Spouse")
        _ae_sources.insert(_ae_sources.index("Annuity") + 1, "Annuity — Spouse")
    filer_dob = D["filer_dob"]
    _filer_age_now = TEA.age_at_date(filer_dob, dt.date.today()) if filer_dob else 70

    if st.button("Add Expense", key="_btn_add_expense"):
        st.session_state.additional_expenses.append({
            "name": "", "net_amount": 0.0,
            "start_age": _filer_age_now, "end_age": 0,
            "inflates": False, "source": "Taxable – Cash",
            "gross_needed": 0.0, "calculated": False, "tax_impact": None,
        })
        st.rerun()

    _ae_to_remove = None
    for idx, ae in enumerate(st.session_state.additional_expenses):
        st.markdown(f"**Expense {idx + 1}**")
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1: ae["name"] = st.text_input("Description", value=ae["name"], key=f"_ae_name_{idx}")
        with c2: ae["net_amount"] = st.number_input("Net Amount $", value=float(ae["net_amount"]), min_value=0.0, step=1000.0, key=f"_ae_amt_{idx}")
        with c3: ae["start_age"] = st.number_input("Start Age", value=int(ae["start_age"]), min_value=50, max_value=110, step=1, key=f"_ae_start_{idx}")
        with c4: ae["end_age"] = st.number_input("End Age (0=one-time)", value=int(ae["end_age"]), min_value=0, max_value=110, step=1, key=f"_ae_end_{idx}")
        c5, c6, c7 = st.columns([2, 3, 1])
        with c5: ae["inflates"] = st.checkbox("Inflates?", value=ae["inflates"], key=f"_ae_inf_{idx}")
        with c6: ae["source"] = st.selectbox("Source", _ae_sources, index=_ae_sources.index(ae["source"]) if ae["source"] in _ae_sources else 0, key=f"_ae_src_{idx}")
        with c7:
            if st.button("Remove", key=f"_ae_rm_{idx}"):
                _ae_to_remove = idx
        st.markdown("---")
    if _ae_to_remove is not None:
        st.session_state.additional_expenses.pop(_ae_to_remove)
        st.rerun()

    st.divider()
    st.subheader("Estate Tax")
    w_check("Include Estate Tax in Projections", "estate_tax_enabled")
    if D["estate_tax_enabled"]:
        col1, col2 = st.columns(2)
        with col1:
            w_num("Federal Exemption (per person)", "federal_estate_exemption", step=100000.0,
                  help="Federal estate tax exemption per individual (~$15M for 2026).")
            w_num("Exemption Growth (%)", "exemption_inflation", step=0.5,
                  help="Annual CPI adjustment to the exemption amount.")
            if is_joint:
                w_check("Use Portability (double exemption)", "use_portability",
                        help="Deceased spouse's unused exemption transfers to survivor.")
        with col2:
            w_num("State Estate Tax Rate (%)", "state_estate_tax_rate", step=1.0,
                  help="State-level estate or inheritance tax rate (0 if none).")
            w_num("State Estate Tax Exemption", "state_estate_exemption", step=100000.0,
                  help="State-level estate tax exemption amount.")

# ══════════════════════════════════════════════════════════════════════
# GIVING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Giving":
    st.header("Giving")
    w_num("Charitable contributions", "charitable", step=500.0)
    _filer_dob_giving = D["filer_dob"]
    if isinstance(_filer_dob_giving, str):
        _filer_dob_giving = dt.date.fromisoformat(_filer_dob_giving)
    # Exact 70½ check: DOB + 70 years 6 months
    _is_70_half = False
    if _filer_dob_giving:
        _70h_m = _filer_dob_giving.month + 6
        _70h_y = _filer_dob_giving.year + 70 + (_70h_m > 12)
        _70h_m = _70h_m - 12 if _70h_m > 12 else _70h_m
        _70h_d = min(_filer_dob_giving.day, [31,28,31,30,31,30,31,31,30,31,30,31][_70h_m - 1])
        _is_70_half = dt.date.today() >= dt.date(_70h_y, _70h_m, _70h_d)
    if _is_70_half and D["charitable"] > 0:
        _qcd_cap = 210000.0 if is_joint else 105000.0
        D["qcd_annual"] = min(D["charitable"], _qcd_cap)
        st.info(f"Age 70½+ — ${D['qcd_annual']:,.0f} of charitable contributions automatically designated "
                f"as QCD (direct IRA-to-charity transfer). Excluded from taxable income; counts toward RMD.")
    else:
        D["qcd_annual"] = 0.0


# ══════════════════════════════════════════════════════════════════════
# PROTECTING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Protecting":
    st.header("Protecting")
    if is_joint:
        w_slider("Survivor Spending % (after first death)", "survivor_spend_pct",
                 min_value=50, max_value=100, step=5)
        w_slider("Pension Survivor Benefit %", "pension_survivor_pct",
                 min_value=0, max_value=100, step=5)
    else:
        st.info("Survivor settings only apply to joint filers.")
    st.divider()
    st.subheader("Insurance Summary")
    st.write(f"Life Insurance CV: {TEA.money(D['life_cash_value'])}  |  Basis: {TEA.money(D['life_basis'])}")
    ann_total = D["annuity_value_f"] + D["annuity_value_s"]
    if ann_total > 0:
        st.write(f"Annuity Value: {TEA.money(ann_total)}  |  Basis: {TEA.money(D['annuity_basis_f'] + D['annuity_basis_s'])}")

    # Investment Real Estate Summary
    _ire_val = D["inv_re_value"]
    _any_addl_props = any(D.get(f"inv_re_{i}_include", False) and D[f"inv_re_{i}_value"] > 0 for i in range(1, 4))
    if _ire_val > 0 or _any_addl_props:
        st.divider()
        st.subheader("Investment Real Estate Summary")
        if _ire_val > 0:
            _ire_basis = D["inv_re_basis"]
            _ire_inc = D["inv_re_net_income"]
            _ire_gain = max(0, _ire_val - _ire_basis)
            _ire_liq = " | **Will liquidate for retirement**" if D["inv_re_liquidate"] else ""
            _ire_inc_label = f"Net Income: {TEA.money(_ire_inc)}/yr" if _ire_inc > 0 else "No net income"
            st.write(f"Base: {TEA.money(_ire_val)}  |  Basis: {TEA.money(_ire_basis)}  |  "
                     f"Gain: {TEA.money(_ire_gain)}  |  {_ire_inc_label}{_ire_liq}")
        for _pi in range(1, 4):
            _pfx = f"inv_re_{_pi}"
            if not D.get(f"{_pfx}_include", False):
                continue
            _pval = D[f"{_pfx}_value"]
            if _pval <= 0:
                continue
            _pname = D[f"{_pfx}_name"] or f"Property {_pi}"
            _pbasis = D[f"{_pfx}_basis"]
            _pmtg = D[f"{_pfx}_mortgage"]
            _pinc = D[f"{_pfx}_net_income"]
            _pliq = D[f"{_pfx}_liquidate"]
            _equity = _pval - _pmtg
            _gain = max(0, _pval - _pbasis)
            _inc_label = f"Net Income: {TEA.money(_pinc)}/yr" if _pinc > 0 else "No net income"
            _liq_label = " | **Will liquidate**" if _pliq else ""
            st.write(f"  - **{_pname}**: Value {TEA.money(_pval)}  |  Basis {TEA.money(_pbasis)}  |  "
                     f"Equity {TEA.money(_equity)}  |  {_inc_label}{_liq_label}")


# ══════════════════════════════════════════════════════════════════════
# ACHIEVING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Achieving":
    st.header("Achieving")

    # --- Gather all input values from D ---
    filer_dob = D["filer_dob"]
    if isinstance(filer_dob, str):
        filer_dob = dt.date.fromisoformat(filer_dob)
    spouse_dob = D["spouse_dob"] if is_joint else None
    if isinstance(spouse_dob, str):
        spouse_dob = dt.date.fromisoformat(spouse_dob)
    current_age_filer = TEA.age_at_date(filer_dob, dt.date.today()) if filer_dob else 70
    current_age_spouse = TEA.age_at_date(spouse_dob, dt.date.today()) if (spouse_dob and is_joint) else None

    tax_year = int(D["tax_year"])
    inflation = D["inflation"] / 100
    bracket_growth = D["bracket_growth"] / 100
    medicare_growth = D["medicare_growth"] / 100

    # Pension COLA — weighted average for engine
    _pcf = D["pension_filer"]; _pcs = D["pension_spouse"]
    _cola_f = D["pension_cola_filer"] / 100; _cola_s = D["pension_cola_spouse"] / 100
    if _pcf > 0 and _pcs > 0:
        pension_cola = (_cola_f * _pcf + _cola_s * _pcs) / (_pcf + _pcs)
    elif _pcf > 0:
        pension_cola = _cola_f
    elif _pcs > 0:
        pension_cola = _cola_s
    else:
        pension_cola = 0.0

    # SS
    filer_ss_already = D["filer_ss_already"]
    filer_ss_current = D["filer_ss_current"] if filer_ss_already else 0.0
    filer_ss_start_year = int(D["filer_ss_start_year"]) if filer_ss_already else 9999
    filer_ss_fra = D["filer_ss_fra"]
    filer_ss_claim = D["filer_ss_claim"]
    spouse_ss_already = D["spouse_ss_already"] if is_joint else False
    spouse_ss_current = D["spouse_ss_current"] if (is_joint and spouse_ss_already) else 0.0
    spouse_ss_start_year = int(D["spouse_ss_start_year"]) if (is_joint and spouse_ss_already) else 9999
    spouse_ss_fra = D["spouse_ss_fra"] if is_joint else 0.0
    spouse_ss_claim = D["spouse_ss_claim"] if is_joint else "FRA"

    # Auto-estimate SS from income if FRA amount not entered and not already receiving
    _salary_f = D["salary_filer"] if D["is_working"] else D.get("wages", 0)
    _salary_s = D["salary_spouse"] if (is_joint and D["is_working"]) else 0.0
    _ret_age_f = int(D["ret_age"])
    _ret_age_s = int(D["spouse_ret_age"]) if is_joint else _ret_age_f
    _sal_growth = D["salary_growth"] / 100

    if not filer_ss_already and filer_ss_fra == 0 and _salary_f > 0:
        filer_ss_fra = engine.estimate_ss_pia(_salary_f, current_age_filer, _ret_age_f, _sal_growth)

    if is_joint and not spouse_ss_already and spouse_ss_fra == 0:
        if _salary_s > 0 and current_age_spouse:
            # Spouse has own earnings — estimate from their salary
            spouse_ss_fra = engine.estimate_ss_pia(_salary_s, current_age_spouse, _ret_age_s, _sal_growth)
        elif filer_ss_fra > 0:
            # No spouse income — spousal benefit = 50% of filer's PIA at FRA
            spouse_ss_fra = filer_ss_fra * 0.50

    # Pensions
    pension_filer = D["pension_filer"]
    pension_spouse = D["pension_spouse"] if is_joint else 0.0

    # Balances
    pretax_bal_filer_current = D["curr_401k_f"] + D["curr_trad_ira_f"]
    pretax_bal_spouse_current = (D["curr_401k_s"] + D["curr_trad_ira_s"]) if is_joint else 0.0
    pretax_bal = pretax_bal_filer_current + pretax_bal_spouse_current
    roth_bal_filer = D["curr_roth_ira_f"]
    roth_bal_spouse = D["curr_roth_ira_s"] if is_joint else 0.0
    roth_bal = roth_bal_filer + roth_bal_spouse
    taxable_brokerage_bal = D["curr_taxable"] + (D["curr_taxable_s"] if is_joint else 0)
    _brokerage_basis_dollars = min(D["taxable_basis"] + (D["taxable_basis_s"] if is_joint else 0), taxable_brokerage_bal)
    brokerage_gain_pct = max(0.0, 1.0 - _brokerage_basis_dollars / taxable_brokerage_bal) if taxable_brokerage_bal > 0 else 0.0
    taxable_cash_bal = D["curr_cash"]
    emergency_fund = D["emergency_fund"]
    life_cash_value = D["life_cash_value"]
    annuity_value_filer = D["annuity_value_f"]
    annuity_basis_filer = D["annuity_basis_f"]
    annuity_value_spouse = D["annuity_value_s"] if is_joint else 0.0
    annuity_basis_spouse = D["annuity_basis_s"] if is_joint else 0.0
    annuity_value = annuity_value_filer + annuity_value_spouse
    annuity_basis = annuity_basis_filer + annuity_basis_spouse

    # Current-year tax inputs (from Receiving)
    # If working, salary from Planning overrides manual wages on Receiving
    if D["is_working"]:
        # Compute pretax retirement deductions from D values for Tab 1 tax estimate.
        # 401(k)/SIMPLE pretax contributions reduce W-2 box 1 wages;
        # SEP-IRA/Traditional IRA/HSA are above-the-line deductions (adjustments).
        _sal_f = D["salary_filer"]
        _sal_s = D["salary_spouse"] if is_joint else 0.0
        _age_f = current_age_filer
        _age_s = current_age_spouse if is_joint else 0
        _50f = _age_f >= 50
        _50s = _age_s >= 50 if is_joint else False

        # Filer plan contribution (pretax portion)
        _pt_f = D["plan_type_f"]
        if _pt_f == "401(k)":
            _lim = 23500 + (7500 if _50f else 0)
            _c = _lim if D["max_401k_f"] else min(D["c401k_f"], _lim)
            _pretax_f = _c * (1 - D["roth_pct_f"] / 100)
        elif _pt_f == "SEP-IRA":
            _sep = min(_sal_f * 0.25, 69000)
            _pretax_f = _sep if D["max_401k_f"] else min(D["c401k_f"], _sep)
        elif _pt_f == "SIMPLE":
            _lim = 16500 + (3500 if _50f else 0)
            _c = _lim if D["max_401k_f"] else min(D["c401k_f"], _lim)
            _pretax_f = _c * (1 - D["roth_pct_f"] / 100)
        else:
            _pretax_f = 0.0

        # Spouse plan contribution (pretax portion)
        _pt_s = D["plan_type_s"]
        _pretax_s = 0.0
        if is_joint and _sal_s > 0:
            if _pt_s == "401(k)":
                _lim = 23500 + (7500 if _50s else 0)
                _c = _lim if D["max_401k_s"] else min(D["c401k_s"], _lim)
                _pretax_s = _c * (1 - D["roth_pct_s"] / 100)
            elif _pt_s == "SEP-IRA":
                _sep = min(_sal_s * 0.25, 69000)
                _pretax_s = _sep if D["max_401k_s"] else min(D["c401k_s"], _sep)
            elif _pt_s == "SIMPLE":
                _lim = 16500 + (3500 if _50s else 0)
                _c = _lim if D["max_401k_s"] else min(D["c401k_s"], _lim)
                _pretax_s = _c * (1 - D["roth_pct_s"] / 100)

        # IRA contributions
        _ira_lim_f = 7000 + (1000 if _50f else 0)
        _ira_lim_s = 7000 + (1000 if _50s else 0)
        _trad_ira_f = _ira_lim_f if D["max_ira_f"] else D["contrib_trad_ira_f"]
        _trad_ira_s = (_ira_lim_s if D["max_ira_s"] else D["contrib_trad_ira_s"]) if is_joint else 0.0

        # HSA
        if D["hsa_eligible"]:
            _hsa_max = (8550 if is_joint else 4300) + (1000 if _50f else 0)
            _hsa = _hsa_max if D["max_hsa"] else min(D["contrib_hsa"], _hsa_max)
        else:
            _hsa = 0.0

        # W-2 reduction (401k/SIMPLE only — these reduce box 1)
        _w2_ded_f = _pretax_f if _pt_f in ("401(k)", "SIMPLE") else 0.0
        _w2_ded_s = _pretax_s if _pt_s in ("401(k)", "SIMPLE") else 0.0
        wages = (_sal_f - _w2_ded_f) + ((_sal_s - _w2_ded_s) if is_joint else 0.0)

        # Above-the-line adjustments (SEP, trad IRA, HSA)
        _above_line = _trad_ira_f + _trad_ira_s + _hsa
        if _pt_f == "SEP-IRA":
            _above_line += _pretax_f
        if is_joint and _pt_s == "SEP-IRA":
            _above_line += _pretax_s
    else:
        wages = D["wages"]
        _above_line = 0.0
    tax_exempt_interest = D["tax_exempt_interest"]
    other_income = D["other_income"]
    # Add net rental income from base investment RE + additional properties
    _inv_re_rental = D["inv_re_net_income"]
    for _ri in range(1, 4):
        if D.get(f"inv_re_{_ri}_include", False) and D[f"inv_re_{_ri}_value"] > 0:
            _inv_re_rental += D[f"inv_re_{_ri}_net_income"]
    if _inv_re_rental > 0:
        other_income += _inv_re_rental

    # Asset allocation helper functions (must be defined before use)
    def _compute_aa_return(acct_key):
        _cma = [D["cma_us_equity"], D["cma_intl_equity"], D["cma_fixed_income"], D["cma_real_assets"], D["cma_cash"]]
        _flds = ["eq", "intl", "fi", "re"]
        allocs = [D[f"aa_{acct_key}_{fk}"] for fk in _flds]
        cash_pct = max(0, 100 - sum(allocs))
        allocs.append(cash_pct)
        return sum(a / 100 * r / 100 for a, r in zip(allocs, _cma))

    def _compute_aa_yields(acct_key):
        """Returns dict with total, div_yield, qual_pct, int_yield, cg_yield, deferred — all as decimals."""
        _cma_k = ["us_equity", "intl_equity", "fixed_income", "real_assets", "cash"]
        _flds = ["eq", "intl", "fi", "re"]
        allocs = [D[f"aa_{acct_key}_{fk}"] for fk in _flds]
        cash_pct = max(0, 100 - sum(allocs))
        allocs.append(cash_pct)
        total = div = qual_w = int_y = cg = 0.0
        for _i, _ck in enumerate(_cma_k):
            w = allocs[_i] / 100
            t = D[f"cma_{_ck}"] / 100
            d = D.get(f"cma_{_ck}_div", 0.0) / 100
            q = D.get(f"cma_{_ck}_qual", 0) / 100
            n = D.get(f"cma_{_ck}_int", 0.0) / 100
            c = D.get(f"cma_{_ck}_cg", 0.0) / 100
            total += w * t
            div += w * d
            qual_w += w * d * q
            int_y += w * n
            cg += w * c
        qual_pct = qual_w / div if div > 0 else 0.0
        deferred = total - div - int_y - cg
        return {"total": total, "div_yield": div, "qual_pct": qual_pct,
                "int_yield": int_y, "cg_yield": cg, "deferred": deferred}

    interest_taxable = D["interest_taxable"]
    total_ordinary_dividends = D["total_ordinary_dividends"]
    qualified_dividends = D["qualified_dividends"]
    cap_gain_loss = D["cap_gain_loss"]
    adjustments = D["adjustments"] + _above_line
    dependents = int(D["dependents"])
    # Auto-compute 65+ from DOB and tax year
    _tax_year_end = dt.date(tax_year, 12, 31)
    filer_65_plus = (TEA.age_at_date(filer_dob, _tax_year_end) >= 65) if filer_dob else D["filer_65_plus"]
    spouse_65_plus = (TEA.age_at_date(spouse_dob, _tax_year_end) >= 65) if (spouse_dob and is_joint) else False
    retirement_deduction = D["retirement_deduction"]
    out_of_state_gain = D["out_of_state_gain"]
    medical_expenses = D["medical_expenses"]
    charitable = D["charitable"]
    qcd_annual = D["qcd_annual"]
    # Deduction method mapping
    _ded_map = {
        "Auto (higher of standard/itemized)": "auto",
        "Force Itemized (auto-calculate)": "force_itemized",
        "Force Itemized (enter amount)": "custom_itemized",
        "Force Standard": "force_standard",
    }
    _ded_method_key = _ded_map.get(D["deduction_method"], "auto")
    home_value = D["home_value"]
    home_appreciation = D["home_appr"] / 100

    # Mortgage
    mortgage_balance = D["mtg_balance"]
    mortgage_rate = D["mtg_rate"] / 100
    mortgage_payment = D["mtg_payment_monthly"] * 12
    property_tax = D["property_tax"]

    # Investment income for projections
    if D["use_asset_alloc"]:
        _taxable_yields = _compute_aa_yields("taxable")
        proj_interest = taxable_brokerage_bal * _taxable_yields["int_yield"]
        proj_dividends = taxable_brokerage_bal * _taxable_yields["div_yield"]
        proj_qual_div = proj_dividends * _taxable_yields["qual_pct"]
        proj_cap_gains = taxable_brokerage_bal * _taxable_yields["cg_yield"]
    elif D["use_invest_assumptions"]:
        proj_div_yield = D["proj_div_yield"] / 100
        proj_cg_pct = D["proj_cg_pct"] / 100
        proj_int_rate = D["proj_int_rate"] / 100
        proj_interest = taxable_brokerage_bal * proj_int_rate
        proj_dividends = taxable_brokerage_bal * proj_div_yield
        proj_qual_div = proj_dividends * 0.8
        proj_cap_gains = taxable_brokerage_bal * proj_cg_pct
    else:
        proj_interest = interest_taxable
        proj_dividends = total_ordinary_dividends
        proj_qual_div = qualified_dividends
        proj_cap_gains = cap_gain_loss
    reinvest_dividends = D["reinvest_dividends"]
    reinvest_cap_gains = D["reinvest_cap_gains"]
    reinvest_interest = D["reinvest_interest"]

    if D["use_asset_alloc"]:
        r_pretax = (_compute_aa_return("pretax_f") + _compute_aa_return("pretax_s")) / 2 if is_joint else _compute_aa_return("pretax_f")
        r_roth = (_compute_aa_return("roth_f") + _compute_aa_return("roth_s")) / 2 if is_joint else _compute_aa_return("roth_f")
        r_taxable = _compute_aa_return("taxable")
        r_annuity = _compute_aa_return("annuity")
        r_cash = D["cma_cash"] / 100
        r_life = D["r_life"] / 100
    else:
        r_cash = D["r_cash"] / 100
        r_taxable = D["r_taxable"] / 100
        r_pretax = D["r_pretax"] / 100
        r_roth = D["r_roth"] / 100
        r_annuity = D["r_annuity"] / 100
        r_life = D["r_life"] / 100

    # RMD
    has_rmd = D["has_rmd"]
    auto_rmd = D["auto_rmd"] and has_rmd
    pretax_balance_filer_prior = D["pretax_bal_filer_prior"] if has_rmd else 0.0
    pretax_balance_spouse_prior = (D["pretax_bal_spouse_prior"] if is_joint else 0.0) if has_rmd else 0.0
    baseline_pretax_distributions = D["baseline_pretax_dist"] if has_rmd else 0.0
    rmd_manual = D["rmd_manual"] if has_rmd else 0.0

    # Spending
    spending_goal = D["living_expenses"]
    heir_tax_rate = D["heir_tax_rate"] / 100
    spending_order = [D["so1"], D["so2"], D["so3"], D["so4"]]
    surplus_dest_raw = D["surplus_dest"]
    surplus_dest = surplus_dest_raw
    surplus_destination = "none" if surplus_dest_raw == "Don't Reinvest" else ("cash" if surplus_dest_raw == "Cash/Savings" else "brokerage")

    # Survivor
    survivor_spending_pct = D["survivor_spend_pct"] if is_joint else 100
    pension_survivor_pct = D["pension_survivor_pct"] if is_joint else 0

    # Plan-through ages
    filer_plan_through_age = int(D["filer_plan_age"])
    spouse_plan_through_age = int(D["spouse_plan_age"]) if is_joint else None

    # --- Compute derived values ---
    tax_year_end = dt.date(tax_year, 12, 31)
    age_filer_eoy = TEA.age_at_date(filer_dob, tax_year_end) if filer_dob else 70
    age_spouse_eoy = TEA.age_at_date(spouse_dob, tax_year_end) if (spouse_dob and is_joint) else None

    # SS for this year — use current_annual directly when already receiving
    if filer_ss_already and filer_ss_current > 0:
        gross_ss_filer = filer_ss_current
    else:
        gross_ss_filer = TEA.annual_ss_in_year(dob=filer_dob, tax_year=tax_year, cola=inflation,
            already_receiving=False, current_annual=0,
            start_year=9999, fra_annual=filer_ss_fra,
            claim_choice=filer_ss_claim, current_year=current_year)
    gross_ss_spouse = 0.0
    if is_joint:
        if spouse_ss_already and spouse_ss_current > 0:
            gross_ss_spouse = spouse_ss_current
        else:
            gross_ss_spouse = TEA.annual_ss_in_year(dob=spouse_dob, tax_year=tax_year, cola=inflation,
                already_receiving=False, current_annual=0,
                start_year=9999, fra_annual=spouse_ss_fra,
                claim_choice=spouse_ss_claim, current_year=current_year)
    gross_ss_total = gross_ss_filer + gross_ss_spouse
    taxable_pensions_total = pension_filer + pension_spouse

    # RMD
    rmd_filer = TEA.compute_rmd_uniform_start73(pretax_balance_filer_prior, age_filer_eoy) if auto_rmd else 0.0
    rmd_spouse = TEA.compute_rmd_uniform_start73(pretax_balance_spouse_prior, age_spouse_eoy) if (auto_rmd and is_joint and age_spouse_eoy) else 0.0
    rmd_total = (rmd_filer + rmd_spouse) if auto_rmd else rmd_manual

    current_age_filer = TEA.age_at_date(filer_dob, dt.date.today()) if filer_dob else 70
    current_age_spouse = TEA.age_at_date(spouse_dob, dt.date.today()) if (spouse_dob and is_joint) else None

    # Compute years
    if is_joint and spouse_plan_through_age and current_age_spouse:
        years = max(filer_plan_through_age - current_age_filer, spouse_plan_through_age - current_age_spouse)
    else:
        years = filer_plan_through_age - current_age_filer
    years = max(1, years)
    start_year = max(current_year, int(tax_year))

    # Sync keys for PDF generation (use _fp_ prefix to avoid
    # colliding with page-2 widget-owned session-state keys)
    st.session_state["_fp_client_name"] = D["client_name"]
    st.session_state["_fp_filer_dob"] = filer_dob
    st.session_state["_fp_spouse_dob"] = spouse_dob
    st.session_state["_fp_tax_year"] = tax_year
    st.session_state["_fp_emergency_fund"] = emergency_fund

    def current_inputs():
        return {
            "wages": float(wages), "tax_exempt_interest": float(tax_exempt_interest),
            "interest_taxable": float(interest_taxable),
            "total_ordinary_dividends": float(total_ordinary_dividends),
            "qualified_dividends": float(qualified_dividends),
            "taxable_ira": float(baseline_pretax_distributions), "rmd_amount": float(rmd_total),
            "taxable_pensions": float(taxable_pensions_total), "gross_ss": float(gross_ss_total),
            "reinvest_dividends": bool(reinvest_dividends), "reinvest_cap_gains": bool(reinvest_cap_gains),
            "reinvest_interest": bool(reinvest_interest),
            "cap_gain_loss": float(cap_gain_loss), "other_income": float(other_income),
            "adjustments": float(adjustments), "dependents": int(dependents),
            "filing_status": filing_status, "filer_65_plus": bool(filer_65_plus),
            "spouse_65_plus": bool(spouse_65_plus),
            "retirement_deduction": float(retirement_deduction), "out_of_state_gain": float(out_of_state_gain),
            "mortgage_balance": float(mortgage_balance), "mortgage_rate": float(mortgage_rate),
            "mortgage_payment": float(mortgage_payment), "property_tax": float(property_tax),
            "medical_expenses": float(medical_expenses), "charitable": float(charitable),
            "qcd_annual": float(qcd_annual),
            "ordinary_tax_only": 0.0, "cashflow_taxfree": 0.0,
            "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0,
            "tax_year": int(tax_year),
            "force_deduction_method": _ded_method_key,
            "custom_deduction_amount": float(D["custom_itemized_amount"]),
            "wages_filer": float(D["salary_filer"]) if D["is_working"] else 0.0,
            "wages_spouse": float(D["salary_spouse"]) if D["is_working"] else 0.0,
            "self_employed_filer": bool(D["self_employed_f"]),
            "self_employed_spouse": bool(D["self_employed_s"]),
        }

    def current_assets():
        _basis = min(float(annuity_basis), float(annuity_value)) if float(annuity_value) > 0 else 0.0
        return {
            "taxable": {"cash": max(0.0, float(taxable_cash_bal) - float(emergency_fund)),
                        "brokerage": float(taxable_brokerage_bal), "emergency_fund": float(emergency_fund)},
            "pretax": {"balance": float(pretax_bal)},
            "taxfree": {
                "roth": float(roth_bal), "roth_filer": float(roth_bal_filer),
                "roth_spouse": float(roth_bal_spouse),
                "life_cash_value": float(life_cash_value),
            },
            "annuity": {
                "value": float(annuity_value), "basis": float(_basis),
                "value_filer": float(annuity_value_filer),
                "basis_filer": float(min(annuity_basis_filer, annuity_value_filer)),
                "value_spouse": float(annuity_value_spouse),
                "basis_spouse": float(min(annuity_basis_spouse, annuity_value_spouse)),
            },
        }

    def _build_tab3_params():
        pf = float(pretax_bal_filer_current)
        ps = float(pretax_bal_spouse_current)
        ratio = pf / (pf + ps) if (pf + ps) > 0 else 1.0
        # Compute non-liquidated investment RE value and weighted appreciation
        _t3_ire_val = 0.0
        _t3_ire_appr = D["inv_re_appr"] / 100
        _t3_wt_num = 0.0; _t3_wt_den = 0.0
        if D["inv_re_value"] > 0 and not D["inv_re_liquidate"]:
            _t3_ire_val += D["inv_re_value"]
            _t3_wt_num += D["inv_re_value"] * (D["inv_re_appr"] / 100)
            _t3_wt_den += D["inv_re_value"]
        for _ri in range(1, 4):
            if D.get(f"inv_re_{_ri}_include", False) and D[f"inv_re_{_ri}_value"] > 0 and not D.get(f"inv_re_{_ri}_liquidate", False):
                _rv = D[f"inv_re_{_ri}_value"]
                _t3_ire_val += _rv
                _t3_wt_num += _rv * (D[f"inv_re_{_ri}_appr"] / 100)
                _t3_wt_den += _rv
        if _t3_wt_den > 0:
            _t3_ire_appr = _t3_wt_num / _t3_wt_den
        return {
            "spending_goal": spending_goal, "start_year": int(start_year),
            "years": int(years), "inflation": float(inflation),
            "bracket_growth": float(bracket_growth), "medicare_growth": float(medicare_growth),
            "pension_cola": float(pension_cola), "heir_tax_rate": heir_tax_rate,
            "r_cash": r_cash, "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
            "r_annuity": r_annuity, "r_life": r_life,
            "gross_ss_total": gross_ss_total, "taxable_pensions_total": taxable_pensions_total,
            "gross_ss_filer": float(gross_ss_filer), "gross_ss_spouse": float(gross_ss_spouse),
            "filer_dob": filer_dob, "spouse_dob": spouse_dob,
            "filer_ss_already": filer_ss_already, "filer_ss_current": float(filer_ss_current),
            "filer_ss_start_year": int(filer_ss_start_year), "filer_ss_fra": float(filer_ss_fra),
            "filer_ss_claim": filer_ss_claim,
            "spouse_ss_already": spouse_ss_already, "spouse_ss_current": float(spouse_ss_current),
            "spouse_ss_start_year": int(spouse_ss_start_year), "spouse_ss_fra": float(spouse_ss_fra),
            "spouse_ss_claim": spouse_ss_claim,
            "current_year": int(current_year),
            "pension_filer": float(pension_filer), "pension_spouse": float(pension_spouse),
            "filer_plan_through_age": filer_plan_through_age,
            "spouse_plan_through_age": spouse_plan_through_age,
            "survivor_spending_pct": survivor_spending_pct,
            "pension_survivor_pct": pension_survivor_pct,
            "filing_status": filing_status,
            "current_age_filer": current_age_filer, "current_age_spouse": current_age_spouse,
            "pretax_filer_ratio": ratio, "brokerage_gain_pct": float(brokerage_gain_pct),
            "interest_taxable": float(proj_interest),
            "reinvest_interest": bool(reinvest_interest),
            "total_ordinary_dividends": float(proj_dividends),
            "qualified_dividends": float(proj_qual_div),
            "cap_gain_loss": float(proj_cap_gains),
            "reinvest_dividends": bool(reinvest_dividends),
            "reinvest_cap_gains": bool(reinvest_cap_gains),
            "retirement_deduction": float(retirement_deduction),
            "out_of_state_gain": float(out_of_state_gain),
            "dependents": int(dependents),
            "property_tax": float(property_tax),
            "medical_expenses": float(medical_expenses),
            "charitable": float(charitable),
            "qcd_annual": float(qcd_annual),
            "mortgage_balance": float(mortgage_balance), "mortgage_rate": float(mortgage_rate),
            "mortgage_payment": float(mortgage_payment),
            "wages": float(wages), "tax_exempt_interest": float(tax_exempt_interest),
            "other_income": float(other_income), "adjustments": float(adjustments),
            "wages_start_age": int(D["wages_start_age"]), "wages_end_age": int(D["wages_end_age"]),
            "other_income_start_age": int(D["other_income_start_age"]), "other_income_end_age": int(D["other_income_end_age"]),
            "tax_exempt_start_age": int(D["tax_exempt_start_age"]), "tax_exempt_end_age": int(D["tax_exempt_end_age"]),
            "home_value": float(home_value), "home_appreciation": float(home_appreciation),
            "inv_re_value": float(_t3_ire_val), "inv_re_appr": float(_t3_ire_appr),
            "additional_expenses": [dict(ae) for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0],
            "future_income": [dict(fi) for fi in st.session_state.future_income if fi.get("amount", 0) > 0],
            "surplus_destination": surplus_destination,
            # Estate tax
            "estate_tax_enabled": bool(D["estate_tax_enabled"]),
            "federal_estate_exemption": float(D["federal_estate_exemption"]),
            "exemption_inflation": float(D["exemption_inflation"]),
            "use_portability": bool(D["use_portability"]),
            "state_estate_tax_rate": float(D["state_estate_tax_rate"]),
            "state_estate_exemption": float(D["state_estate_exemption"]),
            # AA yield decomposition for projection
            "aa_yields": {
                "taxable": _compute_aa_yields("taxable"),
                "pretax": _compute_aa_yields("pretax_f"),
                "roth": _compute_aa_yields("roth_f"),
                "annuity": _compute_aa_yields("annuity"),
            } if D["use_asset_alloc"] else None,
        }

    # --- Auto-compute base taxes ---
    _auto_inputs = current_inputs()
    _auto_assets = current_assets()
    _auto_res = TEA.compute_case(_auto_inputs)
    st.session_state.base_inputs = _auto_inputs
    st.session_state.base_results = _auto_res
    st.session_state.assets = _auto_assets

    # ---- Sub-tabs ----
    _is_working = D["is_working"]
    if _is_working:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Base Tax Estimator", "Retirement Readiness", "Savings Optimizer",
            "Retirement Projection", "Roth Conversion Opportunity"])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Base Tax Estimator", "Income Needs", "Wealth Projection",
            "Multigenerational Optimizer", "Roth Conversion Opportunity"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: Base Tax Estimator
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Estimated Tax Analysis")
        r = st.session_state.base_results
        TEA.display_tax_return(r, mortgage_pmt=float(mortgage_payment),
                               filer_65=filer_65_plus, spouse_65=spouse_65_plus)

    # ================================================================
    # PRE-RETIREMENT TABS (tabs 2-4) — only when working
    # ================================================================
    if _is_working:
        _fp_pre_ret_tabs(tab2, tab4, tab3, D, is_joint, filing_status,
                         filer_dob, spouse_dob, current_year, inflation, bracket_growth,
                         medicare_growth, pension_cola, spending_order, surplus_destination,
                         current_age_filer, current_age_spouse,
                         filer_plan_through_age, spouse_plan_through_age,
                         r_pretax, r_roth, r_taxable, r_cash,
                         r_annuity, r_life,
                         life_cash_value, annuity_value, annuity_basis,
                         filer_65_plus, spouse_65_plus,
                         wages, mortgage_balance, mortgage_rate, mortgage_payment,
                         property_tax, medical_expenses, charitable, qcd_annual,
                         home_value, home_appreciation, years,
                         pension_filer, pension_spouse,
                         filer_ss_already, filer_ss_current, filer_ss_start_year,
                         filer_ss_fra, filer_ss_claim,
                         spouse_ss_already, spouse_ss_current, spouse_ss_start_year,
                         spouse_ss_fra, spouse_ss_claim,
                         heir_tax_rate, survivor_spending_pct, pension_survivor_pct,
                         pretax_bal_filer_current, pretax_bal_spouse_current,
                         taxable_brokerage_bal, brokerage_gain_pct, taxable_cash_bal,
                         emergency_fund, interest_taxable, total_ordinary_dividends,
                         qualified_dividends, cap_gain_loss, other_income, adjustments,
                         dependents, retirement_deduction, out_of_state_gain,
                         tax_exempt_interest, start_year,
                         proj_interest, proj_dividends, proj_qual_div, proj_cap_gains,
                         reinvest_dividends, reinvest_cap_gains, reinvest_interest,
                         spending_goal)

    # ════════════════════════════════════════════════════════════════
    # RETIRED TABS 2-4 — only when NOT working
    # ════════════════════════════════════════════════════════════════
    if not _is_working:

        # ════════════════════════════════════════════════════════════════
        # TAB 2: Income Needs
        # ════════════════════════════════════════════════════════════════
        with tab2:
            st.subheader("Income Needs Analysis")
            if True:
                net_needed = st.number_input("Net income needed", min_value=0.0, value=float(D["net_income_needed"]), step=1000.0, key="fp_net_needed")
                D["net_income_needed"] = net_needed
                taxes_from_cash = st.checkbox("Pay taxes from cash (not from withdrawal)?", value=False, key="fp_taxes_from_cash")
                _wd_sources = ["Taxable \u2013 Cash", "Taxable \u2013 Brokerage", "Pre-Tax \u2013 IRA/401k", "Annuity", "Roth", "Life Insurance (loan)"]
                if is_joint:
                    _wd_sources.insert(_wd_sources.index("Roth") + 1, "Roth \u2014 Spouse")
                    _wd_sources.insert(_wd_sources.index("Annuity") + 1, "Annuity \u2014 Spouse")
                source = st.radio("Withdrawal source", _wd_sources, key="fp_wd_source")
                avail = TEA.max_withdrawable(st.session_state.assets, source)
                st.caption(f"Available in {source}: {TEA.money(avail)}")
                if st.button("Calculate Income Needs", type="primary", key="fp_calc_needs"):
                    for _ae in st.session_state.additional_expenses:
                        _ae["calculated"] = False; _ae["tax_impact"] = None
                    extra, base_case, solved, solved_assets, solved_inputs = TEA.solve_gross_up_with_assets(
                        st.session_state.base_inputs, st.session_state.assets, source,
                        float(brokerage_gain_pct), float(net_needed), taxes_from_cash)
                    if extra is None:
                        st.error(f"Insufficient assets in {source} ({TEA.money(avail)}) to meet the income need of {TEA.money(net_needed)}.")
                    elif extra == 0.0:
                        st.info(f"Your base income already meets or exceeds the {TEA.money(net_needed)} target \u2014 no additional withdrawal needed.")
                        st.session_state.last_solved_results = base_case
                        st.session_state.last_solved_inputs = st.session_state.base_inputs
                        st.session_state.last_solved_assets = st.session_state.assets
                        st.session_state.last_net_needed = net_needed
                        st.session_state.last_source = source
                        st.session_state.gross_from_needs = base_case["spendable_gross"]
                        st.session_state.last_withdrawal_proceeds = 0
                    else:
                        st.success(f"**Withdrawal needed: {TEA.money(extra)}** from {source}")
                        st.session_state.last_solved_results = solved
                        st.session_state.last_solved_inputs = solved_inputs
                        st.session_state.last_solved_assets = solved_assets
                        st.session_state.last_net_needed = net_needed
                        st.session_state.last_source = source
                        st.session_state.gross_from_needs = solved["spendable_gross"]
                        st.session_state.last_withdrawal_proceeds = extra

                        st.divider()
                        st.markdown("## Estimated Tax Analysis \u2014 After Withdrawal")
                        TEA.display_tax_return(solved, mortgage_pmt=float(mortgage_payment),
                                               filer_65=filer_65_plus, spouse_65=spouse_65_plus)

                        st.divider()
                        st.markdown("### Additional Tax Liability from Generating Income Needs")
                        _delta_fed = solved["fed_tax"] - base_case["fed_tax"]
                        _delta_sc = solved["sc_tax"] - base_case["sc_tax"]
                        _delta_med = solved["medicare_premiums"] - base_case["medicare_premiums"]
                        _delta_total = _delta_fed + _delta_sc + _delta_med
                        _impact_table = "| | Before | After | Change |\n|:---|-------:|-------:|-------:|\n"
                        _impact_table += f"| Federal tax | {TEA.money(base_case['fed_tax'])} | {TEA.money(solved['fed_tax'])} | {TEA.money(_delta_fed)} |\n"
                        _impact_table += f"| SC tax | {TEA.money(base_case['sc_tax'])} | {TEA.money(solved['sc_tax'])} | {TEA.money(_delta_sc)} |\n"
                        _impact_table += f"| Medicare premiums | {TEA.money(base_case['medicare_premiums'])} | {TEA.money(solved['medicare_premiums'])} | {TEA.money(_delta_med)} |\n"
                        _impact_table += f"| **Total** | **{TEA.money(base_case['total_tax'] + base_case['medicare_premiums'])}** | **{TEA.money(solved['total_tax'] + solved['medicare_premiums'])}** | **{TEA.money(_delta_total)}** |\n"
                        st.markdown(_impact_table)
                        if _delta_total > 0:
                            st.caption(f"Generating {TEA.money(net_needed)} in income needs from {source} adds {TEA.money(_delta_total)} in total tax liability.")
                        else:
                            st.caption(f"This withdrawal source does not increase your tax liability.")

                # ---------- Additional Expenses ----------
                st.divider()
                with st.expander("Additional Expenses", expanded=len(st.session_state.additional_expenses) > 0):
                    st.caption("Add one-time or recurring expenses beyond base spending (e.g., new car, home repair, long-term care). Each expense has its own withdrawal source.")
                    _ae_sources = ["Taxable \u2013 Cash", "Taxable \u2013 Brokerage", "Pre-Tax \u2013 IRA/401k", "Annuity", "Roth", "Life Insurance (loan)"]
                    if is_joint:
                        _ae_sources.insert(_ae_sources.index("Roth") + 1, "Roth \u2014 Spouse")
                        _ae_sources.insert(_ae_sources.index("Annuity") + 1, "Annuity \u2014 Spouse")
                    _filer_age_now = TEA.age_at_date(filer_dob, dt.date.today()) if filer_dob else 70

                    if st.button("Add Expense", key="fp_btn_add_expense"):
                        st.session_state.additional_expenses.append({
                            "name": "", "net_amount": 0.0,
                            "start_age": _filer_age_now, "end_age": 0,
                            "inflates": False, "source": "Taxable \u2013 Cash",
                            "gross_needed": 0.0, "calculated": False, "tax_impact": None,
                        })
                        st.rerun()

                    _ae_to_remove = None
                    for idx, ae in enumerate(st.session_state.additional_expenses):
                        st.markdown(f"**Expense {idx + 1}**")
                        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
                        with c1: ae["name"] = st.text_input("Description", value=ae["name"], key=f"fp_ae_name_{idx}")
                        with c2: ae["net_amount"] = st.number_input("Net Amount $", value=float(ae["net_amount"]), min_value=0.0, step=1000.0, key=f"fp_ae_amt_{idx}")
                        with c3: ae["start_age"] = st.number_input("Start Age", value=int(ae["start_age"]), min_value=50, max_value=110, step=1, key=f"fp_ae_start_{idx}")
                        with c4: ae["end_age"] = st.number_input("End Age (0=one-time)", value=int(ae["end_age"]), min_value=0, max_value=110, step=1, key=f"fp_ae_end_{idx}")
                        c5, c6, c7 = st.columns([2, 3, 1])
                        with c5: ae["inflates"] = st.checkbox("Inflates?", value=ae["inflates"], key=f"fp_ae_inf_{idx}")
                        with c6: ae["source"] = st.selectbox("Withdrawal Source", _ae_sources, index=_ae_sources.index(ae["source"]) if ae["source"] in _ae_sources else 0, key=f"fp_ae_src_{idx}")
                        with c7:
                            if st.button("Remove", key=f"fp_ae_rm_{idx}"):
                                _ae_to_remove = idx
                        _ae_eff_end = ae["end_age"] if ae["end_age"] > ae["start_age"] else ae["start_age"] + 1
                        _ae_duration = "one-time" if ae["end_age"] == 0 or ae["end_age"] <= ae["start_age"] else f"ages {ae['start_age']}-{ae['end_age']}"
                        st.caption(f"Duration: {_ae_duration} | Source: {ae['source']}")
                        st.markdown("---")

                    if _ae_to_remove is not None:
                        st.session_state.additional_expenses.pop(_ae_to_remove)
                        st.rerun()

                # Calculate button and results OUTSIDE the expander
                _ae_with_amounts = [ae for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0]
                if _ae_with_amounts:
                    if not st.session_state.last_solved_results:
                        st.warning("Run 'Calculate Income Needs' above first to establish the baseline before calculating additional expenses.")
                    else:
                        _ae_calc_clicked = st.button("Calculate Additional Expenses", type="primary", key="fp_btn_calc_addl_expenses")
                        if _ae_calc_clicked:
                            _ae_baseline_inputs = st.session_state.last_solved_inputs
                            _ae_baseline_assets = st.session_state.last_solved_assets
                            _ae_baseline_res = st.session_state.last_solved_results
                            _ae_baseline_net = _ae_baseline_res["net_after_tax"]
                            for ae in st.session_state.additional_expenses:
                                if ae["net_amount"] <= 0:
                                    ae["calculated"] = False; ae["tax_impact"] = None; continue
                                _ae_target = _ae_baseline_net + float(ae["net_amount"])
                                _ae_extra, _ae_bc, _ae_solved, _ae_assets, _ae_inp = TEA.solve_gross_up_with_assets(
                                    _ae_baseline_inputs, _ae_baseline_assets, ae["source"],
                                    float(brokerage_gain_pct), _ae_target, False)
                                if _ae_extra is None:
                                    _ae_avail = TEA.max_withdrawable(_ae_baseline_assets, ae['source'])
                                    ae["gross_needed"] = ae["net_amount"]; ae["calculated"] = True
                                    ae["tax_impact"] = {"error": f"Cannot generate {TEA.money(ae['net_amount'])} net from {ae['source']} (available: {TEA.money(_ae_avail)})."}
                                elif _ae_extra == 0.0:
                                    ae["gross_needed"] = ae["net_amount"]; ae["calculated"] = True
                                    ae["tax_impact"] = {
                                        "gross": ae["net_amount"],
                                        "fed_before": _ae_baseline_res["fed_tax"], "fed_after": _ae_baseline_res["fed_tax"], "delta_fed": 0.0,
                                        "sc_before": _ae_baseline_res["sc_tax"], "sc_after": _ae_baseline_res["sc_tax"], "delta_sc": 0.0,
                                        "med_before": _ae_baseline_res["medicare_premiums"], "med_after": _ae_baseline_res["medicare_premiums"], "delta_med": 0.0,
                                        "total_before": _ae_baseline_res["total_tax"] + _ae_baseline_res["medicare_premiums"],
                                        "total_after": _ae_baseline_res["total_tax"] + _ae_baseline_res["medicare_premiums"], "delta_total": 0.0,
                                    }
                                else:
                                    ae["gross_needed"] = _ae_extra
                                    _d_fed = _ae_solved["fed_tax"] - _ae_baseline_res["fed_tax"]
                                    _d_sc = _ae_solved["sc_tax"] - _ae_baseline_res["sc_tax"]
                                    _d_med = _ae_solved["medicare_premiums"] - _ae_baseline_res["medicare_premiums"]
                                    _d_total = _d_fed + _d_sc + _d_med
                                    ae["calculated"] = True
                                    ae["tax_impact"] = {
                                        "gross": _ae_extra,
                                        "fed_before": _ae_baseline_res["fed_tax"], "fed_after": _ae_solved["fed_tax"], "delta_fed": _d_fed,
                                        "sc_before": _ae_baseline_res["sc_tax"], "sc_after": _ae_solved["sc_tax"], "delta_sc": _d_sc,
                                        "med_before": _ae_baseline_res["medicare_premiums"], "med_after": _ae_solved["medicare_premiums"], "delta_med": _d_med,
                                        "total_before": _ae_baseline_res["total_tax"] + _ae_baseline_res["medicare_premiums"],
                                        "total_after": _ae_solved["total_tax"] + _ae_solved["medicare_premiums"], "delta_total": _d_total,
                                    }

                        for ae in st.session_state.additional_expenses:
                            if ae.get("calculated") and ae.get("tax_impact"):
                                _ti = ae["tax_impact"]
                                if "error" in _ti:
                                    st.error(f"**{ae['name'] or 'Expense'}:** {_ti['error']}")
                                else:
                                    st.markdown(f"### {ae['name'] or 'Expense'} \u2014 Tax Impact")
                                    st.success(f"**Withdrawal needed: {TEA.money(_ti['gross'])}** from {ae['source']}")
                                    _ae_impact = "| | Income Needs | + Expense | Change |\n|:---|-------:|-------:|-------:|\n"
                                    _ae_impact += f"| Federal tax | {TEA.money(_ti['fed_before'])} | {TEA.money(_ti['fed_after'])} | {TEA.money(_ti['delta_fed'])} |\n"
                                    _ae_impact += f"| SC tax | {TEA.money(_ti['sc_before'])} | {TEA.money(_ti['sc_after'])} | {TEA.money(_ti['delta_sc'])} |\n"
                                    _ae_impact += f"| Medicare premiums | {TEA.money(_ti['med_before'])} | {TEA.money(_ti['med_after'])} | {TEA.money(_ti['delta_med'])} |\n"
                                    _ae_impact += f"| **Total** | **{TEA.money(_ti['total_before'])}** | **{TEA.money(_ti['total_after'])}** | **{TEA.money(_ti['delta_total'])}** |\n"
                                    st.markdown(_ae_impact)

                        _ae_summary = []
                        for ae in st.session_state.additional_expenses:
                            if ae["net_amount"] > 0:
                                _ae_eff_end = ae["end_age"] if ae["end_age"] > ae["start_age"] else ae["start_age"] + 1
                                _tax_cost = ae.get("tax_impact", {}).get("delta_total", 0) if ae.get("calculated") else 0
                                _ae_summary.append({
                                    "Expense": ae["name"] or "Expense", "Net Amount": TEA.money(ae["net_amount"]),
                                    "Ages": f"{ae['start_age']}-{_ae_eff_end}" if _ae_eff_end > ae["start_age"] + 1 else f"{ae['start_age']} (one-time)",
                                    "Inflates": "Yes" if ae["inflates"] else "No", "Source": ae["source"],
                                    "Gross W/D": TEA.money(ae.get("gross_needed", 0)), "Tax Cost": TEA.money(_tax_cost),
                                })
                        if _ae_summary:
                            st.markdown("#### Additional Expenses Summary")
                            st.dataframe(pd.DataFrame(_ae_summary), use_container_width=True, hide_index=True)

                # ---------- Future Income ----------
                with st.expander("Future Income", expanded=len(st.session_state.future_income) > 0):
                    st.caption("Add expected future income (e.g., inheritance, rental income, part-time work).")
                    _filer_age_now2 = TEA.age_at_date(filer_dob, dt.date.today()) if filer_dob else 70

                    if st.button("Add Income", key="fp_btn_add_income"):
                        st.session_state.future_income.append({
                            "name": "", "amount": 0.0,
                            "start_age": _filer_age_now2, "end_age": 0,
                            "inflates": False, "taxable": True,
                        })
                        st.rerun()

                    _fi_to_remove = None
                    for idx, fi in enumerate(st.session_state.future_income):
                        st.markdown(f"**Income {idx + 1}**")
                        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
                        with c1: fi["name"] = st.text_input("Description", value=fi["name"], key=f"fp_fi_name_{idx}")
                        with c2: fi["amount"] = st.number_input("Annual Amount $", value=float(fi["amount"]), min_value=0.0, step=1000.0, key=f"fp_fi_amt_{idx}")
                        with c3: fi["start_age"] = st.number_input("Start Age", value=int(fi["start_age"]), min_value=50, max_value=110, step=1, key=f"fp_fi_start_{idx}")
                        with c4: fi["end_age"] = st.number_input("End Age (0=ongoing)", value=int(fi["end_age"]), min_value=0, max_value=110, step=1, key=f"fp_fi_end_{idx}")
                        c5, c6, c7 = st.columns([2, 2, 1])
                        with c5: fi["inflates"] = st.checkbox("Inflates?", value=fi["inflates"], key=f"fp_fi_inf_{idx}")
                        with c6: fi["taxable"] = st.checkbox("Taxable?", value=fi["taxable"], key=f"fp_fi_tax_{idx}")
                        with c7:
                            if st.button("Remove", key=f"fp_fi_rm_{idx}"):
                                _fi_to_remove = idx
                        _fi_eff_end = fi["end_age"] if fi["end_age"] > fi["start_age"] else 999
                        _fi_duration = "ongoing" if fi["end_age"] == 0 or fi["end_age"] <= fi["start_age"] else f"ages {fi['start_age']}-{fi['end_age']}"
                        st.caption(f"Duration: {_fi_duration} | {'Taxable' if fi['taxable'] else 'Non-taxable'}")
                        st.markdown("---")

                    if _fi_to_remove is not None:
                        st.session_state.future_income.pop(_fi_to_remove)
                        st.rerun()

                    if st.session_state.future_income:
                        _fi_summary = []
                        for fi in st.session_state.future_income:
                            if fi["amount"] > 0:
                                _fi_eff_end = fi["end_age"] if fi["end_age"] > fi["start_age"] else "ongoing"
                                _fi_summary.append({
                                    "Income": fi["name"] or "Income", "Annual Amount": TEA.money(fi["amount"]),
                                    "Ages": f"{fi['start_age']}-{_fi_eff_end}",
                                    "Inflates": "Yes" if fi["inflates"] else "No",
                                    "Taxable": "Yes" if fi["taxable"] else "No",
                                })
                        if _fi_summary:
                            st.markdown("#### Summary")
                            st.dataframe(pd.DataFrame(_fi_summary), use_container_width=True, hide_index=True)

        # ════════════════════════════════════════════════════════════════
        # TAB 3: Wealth Projection
        # ════════════════════════════════════════════════════════════════
        with tab3:
            st.subheader("Wealth Projection")
            if st.session_state.assets is None:
                st.warning("Run Base Tax Estimator first.")
            else:
                a0 = st.session_state.assets
                _default_spend = st.session_state.last_net_needed if st.session_state.last_net_needed else spending_goal
                colL, colR = st.columns(2)
                with colL:
                    spending_goal = st.number_input("Annual Spending Goal (After-Tax)", value=_default_spend, step=5000.0, key="fp_spending_goal")
                    if st.session_state.last_net_needed:
                        st.caption("(Pulled from Income Needs tab)")
                    heir_tax_rate = st.number_input("Heir Tax Rate (%)", value=D["heir_tax_rate"], step=1.0, key="fp_heir_tab3",
                        help="Heir's marginal tax rate on inherited IRA distributions. Used in 10-year SECURE Act model.") / 100
                    start_year = st.number_input("Start year", min_value=2020, max_value=2100, value=max(2026, int(tax_year)), step=1, key="fp_start_year")
                    filer_plan_through_age = st.number_input("Filer Plan Through Age", min_value=70, max_value=105, value=int(D["filer_plan_age"]), step=1, key="fp_filer_plan_age")
                    spouse_plan_through_age = st.number_input("Spouse Plan Through Age", min_value=70, max_value=105, value=int(D["spouse_plan_age"]), step=1, key="fp_spouse_plan_age") if is_joint else None
                    if is_joint and spouse_plan_through_age and current_age_spouse:
                        years = max(filer_plan_through_age - current_age_filer, spouse_plan_through_age - current_age_spouse)
                    else:
                        years = filer_plan_through_age - current_age_filer
                    years = max(1, years)
                    st.caption(f"Projection: {years} years")
                    if is_joint:
                        survivor_spending_pct = st.slider("Survivor Spending % (after first death)", 50, 100, D["survivor_spend_pct"], key="fp_survivor_spend_pct")
                        pension_survivor_pct = st.slider("Pension Survivor Benefit %", 0, 100, D["pension_survivor_pct"], key="fp_pension_survivor_pct")
                    else:
                        survivor_spending_pct = 100; pension_survivor_pct = 0
                    st.markdown("#### Liquidation Order (Waterfall)")
                    c1, c2, c3, c4 = st.columns(4)
                    opts = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
                    so1 = c1.selectbox("1st", opts, index=opts.index(D["so1"]), key="fp_so1")
                    so2 = c2.selectbox("2nd", opts, index=opts.index(D["so2"]), key="fp_so2")
                    so3 = c3.selectbox("3rd", opts, index=opts.index(D["so3"]), key="fp_so3")
                    so4 = c4.selectbox("4th", opts, index=opts.index(D["so4"]), key="fp_so4")
                    D["so1"] = so1; D["so2"] = so2; D["so3"] = so3; D["so4"] = so4
                    spending_order = [so1, so2, so3, so4]
                    # Warn if an early-order bucket has no balance
                    _bucket_bals = {
                        "Taxable": float(taxable_brokerage_bal) + float(taxable_cash_bal),
                        "Pre-Tax": float(pretax_bal),
                        "Tax-Free": float(roth_bal) + float(life_cash_value),
                        "Tax-Deferred": float(annuity_value),
                    }
                    _empty_early = [b for b in spending_order[:2] if _bucket_bals.get(b, 0) < 1]
                    if _empty_early:
                        st.warning(f"Waterfall has **{', '.join(_empty_early)}** early in the order but {'it has' if len(_empty_early)==1 else 'they have'} no balance. Withdrawals will skip to the next bucket.")
                    surplus_dest = st.radio("Reinvest surplus income in:", ["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"],
                        index=["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"].index(D["surplus_dest"]),
                        key="fp_ret_surplus_dest", horizontal=True,
                        help="When income exceeds spending + taxes, where does the excess go?")
                    D["surplus_dest"] = surplus_dest
                    surplus_destination = "none" if surplus_dest == "Don't Reinvest" else ("cash" if surplus_dest == "Cash/Savings" else "brokerage")
                with colR:
                    st.markdown("#### Growth Rates")
                    st.write(f"Cash: {r_cash:.1%} | Brokerage: {r_taxable:.1%} | Pre-tax: {r_pretax:.1%} | Roth: {r_roth:.1%} | Annuity: {r_annuity:.1%} | Life: {r_life:.1%}")
                    st.markdown("#### Monte Carlo Settings")
                    mc_simulations = st.number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100, key="fp_mc_sims")
                    mc_volatility = st.number_input("Annual volatility (%)", value=12.0, step=1.0, key="fp_mc_vol") / 100

                    if D["use_asset_alloc"]:
                        st.markdown("#### Portfolio Summary (from Asset Allocation)")
                        _ps_accts = [
                            ("Brokerage", taxable_brokerage_bal, _compute_aa_yields("taxable"), _compute_aa_return("taxable")),
                            ("Pre-Tax", pretax_bal, _compute_aa_yields("pretax_f"), _compute_aa_return("pretax_f")),
                            ("Roth", roth_bal, _compute_aa_yields("roth_f"), _compute_aa_return("roth_f")),
                        ]
                        if annuity_value > 0:
                            _ps_accts.append(("Annuity", annuity_value, _compute_aa_yields("annuity"), _compute_aa_return("annuity")))
                        _ps_total = sum(b for _, b, _, _ in _ps_accts)
                        if _ps_total > 0:
                            _cma_labels_ps = ["US Equity", "Int'l Equity", "Fixed Income", "Real Assets", "Cash"]
                            _flds_ps = ["eq", "intl", "fi", "re"]
                            _acct_keys_ps = [("taxable", taxable_brokerage_bal), ("pretax_f", pretax_bal), ("roth_f", roth_bal)]
                            if annuity_value > 0:
                                _acct_keys_ps.append(("annuity", annuity_value))
                            _overall_alloc = [0.0] * 5
                            for _ak, _ab in _acct_keys_ps:
                                _w = _ab / _ps_total
                                _aa = [D[f"aa_{_ak}_{fk}"] for fk in _flds_ps]
                                _aa_cash = max(0, 100 - sum(_aa))
                                _aa.append(_aa_cash)
                                for _j in range(5):
                                    _overall_alloc[_j] += _w * _aa[_j]
                            _alloc_str = " | ".join(f"{lbl}: {pct:.0f}%" for lbl, pct in zip(_cma_labels_ps, _overall_alloc))
                            _ps_ret = sum(b / _ps_total * r for _, b, _, r in _ps_accts)
                            _ps_div = sum(b / _ps_total * y["div_yield"] for _, b, y, _ in _ps_accts)
                            _ps_int = sum(b / _ps_total * y["int_yield"] for _, b, y, _ in _ps_accts)
                            _ps_cg = sum(b / _ps_total * y["cg_yield"] for _, b, y, _ in _ps_accts)
                            _ps_def = _ps_ret - _ps_div - _ps_int - _ps_cg
                            st.write(f"**Total Portfolio:** \\${_ps_total:,.0f}")
                            st.write(f"**Allocation:** {_alloc_str}")
                            st.write(f"**Blended Return: {_ps_ret * 100:.2f}%** — "
                                     f"Div: {_ps_div * 100:.2f}% (\\${_ps_total * _ps_div:,.0f}) | "
                                     f"Int: {_ps_int * 100:.2f}% (\\${_ps_total * _ps_int:,.0f}) | "
                                     f"CG: {_ps_cg * 100:.2f}% (\\${_ps_total * _ps_cg:,.0f}) | "
                                     f"Deferred: {_ps_def * 100:.2f}%")

                # Read-only summary of additional expenses and future income
                _ae_active = [ae for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0]
                _fi_active = [fi for fi in st.session_state.future_income if fi.get("amount", 0) > 0]
                if _ae_active or _fi_active:
                    st.markdown("#### Additional Items (from Income Needs tab)")
                    if _ae_active:
                        _ae_disp = []
                        for ae in _ae_active:
                            _eff_end = ae["end_age"] if ae["end_age"] > ae["start_age"] else ae["start_age"] + 1
                            _ae_disp.append({
                                "Expense": ae.get("name") or "Expense", "Net $": TEA.money(ae["net_amount"]),
                                "Ages": f"{ae['start_age']}-{_eff_end}" if _eff_end > ae["start_age"] + 1 else f"{ae['start_age']} (one-time)",
                                "Source": ae["source"], "Inflates": "Yes" if ae["inflates"] else "No",
                            })
                        st.caption("Additional Expenses:")
                        st.dataframe(pd.DataFrame(_ae_disp), use_container_width=True, hide_index=True)
                    if _fi_active:
                        _fi_disp = []
                        for fi in _fi_active:
                            _eff_end = fi["end_age"] if fi["end_age"] > fi["start_age"] else "ongoing"
                            _fi_disp.append({
                                "Income": fi.get("name") or "Income", "Annual $": TEA.money(fi["amount"]),
                                "Ages": f"{fi['start_age']}-{_eff_end}",
                                "Taxable": "Yes" if fi["taxable"] else "No", "Inflates": "Yes" if fi["inflates"] else "No",
                            })
                        st.caption("Future Income:")
                        st.dataframe(pd.DataFrame(_fi_disp), use_container_width=True, hide_index=True)

                if st.button("Run Wealth Projection", type="primary", key="fp_run_proj"):
                    tab3_params = _build_tab3_params()
                    result = TEA.run_wealth_projection(a0, tab3_params, spending_order)
                    rows = result["year_details"]
                    st.session_state.tab3_rows = rows
                    st.session_state.tab3_params = {
                        "spending_goal": spending_goal, "inflation": inflation,
                        "years": int(years), "heir_tax_rate": heir_tax_rate,
                        "spending_order": " -> ".join(spending_order),
                        "r_cash": r_cash, "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
                        "r_annuity": r_annuity, "r_life": r_life,
                        "additional_expenses": [dict(ae) for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0],
                        "future_income": [dict(fi) for fi in st.session_state.future_income if fi.get("amount", 0) > 0],
                    }
                    _hide_cols = ["W/D Roth", "W/D Life", "Bal Life", "Total Wealth", "_net_draw"]
                    if all(r.get("Addl Expense", 0) == 0 for r in rows): _hide_cols.append("Addl Expense")
                    if all(r.get("Extra Income", 0) == 0 for r in rows): _hide_cols.append("Extra Income")
                    if all(r.get("Accel PT", 0) == 0 for r in rows): _hide_cols.append("Accel PT")
                    if all(r.get("Harvest Gains", 0) == 0 for r in rows): _hide_cols.append("Harvest Gains")
                    if all(r.get("Other Inc", 0) == 0 for r in rows): _hide_cols.append("Other Inc")
                    _df_display = pd.DataFrame(rows).drop(columns=[c for c in _hide_cols if c in rows[0]], errors="ignore")
                    st.dataframe(_df_display, use_container_width=True, hide_index=True)
                    # Check for portfolio depletion / spending shortfall
                    _depleted_years = [r for r in rows if r.get("Portfolio", 1) <= 0]
                    if _depleted_years:
                        _first_depl = _depleted_years[0]
                        st.error(f"Portfolio depleted at age {_first_depl.get('Age', '?')} "
                                 f"(year {_first_depl.get('Year', '?')}). "
                                 f"Living expenses cannot be fully funded for the remaining "
                                 f"{len(_depleted_years)} year(s) of the plan.")
                    if len(rows) > 0:
                        st.markdown("### Projection Summary")
                        _has_estate_tax = "Estate Tax" in rows[-1]
                        _n_cols = 6 if _has_estate_tax else 5
                        _cols = st.columns(_n_cols)
                        with _cols[0]: st.metric("Portfolio", f"${rows[-1]['Portfolio']:,.0f}")
                        with _cols[1]: st.metric("Gross Estate", f"${rows[-1]['Gross Estate']:,.0f}")
                        _ci = 2
                        if _has_estate_tax:
                            with _cols[_ci]: st.metric("Estate Tax", f"${rows[-1]['Estate Tax']:,.0f}")
                            _ci += 1
                        with _cols[_ci]: st.metric("Net Estate", f"${rows[-1]['Estate (Net)']:,.0f}")
                        with _cols[_ci+1]: st.metric("Total Taxes Paid", f"${sum(r['Taxes'] + r['Medicare'] for r in rows):,.0f}")
                        with _cols[_ci+2]: st.metric("Final Year Spending", f"${rows[-1]['Spending']:,.0f}")

                        # --- Monte Carlo Simulation ---
                        st.divider()
                        st.markdown("### Monte Carlo Analysis")
                        n_sims = int(mc_simulations); vol = mc_volatility; n_years = int(years)
                        mc_init_cash = a0["taxable"]["cash"]; mc_init_brok = a0["taxable"]["brokerage"]
                        mc_init_pf = float(pretax_bal_filer_current)
                        mc_init_ps = float(pretax_bal_spouse_current)
                        mc_init_roth = a0["taxfree"]["roth"]; mc_init_life = a0["taxfree"]["life_cash_value"]
                        mc_init_ann = a0["annuity"]["value"]
                        init_total = mc_init_cash + mc_init_brok + mc_init_pf + mc_init_ps + mc_init_roth + mc_init_life + mc_init_ann
                        det_spending = [r["Spending"] for r in rows]; det_ss = [r["SS"] for r in rows]; det_pension = [r["Pension"] for r in rows]
                        det_inv_inc = [r.get("Inv Inc", 0) for r in rows]
                        det_extra_inc = [r.get("Extra Income", 0) for r in rows]
                        det_eff_tax = []
                        for r in rows:
                            inc = r.get("Total Income", 0); tx = r.get("Taxes", 0) + r.get("Medicare", 0)
                            det_eff_tax.append(tx / inc if inc > 1 else 0.15)
                        rng = np.random.default_rng(42)
                        ending_portfolios = np.zeros(n_sims); ran_out_year = np.full(n_sims, n_years)
                        all_paths = np.zeros((n_sims, n_years + 1)); all_paths[:, 0] = init_total

                        for sim in range(n_sims):
                            s_cash = mc_init_cash; s_brok = mc_init_brok; s_pf = mc_init_pf; s_ps = mc_init_ps
                            s_roth = mc_init_roth; s_life = mc_init_life; s_ann = mc_init_ann
                            for yr_i in range(n_years):
                                _yi = min(yr_i, len(rows) - 1)
                                _spending = det_spending[_yi]; ss = det_ss[_yi]; pension = det_pension[_yi]; eff_tax = det_eff_tax[_yi]
                                inv_inc = det_inv_inc[_yi]; extra_inc = det_extra_inc[_yi]
                                _age_f = current_age_filer + yr_i
                                _age_s = (current_age_spouse + yr_i) if current_age_spouse else None
                                rmd_f = TEA.compute_rmd_uniform_start73(s_pf, _age_f)
                                rmd_s = TEA.compute_rmd_uniform_start73(s_ps, _age_s)
                                rmd = rmd_f + rmd_s; s_pf -= rmd_f; s_ps -= rmd_s
                                total_income = ss + pension + rmd + inv_inc + extra_inc
                                est_taxes = total_income * eff_tax
                                income_available = total_income
                                cash_needed = _spending + est_taxes; shortfall = cash_needed - income_available
                                if shortfall > 0:
                                    for bucket in spending_order:
                                        if shortfall <= 0: break
                                        if bucket == "Taxable":
                                            pull = min(shortfall, max(0.0, s_cash)); s_cash -= pull; shortfall -= pull
                                            if shortfall > 0:
                                                pull = min(shortfall, max(0.0, s_brok)); s_brok -= pull; shortfall -= pull
                                        elif bucket == "Pre-Tax":
                                            _avail = max(0.0, s_pf) + max(0.0, s_ps); pull = min(shortfall, _avail)
                                            if _avail > 0:
                                                pf_pull = pull * max(0.0, s_pf) / _avail; s_pf -= pf_pull; s_ps -= (pull - pf_pull)
                                            shortfall -= pull
                                        elif bucket == "Tax-Free":
                                            pull = min(shortfall, max(0.0, s_roth)); s_roth -= pull; shortfall -= pull
                                            if shortfall > 0:
                                                pull = min(shortfall, max(0.0, s_life)); s_life -= pull; shortfall -= pull
                                        elif bucket == "Tax-Deferred":
                                            pull = min(shortfall, max(0.0, s_ann)); s_ann -= pull; shortfall -= pull
                                else:
                                    s_cash += abs(shortfall)
                                yr_shock = rng.normal(0, vol)
                                s_cash = max(0.0, s_cash) * (1 + r_taxable + yr_shock)
                                s_brok = max(0.0, s_brok) * (1 + r_taxable + yr_shock)
                                s_pf = max(0.0, s_pf) * (1 + r_pretax + yr_shock)
                                s_ps = max(0.0, s_ps) * (1 + r_pretax + yr_shock)
                                s_roth = max(0.0, s_roth) * (1 + r_roth + yr_shock)
                                s_ann = max(0.0, s_ann) * (1 + r_annuity + yr_shock)
                                s_life = max(0.0, s_life) * (1 + r_life + yr_shock)
                                total_port = s_cash + s_brok + s_pf + s_ps + s_roth + s_ann + s_life
                                if total_port <= 0:
                                    ran_out_year[sim] = yr_i
                                    s_cash = s_brok = s_pf = s_ps = s_roth = s_ann = s_life = 0.0; total_port = 0.0
                                all_paths[sim, yr_i + 1] = total_port
                            ending_portfolios[sim] = s_cash + s_brok + s_pf + s_ps + s_roth + s_ann + s_life

                        success_count = np.sum(ending_portfolios > 0); success_rate = success_count / n_sims * 100
                        mc_data = {"success_rate": success_rate, "median": float(np.median(ending_portfolios)),
                                   "p10": float(np.percentile(ending_portfolios, 10)), "p90": float(np.percentile(ending_portfolios, 90))}
                        if success_rate < 100:
                            failed = ran_out_year[ran_out_year < n_years]
                            if len(failed) > 0: mc_data["median_fail_year"] = int(start_year) + int(np.median(failed))
                        st.session_state.tab3_mc = mc_data

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

                        years_axis = list(range(int(start_year), int(start_year) + n_years + 1))
                        p10 = np.percentile(all_paths, 10, axis=0); p25 = np.percentile(all_paths, 25, axis=0)
                        p50 = np.percentile(all_paths, 50, axis=0); p75 = np.percentile(all_paths, 75, axis=0)
                        p90 = np.percentile(all_paths, 90, axis=0)
                        det_line = [init_total] + [rows[i]["Portfolio"] for i in range(len(rows))]
                        while len(det_line) < n_years + 1: det_line.append(0)
                        chart_df = pd.DataFrame({
                            "Year": years_axis, "10th Pctl": p10, "25th Pctl": p25, "Median": p50,
                            "75th Pctl": p75, "90th Pctl": p90, "Deterministic": det_line[:n_years + 1],
                        }).set_index("Year")
                        st.line_chart(chart_df, use_container_width=True)
                        st.caption("Shaded bands: 10th\u201390th percentile range. Bold line: deterministic projection.")

        # ════════════════════════════════════════════════════════════════
        # TAB 4: Multigenerational Optimizer
        # ════════════════════════════════════════════════════════════════

        def _run_single_strategy(a0, params, strat, conv_strat="none",
                                  stop_age=0, target_agi=0, conv_years=0, agi_cap=False):
            """Run one spending+conversion combo through the engine. Returns engine result dict."""
            order_key = strat["key"]; wf_order = strat["wf"]
            is_blend = strat["blend"]; pt_cap = strat["pt_cap"]
            adaptive_key = strat["adaptive"]
            accel_bracket = strat.get("accel_bracket")
            ann_depl_yrs = strat.get("ann_depl_yrs"); ann_gains_only = strat.get("ann_gains_only", False)
            harvest_bracket = strat.get("harvest_bracket")
            _extra_kw = {}
            if accel_bracket is not None: _extra_kw["extra_pretax_bracket"] = accel_bracket
            if ann_depl_yrs is not None:
                _extra_kw["annuity_depletion_years"] = ann_depl_yrs
                if ann_gains_only: _extra_kw["annuity_gains_only"] = True
            if harvest_bracket is not None: _extra_kw["harvest_gains_bracket"] = harvest_bracket
            # Conversion parameters
            if conv_strat != "none":
                _extra_kw["conversion_strategy"] = conv_strat
                _extra_kw["conversion_years_limit"] = conv_years if conv_years > 0 else int(params["years"])
                _extra_kw["stop_conversion_age"] = stop_age if stop_age > 0 else 200
                if target_agi > 0: _extra_kw["target_agi"] = target_agi
                if agi_cap: _extra_kw["agi_cap_bracket_fills"] = True
            # Dispatch by strategy type
            if order_key == "adaptive":
                if conv_strat == "none":
                    # Let built-in adaptive conversions work (pass stop_age from user or default high)
                    return TEA.run_wealth_projection(a0, params, [], adaptive_strategy=adaptive_key,
                                                      stop_conversion_age=stop_age if stop_age > 0 else 200, **_extra_kw)
                else:
                    # Override: explicit conversion strategy replaces built-in
                    return TEA.run_wealth_projection(a0, params, [], adaptive_strategy=adaptive_key, **_extra_kw)
            elif order_key == "blend":
                return TEA.run_wealth_projection(a0, params, [], pretax_annual_cap=pt_cap, **_extra_kw)
            elif order_key == "prorata":
                return TEA.run_wealth_projection(a0, params, [], prorata_blend=True, **_extra_kw)
            elif order_key == "prorata_pt_heavy":
                return TEA.run_wealth_projection(a0, params, [], prorata_blend=True, prorata_weights={"pretax": 2.0, "roth": 0.25}, **_extra_kw)
            elif order_key == "prorata_no_roth":
                return TEA.run_wealth_projection(a0, params, [], prorata_blend=True, prorata_weights={"roth": 0.0, "life": 0.0}, **_extra_kw)
            elif is_blend:
                return TEA.run_wealth_projection(a0, params, wf_order, blend_mode=True, **_extra_kw)
            else:
                return TEA.run_wealth_projection(a0, params, wf_order, **_extra_kw)

        def _package_result(result, strat, spending_label, conv_label, scan_phase):
            """Build unified result dict from engine output."""
            _yd = result["year_details"]
            _pr_weights = None
            order_key = strat["key"]
            if order_key == "prorata_pt_heavy": _pr_weights = {"pretax": 2.0, "roth": 0.25}
            elif order_key == "prorata_no_roth": _pr_weights = {"roth": 0.0, "life": 0.0}
            _strat_info = {"type": order_key, "pt_cap": strat["pt_cap"], "blend": strat["blend"],
                           "wf": strat["wf"], "prorata": order_key.startswith("prorata"),
                           "prorata_weights": _pr_weights, "adaptive_key": strat["adaptive"],
                           "accel_bracket": strat.get("accel_bracket"),
                           "ann_depl_yrs": strat.get("ann_depl_yrs"),
                           "ann_gains_only": strat.get("ann_gains_only", False),
                           "conv_strat": strat.get("conv_strat", "none"),
                           "harvest_bracket": strat.get("harvest_bracket")}
            return {
                "spending_label": spending_label,
                "conversion_label": conv_label,
                "scan_phase": scan_phase,
                "after_tax_estate": result["after_tax_estate"],
                "gross_estate": result.get("gross_estate", result["total_wealth"]),
                "total_wealth": result["total_wealth"],
                "total_taxes": result["total_taxes"],
                "total_converted": result.get("total_converted", 0),
                "final_cash": result["final_cash"],
                "final_brokerage": result["final_brokerage"],
                "final_pretax": result["final_pretax"],
                "final_roth": result["final_roth"],
                "final_annuity": result["final_annuity"],
                "final_life": result["final_life"],
                "tot_wd_cash": sum(r.get("W/D Cash", 0) for r in _yd),
                "tot_wd_taxable": sum(r.get("W/D Taxable", 0) for r in _yd),
                "tot_wd_pretax": sum(r.get("W/D Pre-Tax", 0) for r in _yd),
                "tot_wd_roth": sum(r.get("W/D Roth", 0) + r.get("W/D Life", 0) for r in _yd),
                "tot_wd_annuity": sum(r.get("W/D Annuity", 0) for r in _yd),
                "_strat_info": _strat_info,
                "_year_details": _yd,
                # PDF backward compat
                "waterfall": spending_label,
                "order": strat.get("key", strat.get("type", "")),
            }

        with tab4:
            st.subheader("Multigenerational Optimizer")
            st.write("Jointly optimizes spending strategy and Roth conversions to maximize after-tax estate to heirs.")
            if st.session_state.assets is None:
                st.warning("Run Base Tax Estimator first.")
            else:
                a0 = st.session_state.assets
                st.caption("Uses the same assumptions (spending goal, start year, years, heir tax rate, growth rates) from the Wealth Projection tab.")
                _plan_age_desc = f"Filer {filer_plan_through_age}"
                if spouse_plan_through_age: _plan_age_desc += f" / Spouse {spouse_plan_through_age}"
                st.write(f"**Spending:** {TEA.money(spending_goal)}  |  **Start:** {int(start_year)}  |  **Plan Through Age:** {_plan_age_desc}  |  **Heir Tax:** {heir_tax_rate:.0%}")
                if is_joint:
                    st.caption(f"Survivor: {survivor_spending_pct}% spending, {pension_survivor_pct}% pension benefit after first death")

                st.divider()

                # ---- Controls ----
                col1, col2 = st.columns(2)
                with col1:
                    target_agi_input = st.number_input("Target AGI (conversion ceiling)", value=218000.0 if is_joint else 109000.0, step=10000.0, key="fp_opt_target",
                        help="Conversions won't push AGI above this. Controls bracket-fill and cap strategies.")
                with col2:
                    st.markdown("**Common Thresholds**")
                    if is_joint:
                        st.write("22%: $206,700 | 24%: $394,600")
                        st.write("IRMAA: $218k | $274k | $342k | $410k")
                    else:
                        st.write("22%: $103,350 | 24%: $197,300")
                        st.write("IRMAA: $109k | $137k | $171k | $205k")
                conv_amounts_str = st.text_input("Conversion amounts to test (comma separated)", value="25000, 50000, 75000, 100000, 150000, 200000", key="fp_opt_amounts")
                include_fill = st.checkbox("Also test 'Fill to Target AGI' strategy", value=True, key="fp_opt_fill")
                include_bracket_fill = st.checkbox("Also test bracket-fill strategies", value=True, key="fp_opt_bracket_fill")
                agi_cap_enabled = st.checkbox("Cap all strategies at Target AGI", value=True, key="fp_opt_agi_cap",
                    help="When checked, bracket-fill and fixed-amount strategies won't exceed the Target AGI")
                col_ds1, col_ds2 = st.columns(2)
                with col_ds1:
                    deep_top_k = st.slider("Deep Search: Top N spending strategies", min_value=3, max_value=20, value=10, key="fp_opt_top_k")
                with col_ds2:
                    test_adaptive_overrides = st.checkbox("Test adaptive strategies with override conversions", value=True, key="fp_opt_adaptive_overrides")

                if st.button("Run Optimizer", type="primary", key="fp_run_optimizer"):
                    params = _build_tab3_params()
                    from itertools import permutations

                    # ---- Build spending strategies ----
                    _spending_strategies = []
                    _blend_caps = [0, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 999999]
                    for cap in _blend_caps:
                        if cap == 0: label = "Brokerage First (Pre-Tax $0/yr)"
                        elif cap >= 999999: label = "Pre-Tax First (Unlimited)"
                        else: label = f"Blend: Pre-Tax ${cap:,.0f}/yr + Brokerage"
                        _spending_strategies.append({"key": "blend", "label": label, "wf": [], "blend": False, "pt_cap": cap, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    _spending_strategies.append({"key": "prorata", "label": "Pro-Rata: All Accounts (Equal Weight)", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    _spending_strategies.append({"key": "prorata_pt_heavy", "label": "Pro-Rata: Heavy Pre-Tax (2x), Light Roth (0.25x)", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    _spending_strategies.append({"key": "prorata_no_roth", "label": "Pro-Rata: All Except Roth", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    _spending_strategies.append({"key": "dynamic", "label": "Dynamic Blend (Marginal Cost)", "wf": [], "blend": True, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    _wf_buckets = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
                    _wf_bucket_bals = {
                        "Taxable": a0["taxable"]["cash"] + a0["taxable"]["brokerage"] + a0["taxable"]["emergency_fund"],
                        "Pre-Tax": a0["pretax"]["balance"],
                        "Tax-Free": a0["taxfree"]["roth"] + a0["taxfree"]["life_cash_value"],
                        "Tax-Deferred": a0["annuity"]["value"],
                    }
                    _seen_wf = set()
                    for _perm in permutations(_wf_buckets):
                        _wf_list = list(_perm)
                        if _wf_bucket_bals.get(_wf_list[0], 0) < 1: continue
                        _wf_key = " -> ".join(_wf_list)
                        if _wf_key not in _seen_wf:
                            _seen_wf.add(_wf_key)
                            _spending_strategies.append({"key": "wf", "label": f"WF: {_wf_key}", "wf": _wf_list, "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    for _ab_rate, _ab_label in [(0.12, "Accel PT: Fill 12% -> Brokerage"), (0.22, "Accel PT: Fill 22% -> Brokerage"), (0.24, "Accel PT: Fill 24% -> Brokerage"), ("irmaa", "Accel PT: Fill to IRMAA -> Brokerage")]:
                        _spending_strategies.append({"key": "blend", "label": _ab_label, "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": _ab_rate, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                    if a0["annuity"]["value"] > 0 and a0["annuity"]["value"] > a0["annuity"]["basis"]:
                        for _depl_yrs in [5, 10, 15]:
                            _spending_strategies.append({"key": "blend", "label": f"Draw Ann Gains {_depl_yrs}yr", "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": _depl_yrs, "ann_gains_only": True, "conv_strat": "none", "harvest_bracket": None})
                    for _hv_accel, _hv_label in [(None, "Harvest 0% LTCG Gains"), (None, "Brokerage First + Harvest 0% Gains"), (0.12, "Accel PT Fill 12% + Harvest 0% Gains"), (0.22, "Accel PT Fill 22% + Harvest 0% Gains")]:
                        _spending_strategies.append({"key": "blend", "label": _hv_label, "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": _hv_accel, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": 0.0})

                    # Adaptive strategies (separate list)
                    _adaptive_strategies = []
                    for _akey, _adef in TEA.ADAPTIVE_STRATEGIES.items():
                        _adaptive_strategies.append({"key": "adaptive", "label": _adef["label"], "wf": [], "blend": False, "pt_cap": None, "adaptive": _akey, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})

                    # ---- Build conversion strategies ----
                    try: conv_amounts = [float(x.strip()) for x in conv_amounts_str.split(",")]
                    except Exception: conv_amounts = [25000, 50000, 75000, 100000]
                    _conv_strategies = []
                    for amt in conv_amounts: _conv_strategies.append((amt, f"${amt:,.0f}/yr"))
                    if include_fill: _conv_strategies.append(("fill_to_target", f"Fill to ${target_agi_input:,.0f}"))
                    if include_bracket_fill:
                        _cap_tag = f" (cap ${target_agi_input/1000:.0f}k)" if agi_cap_enabled else ""
                        _conv_strategies += [("fill_bracket_12", f"Fill 12% Bracket{_cap_tag}"), ("fill_bracket_22", f"Fill 22% Bracket{_cap_tag}"), ("fill_bracket_24", f"Fill 24% Bracket{_cap_tag}"), ("fill_irmaa_0", "Fill to IRMAA Tier 1")]

                    # ---- Count total runs for progress ----
                    _n_quick = len(_spending_strategies) + len(_adaptive_strategies)
                    _n_deep_est = min(deep_top_k, len(_spending_strategies)) * len(_conv_strategies)
                    if test_adaptive_overrides: _n_deep_est += min(3, len(_adaptive_strategies)) * len(_conv_strategies)
                    _n_total = _n_quick + _n_deep_est

                    all_results = []; _run_count = 0
                    progress_bar = st.progress(0); status_text = st.empty()

                    # ════════════════════════════════════════════
                    # QUICK SCAN: non-adaptive with NO conversion + adaptive WITH built-in
                    # ════════════════════════════════════════════
                    status_text.text("Quick Scan: testing spending strategies...")

                    # Non-adaptive strategies — no conversions (stop_conversion_age=0)
                    for _strat in _spending_strategies:
                        status_text.text(f"Quick Scan ({_run_count + 1}/{_n_quick}): {_strat['label']}")
                        result = _run_single_strategy(a0, params, _strat, conv_strat="none", stop_age=0)
                        pkg = _package_result(result, _strat, _strat["label"], "No Conversion", "quick")
                        all_results.append(pkg)
                        _run_count += 1
                        progress_bar.progress(min(1.0, _run_count / _n_total * 0.4))

                    # Adaptive strategies — WITH their built-in conversions (the key fix!)
                    for _strat in _adaptive_strategies:
                        status_text.text(f"Quick Scan ({_run_count + 1}/{_n_quick}): {_strat['label']} (built-in conversions)")
                        result = _run_single_strategy(a0, params, _strat, conv_strat="none",
                                                       stop_age=200, target_agi=target_agi_input)
                        pkg = _package_result(result, _strat, _strat["label"], "Built-in", "quick")
                        all_results.append(pkg)
                        _run_count += 1
                        progress_bar.progress(min(1.0, _run_count / _n_total * 0.4))

                    # ---- Rank quick scan, select top K non-adaptive ----
                    _quick_non_adaptive = [r for r in all_results if r["scan_phase"] == "quick" and r["conversion_label"] == "No Conversion"]
                    _quick_non_adaptive.sort(key=lambda x: x["after_tax_estate"], reverse=True)
                    _top_k_spending = _quick_non_adaptive[:deep_top_k]
                    _top_k_labels = set(r["spending_label"] for r in _top_k_spending)
                    _top_k_strats = [s for s in _spending_strategies if s["label"] in _top_k_labels]

                    _quick_adaptive = [r for r in all_results if r["scan_phase"] == "quick" and r["conversion_label"] == "Built-in"]
                    _quick_adaptive.sort(key=lambda x: x["after_tax_estate"], reverse=True)
                    _top_adaptive_strats = []
                    if test_adaptive_overrides:
                        _top_adaptive_labels = set(r["spending_label"] for r in _quick_adaptive[:3])
                        _top_adaptive_strats = [s for s in _adaptive_strategies if s["label"] in _top_adaptive_labels]

                    # ════════════════════════════════════════════
                    # DEEP SEARCH: top K spending × all conversions + top adaptive × overrides
                    # ════════════════════════════════════════════
                    status_text.text("Deep Search: testing spending + conversion combos...")

                    for _strat in _top_k_strats:
                        for _conv_val, _conv_label in _conv_strategies:
                            status_text.text(f"Deep Search ({_run_count + 1 - _n_quick}/{_n_deep_est}): {_strat['label']} + {_conv_label}")
                            result = _run_single_strategy(a0, params, _strat, conv_strat=_conv_val,
                                                           stop_age=200, target_agi=target_agi_input,
                                                           conv_years=int(params["years"]), agi_cap=agi_cap_enabled)
                            pkg = _package_result(result, _strat, _strat["label"], _conv_label, "deep")
                            all_results.append(pkg)
                            _run_count += 1
                            _deep_pct = 0.4 + 0.6 * (_run_count - _n_quick) / max(1, _n_deep_est)
                            progress_bar.progress(min(1.0, _deep_pct))

                    # Adaptive overrides: top 3 adaptive × all explicit conversions
                    for _strat in _top_adaptive_strats:
                        for _conv_val, _conv_label in _conv_strategies:
                            status_text.text(f"Deep Search ({_run_count + 1 - _n_quick}/{_n_deep_est}): {_strat['label']} + {_conv_label}")
                            result = _run_single_strategy(a0, params, _strat, conv_strat=_conv_val,
                                                           stop_age=200, target_agi=target_agi_input,
                                                           conv_years=int(params["years"]), agi_cap=agi_cap_enabled)
                            pkg = _package_result(result, _strat, _strat["label"], _conv_label, "deep")
                            all_results.append(pkg)
                            _run_count += 1
                            _deep_pct = 0.4 + 0.6 * (_run_count - _n_quick) / max(1, _n_deep_est)
                            progress_bar.progress(min(1.0, _deep_pct))

                    progress_bar.progress(1.0)
                    progress_bar.empty(); status_text.empty()

                    # ════════════════════════════════════════════
                    # MERGE & RANK — dedup within $100
                    # ════════════════════════════════════════════
                    all_results.sort(key=lambda x: x["after_tax_estate"], reverse=True)
                    _deduped = []
                    for r in all_results:
                        _merged = False
                        for d in _deduped:
                            if d["spending_label"] == r["spending_label"] and abs(d["after_tax_estate"] - r["after_tax_estate"]) < 100:
                                if "_group_members" not in d: d["_group_members"] = [d["conversion_label"]]
                                d["_group_members"].append(r["conversion_label"]); d["_group_size"] = d.get("_group_size", 1) + 1; _merged = True; break
                        if not _merged: r["_group_members"] = [r["conversion_label"]]; r["_group_size"] = 1; _deduped.append(r)
                    all_results = _deduped

                    # Store in session state
                    _quick_results = [r for r in all_results if r["scan_phase"] == "quick"]
                    st.session_state.opt_quick_results = _quick_results
                    st.session_state.opt_deep_results = [r for r in all_results if r["scan_phase"] == "deep"]
                    st.session_state.opt_all_results = all_results
                    st.session_state.opt_best_combo = all_results[0] if all_results else None
                    st.session_state.opt_best_details = all_results[0]["_year_details"] if all_results else None
                    st.session_state.opt_params = params

                    # Baseline = best spending-only (no conversion)
                    _spending_only = [r for r in all_results if r["conversion_label"] == "No Conversion"]
                    _best_spending_only = _spending_only[0] if _spending_only else all_results[0]
                    st.session_state.opt_baseline_details = _best_spending_only["_year_details"]

                    # ---- Backward compat for PDF ----
                    _p1_compat = [r for r in all_results if r["conversion_label"] in ("No Conversion", "Built-in")]
                    st.session_state.phase1_results = _p1_compat
                    st.session_state.phase1_best_order = _best_spending_only["_strat_info"]
                    st.session_state.phase1_best_details = _best_spending_only["_year_details"]
                    st.session_state.phase1_all_details = {r["spending_label"]: r["_year_details"] for r in _p1_compat}
                    st.session_state.phase1_selected_strategy = None
                    st.session_state.phase1_params = params
                    _p2_compat = [r for r in all_results if r["conversion_label"] not in ("No Conversion", "Built-in")]
                    _global_best = all_results[0]
                    # Map deep results into phase2 format for PDF
                    _p2_mapped = []
                    _baseline_estate = _best_spending_only["after_tax_estate"]
                    for r in _p2_compat:
                        _p2_mapped.append({
                            "strategy_name": f"{r['spending_label']} + {r['conversion_label']}",
                            "after_tax_estate": r["after_tax_estate"],
                            "improvement": r["after_tax_estate"] - _baseline_estate,
                            "total_wealth": r["total_wealth"],
                            "gross_estate": r["gross_estate"],
                            "total_taxes": r["total_taxes"],
                            "total_converted": r["total_converted"],
                            "final_pretax": r["final_pretax"],
                            "final_roth": r["final_roth"],
                            "final_brokerage": r["final_cash"] + r["final_brokerage"],
                            "_year_details": r["_year_details"],
                        })
                    st.session_state.phase2_results = _p2_mapped if _p2_mapped else None
                    st.session_state.phase2_best_details = _global_best["_year_details"]
                    st.session_state.phase2_baseline_details = _best_spending_only["_year_details"]
                    st.session_state.phase2_best_name = f"{_global_best['spending_label']} + {_global_best['conversion_label']}"

                    st.success(f"Tested {_run_count} combinations ({_n_quick} quick scan + {_run_count - _n_quick} deep search)")

                # ════════════════════════════════════════════
                # RESULTS DISPLAY
                # ════════════════════════════════════════════
                if st.session_state.opt_all_results:
                    _all = st.session_state.opt_all_results
                    _best = _all[0]
                    _spending_only = [r for r in _all if r["conversion_label"] == "No Conversion"]
                    _best_spending_only = _spending_only[0] if _spending_only else _best
                    _baseline_estate = _best_spending_only["after_tax_estate"]

                    # ---- Best Combo highlight ----
                    _best_label = f"{_best['spending_label']} + {_best['conversion_label']}"
                    _improvement = _best["after_tax_estate"] - _baseline_estate
                    if _improvement > 100:
                        st.success(f"**Best:** {_best_label} — Estate: **${_best['after_tax_estate']:,.0f}** (+${_improvement:,.0f} vs spending-only)")
                    else:
                        st.success(f"**Best:** {_best_label} — Estate: **${_best['after_tax_estate']:,.0f}**")

                    # ---- Summary metrics ----
                    _has_et = _best.get("estate_tax", 0) > 0
                    _m_cols = st.columns(4)
                    with _m_cols[0]: st.metric("Gross Estate", f"${_best['gross_estate']:,.0f}")
                    with _m_cols[1]: st.metric("Net Estate (After All Tax)", f"${_best['after_tax_estate']:,.0f}")
                    with _m_cols[2]: st.metric("Total Taxes + Medicare", f"${_best['total_taxes']:,.0f}")
                    with _m_cols[3]: st.metric("Total Converted", f"${_best['total_converted']:,.0f}")

                    # ---- Spending-Only Rankings ----
                    _quick_display = [r for r in _all if r["conversion_label"] in ("No Conversion", "Built-in")]
                    if _quick_display:
                        with st.expander("Spending-Only Rankings (Quick Scan)", expanded=False):
                            _comp_rows = []
                            for r in _quick_display:
                                _type_tag = r["order"]
                                if _type_tag == "adaptive": _cat = "Adaptive"
                                elif _type_tag.startswith("prorata"): _cat = "Pro-Rata"
                                elif _type_tag == "blend": _cat = "Blend"
                                elif _type_tag == "dynamic": _cat = "Dynamic"
                                else: _cat = "Waterfall"
                                _row = {"Type": _cat, "Strategy": r["spending_label"], "Conversion": r["conversion_label"],
                                    "Net Estate": f"${r['after_tax_estate']:,.0f}", "Gross Estate": f"${r['gross_estate']:,.0f}",
                                    "Total Taxes": f"${r['total_taxes']:,.0f}", "Converted": f"${r['total_converted']:,.0f}",
                                    "Drew Cash": f"${r.get('tot_wd_cash', 0):,.0f}", "Drew Taxable": f"${r.get('tot_wd_taxable', 0):,.0f}",
                                    "Drew Pre-Tax": f"${r.get('tot_wd_pretax', 0):,.0f}", "Drew Tax-Free": f"${r.get('tot_wd_roth', 0):,.0f}",
                                    "Drew Annuity": f"${r.get('tot_wd_annuity', 0):,.0f}",
                                    "Final Pre-Tax": f"${r['final_pretax']:,.0f}", "Final Roth": f"${r['final_roth']:,.0f}",
                                    "Final Brokerage": f"${r['final_brokerage']:,.0f}", "Final Cash": f"${r.get('final_cash', 0):,.0f}",
                                    "Final Annuity": f"${r.get('final_annuity', 0):,.0f}", "Final Life": f"${r.get('final_life', 0):,.0f}",
                                    "vs Best": f"${r['after_tax_estate'] - _quick_display[0]['after_tax_estate']:+,.0f}"}
                                _gs = r.get("_group_size", 1)
                                _row["# Tied"] = str(_gs) if _gs > 1 else ""
                                _comp_rows.append(_row)
                            _comp_df = pd.DataFrame(_comp_rows)
                            # Drop zero columns
                            for _bcol, _bkey in [("Drew Cash", "tot_wd_cash"), ("Drew Taxable", "tot_wd_taxable"), ("Drew Pre-Tax", "tot_wd_pretax"), ("Drew Tax-Free", "tot_wd_roth"), ("Drew Annuity", "tot_wd_annuity"), ("Final Cash", "final_cash"), ("Final Annuity", "final_annuity"), ("Final Life", "final_life"), ("Final Pre-Tax", "final_pretax"), ("Final Roth", "final_roth"), ("Final Brokerage", "final_brokerage"), ("Converted", "total_converted")]:
                                if _bcol in _comp_df.columns and all(r.get(_bkey, 0) < 1 for r in _quick_display): _comp_df = _comp_df.drop(columns=[_bcol])
                            st.dataframe(_comp_df, use_container_width=True, hide_index=True)

                    # ---- All Tested Combos ----
                    with st.expander("All Tested Combinations", expanded=True):
                        _combo_rows = []
                        for r in _all:
                            _row = {"Spending Strategy": r["spending_label"], "Conversion": r["conversion_label"],
                                "Net Estate": f"${r['after_tax_estate']:,.0f}", "Total Taxes": f"${r['total_taxes']:,.0f}",
                                "Converted": f"${r['total_converted']:,.0f}",
                                "vs Best": f"${r['after_tax_estate'] - _best['after_tax_estate']:+,.0f}"}
                            _gs = r.get("_group_size", 1)
                            _row["# Tied"] = str(_gs) if _gs > 1 else ""
                            _combo_rows.append(_row)
                        st.dataframe(pd.DataFrame(_combo_rows), use_container_width=True, hide_index=True)

                    # ---- Combo selector + year-by-year detail ----
                    _combo_labels = [f"{r['spending_label']} + {r['conversion_label']}" for r in _all]
                    _selected_combo_label = st.selectbox("Select combination to view detail", _combo_labels, index=0, key="fp_opt_combo_select")
                    _selected_idx = _combo_labels.index(_selected_combo_label)
                    _selected_combo = _all[_selected_idx]
                    st.session_state.opt_selected_combo = _selected_combo

                    if _selected_combo_label != f"{_best['spending_label']} + {_best['conversion_label']}":
                        col1b, col2b, col3b = st.columns(3)
                        with col1b: st.metric("Selected Gross Estate", f"${_selected_combo['gross_estate']:,.0f}")
                        with col2b: st.metric("Selected Net Estate", f"${_selected_combo['after_tax_estate']:,.0f}")
                        with col3b: st.metric("vs Best", f"${_selected_combo['after_tax_estate'] - _best['after_tax_estate']:+,.0f}")

                    _opt_hide = ["W/D Roth", "W/D Life", "Bal Life", "Total Wealth", "_net_draw"]
                    _zero_hide = ["Accel PT", "Harvest Gains"]
                    _sel_details = _selected_combo.get("_year_details", [])
                    if _sel_details:
                        with st.expander(f"Year-by-Year Detail: {_selected_combo_label}", expanded=True):
                            _df_det = pd.DataFrame(_sel_details)
                            _df_det = _df_det.drop(columns=[c for c in _opt_hide if c in _df_det.columns], errors="ignore")
                            for _zc in _zero_hide:
                                if _zc in _df_det.columns and (_df_det[_zc] == 0).all(): _df_det = _df_det.drop(columns=[_zc])
                            st.dataframe(_df_det, use_container_width=True, hide_index=True)

                    # ---- Side-by-side: Best combo vs spending-only baseline ----
                    _bl_details = _best_spending_only.get("_year_details", [])
                    if _bl_details and _sel_details:
                        st.divider()
                        yr1_bl = _bl_details[0] if _bl_details else {}; yr1_bc = _sel_details[0] if _sel_details else {}
                        st.markdown(f"### Year 1 Cash Flow: No Conversion vs {_selected_combo_label}")
                        col1, col2 = st.columns(2)
                        for col, yr, label in [(col1, yr1_bl, f"{_best_spending_only['spending_label']} (No Conv)"), (col2, yr1_bc, _selected_combo_label)]:
                            with col:
                                st.markdown(f"#### {label}")
                                st.markdown("**Income (Cash Received)**")
                                income_rows = [("Social Security", yr.get("SS", 0)), ("Pensions", yr.get("Pension", 0)), ("RMD", yr.get("RMD", 0)), ("Investment Income", yr.get("Inv Inc", 0)), ("W/D Cash", yr.get("W/D Cash", 0)), ("W/D Taxable", yr.get("W/D Taxable", 0)), ("W/D Pre-Tax", yr.get("W/D Pre-Tax", 0)), ("Roth Conversion", yr.get("Conversion", 0)), ("W/D Roth", yr.get("W/D Roth", 0)), ("W/D Life Ins", yr.get("W/D Life", 0)), ("W/D Annuity", yr.get("W/D Annuity", 0))]
                                income_rows = [(k, v) for k, v in income_rows if v > 0]
                                total_inc = sum(v for _, v in income_rows)
                                st.dataframe([{"Income": k, "Amount": TEA.money(v)} for k, v in income_rows], use_container_width=True, hide_index=True)
                                st.metric("Total Income", TEA.money(total_inc))
                                st.markdown("**Expenses & Outflows**")
                                out_rows = [("Living Expenses", yr.get("Spending", 0)), ("Taxes", yr.get("Taxes", 0)), ("Medicare", yr.get("Medicare", 0))]
                                out_rows = [(k, v) for k, v in out_rows if v > 0]
                                total_out = sum(v for _, v in out_rows)
                                st.dataframe([{"Expense": k, "Amount": TEA.money(v)} for k, v in out_rows], use_container_width=True, hide_index=True)
                                st.metric("Total Outflows", TEA.money(total_out))

                        st.divider()
                        st.markdown("### End of Projection Comparison")
                        col1, col2 = st.columns(2)
                        final_bl = _bl_details[-1] if _bl_details else {}; final_bc = _sel_details[-1] if _sel_details else {}
                        with col1:
                            st.markdown(f"#### {_best_spending_only['spending_label']} (No Conv)")
                            st.metric("Final Pre-Tax", TEA.money(final_bl.get("Bal Pre-Tax", 0)))
                            st.metric("Final Roth", TEA.money(final_bl.get("Bal Roth", 0)))
                            st.metric("Final Taxable", TEA.money(final_bl.get("Bal Cash", 0) + final_bl.get("Bal Taxable", 0)))
                            st.metric("After-Tax Estate", TEA.money(final_bl.get("Estate (Net)", 0)))
                        with col2:
                            st.markdown(f"#### {_selected_combo_label}")
                            st.metric("Final Pre-Tax", TEA.money(final_bc.get("Bal Pre-Tax", 0)))
                            st.metric("Final Roth", TEA.money(final_bc.get("Bal Roth", 0)))
                            st.metric("Final Taxable", TEA.money(final_bc.get("Bal Cash", 0) + final_bc.get("Bal Taxable", 0)))
                            st.metric("After-Tax Estate", TEA.money(final_bc.get("Estate (Net)", 0)))

    # ════════════════════════════════════════════════════════════════
    # TAB 5: Roth Conversion Opportunity
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Roth Conversion Opportunity")
        st.write("Calculate how much you can convert to fill up to a target income level based on your current tax scenario.")
        # For pre-retirement users, fall back to base results; for retired, use Income Needs
        _t5_has_solved = st.session_state.last_solved_results is not None and st.session_state.last_solved_inputs is not None
        if not _t5_has_solved and st.session_state.base_results is None:
            st.warning("Run Base Tax Estimator first.")
        else:
            if _t5_has_solved:
                solved_res = st.session_state.last_solved_results; solved_inp = st.session_state.last_solved_inputs
                net_needed_val = float(st.session_state.last_net_needed or 0.0); source_used = st.session_state.last_source or "Unknown"
            else:
                solved_res = st.session_state.base_results; solved_inp = st.session_state.base_inputs
                net_needed_val = 0.0; source_used = "Base Scenario"
            st.markdown("### Current Scenario")
            if _t5_has_solved:
                st.write(f"**Funding Source:** {source_used}"); st.write(f"**Net Income Needed:** {TEA.money(net_needed_val)}")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Current AGI", TEA.money(solved_res["agi"]))
            with col2: st.metric("Current Federal Tax", TEA.money(solved_res["fed_tax"]))
            with col3: st.metric("Current Total Tax", TEA.money(solved_res["total_tax"]))
            st.divider()
            st.markdown("### Common Target Thresholds")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tax Bracket Tops (Taxable Income)**")
                if is_joint: st.write("12% bracket: $96,950"); st.write("22% bracket: $206,700"); st.write("24% bracket: $394,600")
                else: st.write("12% bracket: $48,475"); st.write("22% bracket: $103,350"); st.write("24% bracket: $197,300")
            with col2:
                st.markdown("**IRMAA Thresholds (AGI)**")
                if is_joint: st.write("No IRMAA: $218,000"); st.write("Tier 1: $274,000"); st.write("Tier 2: $342,000"); st.write("Tier 3: $410,000")
                else: st.write("No IRMAA: $109,000"); st.write("Tier 1: $137,000"); st.write("Tier 2: $171,000"); st.write("Tier 3: $205,000")
            st.divider()
            target_type = st.radio("Target Type", ["AGI Target", "Taxable Income Target"], horizontal=True, key="fp_tab5_target_type")
            if target_type == "AGI Target":
                default_target = 218000.0 if is_joint else 109000.0
                target_amount = st.number_input("Target AGI", value=default_target, step=1000.0, key="fp_tab5_target_amt"); current_level = solved_res["agi"]
            else:
                default_target = 206700.0 if is_joint else 103350.0
                target_amount = st.number_input("Target Taxable Income", value=default_target, step=1000.0, key="fp_tab5_target_amt"); current_level = solved_res["fed_taxable"]
            conversion_room = max(0.0, target_amount - current_level)
            # Only IRA balances are available for conversion (401k generally not convertible while employed)
            available_ira = D["curr_trad_ira_f"] + (D["curr_trad_ira_s"] if is_joint else 0.0)
            actual_conversion = min(conversion_room, available_ira)
            st.caption(f"Available IRA balance: **{TEA.money(available_ira)}** | Conversion room: **{TEA.money(conversion_room)}** | Actual conversion: **{TEA.money(actual_conversion)}**")
            if actual_conversion < conversion_room and available_ira < conversion_room:
                st.warning(f"Conversion limited to {TEA.money(available_ira)} (available IRA balance — 401(k) not convertible while employed).")
            if st.button("Calculate Conversion Opportunity", type="primary", key="fp_tab5_calc"):
                if conversion_room <= 0:
                    st.session_state.tab5_conv_res = None
                    st.warning(f"You are already at or above the target. Current: {TEA.money(current_level)}, Target: {TEA.money(target_amount)}")
                else:
                    conv_inputs = dict(solved_inp); conv_inputs["taxable_ira"] = float(conv_inputs.get("taxable_ira", 0.0)) + actual_conversion
                    conv_res = TEA.compute_case(conv_inputs)
                    additional_tax = conv_res["total_tax"] - solved_res["total_tax"]
                    additional_medicare = conv_res["medicare_premiums"] - solved_res["medicare_premiums"]
                    total_additional_cost = additional_tax + additional_medicare
                    st.session_state.tab5_conv_res = conv_res; st.session_state.tab5_conv_inputs = conv_inputs
                    st.session_state.tab5_actual_conversion = actual_conversion
                    st.session_state.tab5_conversion_room = conversion_room
                    st.session_state.tab5_total_additional_cost = total_additional_cost

            if st.session_state.tab5_conv_res is not None:
                conv_res = st.session_state.tab5_conv_res; conv_inputs = st.session_state.tab5_conv_inputs
                actual_conversion = st.session_state.tab5_actual_conversion
                conversion_room = st.session_state.tab5_conversion_room
                total_additional_cost = st.session_state.tab5_total_additional_cost
                additional_tax = conv_res["total_tax"] - solved_res["total_tax"]
                net_to_roth = actual_conversion - total_additional_cost
                effective_rate = (total_additional_cost / actual_conversion * 100) if actual_conversion > 0 else 0
                st.markdown("### Conversion Opportunity")
                st.success(f"**Roth Conversion Room: {TEA.money(conversion_room)}** &nbsp;&nbsp;|&nbsp;&nbsp; **Additional Tax: {TEA.money(total_additional_cost)}** &nbsp;&nbsp;|&nbsp;&nbsp; **Net to Roth: {TEA.money(net_to_roth)}**")
                if actual_conversion < conversion_room: st.info(f"Note: Only {TEA.money(available_ira)} available in IRA accounts (401(k) balances not convertible while employed).")
                st.markdown("### Tax Impact Analysis")
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Conversion Amount", TEA.money(actual_conversion)); st.metric("Net to Roth", TEA.money(net_to_roth))
                with col2: st.metric("Tax Before", TEA.money(solved_res["total_tax"])); st.metric("Tax After", TEA.money(conv_res["total_tax"]))
                with col3: st.metric("Additional Tax", TEA.money(additional_tax)); st.metric("Effective Tax Rate", f"{effective_rate:.1f}%")
                st.markdown("### IRMAA Status")
                col1, col2 = st.columns(2)
                with col1: st.write(f"**Before Conversion:** {'IRMAA applies' if solved_res['has_irmaa'] else 'No IRMAA'}"); st.write(f"Medicare Premiums: {TEA.money(solved_res['medicare_premiums'])}")
                with col2: st.write(f"**After Conversion:** {'IRMAA applies' if conv_res['has_irmaa'] else 'No IRMAA'}"); st.write(f"Medicare Premiums: {TEA.money(conv_res['medicare_premiums'])}")
                if conv_res["has_irmaa"] and not solved_res["has_irmaa"]:
                    irmaa_cost = conv_res["medicare_premiums"] - solved_res["medicare_premiums"]
                    st.warning(f"Warning: This conversion would trigger IRMAA, adding {TEA.money(irmaa_cost)} in Medicare premiums!")
                st.markdown("### Detailed Comparison")
                comparison_data = [
                    {"Metric": "AGI", "Before": TEA.money(solved_res["agi"]), "After": TEA.money(conv_res["agi"])},
                    {"Metric": "Federal Taxable", "Before": TEA.money(solved_res["fed_taxable"]), "After": TEA.money(conv_res["fed_taxable"])},
                    {"Metric": "Federal Tax", "Before": TEA.money(solved_res["fed_tax"]), "After": TEA.money(conv_res["fed_tax"])},
                    {"Metric": "SC Tax", "Before": TEA.money(solved_res["sc_tax"]), "After": TEA.money(conv_res["sc_tax"])},
                    {"Metric": "Total Tax", "Before": TEA.money(solved_res["total_tax"]), "After": TEA.money(conv_res["total_tax"])},
                    {"Metric": "Medicare Premiums", "Before": TEA.money(solved_res["medicare_premiums"]), "After": TEA.money(conv_res["medicare_premiums"])},
                ]
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                st.divider(); st.markdown("### Cash Flow Comparison")
                TEA.display_cashflow_comparison(solved_inp, solved_res, conv_inputs, conv_res,
                    net_needed=net_needed_val, roth_conversion=actual_conversion,
                    mortgage_payment=float(mortgage_payment),
                    title_before="Before Conversion", title_after="After Conversion")

                # --- Estate Impact ---
                st.divider()
                st.markdown("### Estate Impact \u2014 Before & After Conversion")
                st.caption("Uses the full wealth projection engine (spending, RMDs, taxes, withdrawals) \u2014 same as Tab 3.")
                _conv_years = st.number_input("Years of Conversion", min_value=1, max_value=30, value=1, step=1, key="fp_conv_years_tab5")
                _ei_params = _build_tab3_params()  # uses client's actual heir_tax_rate, spending, etc.
                _ei_ran = False

                if _is_working:
                    # ── Working client: project to retirement, then model retirement phase ──
                    _accum = st.session_state.get("projection_results")
                    if _accum is None:
                        st.info("Run the **Projection** tab first to see estate impact analysis.")
                    else:
                        _final = _accum["rows"][-1] if _accum.get("rows") else {}
                        _ret_brok = _accum.get("final_brokerage", _final.get("Bal Taxable", 0))
                        _ret_cash = _accum.get("final_cash", 0)
                        _ret_pretax = _final.get("Bal Pre-Tax", 0)
                        _ret_roth = _final.get("Bal Roth", 0)
                        _ret_basis = _accum.get("final_basis", 0)
                        _roth_split = roth_bal_filer / max(roth_bal, 1) if roth_bal > 0 else 1.0
                        _ann_at_ret = annuity_value * ((1 + r_annuity) ** years_to_ret) if annuity_value > 0 else 0.0
                        _life_at_ret = life_cash_value * ((1 + r_life) ** years_to_ret) if life_cash_value > 0 else 0.0
                        _ann_split = annuity_value_filer / max(annuity_value, 1) if annuity_value > 0 else 1.0
                        # Build retirement-start assets in TEA format
                        _ei_a0 = {
                            "taxable": {"cash": _ret_cash, "brokerage": _ret_brok, "emergency_fund": float(emergency_fund)},
                            "pretax": {"balance": _ret_pretax},
                            "taxfree": {
                                "roth": _ret_roth, "roth_filer": _ret_roth * _roth_split,
                                "roth_spouse": _ret_roth * (1 - _roth_split),
                                "life_cash_value": _life_at_ret,
                            },
                            "annuity": {
                                "value": _ann_at_ret, "basis": float(annuity_basis),
                                "value_filer": _ann_at_ret * _ann_split,
                                "basis_filer": float(min(annuity_basis_filer, _ann_at_ret * _ann_split)),
                                "value_spouse": _ann_at_ret * (1 - _ann_split),
                                "basis_spouse": float(min(annuity_basis_spouse, _ann_at_ret * (1 - _ann_split))),
                            },
                        }
                        # Override params for retirement phase (no salary, correct horizon)
                        _inf_factor = (1 + inflation) ** years_to_ret
                        _ei_params["wages"] = 0.0
                        _ei_params["adjustments"] = 0.0
                        _ret_yrs = max(1, filer_plan_through_age - ret_age)
                        if is_joint and current_age_spouse:
                            _spouse_ret_age = current_age_spouse + years_to_ret
                            _ret_yrs = max(_ret_yrs, spouse_plan_through_age - _spouse_ret_age)
                        _ei_params["years"] = _ret_yrs
                        _ei_params["start_year"] = current_year + years_to_ret
                        _ei_params["spending_goal"] = D["living_expenses"] * (D["ret_pct"] / 100) * _inf_factor
                        _ei_params["current_age_filer"] = ret_age
                        _ei_params["current_age_spouse"] = (current_age_spouse + years_to_ret) if current_age_spouse else None
                        _ei_params["home_value"] = home_value * ((1 + home_appreciation) ** years_to_ret)
                        _ei_params["brokerage_gain_pct"] = max(0.0, 1.0 - _ret_basis / _ret_brok) if _ret_brok > 0 else 0.0
                        _ei_params["pretax_filer_ratio"] = pretax_bal_filer_current / max(pretax_bal, 1) if pretax_bal > 0 else 1.0
                        # Mortgage at retirement
                        _mtg_yrs_left = max(0, mtg_years - years_to_ret)
                        if _mtg_yrs_left <= 0:
                            _ei_params["mortgage_balance"] = 0.0
                            _ei_params["mortgage_payment"] = 0.0
                        # Scale investment income to projected brokerage size
                        _cur_brok = float(taxable_brokerage_bal)
                        _scale = _ret_brok / _cur_brok if _cur_brok > 0 and _ret_brok > 0 else 1.0
                        _ei_params["interest_taxable"] = float(proj_interest) * _scale
                        _ei_params["total_ordinary_dividends"] = float(proj_dividends) * _scale
                        _ei_params["qualified_dividends"] = float(proj_qual_div) * _scale
                        _ei_params["cap_gain_loss"] = float(proj_cap_gains) * _scale
                        # Conversion happens during working years — compute compounded impact at retirement
                        _nc_pt_loss = 0.0; _wc_roth_gain = 0.0; _wc_brok_loss = 0.0
                        _eff_conv_yrs = min(_conv_years, years_to_ret)
                        for _ci in range(_eff_conv_yrs):
                            _grow = max(0, years_to_ret - _ci)
                            _nc_pt_loss += actual_conversion * ((1 + r_pretax) ** _grow)
                            _wc_roth_gain += (actual_conversion - total_additional_cost) * ((1 + r_roth) ** _grow)
                            _wc_brok_loss += total_additional_cost * ((1 + r_taxable) ** _grow)
                        _total_converted = actual_conversion * _eff_conv_yrs
                        # No-conversion run: normal retirement projection from projected assets
                        _nc_result = TEA.run_wealth_projection(_ei_a0, _ei_params, spending_order, conversion_strategy="none")
                        # With-conversion run: adjust starting assets for the pre-retirement conversion
                        _wc_a0 = copy.deepcopy(_ei_a0)
                        _wc_a0["pretax"]["balance"] = max(0, _ret_pretax - _nc_pt_loss)
                        _new_roth = _ret_roth + _wc_roth_gain
                        _wc_a0["taxfree"]["roth"] = _new_roth
                        _wc_a0["taxfree"]["roth_filer"] = _new_roth * _roth_split
                        _wc_a0["taxfree"]["roth_spouse"] = _new_roth * (1 - _roth_split)
                        _new_brok = max(0, _ret_brok - _wc_brok_loss)
                        _wc_a0["taxable"]["brokerage"] = _new_brok
                        _wc_result = TEA.run_wealth_projection(_wc_a0, _ei_params, spending_order, conversion_strategy="none")
                        _ei_ran = True
                else:
                    # ── Retired client: conversion happens within the projection ──
                    if _t5_has_solved and st.session_state.last_solved_assets:
                        _ei_a0 = st.session_state.last_solved_assets
                    else:
                        _ei_a0 = st.session_state.assets
                    _ei_stop_age = current_age_filer + _conv_years
                    _total_converted = actual_conversion * _conv_years
                    _nc_result = TEA.run_wealth_projection(_ei_a0, _ei_params, spending_order, conversion_strategy="none")
                    _wc_result = TEA.run_wealth_projection(_ei_a0, _ei_params, spending_order, conversion_strategy=float(actual_conversion), conversion_years_limit=_conv_years, stop_conversion_age=_ei_stop_age)
                    _total_converted = _wc_result.get("total_converted", _total_converted)
                    _ei_ran = True

                if _ei_ran:
                    _nc_net = _nc_result["after_tax_estate"]; _wc_net = _wc_result["after_tax_estate"]
                    _nc_gross = _nc_result.get("gross_estate", _nc_result["total_wealth"]); _wc_gross = _wc_result.get("gross_estate", _wc_result["total_wealth"])
                    _nc_taxes = _nc_result["total_taxes"]; _wc_taxes = _wc_result["total_taxes"]
                    _nc_pretax = _nc_result["final_pretax"]; _wc_pretax = _wc_result["final_pretax"]
                    _nc_roth = _nc_result["final_roth"]; _wc_roth = _wc_result["final_roth"]
                    _nc_brok = _nc_result["final_cash"] + _nc_result["final_brokerage"]; _wc_brok = _wc_result["final_cash"] + _wc_result["final_brokerage"]
                    _yr_label = "Year" if _conv_years == 1 else f"{_conv_years} Years"
                    _proj_label = f"{int(_ei_params['years'])}yr Projection"
                    if _is_working:
                        st.caption(f"Retirement-phase projection ({_proj_label} from age {ret_age}). Conversion impact compounded through accumulation.")
                    _estate_table = f"| Metric | No Conversion | With Conversion ({_yr_label}) | Change |\n|:---|-------:|-------:|-------:|\n"
                    _estate_table += f"| **Net Estate** | **{TEA.money(_nc_net)}** | **{TEA.money(_wc_net)}** | **{TEA.money(_wc_net - _nc_net)}** |\n"
                    _estate_table += f"| Total Taxes Paid | {TEA.money(_nc_taxes)} | {TEA.money(_wc_taxes)} | {TEA.money(_wc_taxes - _nc_taxes)} |\n"
                    _estate_table += f"| Final Pre-Tax | {TEA.money(_nc_pretax)} | {TEA.money(_wc_pretax)} | {TEA.money(_wc_pretax - _nc_pretax)} |\n"
                    _estate_table += f"| Final Roth | {TEA.money(_nc_roth)} | {TEA.money(_wc_roth)} | {TEA.money(_wc_roth - _nc_roth)} |\n"
                    _estate_table += f"| Final Brokerage | {TEA.money(_nc_brok)} | {TEA.money(_wc_brok)} | {TEA.money(_wc_brok - _nc_brok)} |\n"
                    _estate_table += f"| **Portfolio** | **{TEA.money(_nc_gross)}** | **{TEA.money(_wc_gross)}** | **{TEA.money(_wc_gross - _nc_gross)}** |\n"
                    st.markdown(_estate_table)
                    st.caption(f"Full {_proj_label} with spending, RMDs, taxes, and withdrawals. Conversion: {TEA.money(actual_conversion)}/yr for {_conv_years} yr. Total converted: {TEA.money(_total_converted)}.")
                    with st.expander("Year-by-Year Projection Detail (With Conversion)"):
                        _wc_details = _wc_result["year_details"]
                        _detail_cols = ["Year", "Age", "Spending", "SS", "Pension", "RMD", "Roth Conv", "Fed Tax", "Total Tax", "Taxable", "Pre-Tax", "Roth", "Net Estate"]
                        _detail_rows = []
                        for _d in _wc_details:
                            _row = {}
                            for _col in _detail_cols:
                                if _col in _d: _row[_col] = _d[_col]
                            _detail_rows.append(_row)
                        if _detail_rows: st.dataframe(pd.DataFrame(_detail_rows), use_container_width=True, hide_index=True)
                    _net_benefit = _wc_net - _nc_net
                    _heir_pct = _ei_params.get("heir_tax_rate", 0.25)
                    if _net_benefit > 0: st.success(f"Converting {TEA.money(actual_conversion)}/yr for {_conv_years} year(s) improves the net estate by **{TEA.money(_net_benefit)}** (heirs @ {_heir_pct:.0%}).")
                    elif _net_benefit < 0: st.warning(f"Converting {TEA.money(actual_conversion)}/yr for {_conv_years} year(s) reduces the net estate by **{TEA.money(abs(_net_benefit))}** (heirs @ {_heir_pct:.0%}).")
                    else: st.info("The conversion strategy is estate-neutral at this heir tax rate.")

# ══════════════════════════════════════════════════════════════════════
# PDF DOWNLOAD
# ══════════════════════════════════════════════════════════════════════
st.divider()
_has_any_data = any([
    st.session_state.get("base_results"),
    st.session_state.get("last_solved_results"),
    st.session_state.get("tab3_rows"),
    st.session_state.get("phase1_results"),
    st.session_state.get("tab5_conv_res"),
])
if _has_any_data:
    _pdf_bytes = TEA.generate_pdf_report()
    st.download_button(
        label="Download PDF Report",
        data=_pdf_bytes,
        file_name=f"OptiPlan_Report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.caption("Run at least one analysis tab to generate a PDF report.")
