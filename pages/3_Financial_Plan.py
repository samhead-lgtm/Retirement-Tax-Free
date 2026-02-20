import streamlit as st
import pandas as pd
import datetime as dt
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tax_estimator_advanced as TEA

st.set_page_config(page_title="OptiPlan Wealth Optimization Planner", layout="wide")

# ---------- Auth ----------
try:
    _needs_auth = bool(st.secrets.get("APP_PASSWORD", ""))
except FileNotFoundError:
    _needs_auth = False
if _needs_auth and not st.session_state.get("authenticated"):
    st.warning("Please log in from the home page.")
    st.stop()

# ---------- CSS: larger sidebar nav ----------
st.markdown("""<style>
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
    "tax_year": 2025,
    "filer_dob": dt.date(1965, 1, 1),
    "spouse_dob": dt.date(1965, 1, 1),
    "is_working": False,
    "ret_age": 65, "spouse_ret_age": 65,
    "filer_plan_age": 95, "spouse_plan_age": 95,
    "inflation": 3.0, "bracket_growth": 2.0,
    "medicare_growth": 5.0, "salary_growth": 3.0,
    "state_tax_rate": 7.0,

    # --- Growing: Employment ---
    "salary_filer": 100000.0, "salary_spouse": 0.0,
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
    "curr_taxable": 0.0, "taxable_basis_pct": 0.60,
    "curr_cash": 0.0, "emergency_fund": 0.0,
    "curr_hsa": 0.0,

    # --- Growing: Insurance Products ---
    "annuity_value_f": 0.0, "annuity_basis_f": 0.0,
    "annuity_value_s": 0.0, "annuity_basis_s": 0.0,
    "life_cash_value": 0.0, "life_basis": 0.0,

    # --- Growing: Growth Rates ---
    "r_cash": 2.0, "r_taxable": 6.0, "r_pretax": 6.0,
    "r_roth": 6.0, "r_annuity": 4.0, "r_life": 4.0,

    # --- Growing: Asset Allocation ---
    "use_asset_alloc": False,
    "cma_us_equity": 10.0, "cma_intl_equity": 8.0,
    "cma_fixed_income": 4.5, "cma_real_assets": 6.0, "cma_cash": 2.5,
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

    # --- Receiving: RMD ---
    "auto_rmd": True,
    "pretax_bal_filer_prior": 0.0, "pretax_bal_spouse_prior": 0.0,
    "baseline_pretax_dist": 0.0, "rmd_manual": 0.0,

    # --- Receiving: Tax Details ---
    "adjustments": 0.0, "dependents": 0,
    "filer_65_plus": False, "spouse_65_plus": False,
    "retirement_deduction": 0.0, "out_of_state_gain": 0.0,
    "medical_expenses": 0.0,

    # --- Spending ---
    "living_expenses": 100000.0, "ret_pct": 80.0, "heir_tax_rate": 25.0,
    "mtg_balance": 0.0, "mtg_rate": 0.0, "mtg_payment_monthly": 0.0,
    "mtg_years": 0, "property_tax": 0.0,
    "so1": "Taxable", "so2": "Pre-Tax", "so3": "Tax-Free", "so4": "Tax-Deferred",
    "surplus_dest": "Taxable Brokerage",

    # --- Giving ---
    "charitable": 0.0, "qcd_annual": 0.0,

    # --- Protecting ---
    "survivor_spend_pct": 80, "pension_survivor_pct": 50,
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
    "phase1_results": None, "phase1_best_order": None, "phase1_best_details": None,
    "phase1_all_details": None, "phase1_selected_strategy": None, "phase1_params": None,
    "phase2_results": None, "phase2_best_details": None, "phase2_baseline_details": None,
    "phase2_best_name": None,
    "tab5_conv_res": None, "tab5_conv_inputs": None, "tab5_actual_conversion": None,
    "tab5_conversion_room": None, "tab5_total_additional_cost": None,
    "projection_results": None, "retire_projection": None,
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
    return True

_profiles = sorted([f[:-5] for f in os.listdir(_PROFILE_DIR) if f.endswith(".json")])

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
    if _profiles:
        _load_sel = st.selectbox("Existing profiles", [""] + _profiles, key="_sb_load_sel")
        if st.session_state.get("_btn_load") and _load_sel:
            if _load_profile(_load_sel):
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
        w_check("Currently Working?", "is_working")
        w_num("Filer Retirement Age", "ret_age", min_value=50, max_value=80, step=1)
        if is_joint:
            w_num("Spouse Retirement Age", "spouse_ret_age", min_value=50, max_value=80, step=1)
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
        with col2:
            if is_joint:
                w_num("Spouse Salary", "salary_spouse", step=5000.0)

        st.divider()
        st.subheader("Retirement Contributions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Filer**")
            w_check("Max 401(k)?", "max_401k_f")
            if not D["max_401k_f"]:
                w_num("401(k) contribution", "c401k_f", step=1000.0)
            w_slider("Roth % of 401(k)", "roth_pct_f", min_value=0, max_value=100, step=5)
            w_num("Employer match rate (%)", "ematch_rate_f", step=1.0, format="%.1f")
            w_num("Employer match up to (%)", "ematch_upto_f", step=1.0, format="%.1f")
            w_check("Backdoor Roth?", "backdoor_roth_f")
            if not D["backdoor_roth_f"]:
                w_check("Max IRA?", "max_ira_f")
                if not D["max_ira_f"]:
                    w_num("Traditional IRA", "contrib_trad_ira_f", step=500.0)
                    w_num("Roth IRA", "contrib_roth_ira_f", step=500.0)
        with col2:
            if is_joint:
                st.markdown("**Spouse**")
                w_check("Max 401(k)?", "max_401k_s")
                if not D["max_401k_s"]:
                    w_num("401(k) contribution", "c401k_s", step=1000.0)
                w_slider("Roth % of 401(k)", "roth_pct_s", min_value=0, max_value=100, step=5)
                w_num("Employer match rate (%)", "ematch_rate_s", step=1.0, format="%.1f")
                w_num("Employer match up to (%)", "ematch_upto_s", step=1.0, format="%.1f")
                w_check("Backdoor Roth?", "backdoor_roth_s")
                if not D["backdoor_roth_s"]:
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
            w_num("Pre-retirement return (%)", "pre_ret_return", step=0.5, format="%.1f")

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
        w_num("Taxable Brokerage", "curr_taxable", step=10000.0)
        w_slider("Brokerage Gain %", "taxable_basis_pct", min_value=0.0, max_value=1.0, step=0.05)
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
        if is_joint:
            w_num("Annuity Value — Spouse", "annuity_value_s", step=5000.0)
            w_num("Annuity Basis — Spouse", "annuity_basis_s", step=5000.0)
    with col2:
        st.markdown("**Life Insurance**")
        w_num("Cash Value", "life_cash_value", step=1000.0)
        w_num("Cost Basis", "life_basis", step=1000.0)

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
        cols = st.columns(5)
        _cma_labels = ["US Equity", "Int'l Equity", "Fixed Income", "Real Assets", "Cash"]
        _cma_keys = ["cma_us_equity", "cma_intl_equity", "cma_fixed_income", "cma_real_assets", "cma_cash"]
        for i, (lbl, ck) in enumerate(zip(_cma_labels, _cma_keys)):
            with cols[i]:
                w_num(lbl, ck, step=0.5, format="%.1f")

        st.subheader("Account Allocations (%)")
        _AA_ACCOUNTS = [("Pre-Tax Filer", "pretax_f"), ("Pre-Tax Spouse", "pretax_s"),
                        ("Roth Filer", "roth_f"), ("Roth Spouse", "roth_s"),
                        ("Taxable", "taxable"), ("Annuity", "annuity")]
        _AA_FIELDS = [("US Eq", "eq"), ("Int'l", "intl"), ("Fixed", "fi"), ("Real", "re")]

        # Header row
        hdr = st.columns([2, 1, 1, 1, 1, 1, 1])
        hdr[0].markdown("**Account**")
        for i, (fl, _) in enumerate(_AA_FIELDS):
            hdr[i + 1].markdown(f"**{fl}**")
        hdr[5].markdown("**Cash**")
        hdr[6].markdown("**Return**")

        _cma_returns = [D[k] for k in _cma_keys]
        for acct_label, acct_key in _AA_ACCOUNTS:
            if not is_joint and "_s" in acct_key:
                continue
            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
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

    # --- Investment Assumptions for Projections ---
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
    col1, col2 = st.columns(2)
    with col1:
        w_num("Home Value", "home_value", step=10000.0)
    with col2:
        w_num("Home Appreciation (%)", "home_appr", step=0.5, format="%.1f")


# ══════════════════════════════════════════════════════════════════════
# RECEIVING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Receiving":
    st.header("Receiving")

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
            w_num("SS at FRA", "filer_ss_fra", step=1000.0)
            w_select("Claim age", ["62", "63", "64", "65", "66", "FRA", "68", "69", "70"], "filer_ss_claim")
    with col2:
        if is_joint:
            st.markdown("**Spouse**")
            w_check("SSDI?", "ssdi_spouse")
            w_check("Already receiving SS?", "spouse_ss_already")
            if D["spouse_ss_already"]:
                w_num("Current annual SS (Spouse)", "spouse_ss_current", step=1000.0)
                w_num("SS start year (Spouse)", "spouse_ss_start_year", min_value=2000, max_value=2060, step=1)
            else:
                w_num("SS at FRA (Spouse)", "spouse_ss_fra", step=1000.0)
                w_select("Claim age (Spouse)", ["62", "63", "64", "65", "66", "FRA", "68", "69", "70"], "spouse_ss_claim")

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
    col1, col2 = st.columns(2)
    with col1:
        w_num("Wages (if any)", "wages", step=1000.0)
        w_num("Tax-exempt interest", "tax_exempt_interest", step=100.0)
    with col2:
        w_num("Other taxable income", "other_income", step=500.0)

    # --- RMD ---
    st.divider()
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
        w_check("Filer age 65+", "filer_65_plus")
        if is_joint:
            w_check("Spouse age 65+", "spouse_65_plus")
    with col2:
        w_num("SC retirement deduction", "retirement_deduction", step=1000.0)
        w_num("Out-of-state gain (SC)", "out_of_state_gain", step=1000.0)
        w_num("Medical / health expenses", "medical_expenses", step=500.0)


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
    st.subheader("Mortgage")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_num("Mortgage Balance", "mtg_balance", step=10000.0)
        w_num("Mortgage Rate (%)", "mtg_rate", step=0.125, format="%.3f")
    with col2:
        w_num("Monthly Payment", "mtg_payment_monthly", step=100.0)
        w_num("Years Remaining", "mtg_years", min_value=0, step=1)
    with col3:
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
    st.subheader("Future Income")
    st.caption("Add expected future income (e.g., inheritance, rental income, part-time work).")
    if st.button("Add Income", key="_btn_add_income"):
        st.session_state.future_income.append({
            "name": "", "amount": 0.0,
            "start_age": _filer_age_now, "end_age": 0,
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


# ══════════════════════════════════════════════════════════════════════
# GIVING
# ══════════════════════════════════════════════════════════════════════
elif nav == "Giving":
    st.header("Giving")
    w_num("Charitable contributions", "charitable", step=500.0)
    w_num("Qualified Charitable Distribution (QCD)", "qcd_annual", step=500.0,
          help="Direct IRA-to-charity transfer. Age 70½+, up to $105k/person. Counts toward RMD, excluded from income.")
    if D["qcd_annual"] > D["charitable"]:
        st.warning("QCD cannot exceed total charitable giving.")


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
    taxable_brokerage_bal = D["curr_taxable"]
    brokerage_gain_pct = D["taxable_basis_pct"]
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
    wages = D["wages"]
    tax_exempt_interest = D["tax_exempt_interest"]
    other_income = D["other_income"]
    interest_taxable = D["interest_taxable"]
    total_ordinary_dividends = D["total_ordinary_dividends"]
    qualified_dividends = D["qualified_dividends"]
    cap_gain_loss = D["cap_gain_loss"]
    adjustments = D["adjustments"]
    dependents = int(D["dependents"])
    filer_65_plus = D["filer_65_plus"]
    spouse_65_plus = D["spouse_65_plus"] if is_joint else False
    retirement_deduction = D["retirement_deduction"]
    out_of_state_gain = D["out_of_state_gain"]
    medical_expenses = D["medical_expenses"]
    charitable = D["charitable"]
    qcd_annual = D["qcd_annual"]
    home_value = D["home_value"]
    home_appreciation = D["home_appr"] / 100

    # Mortgage
    mortgage_balance = D["mtg_balance"]
    mortgage_rate = D["mtg_rate"] / 100
    mortgage_payment = D["mtg_payment_monthly"] * 12
    property_tax = D["property_tax"]

    # Investment income for projections
    if D["use_invest_assumptions"]:
        proj_div_yield = D["proj_div_yield"] / 100
        proj_cg_pct = D["proj_cg_pct"] / 100
        proj_int_rate = D["proj_int_rate"] / 100
        proj_interest = taxable_brokerage_bal * proj_int_rate
        proj_dividends = taxable_brokerage_bal * proj_div_yield
        proj_qual_div = proj_dividends * 0.8
        proj_cap_gains = taxable_brokerage_bal * proj_cg_pct
        reinvest_dividends = D["reinvest_dividends"]
        reinvest_cap_gains = D["reinvest_cap_gains"]
        reinvest_interest = D["reinvest_interest"]
    else:
        proj_interest = interest_taxable
        proj_dividends = total_ordinary_dividends
        proj_qual_div = qualified_dividends
        proj_cap_gains = cap_gain_loss
    reinvest_dividends = D["reinvest_dividends"]
    reinvest_cap_gains = D["reinvest_cap_gains"]
    reinvest_interest = D["reinvest_interest"]

    # Growth rates (possibly overridden by asset allocation)
    def _compute_aa_return(acct_key):
        _cma = [D["cma_us_equity"], D["cma_intl_equity"], D["cma_fixed_income"], D["cma_real_assets"], D["cma_cash"]]
        _flds = ["eq", "intl", "fi", "re"]
        allocs = [D[f"aa_{acct_key}_{fk}"] for fk in _flds]
        cash_pct = max(0, 100 - sum(allocs))
        allocs.append(cash_pct)
        return sum(a / 100 * r / 100 for a, r in zip(allocs, _cma))

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
    auto_rmd = D["auto_rmd"]
    pretax_balance_filer_prior = D["pretax_bal_filer_prior"]
    pretax_balance_spouse_prior = D["pretax_bal_spouse_prior"] if is_joint else 0.0
    baseline_pretax_distributions = D["baseline_pretax_dist"]
    rmd_manual = D["rmd_manual"]

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

    # SS for this year
    gross_ss_filer = TEA.annual_ss_in_year(dob=filer_dob, tax_year=tax_year, cola=inflation,
        already_receiving=filer_ss_already, current_annual=filer_ss_current,
        start_year=filer_ss_start_year, fra_annual=filer_ss_fra,
        claim_choice=filer_ss_claim, current_year=current_year)
    gross_ss_spouse = 0.0
    if is_joint:
        gross_ss_spouse = TEA.annual_ss_in_year(dob=spouse_dob, tax_year=tax_year, cola=inflation,
            already_receiving=spouse_ss_already, current_annual=spouse_ss_current,
            start_year=spouse_ss_start_year, fra_annual=spouse_ss_fra,
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
            "home_value": float(home_value), "home_appreciation": float(home_appreciation),
            "additional_expenses": [dict(ae) for ae in st.session_state.additional_expenses if ae.get("net_amount", 0) > 0],
            "future_income": [dict(fi) for fi in st.session_state.future_income if fi.get("amount", 0) > 0],
            "surplus_destination": surplus_destination,
        }

    # --- Auto-compute base taxes ---
    _auto_inputs = current_inputs()
    _auto_assets = current_assets()
    _auto_res = TEA.compute_case(_auto_inputs)
    st.session_state.base_inputs = _auto_inputs
    st.session_state.base_results = _auto_res
    st.session_state.assets = _auto_assets

    # ---- Sub-tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Base Tax Estimator", "Income Needs", "Wealth Projection",
        "Income Optimizer", "Roth Conversion Opportunity"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: Base Tax Estimator
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Estimated Tax Analysis")
        r = st.session_state.base_results
        TEA.display_tax_return(r, mortgage_pmt=float(mortgage_payment),
                               filer_65=filer_65_plus, spouse_65=spouse_65_plus)

    # ════════════════════════════════════════════════════════════════
    # TAB 2: Income Needs
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Income Needs Analysis")
        if True:
            net_needed = st.number_input("Net income needed", min_value=0.0, step=1000.0, key="fp_net_needed")
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
                start_year = st.number_input("Start year", min_value=2020, max_value=2100, value=max(2026, int(tax_year) + 1), step=1, key="fp_start_year")
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
                spending_order = [so1, so2, so3, so4]
                surplus_dest = st.radio("Reinvest surplus income in:", ["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"],
                    index=["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"].index(D["surplus_dest"]),
                    key="fp_ret_surplus_dest", horizontal=True,
                    help="When income exceeds spending + taxes, where does the excess go?")
                surplus_destination = "none" if surplus_dest == "Don't Reinvest" else ("cash" if surplus_dest == "Cash/Savings" else "brokerage")
            with colR:
                st.markdown("#### Growth Rates")
                st.write(f"Cash: {r_cash:.1%} | Brokerage: {r_taxable:.1%} | Pre-tax: {r_pretax:.1%} | Roth: {r_roth:.1%} | Annuity: {r_annuity:.1%} | Life: {r_life:.1%}")
                st.markdown("#### Monte Carlo Settings")
                mc_simulations = st.number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100, key="fp_mc_sims")
                mc_volatility = st.number_input("Annual volatility (%)", value=12.0, step=1.0, key="fp_mc_vol") / 100

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
                _df_display = pd.DataFrame(rows).drop(columns=[c for c in _hide_cols if c in rows[0]], errors="ignore")
                st.dataframe(_df_display, use_container_width=True, hide_index=True)
                if len(rows) > 0:
                    st.markdown("### Projection Summary")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1: st.metric("Portfolio", f"${rows[-1]['Portfolio']:,.0f}")
                    with col2: st.metric("Gross Estate", f"${rows[-1]['Gross Estate']:,.0f}")
                    with col3: st.metric("Net Estate", f"${rows[-1]['Estate (Net)']:,.0f}")
                    with col4: st.metric("Total Taxes Paid", f"${sum(r['Taxes'] + r['Medicare'] for r in rows):,.0f}")
                    with col5: st.metric("Final Year Spending", f"${rows[-1]['Spending']:,.0f}")

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
                    det_eff_tax = []
                    for r in rows:
                        inc = r.get("Total Income", 0); tx = r.get("Taxes", 0) + r.get("Medicare", 0)
                        det_eff_tax.append(tx / inc if inc > 1 else 0.15)
                    mc_inv_income = rows[0]["Inv Inc"] if rows else 0.0
                    rng = np.random.default_rng(42)
                    ending_portfolios = np.zeros(n_sims); ran_out_year = np.full(n_sims, n_years)
                    all_paths = np.zeros((n_sims, n_years + 1)); all_paths[:, 0] = init_total

                    for sim in range(n_sims):
                        s_cash = mc_init_cash; s_brok = mc_init_brok; s_pf = mc_init_pf; s_ps = mc_init_ps
                        s_roth = mc_init_roth; s_life = mc_init_life; s_ann = mc_init_ann
                        for yr_i in range(n_years):
                            _yi = min(yr_i, len(rows) - 1)
                            _spending = det_spending[_yi]; ss = det_ss[_yi]; pension = det_pension[_yi]; eff_tax = det_eff_tax[_yi]
                            _age_f = current_age_filer + yr_i
                            _age_s = (current_age_spouse + yr_i) if current_age_spouse else None
                            rmd_f = TEA.compute_rmd_uniform_start73(s_pf, _age_f)
                            rmd_s = TEA.compute_rmd_uniform_start73(s_ps, _age_s)
                            rmd = rmd_f + rmd_s; s_pf -= rmd_f; s_ps -= rmd_s
                            total_income = ss + pension + rmd + mc_inv_income
                            est_taxes = total_income * eff_tax
                            income_available = ss + pension + rmd + mc_inv_income
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
    # TAB 4: Income Optimizer
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Income Optimizer")
        st.write("Finds the optimal spending strategy to maximize after-tax wealth to heirs, then tests Roth conversions for further improvement.")
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
            st.markdown("### Phase 1: Optimal Spending Strategy")
            st.info("Tests smart blend strategies (varying pre-tax vs brokerage mix each year) plus waterfall orderings to find what maximizes after-tax estate.")

            if st.button("Run Optimizer", type="primary", key="fp_run_phase1"):
                params = _build_tab3_params()
                _test_strategies = []
                _blend_caps = [0, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 999999]
                for cap in _blend_caps:
                    if cap == 0: label = "Brokerage First (Pre-Tax $0/yr)"
                    elif cap >= 999999: label = "Pre-Tax First (Unlimited)"
                    else: label = f"Blend: Pre-Tax ${cap:,.0f}/yr + Brokerage"
                    _test_strategies.append({"key": "blend", "label": label, "wf": [], "blend": False, "pt_cap": cap, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "prorata", "label": "Pro-Rata: All Accounts (Equal Weight)", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "prorata_pt_heavy", "label": "Pro-Rata: Heavy Pre-Tax (2x), Light Roth (0.25x)", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "prorata_no_roth", "label": "Pro-Rata: All Except Roth", "wf": [], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "dynamic", "label": "Dynamic Blend (Marginal Cost)", "wf": [], "blend": True, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "wf1", "label": "WF: Taxable -> Pre-Tax -> Tax-Free -> Annuity", "wf": ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _test_strategies.append({"key": "wf2", "label": "WF: Pre-Tax -> Taxable -> Tax-Free -> Annuity", "wf": ["Pre-Tax", "Taxable", "Tax-Free", "Tax-Deferred"], "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                _user_wf_name = " -> ".join(spending_order)
                if _user_wf_name not in ["Taxable -> Pre-Tax -> Tax-Free -> Tax-Deferred", "Pre-Tax -> Taxable -> Tax-Free -> Tax-Deferred"]:
                    _test_strategies.append({"key": "user", "label": f"WF: {_user_wf_name} (Tab 3)", "wf": spending_order, "blend": False, "pt_cap": None, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                for _akey, _adef in TEA.ADAPTIVE_STRATEGIES.items():
                    _test_strategies.append({"key": "adaptive", "label": _adef["label"], "wf": [], "blend": False, "pt_cap": None, "adaptive": _akey, "accel_bracket": None, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                for _ab_rate, _ab_label in [(0.12, "Accel PT: Fill 12% -> Brokerage"), (0.22, "Accel PT: Fill 22% -> Brokerage"), (0.24, "Accel PT: Fill 24% -> Brokerage"), ("irmaa", "Accel PT: Fill to IRMAA -> Brokerage")]:
                    _test_strategies.append({"key": "blend", "label": _ab_label, "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": _ab_rate, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": None})
                if a0["annuity"]["value"] > 0 and a0["annuity"]["value"] > a0["annuity"]["basis"]:
                    for _depl_yrs in [5, 10, 15]:
                        _test_strategies.append({"key": "blend", "label": f"Draw Ann Gains {_depl_yrs}yr", "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": None, "ann_depl_yrs": _depl_yrs, "ann_gains_only": True, "conv_strat": "none", "harvest_bracket": None})
                for _cb_accel, _cb_conv, _cb_label in [(0.12, "fill_bracket_22", "Accel PT Fill 12% + Convert Fill 22%"), (0.22, "fill_bracket_24", "Accel PT Fill 22% + Convert Fill 24%"), (None, "fill_bracket_22", "Brokerage First + Convert Fill 22%"), (None, "fill_bracket_24", "Brokerage First + Convert Fill 24%")]:
                    _test_strategies.append({"key": "blend", "label": _cb_label, "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": _cb_accel, "ann_depl_yrs": None, "conv_strat": _cb_conv, "harvest_bracket": None})
                for _hv_accel, _hv_label in [(None, "Harvest 0% LTCG Gains"), (None, "Brokerage First + Harvest 0% Gains"), (0.12, "Accel PT Fill 12% + Harvest 0% Gains"), (0.22, "Accel PT Fill 22% + Harvest 0% Gains")]:
                    _test_strategies.append({"key": "blend", "label": _hv_label, "wf": [], "blend": False, "pt_cap": 0, "adaptive": None, "accel_bracket": _hv_accel, "ann_depl_yrs": None, "conv_strat": "none", "harvest_bracket": 0.0})

                results = []; all_details = {}; best_estate = -float("inf"); best_order = None; best_details = None
                progress_bar = st.progress(0); status_text = st.empty()
                for idx, _strat in enumerate(_test_strategies):
                    order_key = _strat["key"]; label = _strat["label"]; wf_order = _strat["wf"]
                    is_blend = _strat["blend"]; pt_cap = _strat["pt_cap"]
                    adaptive_key = _strat["adaptive"]; accel_bracket = _strat["accel_bracket"]
                    ann_depl_yrs = _strat.get("ann_depl_yrs"); ann_gains_only = _strat.get("ann_gains_only", False)
                    conv_strat = _strat["conv_strat"]; harvest_bracket = _strat.get("harvest_bracket")
                    status_text.text(f"Testing {idx + 1} of {len(_test_strategies)}: {label}")
                    _extra_kw = {}
                    if accel_bracket is not None: _extra_kw["extra_pretax_bracket"] = accel_bracket
                    if ann_depl_yrs is not None:
                        _extra_kw["annuity_depletion_years"] = ann_depl_yrs
                        if ann_gains_only: _extra_kw["annuity_gains_only"] = True
                    if harvest_bracket is not None: _extra_kw["harvest_gains_bracket"] = harvest_bracket
                    if conv_strat != "none":
                        _extra_kw["conversion_strategy"] = conv_strat
                        _extra_kw["conversion_years_limit"] = int(params["years"])
                        _extra_kw["stop_conversion_age"] = 200
                    if order_key == "adaptive":
                        result = TEA.run_wealth_projection(a0, params, [], conversion_years_limit=int(params["years"]), stop_conversion_age=200, adaptive_strategy=adaptive_key, **_extra_kw)
                    elif order_key == "blend":
                        result = TEA.run_wealth_projection(a0, params, [], pretax_annual_cap=pt_cap, **_extra_kw)
                    elif order_key == "prorata":
                        result = TEA.run_wealth_projection(a0, params, [], prorata_blend=True, **_extra_kw)
                    elif order_key == "prorata_pt_heavy":
                        result = TEA.run_wealth_projection(a0, params, [], prorata_blend=True, prorata_weights={"pretax": 2.0, "roth": 0.25}, **_extra_kw)
                    elif order_key == "prorata_no_roth":
                        result = TEA.run_wealth_projection(a0, params, [], prorata_blend=True, prorata_weights={"roth": 0.0, "life": 0.0}, **_extra_kw)
                    elif is_blend:
                        result = TEA.run_wealth_projection(a0, params, wf_order, blend_mode=True, **_extra_kw)
                    else:
                        result = TEA.run_wealth_projection(a0, params, wf_order, **_extra_kw)
                    estate = result["after_tax_estate"]
                    _pr_weights = None
                    if order_key == "prorata_pt_heavy": _pr_weights = {"pretax": 2.0, "roth": 0.25}
                    elif order_key == "prorata_no_roth": _pr_weights = {"roth": 0.0, "life": 0.0}
                    _strat_info = {"type": order_key, "pt_cap": pt_cap, "blend": is_blend, "wf": wf_order, "prorata": order_key.startswith("prorata"), "prorata_weights": _pr_weights, "adaptive_key": adaptive_key, "accel_bracket": accel_bracket, "ann_depl_yrs": ann_depl_yrs, "ann_gains_only": ann_gains_only, "conv_strat": conv_strat, "harvest_bracket": harvest_bracket}
                    results.append({"order": order_key, "waterfall": label, "after_tax_estate": estate, "total_wealth": result["total_wealth"], "gross_estate": result.get("gross_estate", result["total_wealth"]), "total_taxes": result["total_taxes"], "final_cash": result["final_cash"], "final_brokerage": result["final_brokerage"], "final_pretax": result["final_pretax"], "final_roth": result["final_roth"], "final_annuity": result["final_annuity"], "final_life": result["final_life"], "_pt_cap": pt_cap, "_is_blend": is_blend, "_wf_order": wf_order, "_strat_info": _strat_info, "_year_details": result["year_details"]})
                    all_details[label] = result["year_details"]
                    if estate > best_estate: best_estate = estate; best_order = _strat_info; best_details = result["year_details"]
                    progress_bar.progress((idx + 1) / len(_test_strategies))
                progress_bar.empty(); status_text.empty()
                results.sort(key=lambda x: x["after_tax_estate"], reverse=True)
                _deduped = []
                for r in results:
                    _merged = False
                    for d in _deduped:
                        if abs(d["after_tax_estate"] - r["after_tax_estate"]) < 100:
                            d["_group_members"].append(r["waterfall"]); d["_group_size"] += 1; _merged = True; break
                    if not _merged: r["_group_members"] = [r["waterfall"]]; r["_group_size"] = 1; _deduped.append(r)
                results = _deduped
                st.session_state.phase1_results = results; st.session_state.phase1_best_order = best_order
                st.session_state.phase1_best_details = best_details; st.session_state.phase1_all_details = all_details
                st.session_state.phase1_selected_strategy = None; st.session_state.phase1_params = params
                st.session_state.phase2_results = None; st.session_state.phase2_best_details = None
                _estate_range = max(r["after_tax_estate"] for r in results) - min(r["after_tax_estate"] for r in results)
                if _estate_range < 100:
                    st.info("Your income covers all spending needs \u2014 withdrawal strategy has no impact. The real opportunity is **Roth conversions** (Phase 2 below).")

            if st.session_state.phase1_results:
                p1 = st.session_state.phase1_results; best = p1[0]
                st.success(f"Best Strategy: **{best['waterfall']}**")
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Gross Estate", f"${best.get('gross_estate', best['total_wealth']):,.0f}")
                with col2: st.metric("Net Estate (After Heir Tax)", f"${best['after_tax_estate']:,.0f}")
                with col3: st.metric("Total Taxes + Medicare", f"${best['total_taxes']:,.0f}")
                with st.expander("All Strategies Compared", expanded=True):
                    _comp_rows = []
                    for r in p1:
                        _type_tag = r["order"]
                        if _type_tag == "adaptive": _cat = "Adaptive"
                        elif _type_tag.startswith("prorata"): _cat = "Pro-Rata"
                        elif _type_tag == "blend": _cat = "Blend"
                        elif _type_tag == "dynamic": _cat = "Dynamic"
                        else: _cat = "Waterfall"
                        _row = {"Type": _cat, "Strategy": r["waterfall"], "Net Estate": f"${r['after_tax_estate']:,.0f}", "Gross Estate": f"${r.get('gross_estate', r['total_wealth']):,.0f}", "Total Taxes": f"${r['total_taxes']:,.0f}", "Final Pre-Tax": f"${r['final_pretax']:,.0f}", "Final Roth": f"${r['final_roth']:,.0f}", "Final Brokerage": f"${r['final_brokerage']:,.0f}", "Final Cash": f"${r.get('final_cash', 0):,.0f}", "Final Annuity": f"${r.get('final_annuity', 0):,.0f}", "Final Life": f"${r.get('final_life', 0):,.0f}", "vs Best": f"${r['after_tax_estate'] - best['after_tax_estate']:+,.0f}"}
                        _gs = r.get("_group_size", 1)
                        _row["# Same"] = _gs if _gs > 1 else ""
                        _comp_rows.append(_row)
                    _comp_df = pd.DataFrame(_comp_rows)
                    for _bcol, _bkey in [("Final Cash", "final_cash"), ("Final Annuity", "final_annuity"), ("Final Life", "final_life"), ("Final Pre-Tax", "final_pretax"), ("Final Roth", "final_roth"), ("Final Brokerage", "final_brokerage")]:
                        if _bcol in _comp_df.columns and all(r.get(_bkey, 0) < 1 for r in p1): _comp_df = _comp_df.drop(columns=[_bcol])
                    st.dataframe(_comp_df, use_container_width=True, hide_index=True)

                _strat_labels = [r["waterfall"] for r in p1]
                _selected_label = st.selectbox("Select strategy to view detail & use for Phase 2", _strat_labels, index=0, key="fp_p1_strategy_select", help="Default is the highest-estate strategy.")
                _selected_result = next(r for r in p1 if r["waterfall"] == _selected_label)
                st.session_state.phase1_selected_strategy = _selected_result.get("_strat_info")
                if _selected_label != best["waterfall"]:
                    col1b, col2b, col3b = st.columns(3)
                    with col1b: st.metric("Selected Gross Estate", f"${_selected_result.get('gross_estate', _selected_result['total_wealth']):,.0f}")
                    with col2b: st.metric("Selected Net Estate", f"${_selected_result['after_tax_estate']:,.0f}")
                    with col3b: st.metric("vs Best", f"${_selected_result['after_tax_estate'] - best['after_tax_estate']:+,.0f}")
                _opt_hide = ["W/D Roth", "W/D Life", "Bal Life", "Total Wealth", "_net_draw"]
                _zero_hide = ["Accel PT", "Harvest Gains"]
                _sel_details = _selected_result.get("_year_details", [])
                if _sel_details:
                    with st.expander(f"Year-by-Year Detail: {_selected_label}", expanded=True):
                        _df_det = pd.DataFrame(_sel_details)
                        _df_det = _df_det.drop(columns=[c for c in _opt_hide if c in _df_det.columns], errors="ignore")
                        for _zc in _zero_hide:
                            if _zc in _df_det.columns and (_df_det[_zc] == 0).all(): _df_det = _df_det.drop(columns=[_zc])
                        st.dataframe(_df_det, use_container_width=True, hide_index=True)
                elif st.session_state.phase1_best_details:
                    with st.expander("Year-by-Year Detail (Best Strategy)"):
                        _df_det = pd.DataFrame(st.session_state.phase1_best_details)
                        _df_det = _df_det.drop(columns=[c for c in _opt_hide if c in _df_det.columns], errors="ignore")
                        for _zc in _zero_hide:
                            if _zc in _df_det.columns and (_df_det[_zc] == 0).all(): _df_det = _df_det.drop(columns=[_zc])
                        st.dataframe(_df_det, use_container_width=True, hide_index=True)

            # ---- Phase 2: Roth Conversion Layering ----
            st.divider()
            st.markdown("### Phase 2: Roth Conversion Layering")
            if st.session_state.phase1_best_order is None:
                st.warning("Run Phase 1 first to establish the optimal spending strategy.")
            else:
                winning_order = st.session_state.phase1_selected_strategy or st.session_state.phase1_best_order
                if isinstance(winning_order, dict):
                    _wo_type = winning_order.get("type", "")
                    if _wo_type == "blend":
                        _cap = winning_order.get("pt_cap", 0)
                        if _cap >= 999999: st.write("**Locked Strategy:** Smart Blend \u2014 Pre-Tax First (Unlimited)")
                        elif _cap == 0: st.write("**Locked Strategy:** Smart Blend \u2014 Brokerage First")
                        else: st.write(f"**Locked Strategy:** Smart Blend \u2014 Pre-Tax ${_cap:,.0f}/yr + Brokerage")
                    elif _wo_type == "dynamic": st.write("**Locked Strategy:** Dynamic Blend (Marginal Cost)")
                    elif _wo_type == "prorata": st.write("**Locked Strategy:** Pro-Rata: All Accounts (Equal Weight)")
                    elif _wo_type == "prorata_pt_heavy": st.write("**Locked Strategy:** Pro-Rata: Heavy Pre-Tax (2x), Light Roth (0.25x)")
                    elif _wo_type == "prorata_no_roth": st.write("**Locked Strategy:** Pro-Rata: All Except Roth")
                    elif _wo_type == "adaptive":
                        _ak = winning_order.get("adaptive_key", "")
                        _alabel = TEA.ADAPTIVE_STRATEGIES.get(_ak, {}).get("label", _ak)
                        st.write(f"**Locked Strategy:** {_alabel}")
                        st.caption(TEA.ADAPTIVE_STRATEGIES.get(_ak, {}).get("description", ""))
                    else:
                        _wf = winning_order.get("wf", [])
                        st.write(f"**Locked Strategy:** Waterfall \u2014 {' -> '.join(_wf)}")
                elif winning_order == "dynamic": st.write("**Locked Strategy:** Dynamic Blend")
                else: st.write(f"**Locked Spending Order:** {' -> '.join(winning_order)}")
                if isinstance(winning_order, dict):
                    _extras = []
                    _wo_accel = winning_order.get("accel_bracket"); _wo_depl = winning_order.get("ann_depl_yrs"); _wo_harvest = winning_order.get("harvest_bracket")
                    if _wo_accel is not None: _extras.append(f"Accel PT: fill {_wo_accel}" if _wo_accel != "irmaa" else "Accel PT: fill to IRMAA")
                    if _wo_depl is not None:
                        _wo_go = winning_order.get("ann_gains_only", False)
                        _extras.append(f"Draw ann gains: {_wo_depl}yr" if _wo_go else f"Annuity depletion: {_wo_depl}yr")
                    if _wo_harvest is not None: _extras.append("Harvest 0% LTCG gains")
                    if _extras: st.caption("Inherited: " + ", ".join(_extras))

                col1, col2 = st.columns(2)
                with col1:
                    opt_stop_age = st.number_input("Stop Conversions at Age", min_value=60, max_value=100, value=75, step=1, key="fp_opt_stop")
                    opt_conv_years = st.number_input("Max Years of Conversions", min_value=1, max_value=30, value=15, step=1, key="fp_opt_conv_years")
                with col2:
                    st.markdown("**Common Thresholds**")
                    if is_joint: st.write("22% bracket: $206,700 | 24% bracket: $394,600 | IRMAA: $218,000")
                    else: st.write("22% bracket: $103,350 | 24% bracket: $197,300 | IRMAA: $109,000")
                    target_agi_input = st.number_input("Target AGI (for 'fill to bracket')", value=218000.0 if is_joint else 109000.0, step=10000.0, key="fp_opt_target")
                conv_amounts_str = st.text_input("Conversion amounts to test (comma separated)", value="25000, 50000, 75000, 100000, 150000, 200000", key="fp_opt_amounts")
                include_fill = st.checkbox("Also test 'Fill to Target AGI' strategy", value=True, key="fp_opt_fill")
                include_bracket_fill = st.checkbox("Also test bracket-fill strategies", value=True, key="fp_opt_bracket_fill")

                if st.button("Run Phase 2 - Test Roth Conversions", type="primary", key="fp_run_phase2"):
                    params = st.session_state.phase1_params or _build_tab3_params()
                    try: conv_amounts = [float(x.strip()) for x in conv_amounts_str.split(",")]
                    except Exception: conv_amounts = [25000, 50000, 75000, 100000]
                    strategies = [("none", "No Conversion (baseline)")]
                    for amt in conv_amounts: strategies.append((amt, f"${amt:,.0f}/yr"))
                    if include_fill: strategies.append(("fill_to_target", f"Fill to ${target_agi_input:,.0f}"))
                    if include_bracket_fill:
                        strategies += [("fill_bracket_12", "Fill 12% Bracket"), ("fill_bracket_22", "Fill 22% Bracket"), ("fill_bracket_24", "Fill 24% Bracket"), ("fill_irmaa_0", "Fill to IRMAA Tier 1")]
                    results = []; best_details = None; baseline_details = None; best_estate = -float("inf"); best_name = ""; baseline_estate = 0.0
                    progress_bar = st.progress(0); status_text = st.empty()
                    _p2_blend = False; _p2_order = []; _p2_pt_cap = None; _p2_prorata = False; _p2_prorata_weights = None; _p2_adaptive_key = None
                    _p2_accel_bracket = None; _p2_harvest_bracket = None; _p2_ann_depl_yrs = None; _p2_ann_gains_only = False
                    if isinstance(winning_order, dict):
                        _wo_type = winning_order.get("type", "")
                        if _wo_type == "adaptive": _p2_adaptive_key = winning_order.get("adaptive_key")
                        elif _wo_type == "blend": _p2_pt_cap = winning_order.get("pt_cap")
                        elif _wo_type == "dynamic": _p2_blend = True
                        elif winning_order.get("prorata", False): _p2_prorata = True; _p2_prorata_weights = winning_order.get("prorata_weights")
                        else: _p2_order = winning_order.get("wf", [])
                        _p2_accel_bracket = winning_order.get("accel_bracket"); _p2_harvest_bracket = winning_order.get("harvest_bracket")
                        _p2_ann_depl_yrs = winning_order.get("ann_depl_yrs"); _p2_ann_gains_only = winning_order.get("ann_gains_only", False)
                    elif winning_order == "dynamic": _p2_blend = True
                    else: _p2_order = winning_order if isinstance(winning_order, list) else []
                    _p2_extra_kw = {}
                    if _p2_accel_bracket is not None: _p2_extra_kw["extra_pretax_bracket"] = _p2_accel_bracket
                    if _p2_harvest_bracket is not None: _p2_extra_kw["harvest_gains_bracket"] = _p2_harvest_bracket
                    if _p2_ann_depl_yrs is not None:
                        _p2_extra_kw["annuity_depletion_years"] = _p2_ann_depl_yrs
                        if _p2_ann_gains_only: _p2_extra_kw["annuity_gains_only"] = True
                    for idx, (strategy, strategy_name) in enumerate(strategies):
                        if _p2_adaptive_key:
                            result = TEA.run_wealth_projection(a0, params, _p2_order, conversion_strategy=strategy, target_agi=target_agi_input, stop_conversion_age=opt_stop_age, conversion_years_limit=opt_conv_years, adaptive_strategy=_p2_adaptive_key, **_p2_extra_kw)
                        else:
                            result = TEA.run_wealth_projection(a0, params, _p2_order, conversion_strategy=strategy, target_agi=target_agi_input, stop_conversion_age=opt_stop_age, conversion_years_limit=opt_conv_years, blend_mode=_p2_blend, pretax_annual_cap=_p2_pt_cap, prorata_blend=_p2_prorata, prorata_weights=_p2_prorata_weights, **_p2_extra_kw)
                        estate = result["after_tax_estate"]
                        if strategy == "none": baseline_estate = estate; baseline_details = result["year_details"]
                        results.append({"strategy_name": strategy_name, "after_tax_estate": estate, "improvement": estate - baseline_estate, "total_wealth": result["total_wealth"], "gross_estate": result.get("gross_estate", result["total_wealth"]), "total_taxes": result["total_taxes"], "total_converted": result["total_converted"], "final_pretax": result["final_pretax"], "final_roth": result["final_roth"], "final_brokerage": result["final_cash"] + result["final_brokerage"], "_year_details": result["year_details"]})
                        if estate > best_estate: best_estate = estate; best_details = result["year_details"]; best_name = strategy_name
                        progress_bar.progress((idx + 1) / len(strategies)); status_text.text(f"Testing strategy {idx + 1} of {len(strategies)}...")
                    progress_bar.empty(); status_text.empty()
                    st.session_state.phase2_results = results; st.session_state.phase2_best_details = best_details
                    st.session_state.phase2_baseline_details = baseline_details; st.session_state.phase2_best_name = best_name

                if st.session_state.phase2_results:
                    p2 = st.session_state.phase2_results
                    p2_sorted = sorted(p2, key=lambda x: x["after_tax_estate"], reverse=True)
                    best_conv = p2_sorted[0]; baseline = next((r for r in p2 if "baseline" in r["strategy_name"].lower()), p2[0])
                    if best_conv["strategy_name"] != baseline["strategy_name"]:
                        improvement = best_conv["after_tax_estate"] - baseline["after_tax_estate"]
                        st.success(f"Best Strategy: **{best_conv['strategy_name']}** adds **${improvement:,.0f}** to after-tax estate vs no conversion")
                    else: st.info("No Roth conversion strategy improves upon the baseline (no conversions).")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Baseline Estate (No Conv)", f"${baseline['after_tax_estate']:,.0f}")
                    with col2: st.metric("Best Estate (With Conv)", f"${best_conv['after_tax_estate']:,.0f}")
                    with col3: st.metric("Improvement", f"${best_conv['after_tax_estate'] - baseline['after_tax_estate']:,.0f}")
                    with st.expander("All Conversion Strategies Compared"):
                        df_p2 = pd.DataFrame(p2_sorted); df_p2.index = range(1, len(df_p2) + 1); df_p2.index.name = "Rank"
                        display_cols2 = ["strategy_name", "after_tax_estate", "improvement", "total_wealth", "total_taxes", "total_converted", "final_pretax", "final_roth", "final_brokerage"]
                        df_show2 = df_p2[display_cols2].copy()
                        for c in display_cols2[1:]: df_show2[c] = df_show2[c].apply(lambda x: f"${x:,.0f}")
                        df_show2.columns = ["Strategy", "After-Tax Estate", "vs Baseline", "Total Wealth", "Total Taxes", "Total Converted", "Final Pre-Tax", "Final Roth", "Final Taxable"]
                        st.dataframe(df_show2, use_container_width=True)

                    _p2_labels = [r["strategy_name"] for r in p2_sorted]
                    _p2_selected_label = st.selectbox("Select conversion strategy to view detail", _p2_labels, index=0, key="fp_p2_strategy_select")
                    _p2_selected_result = next(r for r in p2_sorted if r["strategy_name"] == _p2_selected_label)
                    _p2_sel_details = _p2_selected_result.get("_year_details", [])
                    _p2_hide = ["W/D Roth", "W/D Life", "Bal Life", "Total Wealth", "_net_draw"]
                    _p2_zero_hide = ["Accel PT", "Harvest Gains"]
                    if _p2_sel_details:
                        with st.expander(f"Year-by-Year Detail: {_p2_selected_label}", expanded=True):
                            _df_p2det = pd.DataFrame(_p2_sel_details)
                            _df_p2det = _df_p2det.drop(columns=[c for c in _p2_hide if c in _df_p2det.columns], errors="ignore")
                            for _zc in _p2_zero_hide:
                                if _zc in _df_p2det.columns and (_df_p2det[_zc] == 0).all(): _df_p2det = _df_p2det.drop(columns=[_zc])
                            st.dataframe(_df_p2det, use_container_width=True, hide_index=True)

                    _bl_result = next((r for r in p2 if "baseline" in r["strategy_name"].lower()), None)
                    _bl_details = _bl_result.get("_year_details", []) if _bl_result else []
                    if _bl_details and _p2_sel_details:
                        st.divider()
                        yr1_bl = _bl_details[0] if _bl_details else {}; yr1_bc = _p2_sel_details[0] if _p2_sel_details else {}
                        st.markdown(f"### Year 1 Cash Flow: No Conversion vs {_p2_selected_label}")
                        col1, col2 = st.columns(2)
                        for col, yr, label in [(col1, yr1_bl, "No Conversion"), (col2, yr1_bc, _p2_selected_label)]:
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
                        final_bl = _bl_details[-1] if _bl_details else {}; final_bc = _p2_sel_details[-1] if _p2_sel_details else {}
                        with col1:
                            st.markdown("#### No Conversion")
                            st.metric("Final Pre-Tax", TEA.money(final_bl.get("Bal Pre-Tax", 0)))
                            st.metric("Final Roth", TEA.money(final_bl.get("Bal Roth", 0)))
                            st.metric("Final Taxable", TEA.money(final_bl.get("Bal Cash", 0) + final_bl.get("Bal Taxable", 0)))
                            st.metric("After-Tax Estate", TEA.money(final_bl.get("Estate (Net)", 0)))
                        with col2:
                            st.markdown(f"#### {_p2_selected_label}")
                            st.metric("Final Pre-Tax", TEA.money(final_bc.get("Bal Pre-Tax", 0)))
                            st.metric("Final Roth", TEA.money(final_bc.get("Bal Roth", 0)))
                            st.metric("Final Taxable", TEA.money(final_bc.get("Bal Cash", 0) + final_bc.get("Bal Taxable", 0)))
                            st.metric("After-Tax Estate", TEA.money(final_bc.get("Estate (Net)", 0)))

    # ════════════════════════════════════════════════════════════════
    # TAB 5: Roth Conversion Opportunity
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Roth Conversion Opportunity")
        st.write("Calculate how much you can convert to fill up to a target income level based on your Income Needs scenario.")
        if st.session_state.last_solved_results is None or st.session_state.last_solved_inputs is None:
            st.warning("Run Income Needs (Tab 2) first to establish your baseline spending scenario.")
        else:
            solved_res = st.session_state.last_solved_results; solved_inp = st.session_state.last_solved_inputs
            net_needed_val = float(st.session_state.last_net_needed or 0.0); source_used = st.session_state.last_source or "Unknown"
            st.markdown("### Current Scenario (from Income Needs)")
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
            available_pretax = st.session_state.assets["pretax"]["balance"] if st.session_state.assets else 0.0
            actual_conversion = min(conversion_room, available_pretax)
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
                if actual_conversion < conversion_room: st.info(f"Note: You only have {TEA.money(available_pretax)} available in pre-tax accounts.")
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
                st.caption("Uses the full wealth projection engine (spending, RMDs, taxes, withdrawals) \u2014 same as Tab 3 / Phase 2.")
                _ec1, _ec2 = st.columns(2)
                with _ec1: _est_heir_rate = st.number_input("Heir Tax Rate (%)", value=25.0, step=1.0, key="fp_heir_tab5") / 100
                with _ec2: _conv_years = st.number_input("Years of Conversion", min_value=1, max_value=30, value=1, step=1, key="fp_conv_years_tab5")
                _ei_params = _build_tab3_params(); _ei_params["heir_tax_rate"] = _est_heir_rate
                _ei_a0 = st.session_state.assets; _ei_stop_age = current_age_filer + _conv_years
                _nc_result = TEA.run_wealth_projection(_ei_a0, _ei_params, spending_order, conversion_strategy="none")
                _wc_result = TEA.run_wealth_projection(_ei_a0, _ei_params, spending_order, conversion_strategy=float(actual_conversion), conversion_years_limit=_conv_years, stop_conversion_age=_ei_stop_age)
                _nc_net = _nc_result["after_tax_estate"]; _wc_net = _wc_result["after_tax_estate"]
                _nc_gross = _nc_result.get("gross_estate", _nc_result["total_wealth"]); _wc_gross = _wc_result.get("gross_estate", _wc_result["total_wealth"])
                _nc_taxes = _nc_result["total_taxes"]; _wc_taxes = _wc_result["total_taxes"]
                _nc_pretax = _nc_result["final_pretax"]; _wc_pretax = _wc_result["final_pretax"]
                _nc_roth = _nc_result["final_roth"]; _wc_roth = _wc_result["final_roth"]
                _nc_brok = _nc_result["final_cash"] + _nc_result["final_brokerage"]; _wc_brok = _wc_result["final_cash"] + _wc_result["final_brokerage"]
                _yr_label = "Year" if _conv_years == 1 else f"{_conv_years} Years"
                _proj_label = f"{int(_ei_params['years'])}yr Projection"
                _estate_table = f"| Metric | No Conversion | With Conversion ({_yr_label}) | Change |\n|:---|-------:|-------:|-------:|\n"
                _estate_table += f"| **Net Estate** | **{TEA.money(_nc_net)}** | **{TEA.money(_wc_net)}** | **{TEA.money(_wc_net - _nc_net)}** |\n"
                _estate_table += f"| Total Taxes Paid | {TEA.money(_nc_taxes)} | {TEA.money(_wc_taxes)} | {TEA.money(_wc_taxes - _nc_taxes)} |\n"
                _estate_table += f"| Final Pre-Tax | {TEA.money(_nc_pretax)} | {TEA.money(_wc_pretax)} | {TEA.money(_wc_pretax - _nc_pretax)} |\n"
                _estate_table += f"| Final Roth | {TEA.money(_nc_roth)} | {TEA.money(_wc_roth)} | {TEA.money(_wc_roth - _nc_roth)} |\n"
                _estate_table += f"| Final Brokerage | {TEA.money(_nc_brok)} | {TEA.money(_wc_brok)} | {TEA.money(_wc_brok - _nc_brok)} |\n"
                _estate_table += f"| **Portfolio** | **{TEA.money(_nc_gross)}** | **{TEA.money(_wc_gross)}** | **{TEA.money(_wc_gross - _nc_gross)}** |\n"
                st.markdown(_estate_table)
                st.caption(f"Full {_proj_label} with spending, RMDs, taxes, and withdrawals. Conversion: {TEA.money(actual_conversion)}/yr for {_conv_years} yr. Total converted: {TEA.money(_wc_result['total_converted'])}.")
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
                if _net_benefit > 0: st.success(f"Converting {TEA.money(actual_conversion)}/yr for {_conv_years} year(s) improves the net estate by **{TEA.money(_net_benefit)}** (heirs @ {_est_heir_rate:.0%}).")
                elif _net_benefit < 0: st.warning(f"Converting {TEA.money(actual_conversion)}/yr for {_conv_years} year(s) reduces the net estate by **{TEA.money(abs(_net_benefit))}** (heirs @ {_est_heir_rate:.0%}).")
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
