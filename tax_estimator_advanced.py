import streamlit as st
import pandas as pd
import datetime as dt
import json, os
from functools import lru_cache
from itertools import permutations

st.set_page_config(page_title="RTF Tax + Wealth Engine (Master)", layout="wide")

DEFAULT_STATE = {
    "base_results": None, "base_inputs": None, "assets": None,
    "gross_from_needs": None, "last_net_needed": None, "last_taxes_paid_by_cash": None,
    "last_source": None, "last_withdrawal_proceeds": None,
    "last_solved_results": None, "last_solved_inputs": None, "last_solved_assets": None,
    "phase1_results": None, "phase1_best_order": None, "phase1_best_details": None,
    "phase1_params": None, "phase2_results": None, "phase2_best_details": None,
    "phase2_baseline_details": None, "phase2_best_name": None,
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
    "total_ordinary_div", "qualified_div", "cap_gain_loss", "other_income",
    "filer_65_plus", "spouse_65_plus",
    "adjustments", "dependents", "retirement_deduction", "out_of_state_gain",
    "itemizing", "itemized_amount",
    "taxable_cash_bal", "taxable_brokerage_bal", "brokerage_gain_pct",
    "pretax_bal", "roth_bal", "life_cash_value", "annuity_value", "annuity_basis",
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
    out_of_state_gain = float(inputs["out_of_state_gain"]); itemizing = bool(inputs["itemizing"])
    itemized_amount = float(inputs["itemized_amount"])

    base_non_ss = wages + interest_taxable + total_ordinary_dividends + taxable_ira + rmd_amount + taxable_pensions + ordinary_tax_only + cap_gain_loss + other_income
    taxable_ss = calculate_taxable_ss(base_non_ss, tax_exempt_interest, gross_ss, filing_status)
    total_income_for_tax = base_non_ss + taxable_ss
    agi = max(0.0, total_income_for_tax - adjustments)
    base_std = get_federal_base_std(filing_status, inflation_factor)
    traditional_extra = get_federal_traditional_extra(filing_status, filer_65_plus, spouse_65_plus, inflation_factor)
    enhanced_extra = get_federal_enhanced_extra(agi, filer_65_plus, spouse_65_plus, filing_status, inflation_factor)
    fed_std = base_std + traditional_extra + enhanced_extra
    deduction_used = itemized_amount if (itemizing and itemized_amount > fed_std) else fed_std
    fed_taxable = max(0.0, agi - deduction_used)
    preferential_amount = qualified_dividends + max(0.0, cap_gain_loss)
    fed = calculate_federal_tax(fed_taxable, preferential_amount, filing_status, inflation_factor)
    sc = calculate_sc_tax(fed_taxable, dependents, taxable_ss, out_of_state_gain, filer_65_plus, spouse_65_plus, retirement_deduction, cap_gain_loss)
    total_tax = fed["federal_tax"] + sc["sc_tax"]
    medicare_premiums, has_irmaa = estimate_medicare_premiums(agi, filing_status, inflation_factor)
    spendable_gross = wages + gross_ss + taxable_pensions + total_ordinary_dividends + taxable_ira + rmd_amount + other_income + cashflow_taxfree + brokerage_proceeds + annuity_proceeds
    net_before_tax = spendable_gross - medicare_premiums
    net_after_tax = spendable_gross - medicare_premiums - total_tax
    return {
        "total_income_for_tax": total_income_for_tax, "agi": agi, "deduction_used": deduction_used,
        "fed_taxable": fed_taxable, "taxable_ss": taxable_ss, "fed_tax": fed["federal_tax"], "effective_fed": fed["effective_fed"],
        "sc_tax": sc["sc_tax"], "effective_sc": sc["effective_sc"], "sc_taxable": sc["sc_taxable"],
        "total_tax": total_tax, "medicare_premiums": medicare_premiums, "has_irmaa": has_irmaa,
        "spendable_gross": spendable_gross, "net_before_tax": net_before_tax, "net_after_tax": net_after_tax,
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

def display_cashflow_comparison(before_inp, before_res, after_inp, after_res, net_needed, roth_conversion=0.0, title_before="Before", title_after="After"):
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
        out_before = [("Living expenses", net_needed), ("Medicare", float(before_res.get("medicare_premiums", 0.0))), ("Taxes", float(before_res.get("total_tax", 0.0)))]
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
        out_after = [("Living expenses", net_needed), ("Medicare", float(after_res.get("medicare_premiums", 0.0))), ("Taxes", float(after_res.get("total_tax", 0.0)))]
        out_after = [(k, v) for k, v in out_after if v > 0]
        total_out_after = sum(x[1] for x in out_after)
        st.dataframe([{"Expense": k, "Amount": money(v)} for k, v in out_after], use_container_width=True, hide_index=True)
        st.metric("Total Outflows", money(total_out_after))

def run_optimized_projection(initial_assets, params, spending_order, conversion_strategy="none",
                              target_agi=0, stop_conversion_age=100, conversion_years_limit=0):
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
    inv_income_base = params["inv_income_base"]
    retirement_deduction = params["retirement_deduction"]
    filing_status = params["filing_status"]
    current_age_filer = params["current_age_filer"]
    current_age_spouse = params["current_age_spouse"]
    pretax_filer_ratio = params["pretax_filer_ratio"]
    brokerage_gain_pct = params["brokerage_gain_pct"]
    ann_basis_start = params["ann_basis_start"]

    curr_cash = initial_assets["taxable"]["cash"]
    curr_brokerage = initial_assets["taxable"]["brokerage"]
    total_pre = initial_assets["pretax"]["balance"]
    curr_pre_filer = total_pre * pretax_filer_ratio
    curr_pre_spouse = total_pre * (1.0 - pretax_filer_ratio)
    curr_roth = initial_assets["taxfree"]["roth"]
    curr_life = initial_assets["taxfree"]["life_cash_value"]
    curr_ann = initial_assets["annuity"]["value"]
    curr_ann_basis = ann_basis_start

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

        inflated_spend = spending_goal * inf_factor
        ss_now = gross_ss_total * inf_factor
        pen_now = taxable_pensions_total * ((1 + pension_cola) ** i)
        inv_income = inv_income_base * inf_factor

        boy_pretax = curr_pre_filer + curr_pre_spouse

        rmd_f = compute_rmd_uniform_start73(curr_pre_filer, age_f)
        rmd_s = compute_rmd_uniform_start73(curr_pre_spouse, age_s)
        rmd_total = rmd_f + rmd_s
        curr_pre_filer -= rmd_f
        curr_pre_spouse -= rmd_s

        base_taxable = pen_now + rmd_total + inv_income

        conversion_this_year = 0.0
        if do_conversions and i < conversion_years_limit and age_f < stop_conversion_age:
            avail_pretax = curr_pre_filer + curr_pre_spouse
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

        cash_received = ss_now + pen_now + rmd_total + inv_income

        wd_cash = 0.0
        wd_brokerage = 0.0
        wd_pretax = 0.0
        wd_roth = 0.0
        wd_life = 0.0
        wd_annuity = 0.0
        ann_gains_withdrawn = 0.0
        cap_gains_realized = 0.0

        for iteration in range(25):
            tax_result = compute_taxes_only(
                gross_ss=ss_now, taxable_pensions=pen_now, rmd_amount=rmd_total,
                taxable_ira=wd_pretax + conversion_this_year,
                conversion_amount=0.0,
                ordinary_income=ann_gains_withdrawn + inv_income,
                cap_gains=cap_gains_realized,
                filing_status=filing_status, filer_65=filer_65, spouse_65=spouse_65,
                retirement_deduction=retirement_deduction, inf_factor=inf_factor
            )

            taxes = tax_result["total_tax"]
            medicare = tax_result["medicare"]
            cash_needed = inflated_spend + taxes + medicare
            cash_available = cash_received + wd_cash + wd_brokerage + wd_pretax + wd_roth + wd_life + wd_annuity
            shortfall = cash_needed - cash_available

            if shortfall <= 1.0:
                break

            pulled_this_iteration = False
            for bucket in spending_order:
                if shortfall <= 0:
                    break

                if bucket == "Taxable":
                    avail_cash = curr_cash - wd_cash
                    pull = min(shortfall, avail_cash)
                    if pull > 0:
                        wd_cash += pull
                        shortfall -= pull
                        pulled_this_iteration = True
                    if shortfall > 0:
                        avail_brok = curr_brokerage - wd_brokerage
                        pull = min(shortfall, avail_brok)
                        if pull > 0:
                            wd_brokerage += pull
                            cap_gains_realized += pull * brokerage_gain_pct
                            shortfall -= pull
                            pulled_this_iteration = True

                elif bucket == "Tax-Free":
                    if conversion_this_year == 0:
                        avail = curr_roth - wd_roth
                        pull = min(shortfall, avail)
                        if pull > 0:
                            wd_roth += pull
                            shortfall -= pull
                            pulled_this_iteration = True
                    if shortfall > 0:
                        avail = curr_life - wd_life
                        pull = min(shortfall, avail)
                        if pull > 0:
                            wd_life += pull
                            shortfall -= pull
                            pulled_this_iteration = True

                elif bucket == "Pre-Tax":
                    avail = curr_pre_filer + curr_pre_spouse - wd_pretax
                    pull = min(shortfall, avail)
                    if pull > 0:
                        wd_pretax += pull
                        shortfall -= pull
                        pulled_this_iteration = True

                elif bucket == "Tax-Deferred":
                    avail = curr_ann - wd_annuity
                    pull = min(shortfall, avail)
                    if pull > 0:
                        current_gains = max(0.0, (curr_ann - wd_annuity + pull) - curr_ann_basis)
                        new_gains = min(pull, max(0.0, current_gains))
                        wd_annuity += pull
                        ann_gains_withdrawn += new_gains
                        shortfall -= pull
                        pulled_this_iteration = True

            if not pulled_this_iteration:
                break

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

        total_taxes_paid += taxes + medicare

        curr_cash = max(0.0, curr_cash) * (1 + r_taxable)
        curr_brokerage = max(0.0, curr_brokerage) * (1 + r_taxable)
        curr_pre_filer = max(0.0, curr_pre_filer) * (1 + r_pretax)
        curr_pre_spouse = max(0.0, curr_pre_spouse) * (1 + r_pretax)
        curr_roth = max(0.0, curr_roth) * (1 + r_roth)
        curr_ann = max(0.0, curr_ann) * (1 + r_annuity)
        curr_life = max(0.0, curr_life) * (1 + r_life)

        total_wealth_yr = curr_cash + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
        ann_gains_yr = max(0.0, curr_ann - curr_ann_basis)
        heir_tax_yr = (curr_pre_filer + curr_pre_spouse) * heir_tax_rate + ann_gains_yr * heir_tax_rate
        at_estate_yr = total_wealth_yr - heir_tax_yr

        year_details.append({
            "Year": yr, "Age": age_f, "Spending": round(inflated_spend, 0),
            "SS": round(ss_now, 0), "Pension": round(pen_now, 0),
            "BOY Pre-Tax": round(boy_pretax, 0), "RMD": round(rmd_total, 0),
            "Inv Inc": round(inv_income, 0), "Conversion": round(conversion_this_year, 0),
            "W/D Cash": round(wd_cash, 0), "W/D Brokerage": round(wd_brokerage, 0),
            "W/D Pre-Tax": round(wd_pretax, 0), "W/D Roth": round(wd_roth, 0),
            "W/D Life": round(wd_life, 0), "W/D Annuity": round(wd_annuity, 0),
            "Cap Gains": round(cap_gains_realized, 0), "Taxes": round(taxes, 0),
            "Medicare": round(medicare, 0),
            "Bal Cash": round(curr_cash, 0), "Bal Brokerage": round(curr_brokerage, 0),
            "Bal Pre-Tax": round(curr_pre_filer + curr_pre_spouse, 0),
            "Bal Roth": round(curr_roth, 0), "Bal Annuity": round(curr_ann, 0),
            "Bal Life": round(curr_life, 0),
            "Total Wealth": round(total_wealth_yr, 0), "Estate (Net)": round(at_estate_yr, 0),
        })

    total_wealth = curr_cash + curr_brokerage + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
    ann_gains_final = max(0.0, curr_ann - curr_ann_basis)
    heir_tax = (curr_pre_filer + curr_pre_spouse) * heir_tax_rate + ann_gains_final * heir_tax_rate
    after_tax_estate = total_wealth - heir_tax

    return {
        "after_tax_estate": after_tax_estate, "total_wealth": total_wealth,
        "total_taxes": total_taxes_paid, "total_converted": total_converted,
        "final_cash": curr_cash, "final_brokerage": curr_brokerage,
        "final_pretax": curr_pre_filer + curr_pre_spouse,
        "final_roth": curr_roth, "final_annuity": curr_ann, "final_life": curr_life,
        "year_details": year_details,
    }


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

    st.header("Tax Year & Filing")
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
    cap_gain_loss = st.number_input("Baseline net cap gain/(loss)", value=0.0, step=1000.0, key="cap_gain_loss")
    other_income = st.number_input("Other taxable income", value=0.0, step=500.0, key="other_income")
    filer_65_plus = st.checkbox("Filer age 65+", key="filer_65_plus")
    spouse_65_plus = st.checkbox("Spouse age 65+", key="spouse_65_plus") if "joint" in filing_status.lower() else False
    adjustments = st.number_input("Adjustments to income", value=0.0, step=500.0, key="adjustments")
    dependents = st.number_input("Dependents", value=0, step=1, key="dependents")
    retirement_deduction = st.number_input("SC retirement deduction", value=0.0, step=1000.0, key="retirement_deduction")
    out_of_state_gain = st.number_input("Out-of-state gain (SC)", value=0.0, step=1000.0, key="out_of_state_gain")
    itemizing = st.checkbox("Itemizing deductions?", key="itemizing")
    itemized_amount = st.number_input("Itemized amount", value=0.0, step=1000.0, key="itemized_amount") if itemizing else 0.0
    taxable_cash_bal = st.number_input("After-tax cash", value=0.0, step=1000.0, key="taxable_cash_bal")
    taxable_brokerage_bal = st.number_input("After-tax brokerage", value=0.0, step=1000.0, key="taxable_brokerage_bal")
    brokerage_gain_pct = st.slider("Brokerage sale gain %", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="brokerage_gain_pct")
    pretax_bal = st.number_input("Pre-tax (IRA/401k) current", value=0.0, step=1000.0, key="pretax_bal")
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
        "cap_gain_loss": float(cap_gain_loss), "other_income": float(other_income),
        "adjustments": float(adjustments), "dependents": int(dependents),
        "filing_status": filing_status, "filer_65_plus": bool(filer_65_plus), "spouse_65_plus": bool(spouse_65_plus),
        "retirement_deduction": float(retirement_deduction), "out_of_state_gain": float(out_of_state_gain),
        "itemizing": bool(itemizing), "itemized_amount": float(itemized_amount),
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
    st.subheader("Base Tax Year Analysis")
    if st.button("Calculate Base Taxes", type="primary"):
        base_inputs = current_inputs(); base_assets = current_assets()
        res = compute_case(base_inputs)
        st.session_state.base_inputs = base_inputs; st.session_state.base_results = res; st.session_state.assets = base_assets
        st.session_state.gross_from_needs = None; st.session_state.last_solved_results = None
    if st.session_state.base_results:
        r = st.session_state.base_results
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Federal (modeled)"); st.write(f"AGI: {money(r['agi'])}"); st.metric("Federal Tax", money(r["fed_tax"]), f"{r['effective_fed']}%")
        with colB:
            st.markdown("### South Carolina (modeled)"); st.metric("SC Tax", money(r["sc_tax"]), f"{r['effective_sc']}%")
        st.divider(); st.markdown("### Cashflow View"); st.metric("Net After Tax", money(r["net_after_tax"]))

with tab2:
    if st.session_state.base_results:
        net_needed = st.number_input("Net income needed", min_value=0.0, step=1000.0)
        source = st.radio("Preferred source", ["Taxable – Cash", "Taxable – Brokerage", "Pre-Tax – IRA/401k", "Annuity", "Roth", "Life Insurance (loan)"])
        if st.button("Calculate Income Needs", type="primary"):
            extra, base_case, solved, solved_assets, solved_inputs = solve_gross_up_with_assets(
                st.session_state.base_inputs, st.session_state.assets, source, float(brokerage_gain_pct), float(net_needed), False)
            if extra:
                st.success(f"Withdrawal: {money(extra)}")
                st.session_state.last_solved_results = solved; st.session_state.last_solved_inputs = solved_inputs
                st.session_state.last_solved_assets = solved_assets; st.session_state.last_net_needed = net_needed
                st.session_state.last_source = source; st.session_state.gross_from_needs = solved["spendable_gross"]

with tab3:
    st.subheader("Wealth Projection")
    if st.session_state.assets is None: st.warning("Run Base Tax Estimator first.")
    else:
        a0 = st.session_state.assets
        colL, colR = st.columns(2)
        with colL:
            spending_goal = st.number_input("Annual Spending Goal (After-Tax)", value=125000.0, step=5000.0)
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

        if st.button("Run Wealth Projection", type="primary"):
            curr_tax = a0["taxable"]["cash"] + a0["taxable"]["brokerage"]
            total_pre = a0["pretax"]["balance"]
            p_filer = float(pretax_balance_filer_prior); p_spouse = float(pretax_balance_spouse_prior) if "joint" in filing_status.lower() else 0.0
            ratio_filer = p_filer / (p_filer + p_spouse) if (p_filer + p_spouse) > 0 else 1.0
            curr_pre_filer = total_pre * ratio_filer; curr_pre_spouse = total_pre * (1.0 - ratio_filer)
            curr_roth = a0["taxfree"]["roth"]; curr_life = a0["taxfree"]["life_cash_value"]
            curr_ann = a0["annuity"]["value"]; curr_ann_basis = a0["annuity"]["basis"]
            rows = []
            for i in range(int(years)):
                yr = int(start_year) + i; age_f = current_age_filer + i
                age_s = (current_age_spouse + i) if current_age_spouse else None
                inf_factor = (1 + inflation) ** i
                inflated_spend = spending_goal * inf_factor
                ss_now = gross_ss_total * inf_factor; pen_now = taxable_pensions_total * ((1 + pension_cola) ** i)
                fixed_gross = ss_now + pen_now
                inv_income = float(interest_taxable) + float(total_ordinary_dividends) + float(cap_gain_loss)
                curr_tax += inv_income
                rmd_f = compute_rmd_uniform_start73(curr_pre_filer, age_f)
                rmd_s = compute_rmd_uniform_start73(curr_pre_spouse, age_s)
                rmd_val = rmd_f + rmd_s; curr_pre_filer -= rmd_f; curr_pre_spouse -= rmd_s
                base_inputs_yr = {
                    "wages": 0.0, "gross_ss": ss_now, "taxable_pensions": pen_now,
                    "rmd_amount": rmd_val, "total_ordinary_dividends": inv_income,
                    "taxable_ira": 0.0, "tax_exempt_interest": 0.0, "interest_taxable": 0.0, "qualified_dividends": 0.0,
                    "cap_gain_loss": 0.0, "other_income": 0.0, "adjustments": 0.0,
                    "filing_status": filing_status, "filer_65_plus": age_f >= 65, "spouse_65_plus": (age_s >= 65 if age_s else False),
                    "dependents": 0, "retirement_deduction": float(retirement_deduction) * inf_factor,
                    "out_of_state_gain": 0.0, "itemizing": False, "itemized_amount": 0.0
                }
                base_res = compute_case_cached(_serialize_inputs_for_cache(base_inputs_yr), inf_factor)
                net_base = base_res["net_after_tax"]
                gap = inflated_spend - net_base
                wd_taxable = 0.0; wd_pre = 0.0; wd_taxfree = 0.0; wd_deferred = 0.0
                total_tax_paid = base_res["total_tax"] + base_res["medicare_premiums"]
                if gap > 0:
                    for bucket in spending_order:
                        if gap <= 0: break
                        if bucket == "Taxable":
                            pull = min(gap, curr_tax); curr_tax -= pull; wd_taxable += pull; gap -= pull
                        elif bucket == "Tax-Free":
                            pull = min(gap, curr_roth); curr_roth -= pull; wd_taxfree += pull; gap -= pull
                            if gap > 0:
                                pull = min(gap, curr_life); curr_life -= pull; wd_taxfree += pull; gap -= pull
                        elif bucket == "Pre-Tax" or bucket == "Tax-Deferred":
                            marg_est = 0.25; needed_gross = gap / (1 - marg_est)
                            if bucket == "Pre-Tax":
                                avail = curr_pre_filer + curr_pre_spouse; pull = min(needed_gross, avail)
                                if avail > 0: curr_pre_filer -= pull * (curr_pre_filer/avail); curr_pre_spouse -= pull * (curr_pre_spouse/avail)
                                wd_pre += pull; gap -= pull * (1 - marg_est); total_tax_paid += pull * marg_est
                            elif bucket == "Tax-Deferred":
                                pull = min(needed_gross, curr_ann); curr_ann -= pull; wd_deferred += pull
                                gap -= pull * (1 - marg_est); total_tax_paid += pull * marg_est
                else: curr_tax += abs(gap)
                curr_tax *= (1 + r_taxable); curr_pre_filer *= (1 + r_pretax); curr_pre_spouse *= (1 + r_pretax)
                curr_roth *= (1 + r_roth); curr_ann *= (1 + r_annuity); curr_life *= (1 + r_life)
                total_wealth = curr_tax + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
                ann_gains_val = max(0.0, curr_ann - curr_ann_basis)
                heir_tax = ((curr_pre_filer + curr_pre_spouse) * heir_tax_rate) + (ann_gains_val * heir_tax_rate)
                at_wealth = total_wealth - heir_tax
                total_income_disp = fixed_gross + inv_income + rmd_val + wd_taxable + wd_pre + wd_taxfree + wd_deferred
                rows.append({
                    "Year": yr, "Age": age_f, "Spending": round(inflated_spend, 0),
                    "Fixed Inc": round(fixed_gross, 0), "Inv Inc": round(inv_income, 0), "RMD": round(rmd_val, 0),
                    "W/D Taxable": round(wd_taxable, 0), "W/D Pre-Tax": round(wd_pre, 0),
                    "W/D Tax-Free": round(wd_taxfree, 0), "W/D Annuity": round(wd_deferred, 0),
                    "Total Income": round(total_income_disp, 0), "Total Tax": round(total_tax_paid, 0),
                    "Total Wealth": round(total_wealth, 0), "Estate (Net)": round(at_wealth, 0)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if len(rows) > 0:
                st.markdown("### Projection Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Final Wealth", f"${rows[-1]['Total Wealth']:,.0f}")
                with col2: st.metric("Final Estate (Net)", f"${rows[-1]['Estate (Net)']:,.0f}")
                with col3: st.metric("Total Taxes Paid", f"${sum(r['Total Tax'] for r in rows):,.0f}")
                with col4: st.metric("Final Year Spending", f"${rows[-1]['Spending']:,.0f}")

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
            pf = float(pretax_balance_filer_prior)
            ps = float(pretax_balance_spouse_prior) if "joint" in filing_status.lower() else 0.0
            ratio = pf / (pf + ps) if (pf + ps) > 0 else 1.0
            return {
                "spending_goal": opt_spending_goal, "start_year": int(opt_start_year),
                "years": int(opt_years), "inflation": float(inflation),
                "pension_cola": float(pension_cola), "heir_tax_rate": opt_heir_rate,
                "r_taxable": r_taxable, "r_pretax": r_pretax, "r_roth": r_roth,
                "r_annuity": r_annuity, "r_life": r_life,
                "gross_ss_total": gross_ss_total, "taxable_pensions_total": taxable_pensions_total,
                "inv_income_base": float(interest_taxable) + float(total_ordinary_dividends) + float(cap_gain_loss),
                "retirement_deduction": float(retirement_deduction), "filing_status": filing_status,
                "current_age_filer": ca_filer, "current_age_spouse": ca_spouse,
                "pretax_filer_ratio": ratio, "brokerage_gain_pct": float(brokerage_gain_pct),
                "ann_basis_start": a0["annuity"]["basis"],
            }

        st.divider()
        st.markdown("### Phase 1: Optimal Spending Strategy")
        st.info("Tests all 24 waterfall orderings (no Roth conversions) to find the spending order that maximizes after-tax estate.")

        if st.button("Run Phase 1 - Find Best Spending Order", type="primary", key="run_phase1"):
            params = _build_opt_params()
            buckets = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
            all_orders = list(permutations(buckets))
            results = []
            best_details = None
            best_estate = -float("inf")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, order in enumerate(all_orders):
                result = run_optimized_projection(a0, params, list(order))
                estate = result["after_tax_estate"]
                results.append({
                    "order": list(order),
                    "waterfall": " -> ".join(order),
                    "after_tax_estate": estate,
                    "total_wealth": result["total_wealth"],
                    "total_taxes": result["total_taxes"],
                    "final_cash": result["final_cash"],
                    "final_brokerage": result["final_brokerage"],
                    "final_pretax": result["final_pretax"],
                    "final_roth": result["final_roth"],
                    "final_annuity": result["final_annuity"],
                    "final_life": result["final_life"],
                })
                if estate > best_estate:
                    best_estate = estate
                    best_details = result["year_details"]
                progress_bar.progress((idx + 1) / len(all_orders))
                status_text.text(f"Testing order {idx + 1} of {len(all_orders)}...")

            progress_bar.empty()
            status_text.empty()

            results.sort(key=lambda x: x["after_tax_estate"], reverse=True)
            st.session_state.phase1_results = results
            st.session_state.phase1_best_order = results[0]["order"]
            st.session_state.phase1_best_details = best_details
            st.session_state.phase1_params = params
            st.session_state.phase2_results = None
            st.session_state.phase2_best_details = None

        if st.session_state.phase1_results:
            p1 = st.session_state.phase1_results
            best = p1[0]
            worst = p1[-1]
            diff = best["after_tax_estate"] - worst["after_tax_estate"]

            st.success(f"Best Spending Order: **{best['waterfall']}**")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Best Estate (After-Tax)", f"${best['after_tax_estate']:,.0f}")
            with col2: st.metric("Worst Estate (After-Tax)", f"${worst['after_tax_estate']:,.0f}")
            with col3: st.metric("Difference", f"${diff:,.0f}",
                                 delta=f"{(diff/worst['after_tax_estate'])*100:.1f}%" if worst['after_tax_estate'] > 0 else "N/A")

            with st.expander("All 24 Waterfall Rankings"):
                df_p1 = pd.DataFrame(p1)
                df_p1.index = range(1, len(df_p1) + 1)
                df_p1.index.name = "Rank"
                display_cols = ["waterfall", "after_tax_estate", "total_wealth", "total_taxes",
                                "final_cash", "final_brokerage", "final_pretax", "final_roth", "final_annuity", "final_life"]
                df_show = df_p1[display_cols].copy()
                for c in display_cols[1:]:
                    df_show[c] = df_show[c].apply(lambda x: f"${x:,.0f}")
                df_show.columns = ["Waterfall", "After-Tax Estate", "Total Wealth", "Total Taxes",
                                   "Final Cash", "Final Brokerage", "Final Pre-Tax", "Final Roth", "Final Annuity", "Final Life"]
                st.dataframe(df_show, use_container_width=True)

            if st.session_state.phase1_best_details:
                with st.expander("Year-by-Year Detail (Best Order)"):
                    st.dataframe(pd.DataFrame(st.session_state.phase1_best_details), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Phase 2: Roth Conversion Layering")

        if st.session_state.phase1_best_order is None:
            st.warning("Run Phase 1 first to establish the optimal spending order.")
        else:
            winning_order = st.session_state.phase1_best_order
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

                for idx, (strategy, strategy_name) in enumerate(strategies):
                    result = run_optimized_projection(
                        a0, params, winning_order,
                        conversion_strategy=strategy,
                        target_agi=target_agi_input,
                        stop_conversion_age=opt_stop_age,
                        conversion_years_limit=opt_conv_years,
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
                                ("W/D Brokerage", yr.get("W/D Brokerage", 0)),
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
            st.markdown("### Conversion Opportunity")
            if conversion_room <= 0:
                st.warning(f"You are already at or above the target. Current: {money(current_level)}, Target: {money(target_amount)}")
            else:
                st.success(f"**Roth Conversion Room: {money(conversion_room)}**")
                if actual_conversion < conversion_room: st.info(f"Note: You only have {money(available_pretax)} available in pre-tax accounts.")
                conv_inputs = dict(solved_inp); conv_inputs["taxable_ira"] = float(conv_inputs.get("taxable_ira", 0.0)) + actual_conversion
                conv_res = compute_case(conv_inputs)
                additional_tax = conv_res["total_tax"] - solved_res["total_tax"]
                effective_rate = (additional_tax / actual_conversion * 100) if actual_conversion > 0 else 0
                st.markdown("### Tax Impact Analysis")
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Conversion Amount", money(actual_conversion)); st.metric("Amount to Roth", money(actual_conversion))
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
                display_cashflow_comparison(solved_inp, solved_res, conv_inputs, conv_res, net_needed=net_needed_val, roth_conversion=actual_conversion, title_before="Before Conversion", title_after="After Conversion")

st.divider()
st.header("Cash Flow Summary (Income vs Expenses)")
if st.session_state.last_solved_results is not None and st.session_state.last_solved_inputs is not None:
    r = st.session_state.last_solved_results; inp = st.session_state.last_solved_inputs
    net_needed_bottom = float(st.session_state.last_net_needed or 0.0)
    income_rows = [
        ("Wages", float(inp["wages"])), ("Social Security (gross)", float(inp["gross_ss"])),
        ("Pensions (cash)", float(inp["taxable_pensions"])), ("Dividends (cash)", float(inp["total_ordinary_dividends"])),
        ("Pre-tax dist", float(inp["taxable_ira"]) + float(inp["rmd_amount"])), ("Other taxable", float(inp["other_income"])),
        ("Tax-free cash", float(inp.get("cashflow_taxfree", 0.0))), ("Brokerage proceeds", float(inp.get("brokerage_proceeds", 0.0))),
        ("Annuity proceeds", float(inp.get("annuity_proceeds", 0.0))),
    ]
    total_inc = sum(x[1] for x in income_rows)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income (Cash Received)")
        st.dataframe([{"Income": k, "Amount": money(v)} for k, v in income_rows if v > 0], use_container_width=True, hide_index=True)
        st.metric("Total Income", money(total_inc))
    with col2:
        st.subheader("Expenses & Outflows")
        out_rows = [("Living expenses", net_needed_bottom), ("Medicare", float(r["medicare_premiums"])), ("Taxes", float(r["total_tax"]))]
        st.dataframe([{"Outflow": k, "Amount": money(v)} for k, v in out_rows if v > 0], use_container_width=True, hide_index=True)
        st.metric("Total Outflows", money(sum(x[1] for x in out_rows)))
elif st.session_state.base_results is not None:
    r = st.session_state.base_results; st.metric("Taxes (Base Case)", money(r["total_tax"]))