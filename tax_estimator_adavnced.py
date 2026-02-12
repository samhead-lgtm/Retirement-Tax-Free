import streamlit as st
import pandas as pd
import datetime as dt
from functools import lru_cache
from itertools import permutations

st.set_page_config(page_title="RTF Tax + Wealth Engine (Master)", layout="wide")

DEFAULT_STATE = {
    "base_results": None, "base_inputs": None, "assets": None,
    "gross_from_needs": None, "last_net_needed": None, "last_taxes_paid_by_cash": None,
    "last_source": None, "last_withdrawal_proceeds": None,
    "last_solved_results": None, "last_solved_inputs": None, "last_solved_assets": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
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

# ============================================================
# OPTIMIZER FUNCTION
# ============================================================
def run_optimized_projection(
    initial_assets, params, spending_order, conversion_strategy, target_agi, 
    stop_conversion_age, conversion_years_limit
):
    """
    Run a full projection with given waterfall order and conversion strategy.
    Returns final after-tax estate and total taxes paid.
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
    inv_income_base = params["inv_income_base"]
    retirement_deduction = params["retirement_deduction"]
    filing_status = params["filing_status"]
    current_age_filer = params["current_age_filer"]
    current_age_spouse = params["current_age_spouse"]
    pretax_filer_ratio = params["pretax_filer_ratio"]
    brokerage_gain_pct = params["brokerage_gain_pct"]
    ann_basis_start = params["ann_basis_start"]

    # Initialize balances
    curr_tax = initial_assets["taxable"]["cash"] + initial_assets["taxable"]["brokerage"]
    total_pre = initial_assets["pretax"]["balance"]
    curr_pre_filer = total_pre * pretax_filer_ratio
    curr_pre_spouse = total_pre * (1.0 - pretax_filer_ratio)
    curr_roth = initial_assets["taxfree"]["roth"]
    curr_life = initial_assets["taxfree"]["life_cash_value"]
    curr_ann = initial_assets["annuity"]["value"]
    curr_ann_basis = ann_basis_start

    total_taxes_paid = 0.0
    total_converted = 0.0

    for i in range(years):
        yr = start_year + i
        age_f = current_age_filer + i
        age_s = (current_age_spouse + i) if current_age_spouse else None
        inf_factor = (1 + inflation) ** i

        inflated_spend = spending_goal * inf_factor
        ss_now = gross_ss_total * inf_factor
        pen_now = taxable_pensions_total * ((1 + pension_cola) ** i)
        
        # Investment income added to taxable
        inv_income = inv_income_base
        curr_tax += inv_income

        # RMDs
        rmd_f = compute_rmd_uniform_start73(curr_pre_filer, age_f)
        rmd_s = compute_rmd_uniform_start73(curr_pre_spouse, age_s)
        rmd_val = rmd_f + rmd_s
        curr_pre_filer -= rmd_f
        curr_pre_spouse -= rmd_s

        # Calculate base taxable income before spending withdrawals
        base_taxable_income = ss_now + pen_now + rmd_val + inv_income

        # Determine conversion amount based on strategy
        conversion_this_year = 0.0
        if i < conversion_years_limit and age_f < stop_conversion_age:
            if conversion_strategy == "fill_to_target":
                # Fill to target AGI
                room = max(0.0, target_agi * inf_factor - base_taxable_income)
                conversion_this_year = min(room, curr_pre_filer + curr_pre_spouse)
            else:
                # Fixed amount
                conversion_this_year = min(float(conversion_strategy), curr_pre_filer + curr_pre_spouse)

        # Apply conversion from pre-tax to Roth
        if conversion_this_year > 0:
            avail_pre = curr_pre_filer + curr_pre_spouse
            if avail_pre > 0:
                filer_share = curr_pre_filer / avail_pre
                curr_pre_filer -= conversion_this_year * filer_share
                curr_pre_spouse -= conversion_this_year * (1 - filer_share)
            curr_roth += conversion_this_year
            total_converted += conversion_this_year

        # Build tax inputs for the year (before spending withdrawals)
        tax_inputs_yr = {
            "wages": 0.0, "gross_ss": ss_now, "taxable_pensions": pen_now,
            "rmd_amount": rmd_val, "total_ordinary_dividends": inv_income,
            "taxable_ira": conversion_this_year,  # Conversion counts as IRA distribution
            "tax_exempt_interest": 0.0, "interest_taxable": 0.0, "qualified_dividends": 0.0,
            "cap_gain_loss": 0.0, "other_income": 0.0, "adjustments": 0.0,
            "filing_status": filing_status, "filer_65_plus": age_f >= 65,
            "spouse_65_plus": (age_s >= 65 if age_s else False),
            "dependents": 0, "retirement_deduction": retirement_deduction * inf_factor,
            "out_of_state_gain": 0.0, "itemizing": False, "itemized_amount": 0.0,
            "ordinary_tax_only": 0.0, "cashflow_taxfree": 0.0, 
            "brokerage_proceeds": 0.0, "annuity_proceeds": 0.0
        }

        # Calculate base taxes and net income
        base_res = compute_case(tax_inputs_yr, inf_factor)
        base_net = base_res["net_after_tax"]

        # Gap between spending need and net income from fixed sources
        gap = inflated_spend - base_net
        
        wd_taxable = 0.0; wd_pre = 0.0; wd_taxfree = 0.0; wd_deferred = 0.0
        additional_tax = 0.0

        if gap > 0:
            for bucket in spending_order:
                if gap <= 0: break
                
                if bucket == "Taxable":
                    pull = min(gap, curr_tax)
                    curr_tax -= pull
                    wd_taxable += pull
                    gap -= pull
                    
                elif bucket == "Tax-Free":
                    # Roth first
                    pull = min(gap, curr_roth)
                    curr_roth -= pull
                    wd_taxfree += pull
                    gap -= pull
                    # Then life insurance
                    if gap > 0:
                        pull = min(gap, curr_life)
                        curr_life -= pull
                        wd_taxfree += pull
                        gap -= pull
                        
                elif bucket == "Pre-Tax":
                    # Gross up for taxes
                    marg_est = 0.25
                    needed_gross = gap / (1 - marg_est)
                    avail = curr_pre_filer + curr_pre_spouse
                    pull = min(needed_gross, avail)
                    if avail > 0:
                        filer_share = curr_pre_filer / avail
                        curr_pre_filer -= pull * filer_share
                        curr_pre_spouse -= pull * (1 - filer_share)
                    wd_pre += pull
                    additional_tax += pull * marg_est
                    gap -= pull * (1 - marg_est)
                    
                elif bucket == "Tax-Deferred":
                    # Annuity - LIFO (gains first)
                    marg_est = 0.25
                    needed_gross = gap / (1 - marg_est)
                    pull = min(needed_gross, curr_ann)
                    curr_ann -= pull
                    wd_deferred += pull
                    additional_tax += pull * marg_est
                    gap -= pull * (1 - marg_est)
        else:
            # Surplus goes to taxable
            curr_tax += abs(gap)

        # Total tax for the year
        year_tax = base_res["total_tax"] + base_res["medicare_premiums"] + additional_tax
        total_taxes_paid += year_tax

        # Pay taxes from taxable account
        curr_tax -= year_tax

        # Apply growth
        curr_tax *= (1 + r_taxable)
        curr_pre_filer *= (1 + r_pretax)
        curr_pre_spouse *= (1 + r_pretax)
        curr_roth *= (1 + r_roth)
        curr_ann *= (1 + r_annuity)
        curr_life *= (1 + r_life)

    # Final calculations
    total_wealth = curr_tax + curr_pre_filer + curr_pre_spouse + curr_roth + curr_ann + curr_life
    ann_gains = max(0.0, curr_ann - curr_ann_basis)
    heir_tax = (curr_pre_filer + curr_pre_spouse) * heir_tax_rate + ann_gains * heir_tax_rate
    after_tax_estate = total_wealth - heir_tax

    return {
        "after_tax_estate": after_tax_estate,
        "total_wealth": total_wealth,
        "total_taxes": total_taxes_paid,
        "total_converted": total_converted,
        "final_taxable": curr_tax,
        "final_pretax": curr_pre_filer + curr_pre_spouse,
        "final_roth": curr_roth,
        "final_annuity": curr_ann,
        "final_life": curr_life,
    }


st.title("RTF Tax + Income Needs + LT Projection + Roth")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Base Tax Estimator", "Income Needs", "Wealth Projection", "Roth Optimizer", "Roth Conversion Opportunity"])

with st.sidebar:
    st.header("Tax Year & Filing")
    current_year = dt.date.today().year
    tax_year = st.number_input("Tax year", min_value=2020, max_value=2100, value=2025, step=1)
    filing_status = st.selectbox("Filing Status", ["Single", "Married Filing Jointly", "Head of Household"])
    st.divider()
    st.header("Inflation / COLA")
    inflation = st.number_input("Inflation (also SS COLA)", value=0.030, step=0.005, format="%.3f")
    st.divider()
    st.header("DOBs (for RMD + SS not-yet)")
    enter_dobs = st.checkbox("Enter DOBs", value=True)
    if enter_dobs:
        filer_dob = st.date_input("Filer DOB", value=dt.date(1955, 1, 1))
        spouse_dob = st.date_input("Spouse DOB", value=dt.date(1955, 1, 1)) if "joint" in filing_status.lower() else None
    else: filer_dob = None; spouse_dob = None
    st.divider()
    st.header("Social Security")
    filer_ss_already = st.checkbox("Filer already receiving SS?", value=False)
    if filer_ss_already:
        filer_ss_current = st.number_input("Filer current annual SS", value=0.0, step=1000.0)
        filer_ss_start_year = st.number_input("Filer SS start year", value=int(tax_year), step=1)
        filer_ss_fra = 0.0; filer_ss_claim = "FRA"
    else:
        filer_ss_current = 0.0; filer_ss_start_year = 9999
        filer_ss_fra = st.number_input("Filer SS at FRA", value=0.0, step=1000.0)
        filer_ss_claim = st.selectbox("Filer claim choice", ["62", "FRA", "70"], index=1)
    if "joint" in filing_status.lower():
        spouse_ss_already = st.checkbox("Spouse already receiving SS?", value=False)
        if spouse_ss_already:
            spouse_ss_current = st.number_input("Spouse current annual SS", value=0.0, step=1000.0)
            spouse_ss_start_year = st.number_input("Spouse SS start year", value=int(tax_year), step=1)
            spouse_ss_fra = 0.0; spouse_ss_claim = "FRA"
        else:
            spouse_ss_current = 0.0; spouse_ss_start_year = 9999
            spouse_ss_fra = st.number_input("Spouse SS at FRA", value=0.0, step=1000.0)
            spouse_ss_claim = st.selectbox("Spouse claim choice", ["62", "FRA", "70"], index=1)
    else: spouse_ss_already = False; spouse_ss_current = 0.0; spouse_ss_start_year = 9999; spouse_ss_fra = 0.0; spouse_ss_claim = "FRA"
    st.divider()
    st.header("Pensions")
    pension_filer = st.number_input("Filer pension", value=0.0, step=1000.0)
    pension_spouse = st.number_input("Spouse pension", value=0.0, step=1000.0) if "joint" in filing_status.lower() else 0.0
    pension_cola = st.number_input("Pension COLA (%)", value=0.00, step=0.005, format="%.3f")
    st.divider()
    st.header("RMD Inputs")
    auto_rmd = st.checkbox("Auto-calculate RMD", value=True)
    pretax_balance_filer_prior = st.number_input("Filer prior-year 12/31 pre-tax balance", value=0.0, step=1000.0)
    pretax_balance_spouse_prior = st.number_input("Spouse prior-year 12/31 pre-tax balance", value=0.0, step=1000.0) if "joint" in filing_status.lower() else 0.0
    baseline_pretax_distributions = st.number_input("Baseline pre-tax distributions", value=0.0, step=1000.0)
    rmd_manual = st.number_input("RMD manual override", value=0.0, step=1000.0) if not auto_rmd else 0.0
    st.divider()
    st.header("Other Income / Deductions / Assets")
    wages = st.number_input("Wages", value=0.0, step=1000.0)
    tax_exempt_interest = st.number_input("Tax-exempt interest", value=0.0, step=100.0)
    interest_taxable = st.number_input("Taxable interest", value=0.0, step=100.0)
    total_ordinary_dividends = st.number_input("Total ordinary dividends", value=0.0, step=100.0)
    qualified_dividends = st.number_input("Qualified dividends", value=0.0, max_value=total_ordinary_dividends, step=100.0)
    cap_gain_loss = st.number_input("Baseline net cap gain/(loss)", value=0.0, step=1000.0)
    other_income = st.number_input("Other taxable income", value=0.0, step=500.0)
    filer_65_plus = st.checkbox("Filer age 65+")
    spouse_65_plus = st.checkbox("Spouse age 65+") if "joint" in filing_status.lower() else False
    adjustments = st.number_input("Adjustments to income", value=0.0, step=500.0)
    dependents = st.number_input("Dependents", value=0, step=1)
    retirement_deduction = st.number_input("SC retirement deduction", value=0.0, step=1000.0)
    out_of_state_gain = st.number_input("Out-of-state gain (SC)", value=0.0, step=1000.0)
    itemizing = st.checkbox("Itemizing deductions?")
    itemized_amount = st.number_input("Itemized amount", value=0.0, step=1000.0) if itemizing else 0.0
    taxable_cash_bal = st.number_input("After-tax cash", value=0.0, step=1000.0)
    taxable_brokerage_bal = st.number_input("After-tax brokerage", value=0.0, step=1000.0)
    brokerage_gain_pct = st.slider("Brokerage sale gain %", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    pretax_bal = st.number_input("Pre-tax (IRA/401k) current", value=0.0, step=1000.0)
    roth_bal = st.number_input("Roth balance", value=0.0, step=1000.0)
    life_cash_value = st.number_input("Life insurance CV", value=0.0, step=1000.0)
    annuity_value = st.number_input("Annuity value", value=0.0, step=1000.0)
    annuity_basis = st.number_input("Annuity cost basis", value=0.0, step=1000.0)
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

# TAB 4 - ROTH OPTIMIZER
with tab4:
    st.subheader("Roth Conversion Optimizer")
    st.write("Tests all 24 waterfall orderings × multiple conversion strategies to find the optimal approach.")
    
    if st.session_state.assets is None:
        st.warning("Run Base Tax Estimator first.")
    else:
        a0 = st.session_state.assets
        
        col1, col2 = st.columns(2)
        with col1:
            opt_spending_goal = st.number_input("Annual Spending Goal (After-Tax)", value=125000.0, step=5000.0, key="opt_spend")
            opt_years = st.number_input("Years to Project", min_value=5, max_value=50, value=30, step=1, key="opt_years")
            opt_start_year = st.number_input("Start Year", min_value=2024, max_value=2100, value=max(2025, int(tax_year)), step=1, key="opt_start")
        with col2:
            opt_heir_rate = st.number_input("Heir Tax Rate (%)", value=25.0, step=1.0, key="opt_heir") / 100
            opt_stop_age = st.number_input("Stop Conversions at Age", min_value=60, max_value=100, value=75, step=1, key="opt_stop")
            opt_conv_years = st.number_input("Max Years of Conversions", min_value=1, max_value=30, value=15, step=1, key="opt_conv_years")
        
        st.markdown("#### Target Income Thresholds to Test")
        is_joint = "joint" in filing_status.lower()
        col1, col2 = st.columns(2)
        with col1:
            if is_joint:
                st.write("• 22% bracket top: $206,700")
                st.write("• 24% bracket top: $394,600")
                st.write("• IRMAA threshold: $212,000")
            else:
                st.write("• 22% bracket top: $103,350")
                st.write("• 24% bracket top: $197,300")
                st.write("• IRMAA threshold: $106,000")
        with col2:
            target_agi_input = st.number_input("Custom Target AGI (for 'fill to bracket')", 
                value=212000.0 if is_joint else 106000.0, step=10000.0, key="opt_target")
        
        st.markdown("#### Conversion Amounts to Test ($)")
        conv_amounts_str = st.text_input("Enter amounts separated by commas", 
            value="0, 25000, 50000, 75000, 100000, 150000, 200000", key="opt_amounts")
        
        include_fill_to_bracket = st.checkbox("Also test 'Fill to Target AGI' strategy", value=True)
        
        if st.button("Run Optimizer", type="primary"):
            # Parse conversion amounts
            try:
                conv_amounts = [float(x.strip()) for x in conv_amounts_str.split(",")]
            except:
                conv_amounts = [0, 25000, 50000, 75000, 100000]
            
            # Build conversion strategies to test
            strategies = [(amt, f"${amt:,.0f}/yr") for amt in conv_amounts]
            if include_fill_to_bracket:
                strategies.append(("fill_to_target", f"Fill to ${target_agi_input:,.0f}"))
            
            # All 24 waterfall permutations
            buckets = ["Taxable", "Pre-Tax", "Tax-Free", "Tax-Deferred"]
            all_orders = list(permutations(buckets))
            
            # Build params
            current_age_filer = age_at_date(filer_dob, dt.date.today()) if filer_dob else 70
            current_age_spouse = age_at_date(spouse_dob, dt.date.today()) if (spouse_dob and "joint" in filing_status.lower()) else None
            p_filer = float(pretax_balance_filer_prior)
            p_spouse = float(pretax_balance_spouse_prior) if "joint" in filing_status.lower() else 0.0
            ratio_filer = p_filer / (p_filer + p_spouse) if (p_filer + p_spouse) > 0 else 1.0
            
            params = {
                "spending_goal": opt_spending_goal,
                "start_year": int(opt_start_year),
                "years": int(opt_years),
                "inflation": float(inflation),
                "pension_cola": float(pension_cola),
                "heir_tax_rate": opt_heir_rate,
                "r_taxable": r_taxable,
                "r_pretax": r_pretax,
                "r_roth": r_roth,
                "r_annuity": r_annuity,
                "r_life": r_life,
                "gross_ss_total": gross_ss_total,
                "taxable_pensions_total": taxable_pensions_total,
                "inv_income_base": float(interest_taxable) + float(total_ordinary_dividends) + float(cap_gain_loss),
                "retirement_deduction": float(retirement_deduction),
                "filing_status": filing_status,
                "current_age_filer": current_age_filer,
                "current_age_spouse": current_age_spouse,
                "pretax_filer_ratio": ratio_filer,
                "brokerage_gain_pct": float(brokerage_gain_pct),
                "ann_basis_start": a0["annuity"]["basis"],
            }
            
            # Run all combinations
            results = []
            total_scenarios = len(all_orders) * len(strategies)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scenario_count = 0
            for order in all_orders:
                for strategy, strategy_name in strategies:
                    result = run_optimized_projection(
                        a0, params, list(order), strategy, target_agi_input,
                        opt_stop_age, opt_conv_years
                    )
                    results.append({
                        "Waterfall": " → ".join(order),
                        "Conversion Strategy": strategy_name,
                        "After-Tax Estate": result["after_tax_estate"],
                        "Total Wealth": result["total_wealth"],
                        "Total Taxes": result["total_taxes"],
                        "Total Converted": result["total_converted"],
                        "Final Taxable": result["final_taxable"],
                        "Final Pre-Tax": result["final_pretax"],
                        "Final Roth": result["final_roth"],
                        "Final Annuity": result["final_annuity"],
                    })
                    scenario_count += 1
                    progress_bar.progress(scenario_count / total_scenarios)
                    status_text.text(f"Running scenario {scenario_count} of {total_scenarios}...")
            
            progress_bar.empty()
            status_text.empty()
            
            # Sort by after-tax estate (descending)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("After-Tax Estate", ascending=False).reset_index(drop=True)
            results_df.index = results_df.index + 1  # Rank starting at 1
            
            st.success(f"Tested {total_scenarios} scenarios!")
            
            # Show top 10
            st.markdown("### Top 10 Strategies")
            top_10 = results_df.head(10).copy()
            for col in ["After-Tax Estate", "Total Wealth", "Total Taxes", "Total Converted", 
                       "Final Taxable", "Final Pre-Tax", "Final Roth", "Final Annuity"]:
                top_10[col] = top_10[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_10, use_container_width=True)
            
            # Show worst 5 for comparison
            st.markdown("### Bottom 5 Strategies (for comparison)")
            bottom_5 = results_df.tail(5).copy()
            for col in ["After-Tax Estate", "Total Wealth", "Total Taxes", "Total Converted",
                       "Final Taxable", "Final Pre-Tax", "Final Roth", "Final Annuity"]:
                bottom_5[col] = bottom_5[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(bottom_5, use_container_width=True)
            
            # Summary stats
            st.markdown("### Analysis")
            best = results_df.iloc[0]
            worst = results_df.iloc[-1]
            diff = best["After-Tax Estate"] - worst["After-Tax Estate"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Strategy Estate", f"${best['After-Tax Estate']:,.0f}")
            with col2:
                st.metric("Worst Strategy Estate", f"${worst['After-Tax Estate']:,.0f}")
            with col3:
                st.metric("Difference (Best vs Worst)", f"${diff:,.0f}", delta=f"{(diff/worst['After-Tax Estate'])*100:.1f}%")
            
            st.markdown(f"**Best Approach:** {best['Waterfall']} with {best['Conversion Strategy']}")

# TAB 5 - Roth Conversion Opportunity
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