import streamlit as st
import pandas as pd
import datetime as dt
import json, os
import numpy as np
from itertools import permutations


DEFAULT_STATE = {
    "projection_results": None,
    "retire_projection": None,
    "num_future_expenses": 0,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Client Profile Save / Load ----------
_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retirement_profiles")
os.makedirs(_PROFILE_DIR, exist_ok=True)

_PROFILE_KEYS = [
    "filer_dob", "spouse_dob", "filing_status",
    "ret_age", "spouse_ret_age", "life_exp",
    "inflation", "pre_ret_return", "post_ret_return", "salary_growth", "state_tax_rate",
    "salary_filer", "salary_spouse", "other_income", "other_income_years",
    "other_income_tax_free", "other_income_inflation", "other_income_recipient", "living_exp_tab1",
    "ssdi_filer", "ssdi_spouse",
    "ss_claim_f", "ss_ov_f", "ss_claim_s", "ss_ov_s",
    "curr_401k_f", "curr_401k_s", "curr_trad_ira_f", "curr_trad_ira_s",
    "curr_roth_ira_f", "curr_roth_ira_s", "curr_roth_401k",
    "curr_taxable", "taxable_basis", "curr_hsa", "curr_cash",
    "inh_ira_f", "inh_rule_f", "inh_yrs_f", "inh_rmd_req_f", "inh_add_f",
    "inh_ira_s", "inh_rule_s", "inh_yrs_s", "inh_rmd_req_s", "inh_add_s",
    "home_value", "home_appr", "mtg_balance", "mtg_rate", "mtg_pmt_monthly", "mtg_years",
    "pen_filer", "pen_filer_age", "pen_filer_has_cola", "pen_filer_cola_type", "pen_filer_cola",
    "pen_spouse", "pen_spouse_age", "pen_spouse_has_cola", "pen_spouse_cola_type", "pen_spouse_cola",
    "hsa_eligible", "max_401k_f", "c401k_f", "roth_pct_f",
    "ematch_rate_f", "ematch_upto_f",
    "backdoor_roth_f", "max_ira_f", "contrib_trad_ira", "contrib_roth_ira",
    "max_hsa", "contrib_hsa", "contrib_taxable",
    "max_401k_s", "c401k_s", "roth_pct_s",
    "ematch_rate_s", "ematch_upto_s",
    "backdoor_roth_s", "max_ira_s", "trad_ira_s", "roth_ira_s",
    "tax_efficient", "div_yield", "ann_cg_pct", "cash_int_rate", "reinvest_inv_income",
    "deficit_action", "itemize_ded",
    "property_tax", "medical_exp", "charitable",
    "surplus_dest",
    "ret_pct", "rr_so1", "rr_so2", "rr_so3", "rr_surplus_dest", "rr_heir_bracket",
    "ss_opt_life_exp",
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
    n = st.session_state.get("num_future_expenses", 0)
    if n > 0:
        data["_num_future_expenses"] = n
        for i in range(n):
            for pfx in ("fe_name_", "fe_amt_", "fe_start_", "fe_end_", "fe_infl_"):
                fk = f"{pfx}{i}"
                if fk in st.session_state:
                    data[fk] = st.session_state[fk]
    return data

def _apply_profile(data):
    for k, v in data.items():
        if k == "_num_future_expenses":
            st.session_state["num_future_expenses"] = v
        elif k in _DATE_KEYS and v is not None:
            st.session_state[k] = dt.date.fromisoformat(v)
        else:
            st.session_state[k] = v

def money(x): return f"${float(x):,.0f}"

def age_at_date(dob, asof):
    if dob is None: return None
    years = asof.year - dob.year
    if (asof.month, asof.day) < (dob.month, dob.day): years -= 1
    return years

# 2025 contribution limits
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

def ira_phase_out(magi, phase_range, full_limit):
    """Calculate allowed contribution after income phase-out."""
    low, high = phase_range
    if magi <= low:
        return full_limit
    if magi >= high:
        return 0.0
    # Pro-rata reduction, rounded up to nearest $10 per IRS rules
    reduction = full_limit * (magi - low) / (high - low)
    import math
    allowed = math.ceil((full_limit - reduction) / 10) * 10
    return max(0.0, min(full_limit, float(allowed)))

# --- FICA taxes ---
SS_WAGE_BASE_2025 = 176100
MEDICARE_ADDITIONAL_THRESHOLD = {"Married Filing Jointly": 250000, "Single": 200000, "Head of Household": 200000}

def calc_fica(wages_filer, wages_spouse, status, inf=1.0):
    """Calculate employee-side FICA (SS 6.2% + Medicare 1.45% + Additional Medicare 0.9%)."""
    wage_base = SS_WAGE_BASE_2025 * inf
    ss_filer = min(wages_filer, wage_base) * 0.062
    ss_spouse = min(wages_spouse, wage_base) * 0.062
    med_filer = wages_filer * 0.0145
    med_spouse = wages_spouse * 0.0145
    # Additional Medicare tax on combined wages over threshold
    threshold = MEDICARE_ADDITIONAL_THRESHOLD.get(status, 200000) * inf
    combined = wages_filer + wages_spouse
    add_med = max(0.0, combined - threshold) * 0.009
    return ss_filer + ss_spouse + med_filer + med_spouse + add_med

# --- SS estimation from income ---
SS_TAXABLE_MAX_2025 = SS_WAGE_BASE_2025

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


# --- Mortgage calculator helpers ---
def calc_mortgage_payment(balance, annual_rate, years):
    """Calculate annual P&I payment from balance, rate, years."""
    if balance <= 0 or years <= 0: return 0.0
    if annual_rate <= 0: return balance / years
    r = annual_rate / 12
    n = years * 12
    monthly = balance * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(monthly * 12, 0)

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

def calc_mortgage_years(balance, annual_rate, annual_payment):
    """Calculate years to payoff from balance, rate, annual payment."""
    import math
    if balance <= 0 or annual_payment <= 0: return 0
    if annual_rate <= 0: return max(0, math.ceil(balance / annual_payment))
    r = annual_rate / 12
    monthly_pmt = annual_payment / 12
    if monthly_pmt <= balance * r: return 99  # payment doesn't cover interest
    n = math.log(monthly_pmt / (monthly_pmt - balance * r)) / math.log(1 + r)
    return max(0, math.ceil(n / 12))

def ss_claim_factor(claim_age):
    """SS benefit as fraction of PIA based on claim age (FRA=67).
    Before FRA: 5/9 of 1% per month for first 36 months (ages 64-67),
                5/12 of 1% per month for additional months (ages 62-64).
    After FRA:  2/3 of 1% per month (8% per year) delayed retirement credits."""
    if claim_age <= 62: return 0.70
    if claim_age >= 70: return 1.24
    if claim_age >= 67:
        return 1.00 + (claim_age - 67) * 0.08
    months_early = (67 - claim_age) * 12
    if months_early <= 36:
        return 1.00 - months_early * (5 / 9 / 100)
    return 1.00 - 36 * (5 / 9 / 100) - (months_early - 36) * (5 / 12 / 100)


# --- Social Security Claiming Strategy Optimizer ---

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
            # No conversion params → runs accumulation without conversions
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
                # Use baseline accumulation (no conversions) — same as Tab 4
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

    # B) Fixed amount × duration grid — covers small totals to large
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

        # Fill-to-bracket strategies (high stop age — they self-taper when no room)
        for name, target in fill_targets:
            res = _run_strategy("fill_to_target", target, 85, defer_rmd)
            _append_result(f"{name}{defer_label}", res, defer_rmd,
                           conv_strategy_val="fill_to_target", conv_target_agi=target, conv_stop_age=85)

        # Fixed amount × duration strategies
        for name, amt, stop in fixed_grid:
            res = _run_strategy(amt, 0, stop, defer_rmd)
            _append_result(f"{name}{defer_label}", res, defer_rmd,
                           conv_strategy_val=amt, conv_target_agi=0, conv_stop_age=stop)

        # ── Retire-only variants: same strategies but NO pre-retirement conversions ──
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


def run_monte_carlo(run_fn, n_sims=500, mean_return=0.07, return_std=0.12,
                    n_years=30, seed=None):
    """Run Monte Carlo simulation by randomizing year-by-year returns.

    Args:
        run_fn: callable(return_sequence) -> dict with "estate" and "final_total" keys
        n_sims: number of simulations
        mean_return: mean annual return
        return_std: standard deviation of annual returns
        n_years: number of years to generate returns for
        seed: optional RNG seed for reproducibility

    Returns:
        dict with median_estate, p10, p25, p75, p90, mean_estate, success_rate, all_estates
    """
    rng = np.random.default_rng(seed)
    all_returns = rng.normal(mean_return, return_std, (n_sims, n_years))
    all_returns = np.maximum(all_returns, -0.50)

    all_estates = []
    for sim in range(n_sims):
        result = run_fn(all_returns[sim].tolist())
        all_estates.append(result["estate"])

    all_estates = np.array(all_estates)
    success_rate = float(np.mean(all_estates > 0))
    return {
        "median_estate": float(np.median(all_estates)),
        "p10": float(np.percentile(all_estates, 10)),
        "p25": float(np.percentile(all_estates, 25)),
        "p75": float(np.percentile(all_estates, 75)),
        "p90": float(np.percentile(all_estates, 90)),
        "mean_estate": float(np.mean(all_estates)),
        "success_rate": success_rate,
        "all_estates": all_estates.tolist(),
    }


# --- RMD (Required Minimum Distribution) ---
# SECURE 2.0: RMDs begin at age 73. Uniform Lifetime Table (IRS Pub 590-B).
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
    1. Calculate bracket space (top of current bracket − current taxable income).
    2. Recommend that amount each year so all distributions stay in the current bracket.
    3. If that's not enough to deplete by the deadline (10-year), increase just enough
       so the minimum number of years spill into the next bracket.
    4. For lifetime: recommend additional above min RMD to fill the current bracket,
       preventing balance growth that would push future RMDs into higher brackets.
    """
    import math
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
        For lifetime: additional above min RMD = bracket_fill − min_rmd (floored at 0).
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
        # For min-only, override: take only the minimum (simulate with top bracket so fill is huge, but it's capped at bal)
        # Actually, _simulate_bracket_fill with current bracket already takes bracket fill.
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

        # The recommended additional (year 1) = bracket_fill − min_rmd
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
                           f"your current bracket — consider a Roth conversion or other strategies.")
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
    # forced distribution ≈ the annual amount — i.e., evenly spread across all years.
    # Criterion: final year's balance (= its distribution) ≤ flat_annual.
    #   Too low  → final year has a big lump (need to take more each year)
    #   Too high → account depletes early (final year is tiny, wasting years)

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
            # Final year fits within the annual amount — try taking less to stretch
            hi = mid
            best_d = mid
        else:
            # Final year is too big — need to take more each year
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
        warning = "Balance is large relative to available bracket space — distributions will push into higher brackets regardless of strategy."

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

def calc_rmd(age, pretax_balance):
    """Calculate required minimum distribution for the year."""
    if age < 73 or pretax_balance <= 0:
        return 0.0
    divisor = RMD_TABLE.get(age, 2.0)
    return pretax_balance / divisor

# --- Federal tax engine (2025 base, inflation-adjusted) ---
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

_FED_BRACKET_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

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
                    filer_65=False, spouse_65=False, inf=1.0, state_rate=0.05):
    """Calculate federal + state tax for a retirement year. All inputs in nominal dollars.
    cap_gains: long-term capital gains + qualified dividends (0%/15%/20% brackets)
    ord_invest_income: cash interest and other ordinary investment income"""
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
    st_tax = calc_state_tax(fed_taxable, state_rate)
    marginal = get_marginal_fed_rate(ordinary_taxable, status, inf)
    return {"fed_tax": fed_tax, "state_tax": st_tax, "total_tax": fed_tax + st_tax,
            "agi": agi, "taxable_ss": taxable_ss, "fed_taxable": fed_taxable,
            "cap_gains_tax": fed_cg_tax, "marginal_rate": marginal}

# --- Retirement projection engine ---
def run_retirement_projection(balances, params, spending_order):
    bal_pt = balances["pretax"]
    bal_brokerage = balances.get("brokerage", balances["taxable"])
    bal_cash = balances.get("cash", 0.0)
    brokerage_basis = balances.get("brokerage_basis", bal_brokerage)
    bal_ro = balances["roth"]
    bal_hs = balances["hsa"]

    retire_age = params["retire_age"]
    life_exp = params["life_expectancy"]
    retire_year = params["retire_year"]
    inflation = params["inflation"]
    post_return = params["post_retire_return"]
    filing_status = params["filing_status"]
    state_rate = params["state_tax_rate"]
    living_expenses_yr1 = params["expenses_at_retirement"]
    ss_filer_fra = params["ss_filer_fra"]
    ss_spouse_fra = params["ss_spouse_fra"]
    ss_filer_claim_age = params["ss_filer_claim_age"]
    ss_spouse_claim_age = params["ss_spouse_claim_age"]
    ssdi_filer_flag = params.get("ssdi_filer", False)
    ssdi_spouse_flag = params.get("ssdi_spouse", False)
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
    fut_expenses = params.get("future_expenses", [])
    div_yield = params.get("dividend_yield", 0.015)
    cash_int_rate = params.get("cash_interest_rate", 0.04)
    surplus_dest = params.get("surplus_destination", "brokerage")
    heir_bracket_option = params.get("heir_bracket_option", "same")  # "same", "lower", "higher"
    ret_other_income = params.get("other_income", 0.0)
    ret_other_income_tax_free = params.get("other_income_tax_free", False)
    ret_other_income_inflation = params.get("other_income_inflation", False)
    ret_other_income_years = params.get("other_income_years", 0)
    ret_inherited_iras = params.get("inherited_iras", [])
    ret_iira_bals = [ira["balance"] for ira in ret_inherited_iras]
    ret_iira_rules = [ira["rule"] for ira in ret_inherited_iras]
    ret_iira_yrs = [ira["years_remaining"] for ira in ret_inherited_iras]
    ret_iira_ages = [ira["owner_age"] for ira in ret_inherited_iras]
    ret_iira_rmd_required = [ira.get("owner_was_taking_rmds", False) for ira in ret_inherited_iras]
    ret_iira_additional = [ira.get("additional_distribution", 0.0) for ira in ret_inherited_iras]

    return_sequence = params.get("return_sequence", None)

    # Roth conversion params
    conv_strategy = params.get("roth_conversion_strategy", "none")  # "none", numeric, or "fill_to_target"
    conv_target_agi = params.get("roth_conversion_target_agi", 0)
    conv_stop_age = params.get("roth_conversion_stop_age", 100)
    defer_first_rmd = params.get("defer_first_rmd", False)
    do_conversions = conv_strategy != "none" and conv_strategy != 0
    total_converted = 0.0
    deferred_rmd = 0.0

    total_taxes_paid = 0.0
    rows = []
    curr_home_val = home_val

    spouse_age_at_retire = params.get("spouse_age_at_retire")

    for i in range(life_exp - retire_age + 1):
        age = retire_age + i
        spouse_age_now = (spouse_age_at_retire + i) if spouse_age_at_retire else age
        year = retire_year + i
        inf_factor = (1 + inflation) ** i
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

        # SS: fra values are in retirement-year dollars; inf_factor continues COLA from there
        # SSDI: already receiving at 100% PIA, just apply inflation
        ss_filer = 0.0
        if ss_filer_fra > 0:
            if ssdi_filer_flag:
                ss_filer = ss_filer_fra * inf_factor
            elif age >= ss_filer_claim_age:
                ss_filer = ss_filer_fra * ss_claim_factor(ss_filer_claim_age) * inf_factor

        ss_spouse = 0.0
        if ss_spouse_fra > 0:
            if ssdi_spouse_flag:
                ss_spouse = ss_spouse_fra * inf_factor
            elif spouse_age_now >= ss_spouse_claim_age:
                ss_spouse = ss_spouse_fra * ss_claim_factor(ss_spouse_claim_age) * inf_factor

        gross_ss = max(0, ss_filer) + max(0, ss_spouse)

        # Pensions: each has its own start age and COLA
        pen_filer_income = 0.0
        if pen_filer > 0 and age >= pen_filer_start:
            yrs_receiving_f = age - pen_filer_start
            pen_filer_income = pen_filer * ((1 + pen_filer_cola) ** yrs_receiving_f) if pen_filer_cola > 0 else pen_filer

        pen_spouse_income = 0.0
        if pen_spouse > 0 and spouse_age_now >= pen_spouse_start:
            yrs_receiving_s = spouse_age_now - pen_spouse_start
            pen_spouse_income = pen_spouse * ((1 + pen_spouse_cola) ** yrs_receiving_s) if pen_spouse_cola > 0 else pen_spouse

        pen_income = pen_filer_income + pen_spouse_income

        # Investment income from taxable accounts (generated before withdrawals)
        yr_dividends = bal_brokerage * div_yield if bal_brokerage > 0 else 0.0
        yr_cash_interest = bal_cash * cash_int_rate if bal_cash > 0 else 0.0
        # Dividends are qualified -> cap gains rate; cash interest -> ordinary
        inv_cap_gains_income = yr_dividends
        inv_ordinary_income = yr_cash_interest

        # Other income (alimony, disability, etc.) — ends after N years from start
        yr_other_income = 0.0
        if ret_other_income > 0:
            if ret_other_income_years == 0 or i < ret_other_income_years:
                yr_other_income = ret_other_income * inf_factor if ret_other_income_inflation else ret_other_income

        fixed_cash = gross_ss + pen_income + yr_dividends + yr_cash_interest + yr_other_income

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

        # Roth conversion (after RMD, before waterfall)
        conversion_this_year = 0.0
        conv_tax_withheld = 0.0  # portion of conversion tax paid from the conversion itself
        conv_tax_total = 0.0  # total incremental tax from conversion
        if do_conversions and age < conv_stop_age:
            avail_pretax = max(0, bal_pt - wd_pretax)  # what's left after RMD
            if avail_pretax > 0:
                yr_taxable_other = yr_other_income if not ret_other_income_tax_free else 0.0
                if conv_strategy == "fill_to_target":
                    # Estimate current taxable income before conversion
                    base_income = wd_pretax + pen_income + yr_inherited_dist + inv_ordinary_income + inv_cap_gains_income + yr_taxable_other
                    est_taxable_ss = gross_ss * 0.85  # conservative: assume 85% taxable
                    room = max(0, conv_target_agi * inf_factor - base_income - est_taxable_ss)
                    conversion_this_year = min(room, avail_pretax)
                else:
                    conversion_this_year = min(float(conv_strategy), avail_pretax)

                # Tax on conversion: withhold from conversion if no other source to pay
                if conversion_this_year > 0:
                    _no_conv_pretax_inc = wd_pretax + pen_income + yr_inherited_dist + yr_taxable_other
                    _no_conv_tax = calc_year_taxes(gross_ss, _no_conv_pretax_inc,
                                                    inv_cap_gains_income, inv_ordinary_income,
                                                    filing_status, filer_65, spouse_65,
                                                    inf_factor, state_rate)["total_tax"]
                    _with_conv_tax = calc_year_taxes(gross_ss, _no_conv_pretax_inc + conversion_this_year,
                                                     inv_cap_gains_income, inv_ordinary_income,
                                                     filing_status, filer_65, spouse_65,
                                                     inf_factor, state_rate)["total_tax"]
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

        for iteration in range(20):
            # Compute cap gains: gains from brokerage sales + dividend income
            total_cap_gains = yr_cap_gains + inv_cap_gains_income
            yr_taxable_other = yr_other_income if not ret_other_income_tax_free else 0.0
            pretax_income = wd_pretax + pen_income + yr_inherited_dist + yr_taxable_other + conversion_this_year
            tax_result = calc_year_taxes(gross_ss, pretax_income, total_cap_gains,
                                         inv_ordinary_income, filing_status,
                                         filer_65, spouse_65, inf_factor, state_rate)
            taxes = tax_result["total_tax"]
            cash_needed = expenses + taxes - conv_tax_withheld  # withheld portion already paid from conversion
            wd_taxable = wd_cash + wd_brokerage
            cash_available = fixed_cash + wd_pretax + wd_taxable + wd_roth + wd_hsa
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
        pretax_income = wd_pretax + pen_income + yr_inherited_dist + yr_taxable_other + conversion_this_year
        tax_result = calc_year_taxes(gross_ss, pretax_income, total_cap_gains,
                                     inv_ordinary_income, filing_status,
                                     filer_65, spouse_65, inf_factor, state_rate)
        taxes = tax_result["total_tax"]
        wd_taxable = wd_cash + wd_brokerage

        # Surplus: income exceeds expenses + taxes → reinvest
        cash_available_final = fixed_cash + wd_pretax + wd_cash + wd_brokerage + wd_roth + wd_hsa
        cash_needed_final = expenses + taxes - conv_tax_withheld
        yr_surplus = max(0.0, cash_available_final - cash_needed_final)

        # Update balances
        bal_pt -= wd_pretax
        # Reduce brokerage basis proportionally to withdrawal
        if wd_brokerage > 0 and bal_brokerage > 0:
            basis_reduction = brokerage_basis * (wd_brokerage / bal_brokerage)
            brokerage_basis = max(0.0, brokerage_basis - basis_reduction)
        bal_brokerage -= wd_brokerage
        bal_cash -= wd_cash
        bal_ro -= wd_roth
        bal_hs -= wd_hsa

        # Reinvest surplus (after-tax money → 100% cost basis)
        if yr_surplus > 0 and surplus_dest != "none":
            if surplus_dest == "cash":
                bal_cash += yr_surplus
            else:
                bal_brokerage += yr_surplus
                brokerage_basis += yr_surplus

        total_taxes_paid += taxes

        # Growth
        yr_return = return_sequence[i] if return_sequence else post_return
        bal_pt = max(0.0, bal_pt) * (1 + yr_return)
        bal_ro = max(0.0, bal_ro) * (1 + yr_return)
        bal_brokerage = max(0.0, bal_brokerage) * (1 + yr_return)
        bal_cash = max(0.0, bal_cash) * (1 + cash_int_rate)
        bal_hs = max(0.0, bal_hs) * (1 + yr_return)
        bal_tx = bal_brokerage + bal_cash
        # Grow remaining inherited IRA balances
        for idx in range(len(ret_inherited_iras)):
            if ret_iira_bals[idx] > 0:
                ret_iira_bals[idx] *= (1 + yr_return)

        # Appreciate home
        if i > 0:
            curr_home_val *= (1 + home_appr)

        bal_inherited_total = sum(ret_iira_bals)
        total_bal = bal_pt + bal_ro + bal_tx + bal_hs + bal_inherited_total
        gross_estate = total_bal + curr_home_val

        # After-tax estate: what heirs actually receive
        my_marginal = tax_result["marginal_rate"]
        if heir_bracket_option == "lower":
            heir_fed = _heir_rate_from_offset(my_marginal, -1)
        elif heir_bracket_option == "higher":
            heir_fed = _heir_rate_from_offset(my_marginal, +1)
        else:
            heir_fed = my_marginal
        heir_total_rate = heir_fed + state_rate
        # Pre-tax & HSA & inherited IRA → fully taxable as ordinary income to heirs
        # Roth → tax-free; Brokerage → stepped-up basis; Cash → no tax; Home → stepped-up basis
        after_tax_estate = (
            bal_pt * (1 - heir_total_rate) +
            bal_ro +
            bal_tx +  # stepped-up basis
            bal_hs * (1 - heir_total_rate) +
            bal_inherited_total * (1 - heir_total_rate) +
            curr_home_val
        )

        row = {
            "Year": year, "Age": age,
            "SS Income": round(gross_ss, 0), "Pension": round(pen_income, 0),
            "Other Income": round(yr_other_income, 0),
            "Dividends": round(yr_dividends, 0),
            "Interest": round(yr_cash_interest, 0),
            "W/D Taxable": round(wd_taxable, 0),
            "W/D Roth": round(wd_roth, 0), "W/D HSA": round(wd_hsa, 0),
            "Realized CG": round(yr_cap_gains, 0),
            "Conv Gross": round(conversion_this_year, 0),
            "Conv Tax": round(conv_tax_total, 0),
            "Conv to Roth": round(conversion_this_year - conv_tax_withheld, 0),
            "Surplus Reinv": round(yr_surplus, 0),
            "Living Exp": round(living_exp, 0), "Mortgage": round(mtg_pmt, 0),
            "Total Exp": round(expenses, 0),
            "Fed Tax": round(tax_result["fed_tax"], 0),
            "State Tax": round(tax_result["state_tax"], 0),
            "Total Tax": round(taxes, 0),
            "Bal Taxable": round(bal_tx, 0),
            "Bal Roth": round(bal_ro, 0), "Bal HSA": round(bal_hs, 0),
        }
        if balances["pretax"] > 0:
            row["Bal Pre-Tax"] = round(bal_pt, 0)
            row["RMD"] = round(rmd_amount, 0)
            row["W/D Pre-Tax"] = round(max(0, wd_pretax - rmd_amount), 0)
        if any(ira["balance"] > 0 for ira in ret_inherited_iras):
            row["Inherited Dist"] = round(yr_inherited_dist, 0)
            row["Bal Inherited"] = round(bal_inherited_total, 0)
        row["Portfolio"] = round(total_bal, 0)
        row["Home Value"] = round(curr_home_val, 0)
        row["Gross Estate"] = round(gross_estate, 0)
        row["Estate (Net)"] = round(after_tax_estate, 0)
        rows.append(row)

    bal_inherited_final = sum(ret_iira_bals)
    final_total = bal_pt + bal_ro + bal_tx + bal_hs + bal_inherited_final
    gross_estate_final = final_total + curr_home_val
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


# ========== UI ==========
st.title("Retirement Estimator – Accumulation Phase")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Current Situation", "Savings Plan", "Projection", "Retirement Readiness", "Roth Conversion"])

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

    st.header("Personal Info")
    filer_dob = st.date_input("Your Date of Birth", value=dt.date(1980, 1, 1), min_value=dt.date(1930, 1, 1), max_value=dt.date(2000, 12, 31), key="filer_dob")
    spouse_dob = st.date_input("Spouse DOB (if applicable)", value=None, min_value=dt.date(1930, 1, 1), max_value=dt.date(2000, 12, 31), key="spouse_dob")
    filing_status = st.selectbox("Filing Status", ["Single", "Married Filing Jointly", "Head of Household"], key="filing_status")
    is_joint = "joint" in filing_status.lower()

    current_age = age_at_date(filer_dob, dt.date.today())
    spouse_age = age_at_date(spouse_dob, dt.date.today()) if spouse_dob else None
    st.write(f"**Your Age:** {current_age}")
    if spouse_age:
        st.write(f"**Spouse Age:** {spouse_age}")

    st.divider()
    st.header("Retirement Targets")
    target_retirement_age = st.number_input("Your Retirement Age", min_value=40, max_value=80, value=65, step=1, key="ret_age")
    years_to_retirement = max(0, target_retirement_age - current_age)
    st.write(f"**Years to Retirement:** {years_to_retirement}")
    if is_joint and spouse_age is not None:
        spouse_retirement_age = st.number_input("Spouse Retirement Age", min_value=40, max_value=80, value=65, step=1, key="spouse_ret_age")
        spouse_years_to_retirement = max(0, spouse_retirement_age - spouse_age)
        st.write(f"**Spouse Years to Retirement:** {spouse_years_to_retirement}")
    else:
        spouse_retirement_age = target_retirement_age
        spouse_years_to_retirement = years_to_retirement
    life_expectancy = st.number_input("Plan Through Age", min_value=80, max_value=100, value=95, step=1, key="life_exp")

    st.divider()
    st.header("Assumptions")
    inflation = st.number_input("Inflation Rate", value=0.03, step=0.005, format="%.3f", key="inflation")
    pre_retire_return = st.number_input("Pre-Retirement Return", value=0.07, step=0.005, format="%.3f", key="pre_ret_return")
    post_retire_return = st.number_input("Post-Retirement Return", value=0.05, step=0.005, format="%.3f", key="post_ret_return")
    salary_growth = st.number_input("Annual Salary Growth", value=0.02, step=0.005, format="%.3f", key="salary_growth")
    state_tax_rate = st.number_input("State Tax Rate (%)", value=5.0, step=0.5, format="%.1f", key="state_tax_rate") / 100

with tab1:
    st.subheader("Current Financial Situation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Income")
        salary_filer = st.number_input("Your Annual Salary", value=100000.0, step=5000.0, key="salary_filer")
        salary_spouse = st.number_input("Spouse Annual Salary", value=0.0, step=5000.0, key="salary_spouse") if is_joint else 0.0
        st.markdown("**Other Income** (disability, inheritance, etc.)")
        _oi_c1, _oi_c2 = st.columns(2)
        with _oi_c1:
            other_income = st.number_input("Annual Amount", value=0.0, step=1000.0, key="other_income")
        with _oi_c2:
            other_income_years = st.number_input("Years Remaining", min_value=0, max_value=50, value=0, step=1,
                key="other_income_years", help="0 = permanent (continues until retirement)")
        if other_income > 0:
            _oi_c3, _oi_c4, _oi_c5 = st.columns(3)
            with _oi_c3:
                other_income_tax_free = st.checkbox("Tax-free income", value=False, key="other_income_tax_free",
                    help="Check for non-taxable income (e.g. disability, Roth inheritance, gifts)")
            with _oi_c4:
                other_income_inflation = st.checkbox("Adjusts for inflation", value=False, key="other_income_inflation",
                    help="Check if this income grows with inflation (e.g. SSDI, some alimony). Uncheck for fixed amounts.")
            with _oi_c5:
                if is_joint:
                    other_income_recipient = st.selectbox("Recipient", ["Household", "Filer", "Spouse"],
                        key="other_income_recipient")
                else:
                    other_income_recipient = "Filer"
        else:
            other_income_tax_free = False
            other_income_inflation = False
            other_income_years = 0
            other_income_recipient = "Household"
        total_income = salary_filer + salary_spouse + other_income
        st.metric("Total Household Income", money(total_income))
        current_living_expenses = st.number_input("Current Annual Living Expenses", value=80000.0, step=5000.0, key="living_exp_tab1")

        with st.expander("Future Expense Changes"):
            st.caption("Add expenses that will start or end in the future (annual amounts, today's dollars)")
            if st.button("Add Expense Change", key="add_fut_exp"):
                st.session_state.num_future_expenses += 1
            future_expenses = []
            for i in range(st.session_state.num_future_expenses):
                fc1, fc2, fc3, fc4, fc5 = st.columns([3, 2, 1.5, 1.5, 0.8])
                with fc1:
                    fe_name = st.text_input("Description", key=f"fe_name_{i}", placeholder="e.g. College tuition")
                with fc2:
                    fe_amt = st.number_input("Annual $", value=0.0, step=1000.0, key=f"fe_amt_{i}")
                with fc3:
                    fe_start = st.number_input("Start Age", min_value=current_age, max_value=100, value=current_age, step=1, key=f"fe_start_{i}")
                with fc4:
                    fe_end = st.number_input("End Age (0=never)", min_value=0, max_value=100, value=0, step=1, key=f"fe_end_{i}")
                with fc5:
                    fe_inflates = st.checkbox("Inflates", value=True, key=f"fe_infl_{i}")
                if fe_amt != 0:
                    future_expenses.append({
                        "name": fe_name or f"Expense {i+1}",
                        "amount": fe_amt,
                        "start_age": fe_start,
                        "end_age": fe_end if fe_end > 0 else 999,
                        "inflates": fe_inflates,
                    })
            if future_expenses:
                st.markdown("**Summary:**")
                for fe in future_expenses:
                    end_txt = f"age {fe['end_age']}" if fe['end_age'] < 999 else "ongoing"
                    sign = "+" if fe["amount"] > 0 else ""
                    st.write(f"- {fe['name']}: {sign}{money(fe['amount'])}/yr, ages {fe['start_age']}-{end_txt} {'(inflates)' if fe['inflates'] else '(fixed)'}")

        st.divider()
        st.markdown("### Social Security")

        ss_detail_filer = estimate_ss_pia(salary_filer, current_age, target_retirement_age, salary_growth, detailed=True)
        ss_pia_filer = ss_detail_filer["pia_annual"]

        ss_col1, ss_col2 = st.columns(2)
        with ss_col1:
            st.markdown("**Your Social Security**")
            ssdi_filer = st.checkbox("Currently receiving SSDI", value=False, key="ssdi_filer",
                help="Social Security Disability Insurance — pays PIA now, converts to retirement benefits at FRA")
            ss_filer_claim_age = st.number_input("Your SS Claim Age", min_value=62, max_value=70, value=67, step=1, key="ss_claim_f",
                disabled=ssdi_filer, help="Disabled — SSDI pays 100% PIA now" if ssdi_filer else None)
            if ssdi_filer:
                st.write(f"PIA (today's $): **{money(ss_pia_filer)}**/yr ({money(ss_pia_filer / 12)}/mo)")
                st.write(f"**SSDI** — receiving 100% PIA now, converts to retirement SS at FRA")
            else:
                yrs_to_claim_filer = max(0, ss_filer_claim_age - current_age)
                claim_f = ss_claim_factor(ss_filer_claim_age)
                ss_at_claim_filer = ss_pia_filer * claim_f * ((1 + inflation) ** yrs_to_claim_filer)
                st.write(f"PIA at FRA (today's $): **{money(ss_pia_filer)}**/yr ({money(ss_pia_filer / 12)}/mo)")
                st.write(f"Claim factor at {ss_filer_claim_age}: **{claim_f:.0%}** of PIA")
                st.write(f"Benefit at claim (future $): **{money(ss_at_claim_filer)}**/yr ({money(ss_at_claim_filer / 12)}/mo)")
            st.caption(f"{ss_detail_filer['work_years']} work years (age 20-{target_retirement_age}), "
                       f"AIME: ${ss_detail_filer['aime']:,.0f}/mo"
                       + (f", {ss_detail_filer['zero_years']} zero yrs in top 35" if ss_detail_filer['zero_years'] > 0 else ""))
            ss_override_filer = st.number_input("Override PIA (annual, today's $, 0=use estimate)", value=0.0, step=1000.0, key="ss_ov_f")

        ss_pia_spouse = 0.0
        ss_spouse_claim_age = 67
        ssdi_spouse = False
        with ss_col2:
            if is_joint:
                st.markdown("**Spouse Social Security**")
                ssdi_spouse = st.checkbox("Currently receiving SSDI", value=False, key="ssdi_spouse",
                    help="Social Security Disability Insurance — pays PIA now, converts to retirement benefits at FRA")
                if salary_spouse > 0:
                    ss_detail_spouse = estimate_ss_pia(salary_spouse, spouse_age or current_age, spouse_retirement_age, salary_growth, detailed=True)
                    ss_pia_spouse = ss_detail_spouse["pia_annual"]
                    ss_spouse_claim_age = st.number_input("Spouse SS Claim Age", min_value=62, max_value=70, value=67, step=1, key="ss_claim_s",
                        disabled=ssdi_spouse, help="Disabled — SSDI pays 100% PIA now" if ssdi_spouse else None)
                    if ssdi_spouse:
                        st.write(f"PIA (today's $): **{money(ss_pia_spouse)}**/yr ({money(ss_pia_spouse / 12)}/mo)")
                        st.write(f"**SSDI** — receiving 100% PIA now, converts to retirement SS at FRA")
                    else:
                        yrs_to_claim_spouse = max(0, ss_spouse_claim_age - (spouse_age or current_age))
                        claim_s = ss_claim_factor(ss_spouse_claim_age)
                        ss_at_claim_spouse = ss_pia_spouse * claim_s * ((1 + inflation) ** yrs_to_claim_spouse)
                        st.write(f"PIA at FRA (today's $): **{money(ss_pia_spouse)}**/yr ({money(ss_pia_spouse / 12)}/mo)")
                        st.write(f"Claim factor at {ss_spouse_claim_age}: **{claim_s:.0%}** of PIA")
                        st.write(f"Benefit at claim (future $): **{money(ss_at_claim_spouse)}**/yr ({money(ss_at_claim_spouse / 12)}/mo)")
                    st.caption(f"{ss_detail_spouse['work_years']} work years (age 20-{spouse_retirement_age}), "
                               f"AIME: ${ss_detail_spouse['aime']:,.0f}/mo"
                               + (f", {ss_detail_spouse['zero_years']} zero yrs in top 35" if ss_detail_spouse['zero_years'] > 0 else ""))
                else:
                    st.caption("No current salary — enter projected PIA below")
                    ss_spouse_claim_age = st.number_input("Spouse SS Claim Age", min_value=62, max_value=70, value=67, step=1, key="ss_claim_s",
                        disabled=ssdi_spouse, help="Disabled — SSDI pays 100% PIA now" if ssdi_spouse else None)
                ss_override_spouse = st.number_input("Spouse PIA (annual, today's $)", value=0.0, step=1000.0, key="ss_ov_s",
                    help="Enter projected annual PIA from SSA statement or ssa.gov/myaccount" if salary_spouse <= 0
                         else "Override estimated PIA (0 = use estimate above)")
                if ss_override_spouse > 0 and salary_spouse <= 0:
                    claim_s = ss_claim_factor(ss_spouse_claim_age)
                    yrs_to_claim_spouse = max(0, ss_spouse_claim_age - (spouse_age or current_age))
                    ss_at_claim_spouse = ss_override_spouse * claim_s * ((1 + inflation) ** yrs_to_claim_spouse)
                    st.write(f"Claim factor at {ss_spouse_claim_age}: **{claim_s:.0%}** of PIA")
                    st.write(f"Benefit at claim (future $): **{money(ss_at_claim_spouse)}**/yr ({money(ss_at_claim_spouse / 12)}/mo)")
            else:
                ss_override_spouse = 0.0

        with st.expander("SS Impact by Retirement Age"):
            alt = ss_detail_filer["alt_ages"]
            alt_rows = []
            for age_val in sorted(alt.keys()):
                pia_val = alt[age_val]
                work_yrs = max(0, max(current_age, age_val) - 20)
                zeros = max(0, 35 - work_yrs)
                claim_adj = pia_val * ss_claim_factor(ss_filer_claim_age)
                marker = " <-- your plan" if age_val == target_retirement_age else ""
                alt_rows.append({"Retire Age": age_val, "Work Years": work_yrs,
                                 "Zero Years (of 35)": zeros,
                                 "PIA (today's $)": money(pia_val),
                                 f"At claim {ss_filer_claim_age} (today's $)": money(claim_adj),
                                 "Note": marker})
            st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)
            if is_joint and salary_spouse > 0:
                st.markdown("**Spouse:**")
                alt_s = ss_detail_spouse["alt_ages"]
                alt_rows_s = []
                for age_val in sorted(alt_s.keys()):
                    pia_val = alt_s[age_val]
                    work_yrs = max(0, max(spouse_age or current_age, age_val) - 20)
                    zeros = max(0, 35 - work_yrs)
                    claim_adj_s = pia_val * ss_claim_factor(ss_spouse_claim_age)
                    marker = " <-- plan" if age_val == spouse_retirement_age else ""
                    alt_rows_s.append({"Retire Age": age_val, "Work Years": work_yrs,
                                       "Zero Years (of 35)": zeros,
                                       "PIA (today's $)": money(pia_val),
                                       f"At claim {ss_spouse_claim_age} (today's $)": money(claim_adj_s),
                                       "Note": marker})
                st.dataframe(pd.DataFrame(alt_rows_s), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Current Savings")
        current_401k_filer = st.number_input("Your 401(k)/403(b)", value=0.0, step=10000.0, key="curr_401k_f")
        current_401k_spouse = st.number_input("Spouse 401(k)/403(b)", value=0.0, step=10000.0, key="curr_401k_s") if is_joint else 0.0
        current_trad_ira_filer = st.number_input("Your Traditional IRA", value=0.0, step=5000.0, key="curr_trad_ira_f")
        current_trad_ira_spouse = st.number_input("Spouse Traditional IRA", value=0.0, step=5000.0, key="curr_trad_ira_s") if is_joint else 0.0
        current_trad_ira = current_trad_ira_filer + current_trad_ira_spouse
        current_roth_ira_filer = st.number_input("Your Roth IRA", value=0.0, step=5000.0, key="curr_roth_ira_f")
        current_roth_ira_spouse = st.number_input("Spouse Roth IRA", value=0.0, step=5000.0, key="curr_roth_ira_s") if is_joint else 0.0
        current_roth_ira = current_roth_ira_filer + current_roth_ira_spouse
        current_roth_401k = st.number_input("Roth 401(k)", value=0.0, step=5000.0, key="curr_roth_401k")
        current_taxable = st.number_input("Taxable Brokerage", value=0.0, step=10000.0, key="curr_taxable")
        taxable_basis = st.number_input("Cost Basis (taxable brokerage)", value=0.0, step=10000.0, key="taxable_basis",
            help="Your original investment amount. Default 0 = assume basis equals current balance (no embedded gains)")
        if taxable_basis <= 0 and current_taxable > 0:
            taxable_basis = current_taxable  # default: no embedded gains
        current_hsa = st.number_input("HSA Balance", value=0.0, step=1000.0, key="curr_hsa")
        current_cash = st.number_input("Cash/Savings", value=0.0, step=5000.0, key="curr_cash")

        with st.expander("Inherited IRA(s)"):
            st.caption("Inherited IRAs have mandatory distribution rules separate from regular IRAs.")
            inherited_ira_filer = st.number_input("Your Inherited IRA Balance", value=0.0, step=5000.0, key="inh_ira_f")
            inherited_ira_rmd_required_filer = False
            inherited_ira_additional_filer = 0.0
            if inherited_ira_filer > 0:
                inherited_ira_rule_filer = st.selectbox("Distribution Rule", ["10-Year Rule", "Lifetime RMDs"],
                    key="inh_rule_f", help="10-Year Rule: most non-spouse beneficiaries (SECURE Act). Lifetime: surviving spouse, disabled, etc.")
                if inherited_ira_rule_filer == "10-Year Rule":
                    inherited_ira_years_filer = st.number_input("Years Remaining in 10-Year Window",
                        min_value=1, max_value=10, value=10, step=1, key="inh_yrs_f")
                    inherited_ira_rule_filer_code = "10_year"
                    inherited_ira_rmd_required_filer = st.checkbox(
                        "Was the original owner already taking RMDs?", key="inh_rmd_req_f",
                        help="If the original owner had already begun RMDs before passing, you must take annual minimum distributions AND empty the account by year 10.")
                    # Show minimum RMD info
                    if inherited_ira_rmd_required_filer:
                        _filer_divisor = SINGLE_LIFE_TABLE.get(current_age, 1.0)
                        _filer_min_rmd = inherited_ira_filer / _filer_divisor if _filer_divisor > 0 else inherited_ira_filer
                        st.info(f"Minimum annual RMD (year 1): {money(_filer_min_rmd)}")
                        inherited_ira_additional_filer = st.number_input(
                            "Additional Annual Distribution (above minimum)", value=0.0, step=1000.0, key="inh_add_f",
                            help="Extra amount to distribute each year beyond the required minimum.")
                    else:
                        inherited_ira_additional_filer = st.number_input(
                            "Planned Annual Distribution", value=0.0, step=1000.0, key="inh_add_f",
                            help="Your chosen annual distribution. Leave at 0 to default to an even split over remaining years.")
                    # Recommendation engine
                    if inherited_ira_years_filer > 0:
                        _rec = calc_inherited_ira_recommendation(
                            inherited_ira_filer, inherited_ira_years_filer, current_age,
                            inherited_ira_rmd_required_filer, total_income, filing_status,
                            state_rate=state_tax_rate, growth_rate=pre_retire_return)
                        if _rec:
                            st.markdown("---")
                            st.markdown(f"**Distribution Recommendation:** Fill to top of {_rec['target_bracket_rate']:.0%} bracket")
                            rc1, rc2 = st.columns(2)
                            with rc1:
                                st.metric("Recommended Annual", money(_rec["recommended_annual"]))
                                st.caption(f"Stays in {_rec['target_bracket_rate']:.0%} bracket (current rate: {_rec['marginal_without']:.0%})")
                            with rc2:
                                st.metric("Est. Tax Savings vs Final-Year Lump", money(_rec["total_tax_lump"] - _rec["total_tax_recommended"]))
                            if _rec.get("spill_desc"):
                                st.info(_rec["spill_desc"])
                            if _rec.get("final_yr_lump") and _rec["final_yr_lump"] > _rec["recommended_annual"] * 1.5:
                                st.error(f"Without spreading: final year forces {money(_rec['final_yr_lump'])} lump sum → {_rec['final_yr_rate']:.0%} bracket")
                            if _rec["warning"]:
                                st.warning(_rec["warning"])
                            with st.expander("Year-by-Year Schedule"):
                                _sched_df = pd.DataFrame(_rec["schedule"])
                                _sched_df["marginal_rate"] = _sched_df["marginal_rate"].apply(lambda x: f"{x:.0%}")
                                st.dataframe(_sched_df, use_container_width=True, hide_index=True)
                else:
                    inherited_ira_years_filer = 0
                    inherited_ira_rule_filer_code = "lifetime"
                    _filer_divisor = SINGLE_LIFE_TABLE.get(current_age, 1.0)
                    _filer_min_rmd = inherited_ira_filer / _filer_divisor if _filer_divisor > 0 else inherited_ira_filer
                    st.info(f"Minimum annual RMD (year 1): {money(_filer_min_rmd)}")
                    inherited_ira_additional_filer = st.number_input(
                        "Additional Annual Distribution (above minimum)", value=0.0, step=1000.0, key="inh_add_f",
                        help="Extra amount to distribute each year beyond the required minimum.")
                    # Lifetime recommendation
                    _lt_horizon = max(5, life_expectancy - current_age)
                    _lt_rec = calc_inherited_ira_recommendation(
                        inherited_ira_filer, 0, current_age,
                        True, total_income, filing_status,
                        state_rate=state_tax_rate, growth_rate=pre_retire_return, rule="lifetime", horizon_years=_lt_horizon)
                    if _lt_rec:
                        st.markdown("---")
                        st.markdown(f"**Distribution Recommendation:** Fill to top of {_lt_rec['target_bracket_rate']:.0%} bracket")
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            if _lt_rec["recommended_additional"] > 0:
                                st.metric("Recommended Annual Total", money(_lt_rec["recommended_annual"]))
                                st.caption(f"Min RMD + {money(_lt_rec['recommended_additional'])} additional")
                            else:
                                st.metric("Year-1 Minimum RMD", money(_lt_rec["recommended_annual"]))
                                st.caption("Min RMD already fills or exceeds current bracket")
                        with rc2:
                            if _lt_rec["tax_savings"] > 0:
                                st.metric("Est. Tax Savings vs Min-Only", money(_lt_rec["tax_savings"]))
                            if _lt_rec["peak_min_only_dist"] > 0:
                                st.metric("Peak Future RMD (min-only)", money(_lt_rec["peak_min_only_dist"]))
                                st.caption(f"At age {_lt_rec['peak_min_only_age']} → {_lt_rec['peak_min_only_rate']:.0%} bracket")
                        if _lt_rec["warning"]:
                            st.warning(_lt_rec["warning"])
                        with st.expander("Year-by-Year Schedule (recommended)"):
                            _sched_df = pd.DataFrame(_lt_rec["schedule"])
                            _sched_df["marginal_rate"] = _sched_df["marginal_rate"].apply(lambda x: f"{x:.0%}")
                            st.dataframe(_sched_df, use_container_width=True, hide_index=True)
            else:
                inherited_ira_rule_filer_code = "10_year"
                inherited_ira_years_filer = 10

            inherited_ira_spouse = 0.0
            inherited_ira_rule_spouse_code = "10_year"
            inherited_ira_years_spouse = 10
            inherited_ira_rmd_required_spouse = False
            inherited_ira_additional_spouse = 0.0
            if is_joint:
                inherited_ira_spouse = st.number_input("Spouse Inherited IRA Balance", value=0.0, step=5000.0, key="inh_ira_s")
                if inherited_ira_spouse > 0:
                    inherited_ira_rule_spouse = st.selectbox("Spouse Distribution Rule", ["10-Year Rule", "Lifetime RMDs"],
                        key="inh_rule_s")
                    if inherited_ira_rule_spouse == "10-Year Rule":
                        inherited_ira_years_spouse = st.number_input("Spouse Years Remaining in 10-Year Window",
                            min_value=1, max_value=10, value=10, step=1, key="inh_yrs_s")
                        inherited_ira_rule_spouse_code = "10_year"
                        inherited_ira_rmd_required_spouse = st.checkbox(
                            "Was the original owner already taking RMDs? (Spouse)", key="inh_rmd_req_s")
                        if inherited_ira_rmd_required_spouse:
                            _sp_age = spouse_age or current_age
                            _sp_divisor = SINGLE_LIFE_TABLE.get(_sp_age, 1.0)
                            _sp_min_rmd = inherited_ira_spouse / _sp_divisor if _sp_divisor > 0 else inherited_ira_spouse
                            st.info(f"Spouse minimum annual RMD (year 1): {money(_sp_min_rmd)}")
                            inherited_ira_additional_spouse = st.number_input(
                                "Spouse Additional Annual Distribution (above minimum)", value=0.0, step=1000.0, key="inh_add_s")
                        else:
                            inherited_ira_additional_spouse = st.number_input(
                                "Spouse Planned Annual Distribution", value=0.0, step=1000.0, key="inh_add_s")
                        # Spouse recommendation
                        if inherited_ira_years_spouse > 0:
                            _sp_rec = calc_inherited_ira_recommendation(
                                inherited_ira_spouse, inherited_ira_years_spouse, spouse_age or current_age,
                                inherited_ira_rmd_required_spouse, total_income, filing_status,
                                state_rate=state_tax_rate, growth_rate=pre_retire_return)
                            if _sp_rec:
                                st.markdown("---")
                                st.markdown(f"**Spouse Distribution Recommendation:** Fill to top of {_sp_rec['target_bracket_rate']:.0%} bracket")
                                sc1, sc2 = st.columns(2)
                                with sc1:
                                    st.metric("Recommended Annual", money(_sp_rec["recommended_annual"]))
                                    st.caption(f"Stays in {_sp_rec['target_bracket_rate']:.0%} bracket (current rate: {_sp_rec['marginal_without']:.0%})")
                                with sc2:
                                    st.metric("Est. Tax Savings vs Final-Year Lump", money(_sp_rec["total_tax_lump"] - _sp_rec["total_tax_recommended"]))
                                if _sp_rec.get("spill_desc"):
                                    st.info(_sp_rec["spill_desc"])
                                if _sp_rec.get("final_yr_lump") and _sp_rec["final_yr_lump"] > _sp_rec["recommended_annual"] * 1.5:
                                    st.error(f"Without spreading: final year forces {money(_sp_rec['final_yr_lump'])} lump sum → {_sp_rec['final_yr_rate']:.0%} bracket")
                                if _sp_rec["warning"]:
                                    st.warning(_sp_rec["warning"])
                                with st.expander("Spouse Year-by-Year Schedule"):
                                    _sched_df = pd.DataFrame(_sp_rec["schedule"])
                                    _sched_df["marginal_rate"] = _sched_df["marginal_rate"].apply(lambda x: f"{x:.0%}")
                                    st.dataframe(_sched_df, use_container_width=True, hide_index=True)
                    else:
                        inherited_ira_years_spouse = 0
                        inherited_ira_rule_spouse_code = "lifetime"
                        _sp_age = spouse_age or current_age
                        _sp_divisor = SINGLE_LIFE_TABLE.get(_sp_age, 1.0)
                        _sp_min_rmd = inherited_ira_spouse / _sp_divisor if _sp_divisor > 0 else inherited_ira_spouse
                        st.info(f"Spouse minimum annual RMD (year 1): {money(_sp_min_rmd)}")
                        inherited_ira_additional_spouse = st.number_input(
                            "Spouse Additional Annual Distribution (above minimum)", value=0.0, step=1000.0, key="inh_add_s")
                        # Spouse lifetime recommendation
                        _sp_lt_horizon = max(5, life_expectancy - (_sp_age or current_age))
                        _sp_lt_rec = calc_inherited_ira_recommendation(
                            inherited_ira_spouse, 0, _sp_age,
                            True, total_income, filing_status,
                            state_rate=state_tax_rate, growth_rate=pre_retire_return, rule="lifetime", horizon_years=_sp_lt_horizon)
                        if _sp_lt_rec:
                            st.markdown("---")
                            st.markdown(f"**Spouse Distribution Recommendation:** Fill to top of {_sp_lt_rec['target_bracket_rate']:.0%} bracket")
                            sc1, sc2 = st.columns(2)
                            with sc1:
                                if _sp_lt_rec["recommended_additional"] > 0:
                                    st.metric("Recommended Annual Total", money(_sp_lt_rec["recommended_annual"]))
                                    st.caption(f"Min RMD + {money(_sp_lt_rec['recommended_additional'])} additional")
                                else:
                                    st.metric("Year-1 Minimum RMD", money(_sp_lt_rec["recommended_annual"]))
                                    st.caption("Min RMD already fills or exceeds current bracket")
                            with sc2:
                                if _sp_lt_rec["tax_savings"] > 0:
                                    st.metric("Est. Tax Savings vs Min-Only", money(_sp_lt_rec["tax_savings"]))
                                if _sp_lt_rec["peak_min_only_dist"] > 0:
                                    st.metric("Peak Future RMD (min-only)", money(_sp_lt_rec["peak_min_only_dist"]))
                                    st.caption(f"At age {_sp_lt_rec['peak_min_only_age']} → {_sp_lt_rec['peak_min_only_rate']:.0%} bracket")
                            if _sp_lt_rec["warning"]:
                                st.warning(_sp_lt_rec["warning"])
                            with st.expander("Spouse Year-by-Year Schedule (recommended)"):
                                _sched_df = pd.DataFrame(_sp_lt_rec["schedule"])
                                _sched_df["marginal_rate"] = _sched_df["marginal_rate"].apply(lambda x: f"{x:.0%}")
                                st.dataframe(_sched_df, use_container_width=True, hide_index=True)

    total_inherited_ira = inherited_ira_filer + inherited_ira_spouse
    total_pretax = current_401k_filer + current_401k_spouse + current_trad_ira
    total_roth = current_roth_ira + current_roth_401k
    total_taxable = current_taxable + current_cash
    total_savings = total_pretax + total_roth + total_taxable + current_hsa

    # Resolve SS values
    ss_filer_final = ss_override_filer if ss_override_filer > 0 else ss_pia_filer
    ss_spouse_final = ss_override_spouse if ss_override_spouse > 0 else ss_pia_spouse

    # Add current SSDI income to total_income for surplus calculation
    if ssdi_filer and ss_filer_final > 0:
        total_income += ss_filer_final
    if ssdi_spouse and ss_spouse_final > 0:
        total_income += ss_spouse_final

    st.divider()
    st.markdown("### Home & Mortgage")
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        home_value = st.number_input("Home Value", value=0.0, step=25000.0, key="home_value")
        home_appreciation = st.number_input("Annual Home Appreciation (%)", value=3.0, step=0.5, format="%.1f", key="home_appr") / 100
        mortgage_balance = st.number_input("Mortgage Balance", value=0.0, step=10000.0, key="mtg_balance")
        mortgage_rate = st.number_input("Mortgage Interest Rate (%)", value=0.0, step=0.125, format="%.3f", key="mtg_rate") / 100
    with hcol2:
        mortgage_payment_monthly = st.number_input("Monthly Mortgage Payment (P&I)", value=0.0, step=100.0, key="mtg_pmt_monthly")
        mortgage_payment_annual = mortgage_payment_monthly * 12
        mortgage_years_remaining = st.number_input("Years Remaining on Mortgage", min_value=0, max_value=30, value=0, step=1, key="mtg_years")

        # Mortgage calculator
        if mortgage_balance > 0 and mortgage_rate > 0:
            with st.expander("Mortgage Calculator"):
                st.write("Calculate missing values from the ones you've entered above.")
                calc_col1, calc_col2 = st.columns(2)
                with calc_col1:
                    if st.button("Calculate Monthly Payment", key="calc_mtg_pmt"):
                        if mortgage_years_remaining > 0:
                            calc_pmt = calc_mortgage_payment(mortgage_balance, mortgage_rate, mortgage_years_remaining)
                            st.info(f"Estimated monthly payment: **{money(calc_pmt / 12)}**")
                        else:
                            st.warning("Enter years remaining first.")
                with calc_col2:
                    if st.button("Calculate Years", key="calc_mtg_yrs"):
                        if mortgage_payment_annual > 0:
                            calc_yrs = calc_mortgage_years(mortgage_balance, mortgage_rate, mortgage_payment_annual)
                            if calc_yrs >= 99:
                                st.warning("Payment doesn't cover interest.")
                            else:
                                st.info(f"Estimated years to payoff: **{calc_yrs}**")
                        else:
                            st.warning("Enter monthly payment first.")

    home_equity = max(0.0, home_value - mortgage_balance)

    st.divider()
    st.markdown("### Pension Information")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**Your Pension**")
        pension_filer_annual = st.number_input("Annual Pension (today's $)", value=0.0, step=1000.0, key="pen_filer")
        if pension_filer_annual > 0:
            pension_filer_start_age = st.number_input("Pension Start Age", min_value=40, max_value=80,
                value=target_retirement_age, step=1, key="pen_filer_age")
            pen_filer_has_cola = st.checkbox("Has cost-of-living increase", key="pen_filer_has_cola")
            if pen_filer_has_cola:
                pen_filer_cola_type = st.radio("Increase type", ["COLA (matches inflation)", "Custom annual increase"],
                    key="pen_filer_cola_type", horizontal=True)
                if pen_filer_cola_type == "Custom annual increase":
                    pension_filer_cola = st.number_input("Annual increase (%)", value=2.0, step=0.5, format="%.1f",
                        key="pen_filer_cola") / 100
                else:
                    pension_filer_cola = inflation
                    st.caption(f"Will match inflation rate ({inflation*100:.1f}%)")
            else:
                pension_filer_cola = 0.0
        else:
            pension_filer_start_age = target_retirement_age
            pension_filer_cola = 0.0
    with pcol2:
        if is_joint:
            st.markdown("**Spouse Pension**")
            pension_spouse_annual = st.number_input("Annual Pension (today's $)", value=0.0, step=1000.0, key="pen_spouse")
            if pension_spouse_annual > 0:
                pension_spouse_start_age = st.number_input("Pension Start Age", min_value=40, max_value=80,
                    value=spouse_retirement_age, step=1, key="pen_spouse_age")
                pen_spouse_has_cola = st.checkbox("Has cost-of-living increase", key="pen_spouse_has_cola")
                if pen_spouse_has_cola:
                    pen_spouse_cola_type = st.radio("Increase type", ["COLA (matches inflation)", "Custom annual increase"],
                        key="pen_spouse_cola_type", horizontal=True)
                    if pen_spouse_cola_type == "Custom annual increase":
                        pension_spouse_cola = st.number_input("Annual increase (%)", value=2.0, step=0.5, format="%.1f",
                            key="pen_spouse_cola") / 100
                    else:
                        pension_spouse_cola = inflation
                        st.caption(f"Will match inflation rate ({inflation*100:.1f}%)")
                else:
                    pension_spouse_cola = 0.0
            else:
                pension_spouse_start_age = spouse_retirement_age
                pension_spouse_cola = 0.0
        else:
            pension_spouse_annual = 0.0
            pension_spouse_start_age = target_retirement_age
            pension_spouse_cola = 0.0

    # --- SS Claiming Strategy Optimizer ---
    _opt_filer_pia = ss_filer_final
    _opt_spouse_pia = ss_spouse_final if is_joint else 0.0
    _has_ss = _opt_filer_pia > 0

    if _has_ss:
        with st.expander("SS Claiming Strategy Optimizer"):
            if is_joint and _opt_spouse_pia > 0:
                st.caption("Evaluates all 81 claiming-age combinations (62-70 for each spouse) "
                           "using a full retirement projection for each. Compares ending estate "
                           "values, capturing portfolio drawdown during gap years, tax bracket "
                           "interactions, RMDs, pensions, and the complete withdrawal waterfall.")
            else:
                st.caption("Evaluates claim ages 62-70 using a full retirement projection for each. "
                           "Compares ending estate values including portfolio drawdown, taxes, "
                           "RMDs, and the complete withdrawal waterfall.")

            _opt_life_exp = st.number_input(
                "Life Expectancy for SS Optimization",
                min_value=75, max_value=100, value=life_expectancy, step=1, key="ss_opt_life_exp",
                help="Shorter life expectancy favors early claiming. Longer favors delay. Try different values to see the impact.")

            # Build estimated balances at retirement (simplified: grow current, no future contributions)
            _opt_growth = (1 + pre_retire_return) ** years_to_retirement
            _opt_cash_growth = (1 + 0.04) ** years_to_retirement  # default cash rate
            _opt_brokerage = current_taxable * _opt_growth
            _opt_cash = current_cash * _opt_cash_growth
            _opt_balances = {
                "pretax": total_pretax * _opt_growth,
                "roth": total_roth * _opt_growth,
                "taxable": _opt_brokerage + _opt_cash,
                "brokerage": _opt_brokerage,
                "cash": _opt_cash,
                "brokerage_basis": taxable_basis,  # basis doesn't grow without contributions
                "hsa": current_hsa * _opt_growth,
            }

            # Compute effective mortgage years remaining
            _opt_mtg_yrs = mortgage_years_remaining
            if _opt_mtg_yrs == 0 and mortgage_payment_annual > 0 and mortgage_balance > 0 and mortgage_rate > 0:
                _opt_mtg_yrs = calc_mortgage_years(mortgage_balance, mortgage_rate, mortgage_payment_annual)
            elif _opt_mtg_yrs == 0 and mortgage_payment_annual > 0:
                _opt_mtg_yrs = 30

            # Build base params matching run_retirement_projection format
            _opt_inf_retire = (1 + inflation) ** years_to_retirement
            _opt_base_params = {
                "retire_age": target_retirement_age,
                "life_expectancy": _opt_life_exp,
                "retire_year": dt.date.today().year + years_to_retirement,
                "inflation": inflation,
                "post_retire_return": post_retire_return,
                "filing_status": filing_status,
                "state_tax_rate": state_tax_rate,
                "expenses_at_retirement": current_living_expenses * 0.8 * _opt_inf_retire,
                "ss_filer_fra": ss_filer_final * _opt_inf_retire,
                "ss_spouse_fra": ss_spouse_final * _opt_inf_retire,
                "ss_filer_claim_age": 67,  # placeholder, overridden by optimizer
                "ss_spouse_claim_age": 67,  # placeholder, overridden by optimizer
                "ssdi_filer": ssdi_filer,
                "ssdi_spouse": ssdi_spouse,
                "other_income": 0.0 if (other_income_years > 0 and other_income_years <= years_to_retirement) else (other_income * _opt_inf_retire if other_income_inflation else other_income),
                "other_income_tax_free": other_income_tax_free,
                "other_income_inflation": other_income_inflation,
                "other_income_years": max(0, other_income_years - years_to_retirement) if other_income_years > 0 else 0,
                "pension_filer_at_retire": pension_filer_annual * _opt_inf_retire,
                "pension_filer_start_age": pension_filer_start_age,
                "pension_filer_cola": pension_filer_cola,
                "pension_spouse_at_retire": pension_spouse_annual * _opt_inf_retire,
                "pension_spouse_start_age": pension_spouse_start_age,
                "pension_spouse_cola": pension_spouse_cola,
                "spouse_age_at_retire": (spouse_age + years_to_retirement) if spouse_age else None,
                "mortgage_payment": mortgage_payment_annual,
                "mortgage_years_at_retire": max(0, _opt_mtg_yrs - years_to_retirement),
                "home_value_at_retire": home_value * ((1 + home_appreciation) ** years_to_retirement),
                "home_appreciation": home_appreciation,
                "future_expenses": future_expenses,
                "dividend_yield": 0.015,
                "cash_interest_rate": 0.04,
                "inherited_iras": [],
                "surplus_destination": "brokerage",
                "heir_bracket_option": "same",
            }

            _opt_spending_order = ["Pre-Tax", "Taxable", "Tax-Free"]

            if is_joint and _opt_spouse_pia > 0:
                if st.button("Optimize SS Claiming Strategy", key="ss_opt_btn"):
                    result = optimize_ss_claiming(
                        balances_at_retire=_opt_balances,
                        base_params=_opt_base_params,
                        spending_order=_opt_spending_order,
                        filer_current_age=current_age,
                        spouse_current_age=spouse_age or current_age)

                    best = result["best"]
                    st.success(f"Recommended: You claim at {best['filer_claim']}, Spouse claims at {best['spouse_claim']}")

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Your Claim Age", best["filer_claim"])
                    mc2.metric("Spouse Claim Age", best["spouse_claim"])
                    mc3.metric("Estate (After Heir Tax)", money(best["estate"]))
                    mc4.metric("Total Lifetime Taxes", money(best["total_taxes"]))

                    st.info(result["explanation"])

                    # Top 5 strategies
                    st.markdown("**Top 5 Strategies:**")
                    top5 = result["rankings"][:5]
                    top5_rows = []
                    for r in top5:
                        top5_rows.append({
                            "Rank": r["rank"],
                            "Your Claim": r["filer_claim"],
                            "Spouse Claim": r["spouse_claim"],
                            "Ending Estate": money(r["estate"]),
                            "Total Taxes": money(r["total_taxes"]),
                            "vs. Best": money(r["diff_vs_best"]) if r["diff_vs_best"] < 0 else "\u2014",
                        })
                    st.dataframe(pd.DataFrame(top5_rows), use_container_width=True, hide_index=True)

                    # Common strategies comparison
                    st.markdown("**Common Strategies:**")
                    all_rankings = result["rankings"]
                    rank_lookup = {(r["filer_claim"], r["spouse_claim"]): r for r in all_rankings}
                    filer_min = max(62, current_age)
                    spouse_min = max(62, spouse_age or current_age)
                    common_combos = [
                        ("Both at 62", (max(62, filer_min), max(62, spouse_min))),
                        ("Both at 67 (FRA)", (max(67, filer_min), max(67, spouse_min))),
                        ("Both at 70", (70, 70)),
                    ]
                    if _opt_filer_pia >= _opt_spouse_pia:
                        common_combos.append(("Higher earner 70 / Lower 62", (70, max(62, spouse_min))))
                    else:
                        common_combos.append(("Higher earner 70 / Lower 62", (max(62, filer_min), 70)))
                    common_rows = []
                    for label, (fc, sc) in common_combos:
                        r = rank_lookup.get((fc, sc))
                        if r:
                            common_rows.append({
                                "Strategy": label,
                                "Your Claim": fc,
                                "Spouse Claim": sc,
                                "Ending Estate": money(r["estate"]),
                                "Total Taxes": money(r["total_taxes"]),
                                "Rank": f"{r['rank']} of {len(all_rankings)}",
                                "vs. Best": money(r["diff_vs_best"]) if r["diff_vs_best"] < 0 else "Best",
                            })
                    if common_rows:
                        st.dataframe(pd.DataFrame(common_rows), use_container_width=True, hide_index=True)

                    # Year-by-year schedule from full projection
                    with st.expander("Year-by-Year Schedule (Best Strategy)"):
                        st.caption("Full retirement projection showing SS, pensions, withdrawals, taxes, and portfolio balances")
                        st.dataframe(pd.DataFrame(result["schedule"]), use_container_width=True, hide_index=True)

            else:
                # Single filer or no spouse SS
                if st.button("Optimize SS Claiming Age", key="ss_opt_btn"):
                    result = optimize_ss_claiming(
                        balances_at_retire=_opt_balances,
                        base_params=_opt_base_params,
                        spending_order=_opt_spending_order,
                        filer_current_age=current_age,
                        spouse_current_age=None)

                    best = result["best"]
                    st.success(f"Recommended: Claim at age {best['filer_claim']}")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Optimal Claim Age", best["filer_claim"])
                    mc2.metric("Estate (After Heir Tax)", money(best["estate"]))
                    mc3.metric("Total Lifetime Taxes", money(best["total_taxes"]))

                    st.info(result["explanation"])

                    # All strategies table
                    st.markdown("**All Claim Ages:**")
                    all_rows = []
                    for r in result["rankings"]:
                        all_rows.append({
                            "Rank": r["rank"],
                            "Claim Age": r["filer_claim"],
                            "Factor": f"{ss_claim_factor(r['filer_claim']):.0%}",
                            "Ending Estate": money(r["estate"]),
                            "Total Taxes": money(r["total_taxes"]),
                            "vs. Best": money(r["diff_vs_best"]) if r["diff_vs_best"] < 0 else "Best",
                        })
                    st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

                    # Year-by-year schedule from full projection
                    with st.expander("Year-by-Year Schedule"):
                        st.caption("Full retirement projection showing SS, pensions, withdrawals, taxes, and portfolio balances")
                        st.dataframe(pd.DataFrame(result["schedule"]), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Current Savings Summary")
    if total_inherited_ira > 0:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1: st.metric("Pre-Tax", money(total_pretax))
        with col2: st.metric("Roth", money(total_roth))
        with col3: st.metric("Taxable", money(total_taxable))
        with col4: st.metric("HSA", money(current_hsa))
        with col5: st.metric("Inherited IRA", money(total_inherited_ira))
        with col6: st.metric("Home Equity", money(home_equity))
        with col7: st.metric("Total", money(total_savings + total_inherited_ira + home_equity))
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1: st.metric("Pre-Tax", money(total_pretax))
        with col2: st.metric("Roth", money(total_roth))
        with col3: st.metric("Taxable", money(total_taxable))
        with col4: st.metric("HSA", money(current_hsa))
        with col5: st.metric("Home Equity", money(home_equity))
        with col6: st.metric("Total", money(total_savings + home_equity))

with tab2:
    st.subheader("Annual Savings Plan")

    catchup_eligible = current_age >= 50
    spouse_catchup = (spouse_age or current_age) >= 50
    limit_401k = LIMITS_401K + (LIMITS_401K_CATCHUP if catchup_eligible else 0)
    limit_401k_spouse = LIMITS_401K + (LIMITS_401K_CATCHUP if spouse_catchup else 0)
    hsa_eligible = st.checkbox("HSA eligible (enrolled in a high-deductible health plan)?", key="hsa_eligible")
    hsa_limit = ((LIMITS_HSA_FAMILY if is_joint else LIMITS_HSA_SINGLE) + (LIMITS_HSA_CATCHUP if catchup_eligible else 0)) if hsa_eligible else 0

    # IRA limits with income phase-outs
    base_ira = LIMITS_IRA + (LIMITS_IRA_CATCHUP if catchup_eligible else 0)
    base_ira_spouse = LIMITS_IRA + (LIMITS_IRA_CATCHUP if spouse_catchup else 0)
    magi = total_income  # simplified MAGI proxy
    if "joint" in filing_status.lower():
        roth_phase = ROTH_PHASE_MFJ
        trad_phase = TRAD_PHASE_MFJ
    elif "head" in filing_status.lower():
        roth_phase = ROTH_PHASE_HOH
        trad_phase = TRAD_PHASE_HOH
    else:
        roth_phase = ROTH_PHASE_SINGLE
        trad_phase = TRAD_PHASE_SINGLE
    roth_ira_limit = ira_phase_out(magi, roth_phase, base_ira)
    roth_ira_limit_spouse = ira_phase_out(magi, roth_phase, base_ira_spouse)
    trad_ira_deduct_limit = ira_phase_out(magi, trad_phase, base_ira)
    trad_ira_deduct_limit_spouse = ira_phase_out(magi, trad_phase, base_ira_spouse)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Your Contributions")
        max_401k_filer = st.checkbox(f"Maximize 401(k) ({money(limit_401k)}/yr)", key="max_401k_f")
        if max_401k_filer:
            contrib_401k_filer = float(limit_401k)
            contrib_401k_filer_pct = min(100.0, contrib_401k_filer / salary_filer * 100) if salary_filer > 0 else 0
        else:
            contrib_401k_filer_pct = st.slider("Your 401(k) % of Salary", 0, 100, 10, key="c401k_f")
            contrib_401k_filer = min(salary_filer * contrib_401k_filer_pct / 100, limit_401k)

        roth_401k_pct_filer = st.slider("% of 401(k) to Roth", 0, 100, 0, key="roth_pct_f")
        contrib_401k_roth_filer = contrib_401k_filer * roth_401k_pct_filer / 100
        contrib_401k_pretax_filer = contrib_401k_filer - contrib_401k_roth_filer
        st.write(f"401(k) total: **{money(contrib_401k_filer)}** — Pre-tax: {money(contrib_401k_pretax_filer)}, Roth: {money(contrib_401k_roth_filer)}")

        employer_match_rate = st.number_input("Employer Match Rate (%)", value=50.0, step=5.0, key="ematch_rate_f",
            help="e.g. 50 = employer matches 50 cents per dollar")
        employer_match_upto = st.number_input("Match Up To (% of salary)", value=6.0, step=0.5, key="ematch_upto_f",
            help="e.g. 6 = employer matches on first 6% of salary you contribute")
        matchable_pct = min(contrib_401k_filer_pct, employer_match_upto)
        employer_match = salary_filer * matchable_pct / 100 * employer_match_rate / 100
        st.write(f"Employer match: **{money(employer_match)}** ({matchable_pct * employer_match_rate / 100:.1f}% of salary, always pre-tax)")

    with col2:
        st.markdown("### IRA & Other Contributions")
        st.caption(f"MAGI: {money(magi)} | Roth limit: {money(roth_ira_limit)} | Trad deductible: {money(trad_ira_deduct_limit)}")

        # Backdoor Roth eligibility (based on filer's own Traditional IRA balance)
        backdoor_eligible = roth_ira_limit < base_ira and current_trad_ira_filer <= 0
        backdoor_roth = False
        if roth_ira_limit <= 0:
            if current_trad_ira_filer <= 0:
                st.warning(f"Income exceeds Roth IRA limit ({money(roth_phase[1])}). Consider backdoor Roth.")
                backdoor_roth = st.checkbox(f"Use Backdoor Roth ({money(base_ira)}/yr)", key="backdoor_roth_f")
            else:
                st.warning(f"Income exceeds Roth IRA limit ({money(roth_phase[1])}). Backdoor Roth not recommended — your Traditional IRA balance of {money(current_trad_ira_filer)} triggers pro-rata taxation on conversions.")
        elif roth_ira_limit < base_ira:
            st.info(f"Roth IRA partially phased out: max {money(roth_ira_limit)} (of {money(base_ira)})")
            if current_trad_ira_filer <= 0:
                backdoor_roth = st.checkbox(f"Use Backdoor Roth for full {money(base_ira)}", key="backdoor_roth_f")

        # Effective Roth limit (full amount if backdoor)
        effective_roth_limit = base_ira if backdoor_roth else roth_ira_limit

        max_ira_filer = st.checkbox(f"Maximize IRA", key="max_ira_f")
        if max_ira_filer:
            contrib_roth_ira = effective_roth_limit
            contrib_trad_ira = 0.0
            if contrib_roth_ira > 0:
                label = "Backdoor Roth" if backdoor_roth else "Roth IRA"
                st.write(f"{label}: **{money(contrib_roth_ira)}**")
        elif backdoor_roth:
            contrib_roth_ira = float(base_ira)
            contrib_trad_ira = 0.0
            st.write(f"Backdoor Roth: **{money(contrib_roth_ira)}**")
        else:
            contrib_trad_ira = st.number_input("Traditional IRA", value=0.0, step=500.0, max_value=float(base_ira), key="contrib_trad_ira")
            contrib_roth_ira = st.number_input("Roth IRA", value=0.0, step=500.0, max_value=float(effective_roth_limit), key="contrib_roth_ira")
            if contrib_roth_ira > effective_roth_limit:
                st.warning(f"Roth IRA exceeds income-based limit of {money(effective_roth_limit)}")
            if contrib_trad_ira + contrib_roth_ira > base_ira:
                st.warning(f"Total IRA exceeds {dt.date.today().year} limit of {money(base_ira)}")
            if contrib_trad_ira > 0 and contrib_trad_ira > trad_ira_deduct_limit:
                st.info(f"Only {money(trad_ira_deduct_limit)} of Traditional IRA is deductible at your income")

        if hsa_eligible:
            max_hsa = st.checkbox(f"Maximize HSA ({money(hsa_limit)}/yr)", key="max_hsa")
            if max_hsa:
                contrib_hsa = float(hsa_limit)
                st.write(f"HSA: **{money(contrib_hsa)}** (max)")
            else:
                contrib_hsa = st.number_input("HSA", value=0.0, step=500.0, key="contrib_hsa")
        else:
            contrib_hsa = 0.0

        contrib_taxable = st.number_input("Taxable Savings", value=0.0, step=1000.0, key="contrib_taxable")

        st.divider()
        st.markdown("**Taxable Account Tax Treatment**")
        tax_efficient = st.checkbox("Tax-efficient investing (buy & hold, index funds)", value=True, key="tax_efficient",
            help="If checked: only dividends taxable annually, all capital gains deferred until sale (low basis). "
                 "If unchecked: annual capital gain distributions are realized and taxed.")
        dividend_yield = st.number_input("Estimated Dividend Yield (%)", value=1.5, step=0.25, format="%.2f", key="div_yield",
            help="Dividends are taxable annually in either mode") / 100
        if not tax_efficient:
            annual_cap_gain_pct = st.number_input("Annual Taxable Gain Distribution (%)", value=3.0, step=0.5, format="%.1f", key="ann_cg_pct",
                help="% of account value distributed as taxable capital gains each year (from fund distributions, rebalancing, etc.)") / 100
        else:
            annual_cap_gain_pct = 0.0
        cash_interest_rate = st.number_input("Cash/Savings Interest Rate (%)", value=4.0, step=0.25, format="%.2f", key="cash_int_rate",
            help="Interest on cash/savings, taxable as ordinary income") / 100
        reinvest_inv_income = st.checkbox("Reinvest investment income", value=True, key="reinvest_inv_income",
            help="If checked, dividends/cap gains/interest stay in the portfolio (still taxable). "
                 "If unchecked, investment income is received as spendable cash.")

    if is_joint and salary_spouse > 0:
        st.divider()
        st.markdown("### Spouse Contributions")
        col1, col2 = st.columns(2)
        with col1:
            max_401k_spouse = st.checkbox(f"Maximize Spouse 401(k) ({money(limit_401k_spouse)}/yr)", key="max_401k_s")
            if max_401k_spouse:
                contrib_401k_spouse = float(limit_401k_spouse)
                contrib_401k_spouse_pct = min(100.0, contrib_401k_spouse / salary_spouse * 100) if salary_spouse > 0 else 0
            else:
                contrib_401k_spouse_pct = st.slider("Spouse 401(k) % of Salary", 0, 100, 10, key="c401k_s")
                contrib_401k_spouse = min(salary_spouse * contrib_401k_spouse_pct / 100, limit_401k_spouse)

            roth_401k_pct_spouse = st.slider("% of Spouse 401(k) to Roth", 0, 100, 0, key="roth_pct_s")
            contrib_401k_roth_spouse = contrib_401k_spouse * roth_401k_pct_spouse / 100
            contrib_401k_pretax_spouse = contrib_401k_spouse - contrib_401k_roth_spouse
            st.write(f"Spouse 401(k) total: **{money(contrib_401k_spouse)}** — Pre-tax: {money(contrib_401k_pretax_spouse)}, Roth: {money(contrib_401k_roth_spouse)}")

            employer_match_rate_spouse = st.number_input("Spouse Employer Match Rate (%)", value=50.0, step=5.0, key="ematch_rate_s",
                help="e.g. 50 = employer matches 50 cents per dollar")
            employer_match_upto_spouse = st.number_input("Spouse Match Up To (% of salary)", value=6.0, step=0.5, key="ematch_upto_s",
                help="e.g. 6 = employer matches on first 6% of salary contributed")
            matchable_pct_spouse = min(contrib_401k_spouse_pct, employer_match_upto_spouse)
            employer_match_spouse = salary_spouse * matchable_pct_spouse / 100 * employer_match_rate_spouse / 100
            st.write(f"Spouse match: **{money(employer_match_spouse)}** ({matchable_pct_spouse * employer_match_rate_spouse / 100:.1f}% of salary, always pre-tax)")
        with col2:
            st.caption(f"Spouse MAGI: {money(magi)} | Spouse Roth limit: {money(roth_ira_limit_spouse)} | Trad deductible: {money(trad_ira_deduct_limit_spouse)}")

            # Spouse backdoor Roth eligibility (based on spouse's own Traditional IRA balance)
            backdoor_roth_spouse = False
            if roth_ira_limit_spouse <= 0:
                if current_trad_ira_spouse <= 0:
                    st.warning(f"Income exceeds spouse Roth IRA limit ({money(roth_phase[1])}). Consider backdoor Roth.")
                    backdoor_roth_spouse = st.checkbox(f"Spouse Backdoor Roth ({money(base_ira_spouse)}/yr)", key="backdoor_roth_s")
                else:
                    st.warning(f"Income exceeds spouse Roth IRA limit. Backdoor not recommended — spouse's Traditional IRA balance of {money(current_trad_ira_spouse)} triggers pro-rata taxation.")
            elif roth_ira_limit_spouse < base_ira_spouse:
                st.info(f"Spouse Roth IRA partially phased out: max {money(roth_ira_limit_spouse)} (of {money(base_ira_spouse)})")
                if current_trad_ira_spouse <= 0:
                    backdoor_roth_spouse = st.checkbox(f"Spouse Backdoor Roth for full {money(base_ira_spouse)}", key="backdoor_roth_s")

            effective_roth_limit_spouse = base_ira_spouse if backdoor_roth_spouse else roth_ira_limit_spouse

            max_ira_spouse = st.checkbox(f"Maximize Spouse IRA", key="max_ira_s")
            if max_ira_spouse:
                contrib_roth_ira_spouse = effective_roth_limit_spouse
                contrib_trad_ira_spouse = 0.0
                if contrib_roth_ira_spouse > 0:
                    label_s = "Backdoor Roth" if backdoor_roth_spouse else "Roth IRA"
                    st.write(f"Spouse {label_s}: **{money(contrib_roth_ira_spouse)}**")
            elif backdoor_roth_spouse:
                contrib_roth_ira_spouse = float(base_ira_spouse)
                contrib_trad_ira_spouse = 0.0
                st.write(f"Spouse Backdoor Roth: **{money(contrib_roth_ira_spouse)}**")
            else:
                contrib_trad_ira_spouse = st.number_input("Spouse Traditional IRA", value=0.0, step=500.0, max_value=float(base_ira_spouse), key="trad_ira_s")
                contrib_roth_ira_spouse = st.number_input("Spouse Roth IRA", value=0.0, step=500.0, max_value=float(effective_roth_limit_spouse), key="roth_ira_s")
                if contrib_trad_ira_spouse + contrib_roth_ira_spouse > base_ira_spouse:
                    st.warning(f"Total spouse IRA exceeds {dt.date.today().year} limit of {money(base_ira_spouse)}")
    else:
        contrib_401k_spouse = 0.0
        contrib_401k_spouse_pct = 0.0
        contrib_401k_pretax_spouse = 0.0
        contrib_401k_roth_spouse = 0.0
        employer_match_spouse = 0.0
        contrib_roth_ira_spouse = 0.0
        contrib_trad_ira_spouse = 0.0
        employer_match_rate_spouse = 0.0
        employer_match_upto_spouse = 0.0
        backdoor_roth_spouse = False
        effective_roth_limit_spouse = 0.0
        base_ira_spouse = 0.0
        roth_ira_limit_spouse = 0.0
        limit_401k_spouse = 0.0

    st.divider()
    total_annual_contrib = (contrib_401k_filer + employer_match + contrib_401k_spouse + employer_match_spouse +
                           contrib_trad_ira + contrib_roth_ira +
                           contrib_trad_ira_spouse + contrib_roth_ira_spouse +
                           contrib_hsa + contrib_taxable)
    savings_rate = total_annual_contrib / total_income * 100 if total_income > 0 else 0

    # Compute inherited IRA distributions (needed for tax estimation)
    _curr_inherited_dist = 0.0
    if inherited_ira_filer > 0:
        _f_dist = calc_inherited_ira_distribution(inherited_ira_filer, inherited_ira_rule_filer_code,
                    inherited_ira_years_filer, current_age, inherited_ira_rmd_required_filer,
                    inherited_ira_additional_filer)
        _curr_inherited_dist += _f_dist["total_distribution"]
    if inherited_ira_spouse > 0:
        _s_dist = calc_inherited_ira_distribution(inherited_ira_spouse, inherited_ira_rule_spouse_code,
                    inherited_ira_years_spouse, spouse_age or current_age, inherited_ira_rmd_required_spouse,
                    inherited_ira_additional_spouse)
        _curr_inherited_dist += _s_dist["total_distribution"]

    # Compute AGI components for tax estimation
    _curr_pretax_ded = (contrib_401k_pretax_filer + contrib_401k_pretax_spouse +
                        contrib_trad_ira + contrib_trad_ira_spouse + contrib_hsa)
    _curr_cash_interest = current_cash * cash_interest_rate
    _curr_dividends = current_taxable * dividend_yield
    _curr_cap_gain_dist = current_taxable * annual_cap_gain_pct
    _curr_taxable_other = other_income if not other_income_tax_free else 0.0
    _curr_taxable_income = max(0, total_income - other_income + _curr_taxable_other + _curr_inherited_dist + _curr_cash_interest - _curr_pretax_ded)
    _curr_filer_65 = current_age >= 65
    _curr_spouse_65 = (spouse_age or 0) >= 65
    _curr_std_ded = get_std_deduction(filing_status, _curr_filer_65, _curr_spouse_65)
    _curr_agi = _curr_taxable_income + _curr_dividends + _curr_cap_gain_dist

    st.divider()
    deficit_action = st.radio("If spending exceeds income in any year:",
        ["Reduce savings first, then liquidate investments",
         "Keep savings, liquidate investments to cover deficit"],
        key="deficit_action", horizontal=False,
        help="In years where total outflow exceeds income, how should the projection handle the shortfall?")

    st.divider()
    st.markdown("### Deductions")
    itemize_deductions = st.checkbox("Itemize deductions (instead of standard deduction)", key="itemize_ded",
        help="Check if your itemized deductions exceed the standard deduction.")
    property_tax = 0.0
    medical_expenses = 0.0
    charitable = 0.0
    if itemize_deductions:
        _curr_mtg_interest, _ = calc_mortgage_interest_for_year(mortgage_balance, mortgage_rate, mortgage_payment_annual)
        # Estimate state tax for SALT display
        _est_state_tax = calc_state_tax(max(0, _curr_agi - _curr_std_ded), state_tax_rate)
        id1, id2 = st.columns(2)
        with id1:
            st.write(f"**State income tax (est.):** {money(_est_state_tax)}")
            property_tax = st.number_input("Annual Property Tax", value=0.0, step=500.0, key="property_tax")
            _salt = min(10000.0, _est_state_tax + property_tax)
            st.caption(f"SALT deduction (capped at $10,000): {money(_salt)}")
        with id2:
            st.write(f"**Mortgage interest:** {money(_curr_mtg_interest)}")
            medical_expenses = st.number_input("Annual Medical Expenses", value=0.0, step=500.0, key="medical_exp",
                help="Only the amount exceeding 7.5% of AGI is deductible")
            charitable = st.number_input("Annual Charitable Contributions", value=0.0, step=500.0, key="charitable")
        # Compute itemized total
        _medical_ded = max(0.0, medical_expenses - _curr_agi * 0.075)
        _itemized_total = _salt + _curr_mtg_interest + _medical_ded + charitable
        st.write(f"**Itemized total:** {money(_itemized_total)}  |  **Standard deduction:** {money(_curr_std_ded)}")
        if _itemized_total > _curr_std_ded:
            st.success(f"Itemizing saves {money(_itemized_total - _curr_std_ded)} in deductions")
        else:
            st.warning(f"Standard deduction is higher by {money(_curr_std_ded - _itemized_total)}. Consider unchecking.")
    else:
        st.caption(f"Standard deduction: {money(_curr_std_ded)}")

    # Final current-year tax estimate (using proper deduction)
    _curr_deduction = _curr_std_ded
    if itemize_deductions:
        _curr_mtg_interest, _ = calc_mortgage_interest_for_year(mortgage_balance, mortgage_rate, mortgage_payment_annual)
        _est_state_tax = calc_state_tax(max(0, _curr_agi - _curr_std_ded), state_tax_rate)
        _salt = min(10000.0, _est_state_tax + property_tax)
        _medical_ded = max(0.0, medical_expenses - _curr_agi * 0.075)
        _itemized_total = _salt + _curr_mtg_interest + _medical_ded + charitable
        if _itemized_total > _curr_std_ded:
            _curr_deduction = _itemized_total

    _curr_ordinary_taxable = max(0, _curr_taxable_income + _curr_cash_interest - _curr_deduction)
    _curr_fed_tax = calc_federal_tax(_curr_ordinary_taxable, filing_status)
    _curr_fed_tax += calc_cg_tax(_curr_dividends + _curr_cap_gain_dist, _curr_ordinary_taxable, filing_status)
    _curr_state_tax = calc_state_tax(max(0, _curr_agi - _curr_deduction), state_tax_rate)
    _curr_fica = calc_fica(salary_filer, salary_spouse, filing_status)
    _curr_total_tax = _curr_fed_tax + _curr_state_tax + _curr_fica
    _curr_future_exp = sum(fe["amount"] for fe in future_expenses if fe["start_age"] <= current_age < fe["end_age"])
    _curr_total_outflow = current_living_expenses + mortgage_payment_annual + _curr_future_exp + _curr_total_tax + total_annual_contrib
    _curr_surplus = total_income + _curr_inherited_dist - _curr_total_outflow

    st.markdown("### Annual Cash Flow Summary")
    if _curr_future_exp != 0:
        cf1, cf2, cf3, cf4, cf5, cf6 = st.columns(6)
    else:
        cf1, cf2, cf3, cf4, cf5 = st.columns(5)
        cf6 = None
    with cf1: st.metric("Income", money(total_income))
    with cf2: st.metric("Living + Mortgage", money(current_living_expenses + mortgage_payment_annual))
    if cf6:
        with cf3: st.metric("Other Expenses", money(_curr_future_exp))
        with cf4: st.metric("Est. Taxes", money(_curr_total_tax))
        with cf5: st.metric("Total Savings", money(total_annual_contrib))
        with cf6: st.metric("Surplus / (Deficit)", money(_curr_surplus), delta_color="normal")
    else:
        with cf3: st.metric("Est. Taxes", money(_curr_total_tax))
        with cf4: st.metric("Total Savings", money(total_annual_contrib))
        with cf5: st.metric("Surplus / (Deficit)", money(_curr_surplus), delta_color="normal")

    # ── Spending validation: flag if surplus is large but no savings to show for it ──
    if _curr_surplus > 0:
        _existing_taxable_savings = current_taxable + current_cash
        _est_working_years = max(1, current_age - 22)  # rough estimate
        # If surplus is meaningful AND existing savings are very low relative to
        # what years of surplus would have produced, spending is likely understated
        _expected_min_savings = _curr_surplus * min(_est_working_years, 10) * 0.5  # conservative: half of surplus × years, capped at 10 yrs
        if _curr_surplus > 5000 and _existing_taxable_savings < _expected_min_savings:
            st.error(
                f"**Spending Check:** Your income exceeds stated expenses + savings by "
                f"**{money(_curr_surplus)}/yr**, yet you have only **{money(_existing_taxable_savings)}** "
                f"in taxable investments + cash. If this surplus existed for even a few years, "
                f"you should have significantly more in savings. "
                f"Most people underestimate spending — consider whether your actual living expenses "
                f"are closer to **{money(current_living_expenses + _curr_surplus)}** "
                f"(current estimate + surplus)."
            )

    contrib_cash_surplus = 0.0
    if _curr_surplus > 0:
        surplus_dest = st.radio("Reinvest annual surplus:", ["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"],
                                key="surplus_dest", horizontal=True,
                                help="Surplus income above expenses and planned savings. Default reinvests in taxable brokerage.")
        if surplus_dest == "Taxable Brokerage":
            contrib_taxable += _curr_surplus
            total_annual_contrib += _curr_surplus
            st.success(f"Adding {money(_curr_surplus)}/yr surplus to taxable investments")
        elif surplus_dest == "Cash/Savings":
            contrib_cash_surplus = _curr_surplus
            total_annual_contrib += _curr_surplus
            st.info(f"Adding {money(_curr_surplus)}/yr surplus to cash savings")
        else:
            st.warning(f"Not reinvesting {money(_curr_surplus)}/yr surplus — this income will be lost from projections.")
    elif _curr_surplus < 0:
        st.warning(f"Annual deficit of {money(abs(_curr_surplus))}. Expenses + savings exceed income.")

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
    pretax_deductions = income_info.get("pretax_deductions", 0) if income_info else 0
    inf_rate = income_info.get("inflation", 0.03) if income_info else 0.03
    fut_expenses = income_info.get("future_expenses", []) if income_info else []
    div_yield = income_info.get("dividend_yield", 0.015) if income_info else 0.015
    ann_cg_pct = income_info.get("annual_cap_gain_pct", 0.0) if income_info else 0.0
    cash_int_rate = income_info.get("cash_interest_rate", 0.04) if income_info else 0.04
    reinvest_inv_income = income_info.get("reinvest_inv_income", True) if income_info else True
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

    rows = []
    retire_age_accum = current_age + years_to_ret
    for yr in range(years_to_ret + 1):
        age = current_age + yr
        year = dt.date.today().year + yr
        is_retire_year = (age == retire_age_accum) and years_to_ret > 0
        sf = (1 + salary_growth_rate) ** yr if yr > 0 else 1.0
        inf_f = (1 + inf_rate) ** yr

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
        yr_inv_income = yr_dividends + yr_cap_gain_dist + yr_cash_interest

        # Cap gain distributions add to basis (already taxed)
        brokerage_basis += yr_cap_gain_dist

        # Social Security (prorated if starting mid-year in retirement year)
        # SSDI: pays 100% PIA from current age, converts to regular SS at FRA
        yr_ss_filer = 0.0
        if ss_filer_pia > 0:
            if ssdi_filer_flag:
                # SSDI pays 100% PIA immediately, growing with inflation
                yr_ss_filer = ss_filer_pia * inf_f
            elif age >= ss_filer_claim:
                ss_annual = ss_filer_pia * ss_claim_factor(ss_filer_claim) * inf_f
                # If claiming starts at retirement, prorate for post-birthday portion
                if is_retire_year and age == ss_filer_claim:
                    yr_ss_filer = ss_annual * retire_frac
                else:
                    yr_ss_filer = ss_annual
        yr_ss_spouse = 0.0
        if ss_spouse_pia > 0:
            if ssdi_spouse_flag:
                yr_ss_spouse = ss_spouse_pia * inf_f
            elif spouse_age_yr >= ss_spouse_claim:
                ss_sp_annual = ss_spouse_pia * ss_claim_factor(ss_spouse_claim) * inf_f
                # If spouse claiming starts at their retirement, prorate
                if is_spouse_retire_year and spouse_age_yr == ss_spouse_claim:
                    yr_ss_spouse = ss_sp_annual * (1.0 - spouse_work_frac)
                else:
                    yr_ss_spouse = ss_sp_annual
        yr_ss = yr_ss_filer + yr_ss_spouse

        # Calculate income and non-savings expenses
        yr_wages = base_salary_filer * sf * work_frac + base_salary_spouse * sf * spouse_work_frac
        yr_spendable_inv_income = 0.0 if reinvest_inv_income else yr_inv_income
        # Other income (disability, inheritance, etc.) — ends after N years if specified
        if base_other_income > 0 and (other_income_years == 0 or yr < other_income_years):
            yr_other_income = base_other_income * inf_f if other_income_inflation else base_other_income
        else:
            yr_other_income = 0.0
        yr_income = yr_wages + yr_ss + yr_spendable_inv_income + yr_other_income
        yr_living = base_expenses * inf_f
        yr_mortgage = base_mortgage if yr < mortgage_yrs_left else 0.0

        yr_future_exp = 0.0
        for fe in fut_expenses:
            if fe["start_age"] <= age < fe["end_age"]:
                yrs_from_start = age - fe["start_age"]
                amt = fe["amount"] * ((1 + inf_rate) ** yrs_from_start) if fe["inflates"] else fe["amount"]
                yr_future_exp += amt

        # Inherited IRA distributions (before tax calc — they are ordinary income)
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
                _est_base = _est_taxable_wages + yr_cash_interest + yr_inherited_dist + yr_dividends + yr_cap_gain_dist + _est_other_taxable
                _est_taxable_ss = yr_ss * 0.85
                _est_agi = _est_base + _est_taxable_ss
                room = max(0, accum_conv_target_agi * inf_f - _est_agi)
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
                _ac_no_conv_tax = calc_year_taxes(yr_ss, _ac_pretax_inc, _ac_cg_inc,
                                                   yr_cash_interest, filing, _ac_filer65, _ac_sp65,
                                                   inf_f, st_rate)["total_tax"]
                _ac_with_conv_tax = calc_year_taxes(yr_ss, _ac_pretax_inc + accum_conversion_this_year,
                                                     _ac_cg_inc, yr_cash_interest, filing,
                                                     _ac_filer65, _ac_sp65, inf_f, st_rate)["total_tax"]
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
        yr_ordinary_inv_income = yr_cash_interest
        yr_cap_gain_income = yr_dividends + yr_cap_gain_dist
        yr_taxable_other = yr_other_income if not other_income_tax_free else 0.0
        # SS taxability: up to 85% based on provisional income
        pretax_for_ss = taxable_wages + yr_ordinary_inv_income + yr_inherited_dist + yr_cap_gain_income + yr_taxable_other + accum_conversion_this_year
        taxable_ss = calc_taxable_ss(pretax_for_ss, yr_ss, filing)
        total_ordinary = taxable_wages + yr_ordinary_inv_income + yr_inherited_dist + yr_taxable_other + taxable_ss + accum_conversion_this_year
        yr_agi = total_ordinary + yr_cap_gain_income
        std_ded = get_std_deduction(filing, filer_65, spouse_65_yr, inf_f)

        # Determine deduction: itemized vs standard
        deduction = std_ded
        if use_itemize and yr < mortgage_yrs_left + 5:  # itemize while it helps
            yr_mtg_interest, mtg_bal_end = calc_mortgage_interest_for_year(mtg_bal_track, mtg_rate, base_mortgage)
            est_state_tax = calc_state_tax(max(0, yr_agi - std_ded), st_rate)
            yr_property_tax = base_property_tax * inf_f
            salt = min(10000.0 * inf_f, est_state_tax + yr_property_tax)
            yr_medical = base_medical * inf_f
            medical_ded = max(0.0, yr_medical - yr_agi * 0.075)
            yr_charitable = base_charitable * inf_f
            itemized = salt + yr_mtg_interest + medical_ded + yr_charitable
            if itemized > std_ded:
                deduction = itemized
            mtg_bal_track = mtg_bal_end
        elif mtg_bal_track > 0 and yr_mortgage > 0:
            # Still track mortgage balance even if not itemizing
            _, mtg_bal_track = calc_mortgage_interest_for_year(mtg_bal_track, mtg_rate, base_mortgage)

        ordinary_taxable = max(0, total_ordinary - deduction)
        fed_tax = calc_federal_tax(ordinary_taxable, filing, inf_f)
        fed_tax += calc_cg_tax(yr_cap_gain_income, ordinary_taxable, filing, inf_f)
        state_tax = calc_state_tax(max(0, yr_agi - deduction), st_rate)
        # FICA: SS 6.2% (up to wage base per worker) + Medicare 1.45% + Additional Medicare 0.9%
        yr_wages_filer = base_salary_filer * sf * work_frac
        yr_wages_spouse = base_salary_spouse * sf * spouse_work_frac
        yr_fica = calc_fica(yr_wages_filer, yr_wages_spouse, filing, inf_f)
        yr_taxes = fed_tax + state_tax + yr_fica

        # Non-savings expenses (subtract any conversion tax already withheld from the conversion)
        yr_fixed_expenses = yr_living + yr_mortgage + yr_future_exp + yr_taxes - accum_conv_tax_withheld
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

        # Apply actual contributions and growth
        if yr > 0:
            bal_pretax += c_pretax
            bal_roth += c_roth
            bal_brokerage += c_taxable
            brokerage_basis += c_taxable
            bal_cash += c_cash_contrib
            bal_hsa += c_hsa
            yr_return = accum_return_sequence[yr] if accum_return_sequence else pre_ret_return
            bal_pretax *= (1 + yr_return)
            bal_roth *= (1 + yr_return)
            bal_brokerage *= (1 + yr_return)
            bal_cash *= (1 + cash_int_rate)
            bal_hsa *= (1 + yr_return)
            # If spending investment income, withdraw it from balances after growth
            if not reinvest_inv_income:
                bal_brokerage -= (yr_dividends + yr_cap_gain_dist)
                bal_cash -= yr_cash_interest
            # Grow remaining inherited IRA balances
            for idx in range(len(inherited_iras)):
                if iira_bals[idx] > 0:
                    iira_bals[idx] *= (1 + yr_return)

        bal_taxable = bal_brokerage + bal_cash
        bal_inherited_total = sum(iira_bals)
        total = bal_pretax + bal_roth + bal_taxable + bal_hsa + bal_inherited_total
        unrealized_gain = max(0, bal_brokerage - brokerage_basis)

        row = {
            "Year": year, "Age": age,
        }
        if accum_spouse_age:
            row["Spouse Age"] = spouse_age_yr
        row.update({
            "Income": round(yr_wages, 0),
            "Other Income": round(yr_other_income, 0),
            "SS Income": round(yr_ss, 0),
            "Inv Income": round(yr_inv_income, 0),
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
            "Surplus": round(yr_income + yr_inherited_dist - yr_total_expenses, 0),
            "Bal Pre-Tax": round(bal_pretax, 0), "Bal Roth": round(bal_roth, 0),
            "Bal Taxable": round(bal_taxable, 0), "Basis": round(brokerage_basis, 0),
            "Unreal Gain": round(unrealized_gain, 0),
            "Bal HSA": round(bal_hsa, 0),
        })
        if any(ira["balance"] > 0 for ira in inherited_iras):
            row["Inherited Dist"] = round(yr_inherited_dist, 0)
            row["Bal Inherited"] = round(bal_inherited_total, 0)
        row["Total Balance"] = round(total, 0)
        rows.append(row)
    return {"rows": rows, "final_brokerage": bal_brokerage, "final_cash": bal_cash, "final_basis": brokerage_basis,
            "final_inherited": sum(iira_bals),
            "total_converted": accum_total_converted,
            "inherited_iras_state": [{"balance": iira_bals[i], "rule": iira_rules[i],
                                      "years_remaining": iira_yrs[i], "owner_age": iira_ages[i] + years_to_ret,
                                      "owner_was_taking_rmds": iira_rmd_required[i],
                                      "additional_distribution": iira_additional[i]}
                                     for i in range(len(inherited_iras))]}

# Build contribution dict (used by both Tab 3 and Tab 4)
_annual_pretax_contrib_filer = contrib_401k_pretax_filer + employer_match + contrib_trad_ira
_annual_pretax_contrib_spouse = contrib_401k_pretax_spouse + employer_match_spouse + contrib_trad_ira_spouse
_annual_pretax_contrib = _annual_pretax_contrib_filer + _annual_pretax_contrib_spouse
_annual_roth_contrib_filer = contrib_401k_roth_filer + contrib_roth_ira
_annual_roth_contrib_spouse = contrib_401k_roth_spouse + contrib_roth_ira_spouse
_annual_roth_contrib = _annual_roth_contrib_filer + _annual_roth_contrib_spouse

_contrib_dict = {
    "pretax": _annual_pretax_contrib,
    "roth": _annual_roth_contrib,
    "taxable": contrib_taxable,
    "cash": contrib_cash_surplus,
    "hsa": contrib_hsa,
    "pretax_filer": _annual_pretax_contrib_filer,
    "pretax_spouse": _annual_pretax_contrib_spouse,
    "roth_filer": _annual_roth_contrib_filer,
    "roth_spouse": _annual_roth_contrib_spouse,
}
_start_balances = {
    "pretax": total_pretax,
    "roth": total_roth,
    "taxable": total_taxable,
    "brokerage": current_taxable,
    "cash": current_cash,
    "brokerage_basis": taxable_basis,
    "hsa": current_hsa,
}
# Pre-tax deductions that reduce taxable income (401k pretax + trad IRA deductible + HSA)
_pretax_deductions = (contrib_401k_pretax_filer + contrib_401k_pretax_spouse +
                      contrib_trad_ira + contrib_trad_ira_spouse + contrib_hsa)
_effective_mortgage_years = mortgage_years_remaining
if _effective_mortgage_years == 0 and mortgage_payment_annual > 0 and mortgage_balance > 0 and mortgage_rate > 0:
    _effective_mortgage_years = calc_mortgage_years(mortgage_balance, mortgage_rate, mortgage_payment_annual)
elif _effective_mortgage_years == 0 and mortgage_payment_annual > 0:
    _effective_mortgage_years = 30  # default if we can't compute

_income_info = {
    "total_income": total_income,
    "salary_filer": salary_filer,
    "salary_spouse": salary_spouse,
    "other_income": other_income,
    "other_income_tax_free": other_income_tax_free,
    "other_income_inflation": other_income_inflation,
    "other_income_years": other_income_years,
    "other_income_recipient": other_income_recipient,
    "living_expenses": current_living_expenses,
    "mortgage_annual": mortgage_payment_annual,
    "mortgage_years": _effective_mortgage_years,
    "filing_status": filing_status,
    "state_rate": state_tax_rate,
    "pretax_deductions": _pretax_deductions,
    "inflation": inflation,
    "future_expenses": future_expenses,
    "current_age": current_age,
    "dividend_yield": dividend_yield,
    "annual_cap_gain_pct": annual_cap_gain_pct,
    "cash_interest_rate": cash_interest_rate,
    "reinvest_inv_income": reinvest_inv_income,
    "deficit_action": "reduce_savings" if "Reduce" in deficit_action else "liquidate",
    "itemize": itemize_deductions,
    "mortgage_balance": mortgage_balance,
    "mortgage_rate": mortgage_rate,
    "property_tax": property_tax,
    "medical_expenses": medical_expenses,
    "charitable": charitable,
    "ss_filer_pia": ss_filer_final,
    "ss_spouse_pia": ss_spouse_final,
    "ss_filer_claim_age": ss_filer_claim_age,
    "ss_spouse_claim_age": ss_spouse_claim_age,
    "ssdi_filer": ssdi_filer,
    "ssdi_spouse": ssdi_spouse,
    "spouse_age": spouse_age,
    "filer_dob": filer_dob,
    "spouse_dob": spouse_dob,
    "retire_age": target_retirement_age,
    "spouse_retire_age": spouse_retirement_age,
    "inherited_iras": [
        {"balance": inherited_ira_filer, "rule": inherited_ira_rule_filer_code,
         "years_remaining": inherited_ira_years_filer, "owner_age": current_age,
         "owner_was_taking_rmds": inherited_ira_rmd_required_filer,
         "additional_distribution": inherited_ira_additional_filer},
        {"balance": inherited_ira_spouse, "rule": inherited_ira_rule_spouse_code,
         "years_remaining": inherited_ira_years_spouse, "owner_age": spouse_age or current_age,
         "owner_was_taking_rmds": inherited_ira_rmd_required_spouse,
         "additional_distribution": inherited_ira_additional_spouse},
    ],
}

with tab3:
    st.subheader("Accumulation Projection")

    if st.button("Run Projection", type="primary"):
        accum_result = run_accumulation(current_age, years_to_retirement, _start_balances, _contrib_dict, salary_growth, pre_retire_return, _income_info)
        st.session_state.projection_results = accum_result

    if st.session_state.projection_results:
        accum_result = st.session_state.projection_results
        rows = accum_result["rows"] if isinstance(accum_result, dict) else accum_result
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        final = rows[-1]
        st.markdown("### At Retirement")
        if final.get("Bal Inherited", 0) > 0:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1: st.metric("Pre-Tax", money(final["Bal Pre-Tax"]))
            with col2: st.metric("Roth", money(final["Bal Roth"]))
            with col3: st.metric("Taxable", money(final["Bal Taxable"]))
            with col4: st.metric("HSA", money(final["Bal HSA"]))
            with col5: st.metric("Inherited IRA", money(final["Bal Inherited"]))
            with col6: st.metric("Total", money(final["Total Balance"]))
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Pre-Tax", money(final["Bal Pre-Tax"]))
            with col2: st.metric("Roth", money(final["Bal Roth"]))
            with col3: st.metric("Taxable", money(final["Bal Taxable"]))
            with col4: st.metric("HSA", money(final["Bal HSA"]))
            with col5: st.metric("Total", money(final["Total Balance"]))

        # Retirement params builder (shared by MC and Savings Vehicle Optimizer)
        _opt_retire_exp = current_living_expenses * 0.80 * ((1 + inflation) ** years_to_retirement)
        _opt_inf_retire = (1 + inflation) ** years_to_retirement

        def _opt_retire_params(accum_inherited_state=None):
            return {
                "retire_age": target_retirement_age,
                "life_expectancy": life_expectancy,
                "retire_year": dt.date.today().year + years_to_retirement,
                "inflation": inflation,
                "post_retire_return": post_retire_return,
                "filing_status": filing_status,
                "state_tax_rate": state_tax_rate,
                "expenses_at_retirement": _opt_retire_exp,
                "ss_filer_fra": ss_filer_final * _opt_inf_retire,
                "ss_spouse_fra": ss_spouse_final * _opt_inf_retire,
                "ss_filer_claim_age": ss_filer_claim_age,
                "ss_spouse_claim_age": ss_spouse_claim_age,
                "ssdi_filer": ssdi_filer,
                "ssdi_spouse": ssdi_spouse,
                "other_income": 0.0 if (other_income_years > 0 and other_income_years <= years_to_retirement) else (other_income * _opt_inf_retire if other_income_inflation else other_income),
                "other_income_tax_free": other_income_tax_free,
                "other_income_inflation": other_income_inflation,
                "other_income_years": max(0, other_income_years - years_to_retirement) if other_income_years > 0 else 0,
                "pension_filer_at_retire": pension_filer_annual * _opt_inf_retire,
                "pension_filer_start_age": pension_filer_start_age,
                "pension_filer_cola": pension_filer_cola,
                "pension_spouse_at_retire": pension_spouse_annual * _opt_inf_retire,
                "pension_spouse_start_age": pension_spouse_start_age,
                "pension_spouse_cola": pension_spouse_cola,
                "spouse_age_at_retire": (spouse_age + years_to_retirement) if spouse_age else None,
                "mortgage_payment": mortgage_payment_annual,
                "mortgage_years_at_retire": max(0, _effective_mortgage_years - years_to_retirement),
                "home_value_at_retire": home_value * ((1 + home_appreciation) ** years_to_retirement),
                "home_appreciation": home_appreciation,
                "future_expenses": future_expenses,
                "dividend_yield": dividend_yield,
                "cash_interest_rate": cash_interest_rate,
                "inherited_iras": accum_inherited_state or [],
            }

        # ── Monte Carlo Stress Test ──
        st.divider()
        st.markdown("### Monte Carlo Stress Test")
        _mc3_c1, _mc3_c2, _mc3_c3, _mc3_c4 = st.columns([1, 1, 1, 1])
        with _mc3_c1:
            _mc3_run = st.checkbox("Run Monte Carlo", value=False, key="mc3_run")
        with _mc3_c3:
            _mc3_nsims = st.number_input("Simulations", min_value=100, max_value=2000, value=500, step=100, key="mc3_nsims")
        with _mc3_c4:
            _mc3_std = st.number_input("Return Std Dev (%)", min_value=5.0, max_value=25.0, value=12.0, step=1.0, key="mc3_std") / 100.0

        if _mc3_run:
            import copy as _mc3_copy
            _mc3_n_years = years_to_retirement + (life_expectancy - target_retirement_age) + 1
            _mc3_seed = 42

            def _mc3_run_fn_current(return_seq):
                """Full lifecycle MC closure for current savings plan."""
                accum_seq = return_seq[:years_to_retirement + 1]
                retire_seq = return_seq[years_to_retirement:]
                ii = _mc3_copy.deepcopy(_income_info)
                ii["return_sequence"] = accum_seq
                accum = run_accumulation(current_age, years_to_retirement,
                                         _mc3_copy.deepcopy(_start_balances),
                                         _contrib_dict, salary_growth, pre_retire_return, ii)
                ar = accum["rows"]
                af = ar[-1]
                bals = {
                    "pretax": float(af["Bal Pre-Tax"]),
                    "roth": float(af["Bal Roth"]),
                    "taxable": float(af["Bal Taxable"]),
                    "brokerage": float(accum.get("final_brokerage", af["Bal Taxable"])),
                    "cash": float(accum.get("final_cash", 0)),
                    "brokerage_basis": float(accum.get("final_basis", af["Bal Taxable"])),
                    "hsa": float(af["Bal HSA"]),
                }
                rp = _mc3_copy.deepcopy(_opt_retire_params(accum.get("inherited_iras_state", [])))
                rp["return_sequence"] = retire_seq
                return run_retirement_projection(bals, rp, ["Pre-Tax", "Taxable", "Tax-Free"])

            _mc3_progress = st.progress(0, text="Running Monte Carlo on current plan...")
            _mc3_result_current = run_monte_carlo(
                _mc3_run_fn_current, n_sims=int(_mc3_nsims),
                mean_return=pre_retire_return, return_std=_mc3_std,
                n_years=_mc3_n_years, seed=_mc3_seed)
            _mc3_progress.progress(100, text="Monte Carlo complete")
            _mc3_progress.empty()

            with _mc3_c2: st.metric("Success Rate", f"{_mc3_result_current['success_rate']:.0%}")

            st.markdown("#### Current Plan — Monte Carlo Results")
            _mc3m1, _mc3m2, _mc3m3 = st.columns(3)
            with _mc3m1: st.metric("Median Estate", money(_mc3_result_current["median_estate"]))
            with _mc3m2: st.metric("10th Percentile", money(_mc3_result_current["p10"]))
            with _mc3m3: st.metric("90th Percentile", money(_mc3_result_current["p90"]))
            st.caption(f"Based on {int(_mc3_nsims)} simulations, mean return {pre_retire_return:.1%}, std dev {_mc3_std:.1%}")

        # ── Savings Vehicle Optimizer ──
        st.divider()
        st.markdown("### Savings Vehicle Optimizer")
        st.caption("Compares Traditional vs Roth strategies through the full lifecycle — accumulation, RMDs, and retirement taxes. "
                   "Only reallocates your existing tax-advantaged contributions; taxable/liquid savings are unchanged.")

        # Only reallocate tax-advantaged contributions — taxable stays liquid
        _opt_tax_adv_pool = (contrib_401k_filer + contrib_401k_spouse +
                              contrib_roth_ira + contrib_trad_ira +
                              contrib_roth_ira_spouse + contrib_trad_ira_spouse +
                              contrib_hsa)
        _opt_emp_match = employer_match + (employer_match_spouse if is_joint else 0.0)

        # Current marginal rate for context
        _opt_curr_taxable = max(0, _curr_agi - _curr_deduction)
        _opt_curr_marginal = get_marginal_fed_rate(_opt_curr_taxable, filing_status)

        def _build_opt_waterfall(tax_adv_pool, favor_trad):
            """Reallocate tax-advantaged pool only. Returns (alloc_list, contrib_dict, employee_pretax_ded).
            Taxable contributions are passed through unchanged."""
            rem = tax_adv_pool
            alloc = []
            pt = 0.0; ro = 0.0; hs = 0.0

            # 1. 401(k) to employer match — filer
            mc_f = min(salary_filer * employer_match_upto / 100, limit_401k, rem) if salary_filer > 0 else 0.0
            if mc_f > 0:
                if favor_trad: pt += mc_f
                else: ro += mc_f
                rem -= mc_f
                # Only show match recommendation if not already capturing the full match
                if contrib_401k_filer_pct < employer_match_upto:
                    full_match = salary_filer * employer_match_upto / 100 * employer_match_rate / 100
                    alloc.append({"Vehicle": "401(k) to Match (You)", "Recommended": round(mc_f),
                                  "Reason": f"Increase to {employer_match_upto:.0f}% to capture full {money(full_match)} match"})

            # 1b. 401(k) to employer match — spouse
            mc_s = 0.0
            if is_joint and salary_spouse > 0 and employer_match_upto_spouse > 0:
                mc_s = min(salary_spouse * employer_match_upto_spouse / 100, limit_401k_spouse, rem)
                if mc_s > 0:
                    if favor_trad: pt += mc_s
                    else: ro += mc_s
                    rem -= mc_s
                    # Only show match recommendation if not already capturing the full match
                    if contrib_401k_spouse_pct < employer_match_upto_spouse:
                        full_match_s = salary_spouse * employer_match_upto_spouse / 100 * employer_match_rate_spouse / 100
                        alloc.append({"Vehicle": "401(k) to Match (Spouse)", "Recommended": round(mc_s),
                                      "Reason": f"Increase to {employer_match_upto_spouse:.0f}% to capture full {money(full_match_s)} match"})

            # 2. HSA to max
            if hsa_limit > 0:
                h = min(hsa_limit, rem)
                hs += h; rem -= h
                alloc.append({"Vehicle": "HSA", "Recommended": round(h),
                              "Reason": "Triple tax advantage: deductible, tax-free growth & medical withdrawals"})

            # 3. Roth IRA — filer
            rl = effective_roth_limit; note = ""
            if backdoor_roth: rl = base_ira; note = " (backdoor)"
            elif roth_ira_limit < base_ira and backdoor_eligible: rl = base_ira; note = " (backdoor rec.)"
            ri = min(rl, rem)
            if rl > 0:
                ro += ri; rem -= ri
                alloc.append({"Vehicle": f"Roth IRA (You){note}", "Recommended": round(ri),
                              "Reason": "Tax-free growth & withdrawals; no RMDs"})

            # 3b. Roth IRA — spouse
            if is_joint and salary_spouse > 0 and (roth_ira_limit_spouse > 0 or backdoor_roth_spouse):
                sl = effective_roth_limit_spouse; sn = ""
                if backdoor_roth_spouse: sl = base_ira_spouse; sn = " (backdoor)"
                rs = min(sl, rem)
                if sl > 0:
                    ro += rs; rem -= rs
                    alloc.append({"Vehicle": f"Roth IRA (Spouse){sn}", "Recommended": round(rs),
                                  "Reason": "Tax-free growth & withdrawals; no RMDs"})

            # 4. 401(k) beyond match — filer
            kt = "Trad 401(k)" if favor_trad else "Roth 401(k)"
            ab_f = max(0, limit_401k - mc_f) if salary_filer > 0 else 0
            ex_f = min(ab_f, rem)
            if ex_f > 0:
                if favor_trad: pt += ex_f
                else: ro += ex_f
                rem -= ex_f
                alloc.append({"Vehicle": f"{kt} Beyond Match (You)", "Recommended": round(ex_f),
                              "Reason": "Pre-tax deduction now" if favor_trad else "Tax-free withdrawals in retirement"})

            # 4b. 401(k) beyond match — spouse
            if is_joint and salary_spouse > 0:
                ab_s = max(0, limit_401k_spouse - mc_s)
                ex_s = min(ab_s, rem)
                if ex_s > 0:
                    if favor_trad: pt += ex_s
                    else: ro += ex_s
                    rem -= ex_s
                    alloc.append({"Vehicle": f"{kt} Beyond Match (Spouse)", "Recommended": round(ex_s),
                                  "Reason": "Pre-tax deduction now" if favor_trad else "Tax-free withdrawals in retirement"})

            # Any remainder stays in taxable (edge case: pool > all tax-adv space)
            spillover = max(0, rem)

            # Employer match is always pre-tax (not from employee paycheck)
            pt += _opt_emp_match
            # Taxable = user's existing taxable contribution + any spillover
            contrib = {"pretax": pt, "roth": ro, "taxable": contrib_taxable + spillover, "hsa": hs}
            # Employee pretax deductions for tax calc = employee pre-tax 401k + HSA
            emp_pretax_ded = (pt - _opt_emp_match) + hs
            return alloc, contrib, emp_pretax_ded

        # Build both strategies
        _alloc_trad, _cd_trad, _ded_trad = _build_opt_waterfall(_opt_tax_adv_pool, True)
        _alloc_roth, _cd_roth, _ded_roth = _build_opt_waterfall(_opt_tax_adv_pool, False)

        # Adjust income info for each scenario (different pre-tax deductions = different working-year taxes)
        _ii_trad = dict(_income_info); _ii_trad["pretax_deductions"] = _ded_trad
        _ii_roth = dict(_income_info); _ii_roth["pretax_deductions"] = _ded_roth

        def _run_full_lifecycle(contrib_d, income_info_mod):
            """Run accumulation + retirement and return (accum_final_row, retire_result)."""
            accum = run_accumulation(current_age, years_to_retirement, _start_balances, contrib_d,
                                     salary_growth, pre_retire_return, income_info_mod)
            ar = accum["rows"]
            af = ar[-1]
            bals = {
                "pretax": float(af["Bal Pre-Tax"]),
                "roth": float(af["Bal Roth"]),
                "taxable": float(af["Bal Taxable"]),
                "brokerage": float(accum.get("final_brokerage", af["Bal Taxable"])),
                "cash": float(accum.get("final_cash", 0)),
                "brokerage_basis": float(accum.get("final_basis", af["Bal Taxable"])),
                "hsa": float(af["Bal HSA"]),
            }
            retire = run_retirement_projection(bals, _opt_retire_params(accum.get("inherited_iras_state", [])),
                                               ["Pre-Tax", "Taxable", "Tax-Free"])
            return af, retire

        # Run full lifecycle for all three: current, Traditional-favored, Roth-favored
        _lc_curr_af, _lc_curr_ret = _run_full_lifecycle(_contrib_dict, _income_info)
        _lc_trad_af, _lc_trad_ret = _run_full_lifecycle(_cd_trad, _ii_trad)
        _lc_roth_af, _lc_roth_ret = _run_full_lifecycle(_cd_roth, _ii_roth)

        # Determine winner by ESTATE VALUE — the metric that actually matters
        _scenarios = [
            ("Current Plan", _lc_curr_af, _lc_curr_ret, None),
            ("Traditional-Favored", _lc_trad_af, _lc_trad_ret, _alloc_trad),
            ("Roth-Favored", _lc_roth_af, _lc_roth_ret, _alloc_roth),
        ]
        _scenarios.sort(key=lambda x: x[2]["estate"], reverse=True)
        _winner_label, _winner_af, _winner_ret, _winner_alloc = _scenarios[0]
        _current_is_best = _winner_label == "Current Plan"

        # Estimate retirement marginal rates under each scenario (for display)
        def _est_retire_marginal(accum_final_row):
            pt_bal = accum_final_row["Bal Pre-Tax"]
            ss_ret = (ss_filer_final * ss_claim_factor(ss_filer_claim_age) +
                      ss_spouse_final * ss_claim_factor(ss_spouse_claim_age)) * _opt_inf_retire
            pen_ret = (pension_filer_annual + pension_spouse_annual) * _opt_inf_retire
            rmd = calc_rmd(target_retirement_age, pt_bal) if target_retirement_age >= 73 else 0
            other = pen_ret + rmd
            tss = calc_taxable_ss(other, ss_ret, filing_status)
            agi = other + tss
            std = get_std_deduction(filing_status, True, is_joint, _opt_inf_retire)
            return get_marginal_fed_rate(max(0, agi - std), filing_status, _opt_inf_retire)

        _ret_rate_trad = _est_retire_marginal(_lc_trad_af)
        _ret_rate_roth = _est_retire_marginal(_lc_roth_af)

        # Display: Rate Comparison
        rc1, rc2, rc3 = st.columns(3)
        with rc1: st.metric("Current Marginal Fed Rate", f"{_opt_curr_marginal:.0%}")
        with rc2:
            st.metric("Retirement Rate (if Trad-heavy)", f"{_ret_rate_trad:.0%}")
            st.caption(f"Pre-tax at retire: {money(_lc_trad_af['Bal Pre-Tax'])}")
        with rc3:
            st.metric("Retirement Rate (if Roth-heavy)", f"{_ret_rate_roth:.0%}")
            st.caption(f"Pre-tax at retire: {money(_lc_roth_af['Bal Pre-Tax'])}")

        # Display: Lifecycle Comparison
        st.markdown("#### Full Lifecycle Comparison")
        st.caption("Same total savings ({}) — only the tax-advantaged split changes. "
                   "Taxable savings ({}/yr) kept liquid in all scenarios. "
                   "Retirement assumes 80% expenses, Pre-Tax→Taxable→Tax-Free withdrawals.".format(
                       money(_opt_tax_adv_pool + contrib_taxable), money(contrib_taxable)))
        _compare = pd.DataFrame({
            "Metric": ["At-Retire Pre-Tax", "At-Retire Roth", "At-Retire Taxable", "At-Retire HSA",
                        "At-Retire Total", "",
                        "Retirement Taxes Paid", "Portfolio at Death",
                        "Final Estate (age {})".format(life_expectancy),
                        "Est. Retire Marginal Rate"],
            "Current Plan": [
                money(_lc_curr_af["Bal Pre-Tax"]), money(_lc_curr_af["Bal Roth"]),
                money(_lc_curr_af["Bal Taxable"]), money(_lc_curr_af["Bal HSA"]),
                money(_lc_curr_af["Total Balance"]), "",
                money(_lc_curr_ret["total_taxes"]), money(_lc_curr_ret["final_total"]),
                money(_lc_curr_ret["estate"]),
                f"{_est_retire_marginal(_lc_curr_af):.0%}"],
            "Traditional-Favored": [
                money(_lc_trad_af["Bal Pre-Tax"]), money(_lc_trad_af["Bal Roth"]),
                money(_lc_trad_af["Bal Taxable"]), money(_lc_trad_af["Bal HSA"]),
                money(_lc_trad_af["Total Balance"]), "",
                money(_lc_trad_ret["total_taxes"]), money(_lc_trad_ret["final_total"]),
                money(_lc_trad_ret["estate"]),
                f"{_ret_rate_trad:.0%}"],
            "Roth-Favored": [
                money(_lc_roth_af["Bal Pre-Tax"]), money(_lc_roth_af["Bal Roth"]),
                money(_lc_roth_af["Bal Taxable"]), money(_lc_roth_af["Bal HSA"]),
                money(_lc_roth_af["Total Balance"]), "",
                money(_lc_roth_ret["total_taxes"]), money(_lc_roth_ret["final_total"]),
                money(_lc_roth_ret["estate"]),
                f"{_ret_rate_roth:.0%}"],
        })
        st.dataframe(_compare, use_container_width=True, hide_index=True)

        # Winner announcement — estate is the bottom line
        if _current_is_best:
            _runner_up = _scenarios[1]
            _gap = _winner_ret["estate"] - _runner_up[2]["estate"]
            st.success(f"**Your current allocation is already optimal** — {money(_gap)} higher estate than the next best strategy ({_runner_up[0]})")
        else:
            _estate_gain = _winner_ret["estate"] - _lc_curr_ret["estate"]
            _tax_diff = _lc_curr_ret["total_taxes"] - _winner_ret["total_taxes"]
            st.success(f"**Recommended: {_winner_label}** — {money(_estate_gain)} higher estate at death vs current plan")
            if _tax_diff > 0:
                st.caption(f"Also saves {money(_tax_diff)} in retirement taxes")
            elif _tax_diff < 0:
                st.caption(f"Pays {money(abs(_tax_diff))} more in retirement taxes, but higher estate from greater compounding")

            # Show recommended allocation only if there's a change to recommend
            st.markdown("#### Recommended Allocation")
            _rec_df = pd.DataFrame(_winner_alloc)
            _rec_df["Recommended"] = _rec_df["Recommended"].apply(lambda x: money(x) if isinstance(x, (int, float)) else x)
            st.dataframe(_rec_df, use_container_width=True, hide_index=True)

        st.caption(f"Tax-advantaged pool: {money(_opt_tax_adv_pool)} | "
                   f"Taxable (liquid, unchanged): {money(contrib_taxable)}")

        # ── MC on Savings Optimizer Scenarios ──
        if _mc3_run:
            st.divider()
            st.markdown("#### Monte Carlo — Strategy Comparison")
            import copy as _mc3o_copy

            _mc3o_scenarios = [
                ("Current Plan", _contrib_dict, _income_info, _lc_curr_ret["estate"]),
                ("Traditional-Favored", _cd_trad, _ii_trad, _lc_trad_ret["estate"]),
                ("Roth-Favored", _cd_roth, _ii_roth, _lc_roth_ret["estate"]),
            ]
            _mc3o_results = []
            _mc3o_progress = st.progress(0, text="Running Monte Carlo on optimizer scenarios...")
            _mc3o_total = len(_mc3o_scenarios) * int(_mc3_nsims)
            _mc3o_done = 0

            for _mc3o_label, _mc3o_cd, _mc3o_ii, _mc3o_det_estate in _mc3o_scenarios:
                _mc3o_cd_local = _mc3o_cd
                _mc3o_ii_local = _mc3o_ii

                def _mc3o_make_run_fn(cd, ii):
                    def _run_fn(return_seq):
                        accum_seq = return_seq[:years_to_retirement + 1]
                        retire_seq = return_seq[years_to_retirement:]
                        ii_mod = _mc3o_copy.deepcopy(ii)
                        ii_mod["return_sequence"] = accum_seq
                        accum = run_accumulation(current_age, years_to_retirement,
                                                 _mc3o_copy.deepcopy(_start_balances),
                                                 cd, salary_growth, pre_retire_return, ii_mod)
                        ar = accum["rows"]
                        af = ar[-1]
                        bals = {
                            "pretax": float(af["Bal Pre-Tax"]),
                            "roth": float(af["Bal Roth"]),
                            "taxable": float(af["Bal Taxable"]),
                            "brokerage": float(accum.get("final_brokerage", af["Bal Taxable"])),
                            "cash": float(accum.get("final_cash", 0)),
                            "brokerage_basis": float(accum.get("final_basis", af["Bal Taxable"])),
                            "hsa": float(af["Bal HSA"]),
                        }
                        rp = _mc3o_copy.deepcopy(_opt_retire_params(accum.get("inherited_iras_state", [])))
                        rp["return_sequence"] = retire_seq
                        return run_retirement_projection(bals, rp, ["Pre-Tax", "Taxable", "Tax-Free"])
                    return _run_fn

                _mc3o_fn = _mc3o_make_run_fn(_mc3o_cd_local, _mc3o_ii_local)
                _mc3o_mc = run_monte_carlo(
                    _mc3o_fn, n_sims=int(_mc3_nsims),
                    mean_return=pre_retire_return, return_std=_mc3_std,
                    n_years=_mc3_n_years, seed=_mc3_seed)
                _mc3o_done += int(_mc3_nsims)
                _mc3o_progress.progress(min(100, int(_mc3o_done / _mc3o_total * 100)),
                                         text=f"Completed {_mc3o_label}...")
                _mc3o_results.append((_mc3o_label, _mc3o_det_estate, _mc3o_mc))

            _mc3o_progress.empty()

            _mc3o_table = pd.DataFrame([
                {
                    "Strategy": label,
                    "Det. Estate": money(det_est),
                    "MC Median": money(mc["median_estate"]),
                    "MC 10th Pctl": money(mc["p10"]),
                    "MC 90th Pctl": money(mc["p90"]),
                    "MC Success Rate": f"{mc['success_rate']:.0%}",
                }
                for label, det_est, mc in _mc3o_results
            ])
            st.dataframe(_mc3o_table, use_container_width=True, hide_index=True)

            # Check if MC changes the winner
            _mc3o_det_winner = max(_mc3o_results, key=lambda x: x[1])[0]
            _mc3o_mc_winner = max(_mc3o_results, key=lambda x: x[2]["median_estate"])[0]
            if _mc3o_det_winner != _mc3o_mc_winner:
                st.warning(f"MC changes the winner: deterministic favors **{_mc3o_det_winner}**, "
                           f"but MC median favors **{_mc3o_mc_winner}**")
            st.caption(f"All scenarios use the same {int(_mc3_nsims)} return sequences for fair comparison (seed={_mc3_seed})")

with tab4:
    st.subheader("Retirement Readiness Analysis")

    # Always compute balances at retirement fresh from current inputs
    _accum_result = run_accumulation(current_age, years_to_retirement, _start_balances, _contrib_dict, salary_growth, pre_retire_return, _income_info)
    _accum_rows = _accum_result["rows"]
    _retire_final = _accum_rows[-1] if _accum_rows else {"Bal Pre-Tax": total_pretax, "Bal Roth": total_roth, "Bal Taxable": total_taxable, "Bal HSA": current_hsa, "Total Balance": total_pretax + total_roth + total_taxable + current_hsa}

    st.markdown(f"**Projected balances at retirement (age {target_retirement_age}):**")
    _retire_inherited = _accum_result.get("final_inherited", 0)
    if _retire_inherited > 0:
        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
        with rc1: st.metric("Pre-Tax", money(_retire_final["Bal Pre-Tax"]))
        with rc2: st.metric("Roth", money(_retire_final["Bal Roth"]))
        with rc3: st.metric("Taxable", money(_retire_final["Bal Taxable"]))
        with rc4: st.metric("HSA", money(_retire_final["Bal HSA"]))
        with rc5: st.metric("Inherited IRA", money(_retire_inherited))
        with rc6: st.metric("Total", money(_retire_final["Total Balance"]))
    else:
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1: st.metric("Pre-Tax", money(_retire_final["Bal Pre-Tax"]))
        with rc2: st.metric("Roth", money(_retire_final["Bal Roth"]))
        with rc3: st.metric("Taxable", money(_retire_final["Bal Taxable"]))
        with rc4: st.metric("HSA", money(_retire_final["Bal HSA"]))
        with rc5: st.metric("Total", money(_retire_final["Total Balance"]))

    st.divider()
    st.markdown("### Retirement Income & Expense Inputs (Today's Dollars)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Current living expenses (from Tab 1):** {money(current_living_expenses)}")
        retirement_pct = st.slider("% of Current Expenses in Retirement", 70, 100, 80, key="ret_pct")
        retirement_expenses = current_living_expenses * retirement_pct / 100
        inflated_expenses = retirement_expenses * ((1 + inflation) ** years_to_retirement)
        st.write(f"**Retirement expenses (today's $):** {money(retirement_expenses)}")
        st.write(f"**Year 1 of retirement (future $):** {money(inflated_expenses)}")

        if pension_filer_annual > 0 or pension_spouse_annual > 0:
            st.divider()
            st.markdown("**Pension Summary (from Tab 1)**")
            if pension_filer_annual > 0:
                pen_f_retire = pension_filer_annual * ((1 + inflation) ** years_to_retirement)
                cola_desc = "Fixed (no increase)" if pension_filer_cola == 0 else f"COLA {pension_filer_cola*100:.1f}%/yr"
                st.write(f"Your pension: {money(pension_filer_annual)}/yr today → {money(pen_f_retire)}/yr at retire, starts age {pension_filer_start_age}, {cola_desc}")
            if pension_spouse_annual > 0:
                pen_s_retire = pension_spouse_annual * ((1 + inflation) ** years_to_retirement)
                cola_desc_s = "Fixed (no increase)" if pension_spouse_cola == 0 else f"COLA {pension_spouse_cola*100:.1f}%/yr"
                st.write(f"Spouse pension: {money(pension_spouse_annual)}/yr today → {money(pen_s_retire)}/yr at retire, starts age {pension_spouse_start_age}, {cola_desc_s}")

    with col2:
        st.markdown("**Social Security (from Tab 1)**")
        ss_filer_retire_yr = ss_filer_final * ((1 + inflation) ** years_to_retirement)
        if ssdi_filer:
            adj_ss_filer = ss_filer_retire_yr  # 100% PIA for SSDI
            st.write(f"Your PIA (today's $): **{money(ss_filer_final)}**/yr")
            st.write(f"**SSDI** — receiving 100% PIA now, converts to retirement SS at FRA")
            st.write(f"Benefit at retirement (future $): **{money(adj_ss_filer)}**/yr")
        else:
            adj_ss_filer = ss_filer_retire_yr * ss_claim_factor(ss_filer_claim_age)
            st.write(f"Your PIA (today's $): **{money(ss_filer_final)}**/yr")
            st.write(f"Claim age: **{ss_filer_claim_age}** ({ss_claim_factor(ss_filer_claim_age):.0%} of PIA)")
            st.write(f"Benefit at retirement (future $): **{money(adj_ss_filer)}**/yr")
        if is_joint:
            ss_spouse_retire_yr = ss_spouse_final * ((1 + inflation) ** years_to_retirement)
            if ssdi_spouse:
                adj_ss_spouse = ss_spouse_retire_yr
                st.write(f"Spouse PIA (today's $): **{money(ss_spouse_final)}**/yr")
                st.write(f"**Spouse SSDI** — receiving 100% PIA now, converts to retirement SS at FRA")
                st.write(f"Spouse benefit at retirement (future $): **{money(adj_ss_spouse)}**/yr")
            else:
                adj_ss_spouse = ss_spouse_retire_yr * ss_claim_factor(ss_spouse_claim_age)
                st.write(f"Spouse PIA (today's $): **{money(ss_spouse_final)}**/yr")
                st.write(f"Spouse claim age: **{ss_spouse_claim_age}** ({ss_claim_factor(ss_spouse_claim_age):.0%} of PIA)")
                st.write(f"Spouse benefit at retirement (future $): **{money(adj_ss_spouse)}**/yr")
        else:
            adj_ss_spouse = 0

    # Inflate pensions to retirement year
    pen_filer_at_retire = pension_filer_annual * ((1 + inflation) ** years_to_retirement)
    pen_spouse_at_retire = pension_spouse_annual * ((1 + inflation) ** years_to_retirement)
    spouse_age_at_retire = (spouse_age + years_to_retirement) if spouse_age else None
    mortgage_yrs_at_retire = max(0, _effective_mortgage_years - years_to_retirement)
    home_value_at_retire = home_value * ((1 + home_appreciation) ** years_to_retirement)

    st.divider()
    if mortgage_payment_annual > 0:
        st.markdown("### Home & Mortgage at Retirement")
        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.write(f"**Home value at retire:** {money(home_value_at_retire)}")
        with mc2: st.write(f"**Mortgage payment:** {money(mortgage_payment_monthly)}/mo (fixed)")
        with mc3: st.write(f"**Mortgage years left at retire:** {mortgage_yrs_at_retire}")

    st.divider()
    st.markdown("### Withdrawal Order & Surplus Handling")
    st.write("**Default:** Pre-Tax -> Taxable -> Tax-Free (Roth/HSA)")
    col1, col2, col3, col4 = st.columns(4)
    opts = ["Pre-Tax", "Taxable", "Tax-Free"]
    so1 = col1.selectbox("1st", opts, index=0, key="rr_so1")
    so2 = col2.selectbox("2nd", opts, index=1, key="rr_so2")
    so3 = col3.selectbox("3rd", opts, index=2, key="rr_so3")
    spending_order = [so1, so2, so3]
    rr_surplus_dest = col4.selectbox("Reinvest surplus in",
        ["Taxable Brokerage", "Cash/Savings", "Don't Reinvest"], key="rr_surplus_dest")

    st.divider()
    st.markdown("### Heir Tax Assumptions")
    st.caption("Pre-tax (IRA/401k) and HSA are taxable to heirs. Roth is tax-free. Brokerage/home get stepped-up basis.")
    heir_col1, heir_col2 = st.columns(2)
    with heir_col1:
        rr_heir_bracket = st.selectbox("Heir's tax bracket",
            ["Same as mine", "One bracket lower", "One bracket higher"],
            key="rr_heir_bracket")
    with heir_col2:
        _bracket_map = {"Same as mine": "same", "One bracket lower": "lower", "One bracket higher": "higher"}
        st.caption("Based on your marginal federal rate + state rate. "
                   "Example: if you're in the 24% bracket and heir is one lower → 22%.")

    def _build_retire_params():
        return {
            "retire_age": target_retirement_age,
            "life_expectancy": life_expectancy,
            "retire_year": dt.date.today().year + years_to_retirement,
            "inflation": inflation,
            "post_retire_return": post_retire_return,
            "filing_status": filing_status,
            "state_tax_rate": state_tax_rate,
            "expenses_at_retirement": inflated_expenses,
            "ss_filer_fra": ss_filer_final * ((1 + inflation) ** years_to_retirement),
            "ss_spouse_fra": ss_spouse_final * ((1 + inflation) ** years_to_retirement),
            "ss_filer_claim_age": ss_filer_claim_age,
            "ss_spouse_claim_age": ss_spouse_claim_age,
            "ssdi_filer": ssdi_filer,
            "ssdi_spouse": ssdi_spouse,
            "other_income": 0.0 if (other_income_years > 0 and other_income_years <= years_to_retirement) else (other_income * ((1 + inflation) ** years_to_retirement) if other_income_inflation else other_income),
            "other_income_tax_free": other_income_tax_free,
            "other_income_inflation": other_income_inflation,
            "other_income_years": max(0, other_income_years - years_to_retirement) if other_income_years > 0 else 0,
            "pension_filer_at_retire": pen_filer_at_retire,
            "pension_filer_start_age": pension_filer_start_age,
            "pension_filer_cola": pension_filer_cola,
            "pension_spouse_at_retire": pen_spouse_at_retire,
            "pension_spouse_start_age": pension_spouse_start_age,
            "pension_spouse_cola": pension_spouse_cola,
            "spouse_age_at_retire": spouse_age_at_retire,
            "mortgage_payment": mortgage_payment_annual,
            "mortgage_years_at_retire": mortgage_yrs_at_retire,
            "home_value_at_retire": home_value_at_retire,
            "home_appreciation": home_appreciation,
            "future_expenses": future_expenses,
            "dividend_yield": dividend_yield,
            "cash_interest_rate": cash_interest_rate,
            "inherited_iras": _accum_result.get("inherited_iras_state", []),
            "surplus_destination": "none" if rr_surplus_dest == "Don't Reinvest" else ("cash" if rr_surplus_dest == "Cash/Savings" else "brokerage"),
            "heir_bracket_option": _bracket_map.get(rr_heir_bracket, "same"),
        }

    balances = {
        "pretax": float(_retire_final["Bal Pre-Tax"]),
        "roth": float(_retire_final["Bal Roth"]),
        "taxable": float(_retire_final["Bal Taxable"]),
        "brokerage": float(_accum_result.get("final_brokerage", _retire_final["Bal Taxable"])),
        "cash": float(_accum_result.get("final_cash", 0)),
        "brokerage_basis": float(_accum_result.get("final_basis", _retire_final["Bal Taxable"])),
        "hsa": float(_retire_final["Bal HSA"]),
    }

    if st.button("Run Retirement Projection", type="primary", key="run_retire_proj"):
        params = _build_retire_params()
        result = run_retirement_projection(balances, params, spending_order)

        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True, hide_index=True)

        if result["rows"]:
            last = result["rows"][-1]
            st.markdown("### End of Plan Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Portfolio", money(result["final_total"]))
            with col2: st.metric("Home Value", money(result.get("final_home_value", 0)))
            with col3: st.metric("Gross Estate", money(result.get("gross_estate", result["estate"])))
            with col4: st.metric("Estate (After Heir Tax)", money(result["estate"]))
            st.caption("Estate (Net) = Portfolio + Home, minus heir taxes on Pre-Tax/HSA/Inherited IRA balances")
            col5, col6, col7, col8, col9 = st.columns(5)
            with col5: st.metric("Pre-Tax", money(result["final_pretax"]))
            with col6: st.metric("Roth", money(result["final_roth"]))
            with col7: st.metric("Taxable", money(result["final_taxable"]))
            with col8: st.metric("HSA", money(result["final_hsa"]))
            with col9: st.metric("Total Taxes Paid", money(result["total_taxes"]))

            depleted = next((r for r in result["rows"] if r["Portfolio"] <= 0), None)
            if depleted:
                st.error(f"Portfolio depleted at age {depleted['Age']} (year {depleted['Year']})")
            else:
                st.success(f"Portfolio survives through age {life_expectancy} with {money(last['Portfolio'])} remaining")

        st.session_state.retire_projection = result

    st.divider()
    st.markdown("### Waterfall Optimizer")
    st.write("Test all 6 withdrawal orderings to find the best strategy.")

    if st.button("Run Waterfall Optimizer", type="primary", key="run_waterfall_opt"):
        params = _build_retire_params()
        all_orders = list(permutations(["Pre-Tax", "Taxable", "Tax-Free"]))
        results = []
        progress = st.progress(0)

        for idx, order in enumerate(all_orders):
            res = run_retirement_projection(balances, params, list(order))
            results.append({
                "Waterfall": " -> ".join(order),
                "order": list(order),
                "Estate": res["estate"],
                "Total Taxes": res["total_taxes"],
                "Final Pre-Tax": res["final_pretax"],
                "Final Roth": res["final_roth"],
                "Final Taxable": res["final_taxable"],
                "Final Total": res["final_total"],
            })
            progress.progress((idx + 1) / len(all_orders))
        progress.empty()

        results.sort(key=lambda x: x["Estate"], reverse=True)
        best = results[0]
        worst = results[-1]
        diff = best["Estate"] - worst["Estate"]

        st.success(f"Best Order: **{best['Waterfall']}**")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Best Estate (After Heir Tax)", money(best["Estate"]))
        with col2: st.metric("Worst Estate (After Heir Tax)", money(worst["Estate"]))
        with col3: st.metric("Difference", money(diff))

        df_opt = pd.DataFrame(results)
        df_opt.index = range(1, len(df_opt) + 1)
        display_cols = ["Waterfall", "Estate", "Total Taxes", "Final Pre-Tax", "Final Roth", "Final Taxable", "Final Total"]
        df_show = df_opt[display_cols].copy()
        for c in display_cols[1:]:
            df_show[c] = df_show[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_show, use_container_width=True)

        # Show year-by-year for the best
        with st.expander("Year-by-Year Detail (Best Order)"):
            best_result = run_retirement_projection(balances, params, best["order"])
            st.dataframe(pd.DataFrame(best_result["rows"]), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Roth Conversion Optimizer")
    st.write("Test multiple Roth conversion strategies to find the one that maximizes after-tax estate. "
             "Converting pre-tax to Roth increases current-year taxes but reduces future RMDs and shifts wealth into tax-free accounts for heirs.")

    st.markdown("**Settings from Retirement Readiness tab (Tab 4):**")
    _rc5_so = st.session_state.get("rr_so1", "Pre-Tax") + " → " + st.session_state.get("rr_so2", "Taxable") + " → " + st.session_state.get("rr_so3", "Tax-Free")
    _rc5_surplus = st.session_state.get("rr_surplus_dest", "Taxable Brokerage")
    _rc5_heir = st.session_state.get("rr_heir_bracket", "Same as mine")
    st.caption(f"Withdrawal order: {_rc5_so}  |  Surplus: {_rc5_surplus}  |  Heir bracket: {_rc5_heir}")

    st.divider()
    st.markdown("### Conversion Parameters")

    _rc_col1, _rc_col2 = st.columns(2)
    with _rc_col1:
        rc_start_age = st.number_input("Start conversions at age", min_value=40, max_value=100,
                                        value=target_retirement_age, step=1, key="rc_start_age",
                                        help="Set below retirement age to convert while still working (higher taxes but more years of tax-free growth)")
    with _rc_col2:
        rc_defer_rmd = st.checkbox("Defer first RMD to following year",
                                    value=False, key="rc_defer_rmd",
                                    help="Defer age-73 RMD to age 74 (doubles RMD that year). Frees up one extra conversion year.")

    st.caption("The optimizer auto-generates strategies: fill-to-bracket (22%, 24%, 32%, IRMAA) "
               "and fixed-duration (3, 5, 7, 10, 15 years), each tested with and without RMD deferral.")

    # Reference info
    if is_joint:
        st.caption("Reference brackets (joint 2025): 22% up to $206,700 | 24% up to $394,600 | "
                   "32% up to $501,050 | IRMAA threshold $212,000")
    else:
        st.caption("Reference brackets (single 2025): 22% up to $103,350 | 24% up to $197,300 | "
                   "32% up to $250,525 | IRMAA threshold $106,000")

    st.divider()
    st.markdown("### Monte Carlo Validation")
    _mc5_c1, _mc5_c2, _mc5_c3 = st.columns(3)
    with _mc5_c1:
        _mc5_run = st.checkbox("Include Monte Carlo validation", value=False, key="mc5_run")
    with _mc5_c2:
        _mc5_nsims = st.number_input("Simulations", min_value=100, max_value=2000, value=500, step=100, key="mc5_nsims")
    with _mc5_c3:
        _mc5_std = st.number_input("Return Std Dev (%)", min_value=5.0, max_value=25.0, value=12.0, step=1.0, key="mc5_std") / 100.0

    st.divider()

    if st.button("Run Roth Conversion Optimizer", type="primary", key="run_roth_opt"):
        rc_params = _build_retire_params()
        import copy as _rc_copy
        rc_balances = _rc_copy.deepcopy(balances)

        # Build accum_inputs for pre-retirement conversions
        rc_accum_inputs = None
        if rc_start_age < target_retirement_age:
            rc_accum_inputs = {
                "current_age": current_age,
                "years_to_ret": years_to_retirement,
                "start_balances": _rc_copy.deepcopy(_start_balances),
                "contributions": _contrib_dict,
                "salary_growth": salary_growth,
                "pre_ret_return": pre_retire_return,
                "income_info": _rc_copy.deepcopy(_income_info),
            }

        with st.spinner("Testing conversion strategies..."):
            rc_result = optimize_roth_conversions(
                rc_balances, rc_params, spending_order,
                start_age=rc_start_age,
                accum_inputs=rc_accum_inputs,
            )

        best_rc = rc_result["best"]
        baseline_est = rc_result["baseline"]
        improvement = best_rc["estate"] - baseline_est

        # Recommendation threshold: lesser of 5% of baseline or $500K
        _rc_threshold = min(baseline_est * 0.05, 500000) if baseline_est > 0 else 500000
        _rc_meets_threshold = improvement >= _rc_threshold
        _rc_pct_improvement = (improvement / baseline_est * 100) if baseline_est > 0 else 0

        # Summary metrics
        st.markdown("### Results")
        _best_is_retire_only = "[retire only]" in best_rc.get("strategy", "")
        _best_is_no_conv = best_rc.get("strategy", "") == "No Conversion"

        if _best_is_no_conv or improvement <= 0:
            st.info("No conversion strategy improves the after-tax estate. **Recommendation: No conversions.**")
        elif _rc_meets_threshold:
            st.success(f"**Recommended:** {best_rc['strategy']} — improves estate by "
                       f"{money(improvement)} ({_rc_pct_improvement:.1f}%)")
        else:
            st.warning(f"Best strategy ({best_rc['strategy']}) improves estate by only "
                       f"**{money(improvement)} ({_rc_pct_improvement:.1f}%)**. "
                       f"This is below the recommendation threshold "
                       f"(lesser of 5% or $500K = {money(_rc_threshold)}). "
                       f"The improvement may not justify the effort and complexity of executing conversions. "
                       f"**Recommendation: No conversions.**")

        # Show baseline estate regardless of threshold
        _sm1, _sm2 = st.columns(2)
        with _sm1: st.metric("Baseline Estate (No Conversions)", money(baseline_est))
        with _sm2: st.metric("Best Possible Improvement", money(improvement),
                              delta=f"{_rc_pct_improvement:.1f}%" if baseline_est > 0 else "N/A")

        # Only show full detail if the improvement meets the recommendation threshold
        if _rc_meets_threshold:
            # Timing insight: compare best pre-retire vs best retire-only
            _rc_pre_retire = rc_result.get("pre_retire", False)
            if _rc_pre_retire:
                _all_res = rc_result.get("all_results", [])
                _best_early = next((r for r in _all_res if "[retire only]" not in r["strategy"] and r["strategy"] != "No Conversion"), None)
                _best_wait = next((r for r in _all_res if "[retire only]" in r["strategy"]), None)
                if _best_early and _best_wait:
                    if _best_wait["estate"] > _best_early["estate"]:
                        _timing_diff = _best_wait["estate"] - _best_early["estate"]
                        st.info(f"Waiting to convert in retirement beats converting during working years by **{money(_timing_diff)}**. "
                                f"Best retire-only: {_best_wait['strategy']} ({money(_best_wait['estate'])}). "
                                f"Best early-start: {_best_early['strategy']} ({money(_best_early['estate'])}). "
                                f"Converting in a lower bracket is more tax-efficient.")
                    else:
                        _timing_diff = _best_early["estate"] - _best_wait["estate"]
                        st.warning(f"Starting conversions early (age {rc_start_age}) beats waiting for retirement by **{money(_timing_diff)}**. "
                                   f"Best early-start: {_best_early['strategy']} ({money(_best_early['estate'])}). "
                                   f"Best retire-only: {_best_wait['strategy']} ({money(_best_wait['estate'])}). "
                                   f"Extra years of tax-free growth outweigh the higher conversion tax rate.")

            _sm3, _sm4, _sm5 = st.columns(3)
            with _sm3: st.metric("Best Estate", money(best_rc["estate"]))
            with _sm4: st.metric("Total Converted", money(best_rc["total_converted"]))
            with _sm5: st.metric("Total Taxes", money(best_rc["total_taxes"]))

            # Show conversion breakdown: accumulation vs retirement
            _best_accum_conv = best_rc.get("accum_converted", 0)
            _best_retire_conv = best_rc.get("retire_converted", 0)
            if _best_accum_conv > 0:
                _cv1, _cv2, _cv3 = st.columns(3)
                with _cv1: st.metric("Converted (Pre-Retirement)", money(_best_accum_conv))
                with _cv2: st.metric("Converted (In Retirement)", money(_best_retire_conv))
                with _cv3: st.metric("Total Converted", money(best_rc["total_converted"]))

            # Show starting balances: baseline (Tab 4) vs best strategy
            _baseline_bals = rc_result.get("baseline_start_bals", {})

            st.markdown("#### Balances at Start of Retirement")
            if _rc_pre_retire:
                st.info(f"Conversions start at age {rc_start_age}, before retirement at age {target_retirement_age}. "
                        f"The optimizer re-runs the accumulation projection WITH conversions for each strategy, "
                        f"so starting retirement balances differ from the Projection tab (which has no conversions).")
                st.markdown("**Without conversions (Projection tab):**")
                _bl1, _bl2, _bl3, _bl4 = st.columns(4)
                with _bl1: st.metric("Pre-Tax", money(_baseline_bals.get("pretax", 0)))
                with _bl2: st.metric("Roth", money(_baseline_bals.get("roth", 0)))
                with _bl3: st.metric("Taxable", money(_baseline_bals.get("taxable", 0)))
                _bl_total = sum(_baseline_bals.values())
                with _bl4: st.metric("Total", money(_bl_total))
                st.markdown("**With best strategy's conversions applied:**")

            _best_start_pt = best_rc.get("starting_pretax", 0)
            _best_start_ro = best_rc.get("starting_roth", 0)
            _best_start_tx = best_rc.get("starting_taxable", 0)
            _best_start_total = _best_start_pt + _best_start_ro + _best_start_tx
            _sb1, _sb2, _sb3, _sb4 = st.columns(4)
            with _sb1: st.metric("Pre-Tax", money(_best_start_pt))
            with _sb2: st.metric("Roth", money(_best_start_ro))
            with _sb3: st.metric("Taxable", money(_best_start_tx))
            with _sb4: st.metric("Total", money(_best_start_total))
            if _rc_pre_retire and _best_accum_conv > 0:
                st.caption(f"Roth includes {money(_best_accum_conv)} of pre-retirement conversions + growth. "
                           f"Pre-Tax is reduced by the same conversions.")

            # All strategies table
            st.markdown("### All Strategies Ranked by After-Tax Estate")
            df_rc = pd.DataFrame(rc_result["rankings"])
            df_rc_show = df_rc.copy()
            _money_cols = ["Estate (Net)", "vs Baseline", "Total Taxes", "Total Converted",
                           "Start Pre-Tax", "Start Roth", "Start Taxable",
                           "Final Pre-Tax", "Final Roth", "Final Taxable"]
            if _rc_pre_retire:
                _money_cols.insert(4, "Accum Conv")
                _money_cols.insert(5, "Retire Conv")
            for c in _money_cols:
                if c in df_rc_show.columns:
                    df_rc_show[c] = df_rc_show[c].apply(lambda x: f"${x:,.0f}")
            st.dataframe(df_rc_show, use_container_width=True, hide_index=True)

            # Accumulation schedule for best strategy (pre-retirement conversions)
            _best_accum_rows = best_rc.get("accum_rows", [])
            if _best_accum_rows:
                with st.expander("Pre-Retirement Accumulation (Best Strategy) — shows conversions during working years"):
                    st.caption("This is the accumulation re-run WITH the best conversion strategy applied. "
                               "Conv Gross shows pre-retirement conversions moving Pre-Tax → Roth.")
                    _accum_cols = ["Year", "Age", "Income", "Conv Gross", "Conv Tax", "Conv to Roth",
                                   "Bal Pre-Tax", "Bal Roth", "Bal Taxable", "Bal HSA", "Total Balance"]
                    _df_accum_best = pd.DataFrame(_best_accum_rows)
                    _show_cols = [c for c in _accum_cols if c in _df_accum_best.columns]
                    st.dataframe(_df_accum_best[_show_cols], use_container_width=True, hide_index=True)

            # Year-by-year retirement schedule for best strategy
            with st.expander("Retirement Year-by-Year Schedule (Best Strategy)"):
                st.caption("W/D columns = withdrawals FROM that account. Conv columns = Roth conversions during retirement. "
                           "Bal columns = end-of-year balance after withdrawals and growth.")
                df_schedule = pd.DataFrame(rc_result["schedule"])
                st.dataframe(df_schedule, use_container_width=True, hide_index=True)

            # ── Monte Carlo Validation ──
            if _mc5_run and rc_result.get("all_results"):
                st.divider()
                st.markdown("### Monte Carlo Validation")
                import copy as _mc5_copy

                _mc5_n_retire_years = life_expectancy - target_retirement_age + 1
                _mc5_seed = 42
                _mc5_top = rc_result["all_results"][:10]

                _mc5_results = []
                _mc5_progress = st.progress(0, text="Running Monte Carlo on top strategies...")
                _mc5_total = len(_mc5_top)

                for _mc5_idx, _mc5_strat in enumerate(_mc5_top):
                    _mc5_s_val = _mc5_strat["conv_strategy_val"]
                    _mc5_s_target = _mc5_strat["conv_target_agi"]
                    _mc5_s_stop = _mc5_strat["conv_stop_age"]
                    _mc5_s_defer = _mc5_strat["defer"]

                    def _mc5_make_run_fn(s_val, s_target, s_stop, s_defer):
                        def _run_fn(return_seq):
                            rp = _mc5_copy.deepcopy(rc_params)
                            rp["roth_conversion_strategy"] = s_val
                            rp["roth_conversion_target_agi"] = s_target
                            rp["roth_conversion_stop_age"] = s_stop
                            rp["defer_first_rmd"] = s_defer
                            rp["return_sequence"] = return_seq
                            return run_retirement_projection(
                                _mc5_copy.deepcopy(rc_balances), rp, spending_order)
                        return _run_fn

                    _mc5_fn = _mc5_make_run_fn(_mc5_s_val, _mc5_s_target, _mc5_s_stop, _mc5_s_defer)
                    _mc5_mc = run_monte_carlo(
                        _mc5_fn, n_sims=int(_mc5_nsims),
                        mean_return=post_retire_return, return_std=_mc5_std,
                        n_years=_mc5_n_retire_years, seed=_mc5_seed)
                    _mc5_results.append((_mc5_strat["strategy"], _mc5_strat["estate"], _mc5_mc))
                    _mc5_progress.progress(int((_mc5_idx + 1) / _mc5_total * 100),
                                            text=f"Completed {_mc5_idx + 1}/{_mc5_total} strategies...")

                _mc5_progress.empty()

                _mc5_table = pd.DataFrame([
                    {
                        "Strategy": label,
                        "Det. Estate": money(det_est),
                        "MC Median": money(mc["median_estate"]),
                        "MC 10th Pctl": money(mc["p10"]),
                        "MC 90th Pctl": money(mc["p90"]),
                        "MC Success Rate": f"{mc['success_rate']:.0%}",
                    }
                    for label, det_est, mc in _mc5_results
                ])
                st.dataframe(_mc5_table, use_container_width=True, hide_index=True)

                # Check if MC changes the winner
                _mc5_det_winner = _mc5_results[0][0]  # already sorted by det. estate
                _mc5_mc_winner = max(_mc5_results, key=lambda x: x[2]["median_estate"])[0]
                if _mc5_det_winner != _mc5_mc_winner:
                    st.warning(f"MC changes the winner: deterministic favors **{_mc5_det_winner}**, "
                               f"but MC median favors **{_mc5_mc_winner}**")
                st.caption(f"Top {len(_mc5_top)} strategies tested with {int(_mc5_nsims)} simulations each, "
                           f"mean return {post_retire_return:.1%}, std dev {_mc5_std:.1%} (seed={_mc5_seed})")
