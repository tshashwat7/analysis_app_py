# services/fundamentals_ext.py (Final Patched Version)
import logging
from typing import Optional
import pandas as pd
import math

import yfinance
from services import fundamentals
from services.data_fetch import _retry, safe_float, normalize_ratio, safe_get
from services.metrics_ext import safe_div

logger = logging.getLogger(__name__)

def compute_fundamentals_extended(symbol:str):
    """
    Compute advanced / supplemental fundamental metrics not covered in fundamentals.py.
    Includes:
      - Asset Turnover
      - FCF Growth (3Y CAGR)
      - Operating CF vs Net Profit alignment
      - Interest Coverage Ratio
      - ROCE (%)
      - PEG Ratio
      - Current Ratio
    """

    fundamentals_ext = {}
    ticker_obj = yfinance.Ticker(symbol)

    def getWACC(symbol: str) -> Optional[float]:
        """
        Approximate WACC based on industry beta and interest rates.
        Fallbacks to 10% if not available.
        """
        try:
            t = yfinance.Ticker(symbol)
            info = t.info or {}
            beta = safe_float(info.get("beta")) or 1.0
            risk_free = 6.8  # e.g., India 10Y bond yield
            market_return = 12.0
            cost_of_equity = risk_free + beta * (market_return - risk_free)
            cost_of_debt = 8.0  # assume average corporate debt rate
            tax_rate = safe_float(getTaxRate()) or 0.25

            debt_to_equity = safe_float(info.get("debtToEquity")) or 0
            wacc = (cost_of_equity * (1 / (1 + debt_to_equity / 100))) + \
                (cost_of_debt * (debt_to_equity / 100 / (1 + debt_to_equity / 100))) * (1 - tax_rate)

            return round(wacc, 2)
        except Exception as e:
            logger.debug(f"WACC calc failed: {e}")
            return 10.0  # fallback


    def getTaxRate():
        fs = getattr(ticker_obj, "financials", None)
        qfs = getattr(ticker_obj, "quarterly_financials", None)

        def _find_fin_item(fs, keys):
            if fs is None or fs.empty:
                return None
            for k in fs.index:
                lk = k.strip().lower()
                for key in keys:
                    if key in lk:
                        try:
                            val = fs.loc[k].iloc[0]
                            if val is not None and val != 0:
                                return val
                        except Exception:
                            pass
            return None

        income_tax_exp = (
            _find_fin_item(fs, ["income tax", "tax expense", "provision for tax", "taxation", "total tax"])
            or _find_fin_item(qfs, ["income tax", "tax expense", "provision for tax", "taxation", "total tax"])
        )
        pre_tax_income = (
            _find_fin_item(fs, ["pretax income", "income before tax", "ebt", "profit before tax"])
            or _find_fin_item(qfs, ["pretax income", "income before tax", "ebt", "profit before tax"])
        )

        # Derive from net income if missing
        if not income_tax_exp:
            net_income = (
                _find_fin_item(fs, ["net income", "net profit"])
                or _find_fin_item(qfs, ["net income", "net profit"])
            )
            if net_income and pre_tax_income and pre_tax_income > net_income:
                income_tax_exp = pre_tax_income - net_income

        # Compute tax rate
        if income_tax_exp and pre_tax_income and pre_tax_income != 0:
            tax_rate = income_tax_exp / pre_tax_income
            if 0 < tax_rate < 1:
                logger.info(f"Derived effective tax rate: {tax_rate:.2%}")
            else:
                tax_rate = 0.25  # fallback
        else:
            tax_rate = 0.25
        return tax_rate


    def _inner():
        fin = getattr(ticker_obj, "financials", pd.DataFrame())
        bs = getattr(ticker_obj, "balance_sheet", pd.DataFrame())
        cf = getattr(ticker_obj, "cashflow", pd.DataFrame())
        info = getattr(ticker_obj, "info", {})

        if fin.empty or bs.empty:
            raise ValueError("Missing financial or balance sheet data")

        # ✅ FCF Growth Sorting Fix
        # Ensure chronological order by sorting columns (which are dates in YF data)
        if not cf.empty:
            cf = cf.loc[:, sorted(cf.columns)]

        # --- Safely Extract Base Values ---

        # Income Statement (Latest Annual)
        latest_fin = fin.iloc[:, 0] if not fin.empty and len(fin.columns) > 0 else {}
        net_income = safe_float(latest_fin.get("Net Income"))
        revenue = safe_float(latest_fin.get("Total Revenue"))
        ebit = safe_float(latest_fin.get("Operating Income", latest_fin.get("EBIT")))

        # Interest Expense with fallbacks
        interest_expense = safe_float(
            latest_fin.get(
                "Interest Expense",
                latest_fin.get(
                    "Interest And Debt Expense",
                    latest_fin.get("Finance Cost", latest_fin.get("Interest", 0)),
                ),
            )
        )

        # Balance Sheet (Latest Annual)
        latest_bs = bs.iloc[:, 0] if not bs.empty and len(bs.columns) > 0 else {}
        total_assets = safe_float(latest_bs.get("Total Assets"))
        total_equity = safe_float(
            latest_bs.get(
                "Total Stockholder Equity", latest_bs.get("Stockholders Equity")
            )
        )
        total_debt = safe_float(latest_bs.get("Total Debt", latest_bs.get("Long Term Debt"))) # Fallback for Total Debt
        cash_equivalents = safe_float(latest_bs.get("Cash And Cash Equivalents"))

        current_assets = safe_float(
            latest_bs.get("Total Current Assets", latest_bs.get("Current Assets"))
        )
        current_liabilities = safe_float(
            latest_bs.get(
                "Total Current Liabilities", latest_bs.get("Current Liabilities")
            )
        )

        # Cash Flow (Latest Annual)
        latest_cf = cf.iloc[:, 0] if not cf.empty and len(cf.columns) > 0 else {}
        ocf = safe_float(latest_cf.get("Total Cash From Operating Activities"))
        capex = abs(safe_float(latest_cf.get("Capital Expenditures", 0)))


        # Calculated FCF
        fcf = safe_float(ocf - capex) if ocf and capex is not None else None
        # ----------------------------------------------------------------------
        # 🧩 Piotroski F-Score (9-point checklist for financial quality)
        # ----------------------------------------------------------------------
        try:
            piotroski_points = 0
            details = {}

            if fin.empty or bs.empty:
                raise ValueError("Missing financials or balance sheet for Piotroski computation.")

            fin = fin.loc[:, sorted(fin.columns)] if not fin.empty else fin
            bs = bs.loc[:, sorted(bs.columns)] if not bs.empty else bs
            cf = cf.loc[:, sorted(cf.columns)] if not cf.empty else cf

            def safe_df_get(df, row, idx=0):
                try:
                    if row in df.index and len(df.columns) > abs(idx):
                        return safe_float(df.loc[row].iloc[idx])
                except Exception:
                    pass
                return None

            # --- Core data ---
            net_income = safe_df_get(fin, "Net Income", -1)
            prev_ni = safe_df_get(fin, "Net Income", -2)
            total_assets = safe_df_get(bs, "Total Assets", -1)
            prev_assets = safe_df_get(bs, "Total Assets", -2)
            ocf = safe_df_get(cf, "Total Cash From Operating Activities", -1)
            total_debt = safe_df_get(bs, "Total Debt", -1)
            prev_debt = safe_df_get(bs, "Total Debt", -2)
            current_assets = safe_df_get(bs, "Total Current Assets", -1)
            current_liabilities = safe_df_get(bs, "Total Current Liabilities", -1)
            prev_ca = safe_df_get(bs, "Total Current Assets", -2)
            prev_cl = safe_df_get(bs, "Total Current Liabilities", -2)
            shares = safe_df_get(fin, "Basic Average Shares", -1)
            prev_shares = safe_df_get(fin, "Basic Average Shares", -2)
            revenue = safe_df_get(fin, "Total Revenue", -1)
            prev_revenue = safe_df_get(fin, "Total Revenue", -2)
            cogs = safe_df_get(fin, "Cost Of Revenue", -1)
            prev_cogs = safe_df_get(fin, "Cost Of Revenue", -2)

            # --- Derived values ---
            roa_now = safe_div(net_income, total_assets)
            roa_prev = safe_div(prev_ni, prev_assets)
            cr_now = safe_div(current_assets, current_liabilities)
            cr_prev = safe_div(prev_ca, prev_cl)
            gm_now = safe_div((revenue - cogs), revenue)
            gm_prev = safe_div((prev_revenue - prev_cogs), prev_revenue)
            at_now = safe_div(revenue, total_assets)
            at_prev = safe_div(prev_revenue, prev_assets)

            # -----------------------------
            # F1–F3: Profitability
            # -----------------------------
            # F1: Positive Net Income
            if net_income and net_income > 0:
                piotroski_points += 1
                details["Positive Net Income (F1)"] = 1
            else:
                details["Positive Net Income (F1)"] = 0

            # F2: Positive Operating Cash Flow
            if ocf and ocf > 0:
                piotroski_points += 1
                details["Positive Operating Cash Flow (F2)"] = 1
            else:
                details["Positive Operating Cash Flow (F2)"] = 0

            # F3: Increasing ROA
            if roa_now is not None and roa_prev is not None:
                if roa_now > roa_prev:
                    piotroski_points += 1
                    details["Improving ROA (F3)"] = 1
                else:
                    details["Improving ROA (F3)"] = 0
            else:
                logger.debug(f"[{symbol}] Missing ROA data for Piotroski F3.")
                details["Improving ROA (F3)"] = 0

            # -----------------------------
            # F4–F9: Leverage, Liquidity, and Efficiency
            # -----------------------------
            # F4: CFO > Net Income (Earnings quality)
            if ocf and net_income and ocf > net_income:
                piotroski_points += 1
                details["CFO > Net Income (F4)"] = 1
            else:
                details["CFO > Net Income (F4)"] = 0

            # F5: Lower Leverage
            if total_debt and prev_debt and total_assets and prev_assets:
                debt_ratio_now = total_debt / total_assets
                debt_ratio_prev = prev_debt / prev_assets
                if debt_ratio_now < debt_ratio_prev:
                    piotroski_points += 1
                    details["Lower Leverage (F5)"] = 1
                else:
                    details["Lower Leverage (F5)"] = 0
            else:
                logger.debug(f"[{symbol}] Missing leverage data for Piotroski F5.")
                details["Lower Leverage (F5)"] = 0

            # F6: Higher Current Ratio
            if cr_now and cr_prev and cr_now > cr_prev:
                piotroski_points += 1
                details["Higher Current Ratio (F6)"] = 1
            else:
                details["Higher Current Ratio (F6)"] = 0

            # F7: No New Shares Issued
            if shares and prev_shares and shares <= prev_shares:
                piotroski_points += 1
                details["No Dilution (F7)"] = 1
            else:
                details["No Dilution (F7)"] = 0

            # F8: Higher Gross Margin
            if gm_now and gm_prev and gm_now > gm_prev:
                piotroski_points += 1
                details["Improving Gross Margin (F8)"] = 1
            else:
                details["Improving Gross Margin (F8)"] = 0

            # F9: Higher Asset Turnover
            if at_now and at_prev and at_now > at_prev:
                piotroski_points += 1
                details["Improving Asset Turnover (F9)"] = 1
            else:
                details["Improving Asset Turnover (F9)"] = 0

            # --- Final composite ---
            fundamentals_ext["Piotroski F-Score"] = {
                "raw": piotroski_points,
                "value": f"{piotroski_points}/9",
                "score": round((piotroski_points / 9) * 10, 1),
                "breakdown": details,
            }

            logger.info(
                f"[{symbol}] Piotroski F-Score: {piotroski_points}/9 → "
                f"{fundamentals_ext['Piotroski F-Score']['score']}/10"
            )

        except Exception as e:
            logger.error(f"[{symbol}] Piotroski F-Score computation failed: {e}")
            fundamentals_ext["Piotroski F-Score"] = {"raw": None, "value": "N/A", "score": 0}


        # --- Asset Turnover Ratio ---
        try:
            if revenue and total_assets and total_assets > 0:
                ratio = revenue / total_assets
                fundamentals_ext["Asset Turnover Ratio"] = {
                    "raw": ratio,
                    "value": round(ratio, 2),
                    "score": 10 if ratio > 1 else (5 if ratio >= 0.5 else 0),
                }
        except Exception as e:
            logger.debug("Asset Turnover calc failed: %s", e)
            fundamentals_ext["Asset Turnover Ratio"] = {
                "raw": None,
                "value": "N/A",
                "score": 0,
            }

        # --- FCF Growth (3Y CAGR) --- not coming
        try:
            if (
                "Total Cash From Operating Activities" in cf.index
                and "Capital Expenditures" in cf.index
            ):
                fcf_series = (
                    cf.loc["Total Cash From Operating Activities"]
                    + cf.loc["Capital Expenditures"]
                )

                if isinstance(fcf_series, pd.Series) and len(fcf_series) >= 4:
                    fcf_latest_4 = fcf_series.dropna().tail(4)
                    if len(fcf_latest_4) == 4:
                        start = safe_float(fcf_latest_4.iloc[0])
                        end = safe_float(fcf_latest_4.iloc[-1])
                        if start and end and start > 0:
                            cagr = ((end / start) ** (1 / 3) - 1) * 100
                            fundamentals_ext["FCF Growth (3Y CAGR)"] = {
                                "raw": cagr,
                                "value": f"{cagr:.2f}%",
                                "score": 10 if cagr > 10 else (5 if cagr >= 5 else 0),
                            }
        except Exception as e:
            logger.debug("FCF Growth calc failed: %s", e)
            fundamentals_ext["FCF Growth (3Y CAGR)"] = {
                "raw": None,
                "value": "N/A",
                "score": 0,
            }

        # --- OCF vs Net Profit Alignment --- not coming
        try:
            if ocf and net_income and net_income != 0:
                ratio = ocf / net_income
                if math.isnan(ratio):
                    raise ValueError("NaN ratio detected")
                status = "Aligned" if 0.8 <= ratio <= 1.5 else "Diverging"
                fundamentals_ext["Operating CF vs Net Profit"] = {
                    "raw": ratio,
                    "value": status,
                    "score": 10 if status == "Aligned" else 0,
                }
        except Exception as e:
            logger.debug("OCF vs Profit calc failed: %s", e)
            fundamentals_ext["Operating CF vs Net Profit"] = {
                "raw": None,
                "value": "N/A",
                "score": 0,
            }

        # ----------------------------------------------------------------------
        # --- LIQUIDITY AND EFFICIENCY METRICS ---
        # ----------------------------------------------------------------------

        # --- Interest Coverage Ratio ---
        try:
            if ebit and interest_expense is not None:
                # ✅ Interest Coverage Zero Guard Fix
                # Treat negligible interest (< 100,000) as zero to avoid noise-driven 500x+ ratios
                if not interest_expense or interest_expense < 1e5:
                    ratio = 999.0
                    score = 10
                else:
                    ratio = ebit / interest_expense
                    score = 10 if ratio > 5 else (5 if ratio >= 3 else 0)

                fundamentals_ext["Interest Coverage Ratio"] = {
                    "raw": ratio,
                    "value": f"{ratio:,.2f}x",
                    "score": score,
                }
                fundamentals_ext["Interest Coverage"] = fundamentals_ext[
                    "Interest Coverage Ratio"
                ]
        except Exception as e:
            logger.debug("Interest Coverage calc failed: %s", e)
            fundamentals_ext["Interest Coverage Ratio"] = {
                "raw": None,
                "value": "N/A",
                "score": 0,
            }

        # --- NEW: Return on Invested Capital (ROIC) ---
        # --- NEW: Return on Invested Capital (ROIC) ---
        try:
            # Step 1: Compute dynamic effective tax rate (fallback = 25%)
            TAX_RATE = safe_float(getTaxRate())
            if TAX_RATE is None or TAX_RATE <= 0 or TAX_RATE >= 1:
                TAX_RATE = 0.25
            logger.info(f"[{symbol}] Effective Tax Rate used: {TAX_RATE:.2%}")

            # Step 2: Compute NOPAT (Net Operating Profit After Tax)
            nopat = ebit * (1 - TAX_RATE) if ebit is not None else None

            # Step 3: Compute Net Invested Capital (NIC)
            if all(v is not None for v in [total_equity, total_debt]) and cash_equivalents is not None:
                net_invested_capital = total_equity + total_debt - cash_equivalents
                nic_source = "equity+debt-cash"
            elif total_assets and current_liabilities:
                net_invested_capital = total_assets - current_liabilities
                nic_source = "assets-liabilities"
            else:
                net_invested_capital = None
                nic_source = "none"

            # Step 4: Calculate ROIC
            if nopat is not None and net_invested_capital and net_invested_capital > 0:
                roic = (nopat / net_invested_capital) * 100
                wacc = safe_float(getWACC(symbol)) or 10.0
                spread = roic - wacc

                if spread >= 10:
                    score = 10
                elif spread >= 5:
                    score = 8
                elif spread >= 2:
                    score = 6
                elif spread >= 0:
                    score = 4
                else:
                    score = 1

                fundamentals_ext["ROIC (%)"] = {
                    "raw": roic,
                    "value": f"{roic:.2f}%",
                    "score": score,
                    "meta": {"wacc": wacc, "spread": spread},
                }

                logger.info(
                    f"[{symbol}] ROIC={roic:.2f}%, WACC={wacc:.2f}%, Spread={spread:.2f}%, "
                    f"EBIT={ebit}, TaxRate={TAX_RATE:.2%}, NOPAT={nopat:.2f}, NIC={net_invested_capital:.2f}"
                )
            else:
                raise ValueError("Required ROIC components missing or IC is zero.")

        except Exception as e:
            logger.debug(f"ROIC calculation failed for {symbol}: {e}")
            fundamentals_ext["ROIC (%)"] = {"raw": None, "value": "N/A", "score": 0}


        # --- ROCE(TTM) (%) ---
        try:
            if ebit and total_assets is not None:
                capital_employed = total_assets - (
                    current_liabilities if current_liabilities is not None else 0
                )
                if capital_employed > 0:
                    roce = (ebit / capital_employed) * 100
                    score = 10 if roce > 20 else (5 if roce >= 10 else 0)

                    fundamentals_ext["ROCE (%)"] = {
                        "raw": roce,
                        "value": f"{roce:.2f}%",
                        "score": score,
                    }
        except Exception as e:
            logger.debug("ROCE calc failed: %s", e)
            fundamentals_ext["ROCE (%)"] = {"raw": None, "value": "N/A", "score": 0}

        # --- Current Ratio ---
        try:
            if (
                current_assets
                and current_liabilities is not None
                and current_liabilities > 0
            ):
                ratio = current_assets / current_liabilities
                score = 10 if ratio >= 2.0 else (5 if ratio >= 1.5 else 0)

                fundamentals_ext["Current Ratio"] = {
                    "raw": ratio,
                    "value": f"{ratio:.2f}",
                    "score": score,
                }
            else:
                fundamentals_ext["Current Ratio"] = {
                    "raw": None,
                    "value": "N/A (Missing data)",
                    "score": 0,
                }

        except Exception as e:
            logger.debug("Current Ratio calc failed: %s", e)
            fundamentals_ext["Current Ratio"] = {
                "raw": None,
                "value": "N/A",
                "score": 0,
            }

        # --- Fallback ROE Calculation ---
        try:
            if net_income and total_equity and total_equity != 0:
                roe_fb = (net_income / total_equity) * 100
                fundamentals_ext["Return on Equity (ROE) (FB)"] = {
                    "raw": roe_fb,
                    "value": f"{roe_fb:.2f}%",
                    "score": 0,
                }
        except Exception as e:
            logger.debug("ROE Fallback calc failed: %s", e)

        return fundamentals_ext

    try:
        return _retry(_inner, retries=2, backoff=0.5)
    except Exception as e:
        logger.error("compute_fundamentals_extended failed: %s", e)
        return fundamentals_ext
