import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize environment
load_dotenv()
st.set_page_config(layout="wide", page_title="Intrinsic", page_icon=":chart_with_upwards_trend:")

# =============================================
# CORE CONFIGURATION
# =============================================
class ValuationConfig:
    def __init__(self):
        self.data_sources = {
            "Yahoo Finance": "yfinance",
            "Alpha Vantage": "alphavantage",
            "Financial Modeling Prep": "fmp"
        }

        self.industry_db = {
            'Technology': {
                'Software': ['MSFT', 'ORCL', 'ADBE', 'CRM', 'NOW', 'INFY.NS', 'TCS.NS'],
                'Semiconductors': ['NVDA', 'INTC', 'AMD', 'QCOM', 'AVGO'],
                'Hardware': ['AAPL', 'HPQ', 'DELL', 'NTAP', 'STX']
            },
            'Financial Services': {
                'Banks': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
                'Insurance': ['BRK-B', 'MET', 'AIG', 'PRU', 'ALL', 'HDFCLIFE.NS', 'SBILIFE.NS']
            },
            'Healthcare': {
                'Pharma': ['PFE', 'MRK', 'JNJ', 'ABBV', 'BMY', 'SUNPHARMA.NS', 'DRREDDY.NS'],
                'Biotech': ['AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB']
            },
            'Consumer Goods': {
                'Retail': ['WMT', 'AMZN', 'TGT', 'COST', 'LOW'],
                'Food & Beverage': ['KO', 'PEP', 'MDLZ', 'K', 'GIS']
            },
            'Energy': {
                'Oil & Gas': ['XOM', 'CVX', 'COP', 'BP', 'RELIANCE.NS'],
                'Renewables': ['NEE', 'ENPH', 'SEDG', 'RUN', 'FSLR']
            }
        }

        self.valuation_params = {
            'risk_free': 0.04,
            'market_premium': 0.06,
            'terminal_growth': 0.025,
            'high_growth_period': 5
        }


config = ValuationConfig()

# =============================================
# UTILITY FUNCTIONS
# =============================================
CURRENCY_SYMBOLS = {
    'USD': '$',
    'INR': '₹',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'CA$',
    'AUD': 'A$',
    'CNY': '¥',
    'KRW': '₩',
    'BRL': 'R$',
    'SGD': 'S$',
    'HKD': 'HK$'
}


def get_currency_symbol(currency_code):
    """Get currency symbol based on currency code"""
    return CURRENCY_SYMBOLS.get(currency_code, f"{currency_code} ")


def get_industry_peers(sector, industry, current_ticker=None):
    """Get pre-defined industry peers with error handling"""
    try:
        peers = config.industry_db.get(sector, {}).get(industry, ['MSFT', 'AAPL', 'GOOG'])
        # Remove the current ticker from the peers list if present
        if current_ticker and current_ticker in peers:
            peers = [peer for peer in peers if peer != current_ticker]
        return peers
    except Exception as e:
        st.warning(f"Error getting peers: {str(e)}")
        return ['MSFT', 'AAPL', 'GOOG']


# =============================================
# DATA LAYER
# =============================================
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def get_market_data(ticker, source="yfinance", api_keys=None):
    """Get comprehensive market data for a given ticker with retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if source == "yfinance":
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")

                if len(info) == 0 or len(hist) == 0:
                    if attempt == max_retries - 1:
                        st.error(f"No data found for ticker: {ticker}")
                        return None
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                if 'currency' not in info:
                    info['currency'] = 'INR' if ticker.endswith('.NS') else 'USD'

                # Parallel fetching of financial data
                with ThreadPoolExecutor() as executor:
                    financials_future = executor.submit(lambda: stock.financials)
                    cashflow_future = executor.submit(lambda: stock.cashflow)
                    balance_sheet_future = executor.submit(lambda: stock.balance_sheet)

                    financials = financials_future.result()
                    cashflow = cashflow_future.result()
                    balance_sheet = balance_sheet_future.result()

                return {
                    'info': info,
                    'history': hist,
                    'financials': financials,
                    'cashflow': cashflow,
                    'balance_sheet': balance_sheet
                }

            elif source == "alphavantage":
                if not api_keys or 'alphavantage' not in api_keys:
                    st.warning("Alpha Vantage API key not configured")
                    return None
                return None

            elif source == "fmp":
                if not api_keys or 'fmp' not in api_keys:
                    st.warning("Financial Modeling Prep API key not configured")
                    return None
                return None

        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Data fetch error for {ticker}: {str(e)}")
                return None
            time.sleep(2 ** attempt)


# =============================================
# VALUATION MODULES
# =============================================
class DCFValuation:
    def __init__(self, ticker_data):
        self.data = ticker_data
        self.free_cash_flows = []
        self.scenario_results = {}
        self.currency = self.data['info'].get('currency', 'USD')
        self.currency_symbol = get_currency_symbol(self.currency)

    def _calculate_single_scenario(self, growth_rate, discount_rate):
        """Core DCF calculation for a single scenario"""
        try:
            # Get required financial data with validation
            shares_outstanding = self.data['info'].get('sharesOutstanding')
            if not shares_outstanding or shares_outstanding <= 0:
                return None

            # Calculate Free Cash Flow with multiple fallbacks
            fcf = self.data['info'].get('freeCashflow')

            if not fcf or fcf == 0:
                if 'cashflow' in self.data:
                    try:
                        last_year = self.data['cashflow'].columns[0]
                        operating_cashflow = self.data['cashflow'].loc['Operating Cash Flow', last_year]
                        capital_expenditure = abs(self.data['cashflow'].loc['Capital Expenditure', last_year])
                        fcf = operating_cashflow - capital_expenditure
                    except (KeyError, IndexError):
                        try:
                            fcf = self.data['cashflow'].loc['Operating Cash Flow', self.data['cashflow'].columns[0]]
                        except (KeyError, IndexError):
                            fcf = self.data['info'].get('operatingCashflow', 0) - self.data['info'].get(
                                'capitalExpenditures', 0)

            if not fcf or fcf == 0:
                fcf = self.data['info'].get('netIncomeToCommon', 0)

            if not fcf or fcf == 0:
                return None

            # Normalize to millions
            fcf_millions = fcf / 1e6

            # Project FCFs for high growth period
            self.free_cash_flows = [
                fcf_millions * (1 + growth_rate) ** year
                for year in range(1, config.valuation_params['high_growth_period'] + 1)
            ]

            # Calculate terminal value (in millions)
            terminal_value = (self.free_cash_flows[-1] *
                              (1 + config.valuation_params['terminal_growth'])) / (
                                     discount_rate - config.valuation_params['terminal_growth'])

            # Discount all cash flows
            discounted_cashflows = [
                cf / (1 + discount_rate) ** year
                for year, cf in enumerate(self.free_cash_flows, 1)
            ]

            discounted_terminal = terminal_value / (1 + discount_rate) ** config.valuation_params['high_growth_period']

            # Calculate equity value per share
            enterprise_value = sum(discounted_cashflows) + discounted_terminal

            # Adjust for net debt
            total_debt = self.data['info'].get('totalDebt', 0)
            cash = self.data['info'].get('totalCash', 0)
            net_debt = total_debt - cash
            net_debt_millions = net_debt / 1e6

            equity_value_total = enterprise_value - net_debt_millions
            equity_value = equity_value_total / (shares_outstanding / 1e6)

            return {
                'intrinsic_value': equity_value,
                'terminal_value': terminal_value,
                'discounted_cashflows': discounted_cashflows,
                'discounted_terminal': discounted_terminal
            }

        except Exception as e:
            st.error(f"DCF scenario calculation error: {str(e)}")
            return None

    def _run_scenario_analysis(self, base_growth, base_discount):
        """Run multiple valuation scenarios"""
        self.scenario_results = {
            'Base Case': self._calculate_single_scenario(base_growth, base_discount),
            'Optimistic Growth': self._calculate_single_scenario(base_growth * 1.25, base_discount * 0.9),
            'Pessimistic Growth': self._calculate_single_scenario(base_growth * 0.75, base_discount * 1.1),
            'High Discount Rate': self._calculate_single_scenario(base_growth, base_discount * 1.25),
            'Low Discount Rate': self._calculate_single_scenario(base_growth, base_discount * 0.75)
        }

    def calculate(self, growth_rate, discount_rate):
        """Calculate DCF with scenario analysis"""
        self._run_scenario_analysis(growth_rate, discount_rate)
        return self.scenario_results['Base Case']['intrinsic_value'] if self.scenario_results['Base Case'] else None

    def display(self):
        """Enhanced DCF display with scenario analysis"""
        with st.expander("Discounted Cash Flow Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                growth = st.slider("Revenue Growth (%)", 0.0, 25.0, 8.0, 0.5) / 100
            with col2:
                discount = st.slider("Discount Rate (%)", 5.0, 25.0, 10.0, 0.5) / 100
            with col3:
                terminal_growth = st.slider("Terminal Growth (%)", 0.0, 5.0,
                                            float(config.valuation_params['terminal_growth'] * 100), 0.1) / 100
                config.valuation_params['terminal_growth'] = terminal_growth

            intrinsic_value = self.calculate(growth, discount)
            current_price = self.data['info'].get('currentPrice')

            if intrinsic_value and current_price:
                # Main metrics
                cols = st.columns(3)
                cols[0].metric("Current Price", f"{self.currency_symbol}{current_price:,.2f}")
                cols[1].metric("DCF Value", f"{self.currency_symbol}{intrinsic_value:,.2f}")

                margin_safety = ((intrinsic_value - current_price) / current_price) * 100
                cols[2].metric("Margin of Safety",
                               f"{margin_safety:.1f}%",
                               delta=f"{margin_safety:.1f}%",
                               delta_color="normal" if margin_safety > 0 else "inverse")

                # Scenario Analysis
                st.subheader("Scenario Analysis")
                scenario_df = pd.DataFrame.from_dict({
                    k: {
                        'Value': f"{self.currency_symbol}{v['intrinsic_value']:,.2f}" if v else 'N/A',
                        'Difference': f"{(v['intrinsic_value'] - current_price) / current_price * 100:.1f}%" if v else 'N/A'
                    }
                    for k, v in self.scenario_results.items() if v
                }, orient='index')
                st.dataframe(scenario_df, use_container_width=True)

                # FCF projection chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Year {i}" for i in range(1, config.valuation_params['high_growth_period'] + 1)],
                    y=self.free_cash_flows,
                    name="Projected FCF",
                    marker_color='#636EFA'
                ))
                fig.update_layout(
                    title="Free Cash Flow Projection (in millions)",
                    yaxis_title=f"Free Cash Flow ({self.currency_symbol} millions)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # DCF Breakdown
                st.subheader("DCF Calculation Breakdown")
                breakdown_data = {
                    "Component": ["Initial Free Cash Flow", "Discount Rate", "Terminal Growth Rate",
                                  "High Growth Period", "Terminal Value", "DCF Value"],
                    "Value": [
                        f"{self.currency_symbol}{self.free_cash_flows[0]:,.2f}M",
                        f"{discount:.1%}",
                        f"{terminal_growth:.1%}",
                        f"{config.valuation_params['high_growth_period']} years",
                        f"{self.currency_symbol}{sum(self.free_cash_flows) * (1 + terminal_growth) / (discount - terminal_growth):,.2f}M",
                        f"{self.currency_symbol}{intrinsic_value:,.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
            else:
                st.warning(
                    "Insufficient data for DCF calculation. Try selecting a different ticker with more complete financial data.")


class ComparableAnalysis:
    def __init__(self, ticker_data):
        self.data = ticker_data
        self.peers = []
        self.ticker = self.data['info'].get('symbol', '')
        self.currency = self.data['info'].get('currency', 'USD')
        self.currency_symbol = get_currency_symbol(self.currency)

    def get_comparables(self):
        """Identify comparable companies with better error handling"""
        try:
            sector = self.data['info'].get('sector', 'Technology')
            industry = self.data['info'].get('industry', 'Software')
            return get_industry_peers(sector, industry, self.ticker)
        except Exception as e:
            st.warning(f"Error identifying comparables: {str(e)}")
            return ['MSFT', 'AAPL', 'GOOG']

    def analyze(self):
        """Run comparable analysis with improved error handling and data validation"""
        try:
            self.peers = self.get_comparables()
            peer_data = []

            # Add current ticker data first
            if self.data and self.data['info']:
                current_ticker = self.data['info'].get('symbol', self.ticker)
                current_price = self.data['info'].get('currentPrice')
                trailing_pe = self.data['info'].get('trailingPE')
                forward_pe = self.data['info'].get('forwardPE')
                price_to_sales = self.data['info'].get('priceToSalesTrailing12Months')
                ev_to_ebitda = self.data['info'].get('enterpriseToEbitda')
                roe = self.data['info'].get('returnOnEquity')
                price_to_book = self.data['info'].get('priceToBook')

                peer_data.append({
                    'Ticker': current_ticker,
                    'Name': self.data['info'].get('shortName', current_ticker),
                    'Price': current_price if current_price else None,
                    'P/E': trailing_pe if trailing_pe else None,
                    'Forward P/E': forward_pe if forward_pe else None,
                    'P/S': price_to_sales if price_to_sales else None,
                    'EV/EBITDA': ev_to_ebitda if ev_to_ebitda else None,
                    'P/B': price_to_book if price_to_book else None,
                    'ROE': roe * 100 if roe else None,
                    '1Y Return': self._get_1y_return(current_ticker)
                })

            with st.spinner("Gathering peer data..."):
                for ticker in self.peers[:5]:  # Limit to 5 peers
                    data = get_market_data(ticker)
                    if data and data['info']:
                        current_price = data['info'].get('currentPrice')
                        trailing_pe = data['info'].get('trailingPE')
                        forward_pe = data['info'].get('forwardPE')
                        price_to_sales = data['info'].get('priceToSalesTrailing12Months')
                        ev_to_ebitda = data['info'].get('enterpriseToEbitda')
                        roe = data['info'].get('returnOnEquity')
                        price_to_book = data['info'].get('priceToBook')

                        peer_data.append({
                            'Ticker': ticker,
                            'Name': data['info'].get('shortName', ticker),
                            'Price': current_price if current_price else None,
                            'P/E': trailing_pe if trailing_pe else None,
                            'Forward P/E': forward_pe if forward_pe else None,
                            'P/S': price_to_sales if price_to_sales else None,
                            'EV/EBITDA': ev_to_ebitda if ev_to_ebitda else None,
                            'P/B': price_to_book if price_to_book else None,
                            'ROE': roe * 100 if roe else None,
                            '1Y Return': self._get_1y_return(ticker)
                        })

            return pd.DataFrame(peer_data)
        except Exception as e:
            st.error(f"Comparable analysis error: {str(e)}")
            return pd.DataFrame()

    def _get_1y_return(self, ticker):
        """Calculate 1-year return with better error handling"""
        try:
            data = yf.download(ticker, period='1y', progress=False)
            if len(data) > 20:  # Ensure we have enough data
                return (data['Close'][-1] / data['Close'][0] - 1) * 100
            return None
        except Exception:
            return None

    def display(self):
        """Display comparable analysis with improved visualization"""
        with st.expander("Comparable Company Analysis", expanded=True):
            st.subheader("Relative Valuation Multiples")

            df_comps = self.analyze()
            if df_comps.empty:
                st.warning("No comparable companies found with sufficient data")
                return

            try:
                # Get current company metrics
                current_eps = self.data['info'].get('trailingEps')
                forward_eps = self.data['info'].get('forwardEps')
                current_revenue = self.data['info'].get('totalRevenue')
                current_ebitda = self.data['info'].get('ebitda')
                shares_outstanding = self.data['info'].get('sharesOutstanding', 1)
                current_price = self.data['info'].get('currentPrice')
                book_value = self.data['info'].get('bookValue')

                if not all([current_price, shares_outstanding]):
                    st.warning("Insufficient data for valuation comparison")

                # Convert numeric columns to float first
                numeric_cols = ['P/E', 'Forward P/E', 'P/S', 'EV/EBITDA', 'P/B']
                for col in numeric_cols:
                    if col in df_comps.columns:
                        df_comps[col] = pd.to_numeric(df_comps[col], errors='coerce')

                # Calculate implied valuations - avoid division by zero
                valuation = {}

                # Only add metrics with valid data
                if current_eps and not pd.isna(df_comps['P/E'].median()) and not np.isnan(df_comps['P/E'].median()):
                    valuation['P/E Implied'] = current_eps * df_comps['P/E'].median()

                if forward_eps and not pd.isna(df_comps['Forward P/E'].median()) and not np.isnan(df_comps['Forward P/E'].median()):
                    valuation['Forward P/E Implied'] = forward_eps * df_comps['Forward P/E'].median()

                if current_revenue and shares_outstanding and not pd.isna(df_comps['P/S'].median()) and not np.isnan(df_comps['P/S'].median()):
                    valuation['P/S Implied'] = (current_revenue / shares_outstanding) * df_comps['P/S'].median()

                if current_ebitda and shares_outstanding and not pd.isna(df_comps['EV/EBITDA'].median()) and not np.isnan(df_comps['EV/EBITDA'].median()):
                    valuation['EV/EBITDA Implied'] = (current_ebitda * df_comps['EV/EBITDA'].median()) / shares_outstanding

                if book_value and not pd.isna(df_comps['P/B'].median()) and not np.isnan(df_comps['P/B'].median()):
                    valuation['P/B Implied'] = book_value * df_comps['P/B'].median()

                if current_price:
                    valuation['Current Price'] = current_price

                if valuation:
                    # Football field chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(valuation.keys()),
                        y=list(valuation.values()),
                        marker_color=['#636EFA', '#00CC96', '#FFA15A', '#AB63FA', '#19D3F3', '#FF6692'],
                        name='Valuation'
                    ))
                    fig.update_layout(
                        title="Valuation Range Comparison",
                        yaxis_title=f"Price ({self.currency_symbol})",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate implied valuation stats
                    if current_price and len(valuation) > 1:
                        fair_values = [v for k, v in valuation.items() if k != 'Current Price' and v is not None]
                        if fair_values:
                            avg_fair_value = sum(fair_values) / len(fair_values)
                            upside_potential = (avg_fair_value / current_price - 1) * 100

                            stat_cols = st.columns(3)
                            stat_cols[0].metric("Average Fair Value", f"{self.currency_symbol}{avg_fair_value:.2f}")
                            stat_cols[1].metric("Current Price", f"{self.currency_symbol}{current_price:.2f}")
                            stat_cols[2].metric("Upside Potential",
                                                f"{upside_potential:.1f}%",
                                                delta=f"{upside_potential:.1f}%",
                                                delta_color="normal" if upside_potential > 0 else "inverse")

                # Peer comparison table
                st.subheader("Peer Comparison Metrics")

                # Format the dataframe safely
                def safe_format(x, fmt):
                    if pd.isna(x) or x == "N/A":
                        return "N/A"
                    try:
                        return fmt.format(float(x))
                    except:
                        return str(x)

                # Split the dataframe display into two columns for better UI
                cols = st.columns([3, 1])

                styled_df = df_comps.copy()
                highlight_first_row = False

                if not styled_df.empty and styled_df.iloc[0]['Ticker'] == self.ticker:
                    highlight_first_row = True

                # Format columns
                for col in styled_df.columns:
                    if col in ['Price']:
                        styled_df[col] = styled_df[col].apply(
                            lambda x: safe_format(x, f"{self.currency_symbol}{x:,.2f}"))
                    elif col in ['ROE', '1Y Return']:
                        styled_df[col] = styled_df[col].apply(lambda x: safe_format(x, f"{x:.1f}%"))
                    elif col in ['P/E', 'Forward P/E', 'P/S', 'EV/EBITDA', 'P/B']:
                        styled_df[col] = styled_df[col].apply(lambda x: safe_format(x, f"{x:.2f}"))

                with cols[0]:
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                    )

                with cols[1]:
                    st.subheader("Multiple Comparison")
                    # Create a radar chart of multiples
                    if not df_comps.empty and len(df_comps) > 1:
                        try:
                            metrics = ['P/E', 'P/S', 'EV/EBITDA', 'P/B']
                            available_metrics = [m for m in metrics if m in df_comps.columns]

                            # Convert to numeric, handling "N/A" values
                            chart_data = df_comps.copy()
                            for col in available_metrics:
                                chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')

                            # Get median values
                            median_values = chart_data[available_metrics].median()

                            # Create radar chart
                            if not median_values.empty and not median_values.isna().all():
                                radar_fig = go.Figure()

                                # Add industry median trace
                                radar_fig.add_trace(go.Scatterpolar(
                                    r=median_values.values,
                                    theta=available_metrics,
                                    fill='toself',
                                    name='Industry Median'
                                ))

                                # Add current company trace if we have the values
                                if chart_data.iloc[0]['Ticker'] == self.ticker:
                                    company_values = chart_data.iloc[0][available_metrics]
                                    if not company_values.isna().all():
                                        radar_fig.add_trace(go.Scatterpolar(
                                            r=company_values.values,
                                            theta=available_metrics,
                                            fill='toself',
                                            name=self.ticker
                                        ))

                                radar_fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                        ),
                                    ),
                                    showlegend=True
                                )
                                st.plotly_chart(radar_fig, use_container_width=True)

                        except Exception as e:
                            st.warning(f"Could not create comparison chart: {e}")

            except Exception as e:
                st.error(f"Valuation comparison error: {str(e)}")


class DividendValuation:
    def __init__(self, ticker_data):
        self.data = ticker_data
        self.dividend_history = []
        self.currency = self.data['info'].get('currency', 'USD')
        self.currency_symbol = get_currency_symbol(self.currency)

    def calculate(self, required_return, growth_rate):
        """Calculate dividend discount model valuation"""
        try:
            # Get current dividend
            current_dividend = self.data['info'].get('dividendRate', 0)

            # Handle case where dividend is zero or missing
            if not current_dividend or current_dividend <= 0:
                return None

            # Apply Gordon Growth Model (DDM)
            if required_return <= growth_rate:
                st.warning("Required return must be greater than growth rate for DDM")
                return None

            ddm_value = current_dividend * (1 + growth_rate) / (required_return - growth_rate)
            return ddm_value

        except Exception as e:
            st.error(f"Dividend valuation error: {str(e)}")
            return None

    def get_dividend_history(self):
        """Get dividend payment history"""
        try:
            ticker = self.data['info'].get('symbol', '')
            if not ticker:
                return []

            # Get dividend history from yfinance
            hist = self.data.get('history', pd.DataFrame())
            if not hist.empty and 'Dividends' in hist.columns:
                div_history = hist[hist['Dividends'] > 0]['Dividends']
                if not div_history.empty:
                    # Convert to readable format
                    history = []
                    for date, value in div_history.items():
                        history.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Amount': value
                        })
                    return history
            return []

        except Exception as e:
            st.warning(f"Could not fetch dividend history: {str(e)}")
            return []

    def display(self):
        """Display dividend valuation model"""
        with st.expander("Dividend Discount Model", expanded=True):
            # Check if company pays dividends
            current_dividend = self.data['info'].get('dividendRate', 0)
            dividend_yield = self.data['info'].get('dividendYield', 0) * 100
            payout_ratio = self.data['info'].get('payoutRatio', 0) * 100

            if not current_dividend or current_dividend <= 0:
                st.warning("This company does not pay dividends or dividend data is unavailable.")
                return

            # Input parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                required_return = st.slider("Required Return (%)", 2.0, 20.0, 8.0, 0.5) / 100
            with col2:
                growth_rate = st.slider("Dividend Growth Rate (%)", 0.0, 15.0, 3.0, 0.5) / 100
            with col3:
                years_to_project = st.slider("Projection Years", 1, 20, 10)

            # Calculate DDM value
            ddm_value = self.calculate(required_return, growth_rate)
            current_price = self.data['info'].get('currentPrice')

            # Display metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric("Current Dividend", f"{self.currency_symbol}{current_dividend:.2f}")
            metric_cols[1].metric("Dividend Yield", f"{dividend_yield:.2f}%")
            metric_cols[2].metric("Payout Ratio", f"{payout_ratio:.2f}%")

            if ddm_value and current_price:
                metric_cols[3].metric(
                    "DDM Value",
                    f"{self.currency_symbol}{ddm_value:.2f}",
                    delta=f"{((ddm_value / current_price) - 1) * 100:.1f}%",
                    delta_color="normal" if ddm_value > current_price else "inverse"
                )

                # Dividend projection chart
                projected_dividends = [current_dividend * (1 + growth_rate) ** year for year in range(years_to_project)]
                projected_years = [datetime.now().year + year for year in range(years_to_project)]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=projected_years,
                    y=projected_dividends,
                    name="Projected Dividends",
                    marker_color='#00CC96'
                ))

                # Add cumulative line
                cumulative_dividends = np.cumsum(projected_dividends)
                fig.add_trace(go.Scatter(
                    x=projected_years,
                    y=cumulative_dividends,
                    name="Cumulative Dividends",
                    line=dict(color='#EF553B', width=2)
                ))

                fig.update_layout(
                    title="Projected Dividend Growth",
                    xaxis_title="Year",
                    yaxis_title=f"Dividend ({self.currency_symbol})",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show dividend history
                dividend_history = self.get_dividend_history()
                if dividend_history:
                    st.subheader("Dividend Payment History")

                    # Convert to DataFrame for display
                    hist_df = pd.DataFrame(dividend_history)

                    # Format amounts with currency symbol
                    hist_df['Amount'] = hist_df['Amount'].apply(lambda x: f"{self.currency_symbol}{x:.4f}")

                    st.dataframe(
                        hist_df,
                        use_container_width=True,
                        height=200
                    )

                # Show calculation breakdown
                st.subheader("DDM Calculation Breakdown")
                breakdown_data = {
                    "Component": ["Current Dividend", "Required Return", "Growth Rate", "DDM Formula", "DDM Value"],
                    "Value": [
                        f"{self.currency_symbol}{current_dividend:.2f}",
                        f"{required_return:.1%}",
                        f"{growth_rate:.1%}",
                        f"D1 / (r - g) = {self.currency_symbol}{current_dividend * (1 + growth_rate):.2f} / ({required_return:.3f} - {growth_rate:.3f})",
                        f"{self.currency_symbol}{ddm_value:.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
            else:
                st.warning("Unable to calculate DDM value with the current parameters.")


class TechnicalAnalysis:
    def __init__(self, ticker_data):
        self.data = ticker_data
        self.currency = self.data['info'].get('currency', 'USD')
        self.currency_symbol = get_currency_symbol(self.currency)

    def calculate_indicators(self):
        """Calculate technical indicators from price history"""
        try:
            history = self.data.get('history', None)
            if history is None or history.empty:
                return None

            # Make a copy to avoid modifying original data
            df = history.copy()

            # Calculate simple moving averages
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()

            # Calculate Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))

            # Calculate MACD
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']

            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            sigma = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * sigma
            df['BB_Lower'] = df['BB_Middle'] - 2 * sigma

            # Filter for the most recent data
            recent = df.iloc[-180:].copy()  # Use past 180 days or available data

            return recent

        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return None

    def display(self):
        """Display technical analysis charts"""
        with st.expander("Technical Analysis", expanded=True):
            indicator_data = self.calculate_indicators()

            if indicator_data is None or indicator_data.empty:
                st.warning("Insufficient price history data for technical analysis")
                return

            # Tabs for different charts
            tab1, tab2, tab3 = st.tabs(["Price & Moving Averages", "RSI & MACD", "Bollinger Bands"])

            with tab1:
                # Price and Moving Averages
                fig1 = go.Figure()

                # Add price candlesticks
                fig1.add_trace(go.Candlestick(
                    x=indicator_data.index,
                    open=indicator_data['Open'],
                    high=indicator_data['High'],
                    low=indicator_data['Low'],
                    close=indicator_data['Close'],
                    name="Price"
                ))

                # Add moving averages
                fig1.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['SMA20'],
                    line=dict(color='blue', width=1),
                    name="SMA20"
                ))

                fig1.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['SMA50'],
                    line=dict(color='orange', width=1),
                    name="SMA50"
                ))

                fig1.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['SMA200'],
                    line=dict(color='red', width=1),
                    name="SMA200"
                ))

                fig1.update_layout(
                    title="Price and Moving Averages",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({self.currency_symbol})",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )
                st.plotly_chart(fig1, use_container_width=True)

                # Moving average crossover signals
                current_close = indicator_data['Close'].iloc[-1]
                current_sma20 = indicator_data['SMA20'].iloc[-1]
                current_sma50 = indicator_data['SMA50'].iloc[-1]
                current_sma200 = indicator_data['SMA200'].iloc[-1]

                signal_cols = st.columns(3)

                # Golden/Death Cross (SMA50 vs SMA200)
                golden_cross = current_sma50 > current_sma200
                signal_cols[0].metric(
                    "50-200 SMA Cross",
                    "Golden Cross" if golden_cross else "Death Cross",
                    delta="Bullish" if golden_cross else "Bearish",
                    delta_color="normal" if golden_cross else "inverse"
                )

                # Short term trend (Price vs SMA20)
                short_bullish = current_close > current_sma20
                signal_cols[1].metric(
                    "Short-term Trend",
                    "Above SMA20" if short_bullish else "Below SMA20",
                    delta="Bullish" if short_bullish else "Bearish",
                    delta_color="normal" if short_bullish else "inverse"
                )

                # Medium term trend (Price vs SMA50)
                medium_bullish = current_close > current_sma50
                signal_cols[2].metric(
                    "Medium-term Trend",
                    "Above SMA50" if medium_bullish else "Below SMA50",
                    delta="Bullish" if medium_bullish else "Bearish",
                    delta_color="normal" if medium_bullish else "inverse"
                )

            with tab2:
                # Create subplot with shared x-axis
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['MACD'],
                    line=dict(color='blue', width=1),
                    name="MACD"
                ))

                fig2.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['Signal'],
                    line=dict(color='red', width=1),
                    name="Signal"
                ))

                # MACD Histogram as bar chart
                fig2.add_trace(go.Bar(
                    x=indicator_data.index,
                    y=indicator_data['MACD_Hist'],
                    marker_color=np.where(indicator_data['MACD_Hist'] >= 0, 'green', 'red'),
                    name="MACD Histogram"
                ))

                fig2.update_layout(
                    title="MACD (12,26,9)",
                    xaxis_title="Date",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)

                # RSI Chart
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['RSI'],
                    line=dict(color='purple', width=1),
                    name="RSI"
                ))

                # Add overbought/oversold lines
                fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

                fig3.update_layout(
                    title="Relative Strength Index (14)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_white"
                )
                st.plotly_chart(fig3, use_container_width=True)

                # RSI and MACD signals
                current_rsi = indicator_data['RSI'].iloc[-1]
                current_macd = indicator_data['MACD'].iloc[-1]
                current_signal = indicator_data['Signal'].iloc[-1]

                signal_cols = st.columns(2)

                # RSI signal
                rsi_condition = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                rsi_signal = "Bearish" if current_rsi > 70 else "Bullish" if current_rsi < 30 else "Neutral"
                signal_cols[0].metric(
                    "RSI Signal",
                    f"{rsi_condition} ({current_rsi:.1f})",
                    delta=rsi_signal,
                    delta_color="normal" if rsi_signal == "Bullish" else "inverse" if rsi_signal == "Bearish" else "off"
                )

                # MACD signal
                macd_bullish = current_macd > current_signal
                signal_cols[1].metric(
                    "MACD Signal",
                    "Bullish Crossover" if macd_bullish else "Bearish Crossover",
                    delta="Bullish" if macd_bullish else "Bearish",
                    delta_color="normal" if macd_bullish else "inverse"
                )

            with tab3:
                # Bollinger Bands
                fig4 = go.Figure()

                # Add price
                fig4.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['Close'],
                    line=dict(color='black', width=1),
                    name="Price"
                ))

                # Add Bollinger Bands
                fig4.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['BB_Upper'],
                    line=dict(color='rgba(0, 176, 246, 0.7)', width=1, dash='dash'),
                    name="Upper Band (+2σ)"
                ))

                fig4.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['BB_Middle'],
                    line=dict(color='rgba(255, 127, 14, 0.7)', width=1),
                    name="Middle Band (SMA20)"
                ))

                fig4.add_trace(go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data['BB_Lower'],
                    line=dict(color='rgba(0, 176, 246, 0.7)', width=1, dash='dash'),
                    name="Lower Band (-2σ)"
                ))

                # Fill between upper and lower bands
                fig4.add_trace(go.Scatter(
                    x=indicator_data.index.tolist() + indicator_data.index.tolist()[::-1],
                    y=indicator_data['BB_Upper'].tolist() + indicator_data['BB_Lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 176, 246, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name="Band Range"
                ))

                fig4.update_layout(
                    title="Bollinger Bands (20,2)",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({self.currency_symbol})",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )
                st.plotly_chart(fig4, use_container_width=True)

                # Bollinger Band signals
                current_close = indicator_data['Close'].iloc[-1]
                current_upper = indicator_data['BB_Upper'].iloc[-1]
                current_lower = indicator_data['BB_Lower'].iloc[-1]
                current_middle = indicator_data['BB_Middle'].iloc[-1]

                # Calculate % bandwidth and %B
                bandwidth = (current_upper - current_lower) / current_middle * 100
                percent_b = (current_close - current_lower) / (current_upper - current_lower) if (
                                                                                                             current_upper - current_lower) != 0 else 0.5

                signal_cols = st.columns(3)

                # Position within bands
                position = "Upper Band" if current_close >= current_upper else "Lower Band" if current_close <= current_lower else "Middle"
                signal = "Overbought" if current_close >= current_upper else "Oversold" if current_close <= current_lower else "Neutral"

                signal_cols[0].metric(
                    "Position within Bands",
                    position,
                    delta=signal,
                    delta_color="inverse" if signal == "Overbought" else "normal" if signal == "Oversold" else "off"
                )

                # %B indicator
                signal_cols[1].metric(
                    "%B Indicator",
                    f"{percent_b:.2f}",
                    delta="High" if percent_b > 0.8 else "Low" if percent_b < 0.2 else "Neutral",
                    delta_color="inverse" if percent_b > 0.8 else "normal" if percent_b < 0.2 else "off"
                )

                # Bandwidth
                signal_cols[2].metric(
                    "Bandwidth",
                    f"{bandwidth:.2f}%",
                    delta="Expanding" if bandwidth > 20 else "Contracting" if bandwidth < 10 else "Average",
                    delta_color="normal" if bandwidth > 20 else "off"
                )


# =============================================
# VALUATION SUMMARY
# =============================================
class ValuationSummary:
    def __init__(self, ticker_data):
        self.data = ticker_data
        self.currency = self.data['info'].get('currency', 'USD')
        self.currency_symbol = get_currency_symbol(self.currency)

    def display(self):
        """Enhanced company summary with better visual design"""
        try:
            # Extract company info
            company_name = self.data['info'].get('shortName', '')
            ticker = self.data['info'].get('symbol', '')
            sector = self.data['info'].get('sector', 'N/A')
            industry = self.data['info'].get('industry', 'N/A')
            current_price = self.data['info'].get('currentPrice')
            market_cap = self.data['info'].get('marketCap', 0) / 1e9  # Convert to billions
            pe_ratio = self.data['info'].get('trailingPE')
            eps = self.data['info'].get('trailingEps')
            revenue = self.data['info'].get('totalRevenue', 0) / 1e9  # Convert to billions
            profit_margin = self.data['info'].get('profitMargins', 0) * 100
            dividend_yield = self.data['info'].get('dividendYield', 0) * 100

            # Company header with improved styling
            st.markdown(f"""
            <div style="background-color:#f8f9fa;padding:20px;border-radius:5px;margin-bottom:20px;border-left:4px solid #2c3e50">
                <h1 style="color:#2c3e50;margin-bottom:5px">{company_name} ({ticker})</h1>
                <p style="color:#6c757d;font-size:16px">{sector} > {industry}</p>
            </div>
            """, unsafe_allow_html=True)

            # Create metrics layout with cards
            cols = st.columns(4)

            cols[0].metric(
                "Current Price",
                f"{self.currency_symbol}{current_price:,.2f}" if current_price else "N/A",
                delta=f"{self.data['info'].get('52WeekChange', 0) * 100:.1f}%" if 'info' in self.data and '52WeekChange' in
                                                                                  self.data['info'] else None
            )

            cols[1].metric("Market Cap", f"{market_cap:.2f}B {self.currency}" if market_cap else "N/A")
            cols[2].metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
            cols[3].metric("EPS", f"{self.currency_symbol}{eps:.2f}" if eps else "N/A")

            cols = st.columns(4)
            cols[0].metric("Revenue (TTM)", f"{revenue:.2f}B {self.currency}" if revenue else "N/A")
            cols[1].metric("Profit Margin", f"{profit_margin:.2f}%" if profit_margin else "N/A")
            cols[2].metric("Dividend Yield", f"{dividend_yield:.2f}%" if dividend_yield else "N/A")

            # 52-week range with visual indicator
            week_low = self.data['info'].get('fiftyTwoWeekLow')
            week_high = self.data['info'].get('fiftyTwoWeekHigh')

            if week_low and week_high and current_price:
                range_percent = (current_price - week_low) / (week_high - week_low) * 100

                cols[3].metric("52-Week Range",
                               f"{self.currency_symbol}{week_low:.2f} - {self.currency_symbol}{week_high:.2f}")

                st.write(f"Current price is at **{range_percent:.1f}%** of 52-week range")

                # Enhanced visual representation
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=range_percent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "52-Week Range Position"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 30], 'color': "#e74c3c"},
                            {'range': [30, 70], 'color': "#f39c12"},
                            {'range': [70, 100], 'color': "#2ecc71"}]
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Description with expandable section
            with st.expander("Company Description", expanded=False):
                desc = self.data['info'].get('longBusinessSummary', 'No description available.')
                st.markdown(f"<div style='text-align: justify;'>{desc}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error displaying company summary: {str(e)}")


# =============================================
# MAIN APP LAYOUT
# =============================================
def main():
    # Initialize session state for UI persistence
    if 'expanded_sections' not in st.session_state:
        st.session_state.expanded_sections = {
            'summary': True,
            'dcf': True,
            'comps': True,
            'dividend': True,
            'technical': True
        }

    # Main title with professional styling
    st.markdown("""
    <div style="background-color:#2c3e50;padding:20px;border-radius:5px;margin-bottom:20px">
        <h1 style="color:white;text-align:center;">Intrinsic</h1>
        <p style="color:#bdc3c7;text-align:center;">Comprehensive stock valuation and analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration with improved organization
    with st.sidebar:
        st.title("Analysis Settings")

        # Ticker input
        default_ticker = "AAPL"
        ticker = st.text_input("Enter Ticker Symbol", default_ticker).upper().strip()

        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            list(config.data_sources.keys()),
            index=0
        )
        source_key = config.data_sources[data_source]

        # API keys for additional data sources
        api_keys = {}
        if source_key != "yfinance":
            with st.expander(f"Configure {data_source} API"):
                api_key = st.text_input(f"{data_source} API Key", type="password")
                if api_key:
                    api_keys[source_key.lower()] = api_key

        # Advanced settings
        with st.expander("Advanced Settings"):
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                0.0, 10.0,
                float(config.valuation_params['risk_free'] * 100),
                0.1
            ) / 100
            market_premium = st.slider(
                "Market Risk Premium (%)",
                1.0, 12.0,
                float(config.valuation_params['market_premium'] * 100),
                0.1
            ) / 100
            terminal_growth = st.slider(
                "Terminal Growth Rate (%)",
                0.0, 5.0,
                float(config.valuation_params['terminal_growth'] * 100),
                0.1
            ) / 100

            config.valuation_params['risk_free'] = risk_free_rate
            config.valuation_params['market_premium'] = market_premium
            config.valuation_params['terminal_growth'] = terminal_growth

        # Custom peer selection
        with st.expander("Custom Peer Companies"):
            custom_peers = st.text_input("Enter tickers separated by commas")
            use_custom_peers = st.checkbox("Use custom peers instead of industry defaults")

        # Display settings
        with st.expander("Display Settings"):
            show_technical = st.checkbox("Show Technical Analysis", True)
            show_dividends = st.checkbox("Show Dividend Analysis", True)

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Valuation Analysis", "Financial Data"])

    with tab1:
        # Fetch data with progress indicator
        with st.spinner(f"Loading data for {ticker}..."):
            ticker_data = get_market_data(ticker, source=source_key, api_keys=api_keys)

        if ticker_data:
            # Company summary section
            summary = ValuationSummary(ticker_data)
            summary.display()

            # Main valuation modules
            dcf = DCFValuation(ticker_data)
            dcf.display()

            comps = ComparableAnalysis(ticker_data)
            if use_custom_peers and custom_peers:
                custom_peer_list = [p.strip().upper() for p in custom_peers.split(',')]
                comps.peers = custom_peer_list
            comps.display()

            if show_dividends:
                div = DividendValuation(ticker_data)
                div.display()

            if show_technical:
                tech = TechnicalAnalysis(ticker_data)
                tech.display()

    with tab2:
        if ticker_data:
            st.subheader("Financial Statements")

            # Financial statement tabs
            fin_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

            with fin_tabs[0]:
                if 'financials' in ticker_data:
                    st.dataframe(ticker_data['financials'], use_container_width=True)
                else:
                    st.warning("Income statement data not available")

            with fin_tabs[1]:
                if 'balance_sheet' in ticker_data:
                    st.dataframe(ticker_data['balance_sheet'], use_container_width=True)
                else:
                    st.warning("Balance sheet data not available")

            with fin_tabs[2]:
                if 'cashflow' in ticker_data:
                    st.dataframe(ticker_data['cashflow'], use_container_width=True)
                else:
                    st.warning("Cash flow statement data not available")
        else:
            st.warning("No financial data available for the selected ticker")


if __name__ == "__main__":
    main()
