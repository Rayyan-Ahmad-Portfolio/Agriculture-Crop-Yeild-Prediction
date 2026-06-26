"""
AgriYield AI — Predictive Harvest Console
Streamlit frontend for the local FastAPI crop yield forecasting microservice.
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

CROP_OPTIONS = [
    "Maize",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Wheat",
    "Cassava",
    "Sweet potatoes",
    "Plantains and others",
    "Yams",
]

REGION_OPTIONS = [
    "Albania",
    "Pakistan",
    "India",
    "Indonesia",
    "United Kingdom",
    "United States",
    "Afghanistan",
    "Argentina",
    "Australia",
    "Bangladesh",
    "Brazil",
    "Canada",
    "China",
    "Egypt",
    "Ethiopia",
    "France",
    "Germany",
    "Kenya",
    "Mexico",
    "Nigeria",
    "South Africa",
    "Spain",
    "Turkey",
    "Ukraine",
    "Viet Nam",
]

# Standard market commodity price index (USD per metric ton)
COMMODITY_PRICE_USD_PER_TON = {
    "Maize": 198.0,
    "Potatoes": 245.0,
    "Rice, paddy": 362.0,
    "Sorghum": 175.0,
    "Soybeans": 448.0,
    "Wheat": 285.0,
    "Cassava": 118.0,
    "Sweet potatoes": 305.0,
    "Plantains and others": 410.0,
    "Yams": 355.0,
}

# 1 hectogram = 0.0001 metric tons
HG_TO_METRIC_TONS = 0.0001


# ---------------------------------------------------------------------------
# Custom CSS — injected once via st.markdown(unsafe_allow_html=True)
# The <style> block targets Streamlit's DOM classes to override defaults.
# ---------------------------------------------------------------------------
def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        /* --- Global canvas: deep dark slate foundation --- */
        .stApp {
            background: linear-gradient(160deg, #0f1419 0%, #1a2332 45%, #121820 100%);
            color: #f8fafc;
        }

        /* Hide default Streamlit chrome for a cleaner single-page dashboard */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* --- Header bar wrapper (applied via custom HTML below inputs) --- */
        .agri-header {
            text-align: center;
            padding: 1.5rem 0 2rem 0;
            border-bottom: 1px solid rgba(74, 222, 128, 0.15);
            margin-bottom: 2rem;
        }
        .agri-header h1 {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #f8fafc;
            margin: 0;
        }
        .agri-header .accent {
            color: #4ade80;
        }
        .agri-header p {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-top: 0.6rem;
            font-weight: 400;
        }

        /* --- Section labels --- */
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #e2e8f0;
            margin: 1.5rem 0 1rem 0;
            letter-spacing: 0.02em;
        }

        /* --- Input card panels: subtle glass effect on dark slate --- */
        .input-card {
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 0.5rem;
            backdrop-filter: blur(8px);
        }

        /* Streamlit widget labels — crisp white for readability */
        label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #f1f5f9 !important;
            font-weight: 500 !important;
        }

        /* Dropdowns and numeric inputs: dark inset fields */
        .stSelectbox > div > div,
        .stNumberInput input {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }

        /* Primary action button: matrix-green glow */
        .stButton > button {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
            color: #0f1419 !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.06em !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 2.5rem !important;
            box-shadow: 0 0 24px rgba(74, 222, 128, 0.35) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 0 32px rgba(74, 222, 128, 0.55) !important;
        }

        /* --- Metric display cards (yield = green, revenue = amber) --- */
        .metric-card {
            border-radius: 16px;
            padding: 2rem 1.75rem;
            text-align: center;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border: 1px solid transparent;
        }
        .metric-card.yield {
            background: linear-gradient(145deg, rgba(34, 197, 94, 0.18) 0%, rgba(22, 163, 74, 0.08) 100%);
            border-color: rgba(74, 222, 128, 0.35);
            box-shadow: 0 8px 32px rgba(34, 197, 94, 0.12);
        }
        .metric-card.revenue {
            background: linear-gradient(145deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 100%);
            border-color: rgba(251, 191, 36, 0.35);
            box-shadow: 0 8px 32px rgba(245, 158, 11, 0.12);
        }
        .metric-card .label {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #cbd5e1;
            margin-bottom: 0.75rem;
        }
        .metric-card.yield .value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #4ade80;
            line-height: 1.2;
        }
        .metric-card.revenue .value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #fbbf24;
            line-height: 1.2;
        }
        .metric-card .sub {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.6rem;
        }

        /* Error alert banner */
        .system-error {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(248, 113, 113, 0.4);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: #fecaca;
            text-align: center;
            font-size: 1rem;
            margin: 1.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hg_ha_to_metric_tons_per_hectare(hg_per_ha: float) -> float:
    """Convert model output (hectograms per hectare) to metric tons per hectare."""
    return hg_per_ha * HG_TO_METRIC_TONS


def estimate_revenue_usd_per_hectare(crop: str, yield_tons_per_ha: float) -> float:
    """Project gross revenue per hectare using the commodity price index."""
    price_per_ton = COMMODITY_PRICE_USD_PER_TON.get(crop, 200.0)
    return yield_tons_per_ha * price_per_ton


def render_metric_cards(yield_tons: float, revenue_usd: float, raw_hg: float) -> None:
    """Render side-by-side analytics cards via custom HTML wrappers."""
    col_yield, col_revenue = st.columns(2)

    with col_yield:
        st.markdown(
            f"""
            <div class="metric-card yield">
                <div class="label">Projected Yield</div>
                <div class="value">{yield_tons:,.2f}</div>
                <div class="sub">Metric Tons per Hectare</div>
                <div class="sub">Raw model reading: {raw_hg:,.0f} hg/ha</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_revenue:
        st.markdown(
            f"""
            <div class="metric-card revenue">
                <div class="label">Estimated Gross Revenue</div>
                <div class="value">${revenue_usd:,.2f}</div>
                <div class="sub">USD per Hectare (market index)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="AgriYield AI • Predictive Harvest Console",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_custom_styles()

    # Header — custom HTML wrapper applies .agri-header styles from CSS block
    st.markdown(
        """
        <div class="agri-header">
            <h1><span class="accent">AgriYield AI</span> • Predictive Harvest Console</h1>
            <p>Real-time crop intelligence and economic forecasting powered by Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Field & Environmental Telemetry (input section)
    # -----------------------------------------------------------------------
    st.markdown(
        '<p class="section-title">Field &amp; Environmental Telemetry</p>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        crop_type = st.selectbox("Crop Type", options=CROP_OPTIONS, key="crop")
        region = st.selectbox("Target Region", options=REGION_OPTIONS, key="region")
        year = st.slider("Target Production Year", min_value=2000, max_value=2035, value=2026)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        rainfall = st.number_input(
            "Annual Rainfall (mm)",
            min_value=0.0,
            value=850.0,
            step=10.0,
            help="Average rainfall in millimeters per year.",
        )
        pesticides = st.number_input(
            "Pesticides Used (metric tonnes)",
            min_value=0.0,
            value=1.5,
            step=0.1,
        )
        avg_temp = st.number_input(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=22.0,
            step=0.5,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Centered action button
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_center, _ = st.columns([1, 2, 1])
    with btn_center:
        compute = st.button("🚀 COMPUTE HARVEST ANALYSIS", use_container_width=True)

    # -----------------------------------------------------------------------
    # Yield Analytics & Revenue Estimates (output section)
    # -----------------------------------------------------------------------
    if compute:
        payload = {
            "Area": region,
            "Item": crop_type,
            "Year": int(year),
            "average_rain_fall_mm_per_year": float(rainfall),
            "pesticides_tonnes": float(pesticides),
            "avg_temp": float(avg_temp),
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()

            raw_hg_yield = float(result.get("hg_ha_yield", 0))
            yield_tons_per_ha = hg_ha_to_metric_tons_per_hectare(raw_hg_yield)
            revenue_usd = estimate_revenue_usd_per_hectare(crop_type, yield_tons_per_ha)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-title">Yield Analytics &amp; Revenue Estimates</p>',
                unsafe_allow_html=True,
            )
            render_metric_cards(yield_tons_per_ha, revenue_usd, raw_hg_yield)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            st.markdown(
                """
                <div class="system-error">
                    System Link Error: Ensure your local AgriYield ML model microservice is actively running.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except requests.exceptions.HTTPError:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                pass
            st.markdown(
                f"""
                <div class="system-error">
                    System Link Error: Ensure your local AgriYield ML model microservice is actively running.
                    {"<br><small>" + str(detail) + "</small>" if detail else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except (requests.exceptions.RequestException, ValueError, KeyError):
            st.markdown(
                """
                <div class="system-error">
                    System Link Error: Ensure your local AgriYield ML model microservice is actively running.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
