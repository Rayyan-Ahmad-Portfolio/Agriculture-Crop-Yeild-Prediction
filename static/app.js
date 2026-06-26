/**
 * AgriYield AI — HTML Harvest Console
 * Fetches yield predictions from the local FastAPI /predict endpoint.
 */

const API_URL = `${window.location.origin}/predict`;
const HG_TO_METRIC_TONS = 0.0001;

const COMMODITY_PRICE_USD_PER_TON = {
  Maize: 198.0,
  Potatoes: 245.0,
  "Rice, paddy": 362.0,
  Sorghum: 175.0,
  Soybeans: 448.0,
  Wheat: 285.0,
  Cassava: 118.0,
  "Sweet potatoes": 305.0,
  "Plantains and others": 410.0,
  Yams: 355.0,
};

const CROP_OPTIONS = [
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
];

const REGION_OPTIONS = [
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
];

function formatNumber(value, decimals = 2) {
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function hgHaToMetricTonsPerHectare(hgPerHa) {
  return hgPerHa * HG_TO_METRIC_TONS;
}

function estimateRevenueUsdPerHectare(crop, yieldTonsPerHa) {
  const pricePerTon = COMMODITY_PRICE_USD_PER_TON[crop] ?? 200.0;
  return yieldTonsPerHa * pricePerTon;
}

function populateSelect(selectEl, options) {
  options.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option;
    opt.textContent = option;
    selectEl.appendChild(opt);
  });
}

function showError(message, detail = "") {
  const errorEl = document.getElementById("error-banner");
  errorEl.classList.remove("hidden");
  errorEl.innerHTML = detail
    ? `${message}<small>${detail}</small>`
    : message;
  document.getElementById("results-section").classList.add("hidden");
}

function hideError() {
  document.getElementById("error-banner").classList.add("hidden");
}

function renderResults(yieldTons, revenueUsd, rawHg) {
  document.getElementById("yield-value").textContent = formatNumber(yieldTons);
  document.getElementById("yield-raw").textContent = `Raw model reading: ${formatNumber(rawHg, 0)} hg/ha`;
  document.getElementById("revenue-value").textContent = `$${formatNumber(revenueUsd)}`;
  document.getElementById("results-section").classList.remove("hidden");
}

async function computeHarvestAnalysis(event) {
  event.preventDefault();
  hideError();

  const btn = document.getElementById("compute-btn");
  btn.disabled = true;
  btn.textContent = "⏳ ANALYZING…";

  const payload = {
    Area: document.getElementById("region").value,
    Item: document.getElementById("crop").value,
    Year: parseInt(document.getElementById("year").value, 10),
    average_rain_fall_mm_per_year: parseFloat(document.getElementById("rainfall").value),
    pesticides_tonnes: parseFloat(document.getElementById("pesticides").value),
    avg_temp: parseFloat(document.getElementById("avg-temp").value),
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = "";
      try {
        const err = await response.json();
        detail = err.detail ?? "";
      } catch {
        /* ignore parse errors */
      }
      showError(
        "System Link Error: Ensure your local AgriYield ML model microservice is actively running.",
        detail
      );
      return;
    }

    const result = await response.json();
    const rawHg = parseFloat(result.hg_ha_yield ?? 0);
    const yieldTons = hgHaToMetricTonsPerHectare(rawHg);
    const crop = document.getElementById("crop").value;
    const revenueUsd = estimateRevenueUsdPerHectare(crop, yieldTons);

    renderResults(yieldTons, revenueUsd, rawHg);
  } catch {
    showError(
      "System Link Error: Ensure your local AgriYield ML model microservice is actively running."
    );
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 COMPUTE HARVEST ANALYSIS";
  }
}

function initYearSlider() {
  const slider = document.getElementById("year");
  const display = document.getElementById("year-display");
  display.textContent = slider.value;
  slider.addEventListener("input", () => {
    display.textContent = slider.value;
  });
}

function init() {
  populateSelect(document.getElementById("crop"), CROP_OPTIONS);
  populateSelect(document.getElementById("region"), REGION_OPTIONS);
  initYearSlider();
  document.getElementById("forecast-form").addEventListener("submit", computeHarvestAnalysis);
}

document.addEventListener("DOMContentLoaded", init);
