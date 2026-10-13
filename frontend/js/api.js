/**
 * FraudShield AI - API Client
 * All fetch calls to the Flask backend.
 */

const BASE = (typeof window !== "undefined" && window.location.origin && window.location.origin.includes(":5000"))
  ? `${window.location.origin}/api`
  : "http://localhost:5000/api";

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await res.json();
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    console.error("API error:", path, err);
    return { ok: false, status: 0, data: { error: err.message } };
  }
}

const API = {
  health:            () => apiFetch("/health"),
  datasetSummary:    () => apiFetch("/dataset/summary"),
  eda:               () => apiFetch("/eda"),
  edaFeature:        (name) => apiFetch(`/eda/feature?name=${encodeURIComponent(name)}`),
  modelsStatus:      () => apiFetch("/models/status"),
  modelsComparison:  () => apiFetch("/models/comparison"),
  predict:           (body) => apiFetch("/predict",  { method: "POST", body: JSON.stringify(body) }),
  explain:           (body) => apiFetch("/explain",  { method: "POST", body: JSON.stringify(body) }),
  explainGlobal:     () => apiFetch("/explain/global"),
  anomaly:           () => apiFetch("/anomaly"),
  businessInsights:  () => apiFetch("/business-insights"),
  transactions:      (page, perPage, cls) => {
    let q = `?page=${page}&per_page=${perPage}`;
    if (cls !== null && cls !== undefined) q += `&class=${cls}`;
    return apiFetch("/transactions" + q);
  },
  thresholdAnalysis: () => apiFetch("/threshold-analysis"),
  cases:             (status, risk, search) => {
    const q = [];
    if (status && status !== "All") q.push(`status=${encodeURIComponent(status)}`);
    if (risk && risk !== "All")     q.push(`risk_level=${encodeURIComponent(risk)}`);
    if (search)                     q.push(`search=${encodeURIComponent(search)}`);
    return apiFetch("/cases" + (q.length ? `?${q.join("&")}` : ""));
  },
  createCase:        (data) => apiFetch("/cases", { method: "POST", body: JSON.stringify(data) }),
  updateCase:        (id, data) => apiFetch(`/cases/${id}`, { method: "PATCH", body: JSON.stringify(data) }),
  deleteCase:        (id) => apiFetch(`/cases/${id}`, { method: "DELETE" }),
};

window.API = API;
