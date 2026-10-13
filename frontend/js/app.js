/**
 * FraudShield AI - Main Application Controller
 * Handles navigation, data loading, and UI rendering for all sections.
 */

// ── Global state ─────────────────────────────────────────────────────────────
const STATE = {
  page: "overview",
  backendOnline: false,
  datasetAvailable: false,
  modelsAvailable: false,
  datasetSummary: null,
  comparison: null,
  txPage: 1,
  txTotal: 0,
  txPerPage: 50,
  txFilter: null,
  predForm: null,
};

// V-features for prediction form
const V_FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const ALL_FEATURES = ["Time", "Amount", ...V_FEATURES];

// ── Utilities ─────────────────────────────────────────────────────────────────
function fmt(v, d = 4) { return typeof v === "number" ? v.toFixed(d) : (v ?? "—"); }
function fmtInt(v) { return typeof v === "number" ? v.toLocaleString() : (v ?? "—"); }
function fmtPct(v) { return typeof v === "number" ? v.toFixed(4) + "%" : (v ?? "—"); }

function toast(msg, type = "info", dur = 3500) {
  const c = document.getElementById("toast-container");
  const el = document.createElement("div");
  const icons = { info: "ℹ️", success: "✅", error: "❌", warning: "⚠️" };
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || "ℹ️"}</span><span>${msg}</span>`;
  c.appendChild(el);
  setTimeout(() => el.remove(), dur);
}

function setLoading(containerId, msg = "Loading...") {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>${msg}</p></div>`;
}

function setEmpty(containerId, icon, title, body) {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `
    <div class="empty-state">
      <div class="icon">${icon}</div>
      <h3>${title}</h3>
      <p>${body}</p>
    </div>`;
}

// ── Navigation ────────────────────────────────────────────────────────────────
function navigateTo(page) {
  STATE.page = page;

  // Update nav items
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.page === page);
  });

  // Show/hide pages
  document.querySelectorAll(".page").forEach(el => {
    el.classList.toggle("active", el.id === `page-${page}`);
  });

  // Update top bar
  const titles = {
    overview:       ["Executive Overview",      "Real-time fraud monitoring dashboard"],
    explorer:       ["Transaction Explorer",    "Browse, search, and filter transactions"],
    prediction:     ["Fraud Prediction",        "Run inference with the trained model"],
    eda:            ["Exploratory Data Analysis","Statistical insights from the dataset"],
    models:         ["Model Laboratory",        "Compare and evaluate all ML models"],
    explainability: ["Explainable AI (SHAP)",   "Understand model predictions"],
    anomaly:        ["Anomaly Detection",        "Isolation Forest unsupervised analysis"],
    business:       ["Business Intelligence",   "Actionable insights from evaluation results"],
    cases:          ["Fraud Investigation Workspace", "Triage and track cases for flagged suspicious activity"],
    health:         ["System Health",           "Backend, dataset, and model status"],
  };
  const [title, sub] = titles[page] || ["FraudShield AI", ""];
  document.getElementById("top-bar-title").textContent = title;
  document.getElementById("top-bar-sub").textContent   = sub;

  // Close mobile sidebar
  document.getElementById("sidebar").classList.remove("open");

  // Lazy-load page data
  loadPageData(page);
}

// ── App Bootstrap ─────────────────────────────────────────────────────────────
async function init() {
  // Wire navigation
  document.querySelectorAll(".nav-item[data-page]").forEach(btn => {
    btn.addEventListener("click", () => navigateTo(btn.dataset.page));
  });

  // Burger menu
  document.getElementById("burger").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
  });

  // Check health
  await checkHealth();
  navigateTo("overview");
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const { ok, data } = await API.health();
  STATE.backendOnline = ok;
  STATE.datasetAvailable = data?.dataset?.dataset_available ?? false;
  STATE.modelsAvailable  = data?.dataset?.models_trained     ?? false;

  const dot  = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  if (ok) {
    dot.className = "health-dot online";
    text.textContent = "Backend Online";
  } else {
    dot.className = "health-dot offline";
    text.textContent = "Backend Offline";
    toast("Cannot reach backend. Is Flask running on port 5000?", "error", 8000);
  }
}

// ── Setup banner ──────────────────────────────────────────────────────────────
function updateSetupBanner(visible, extra = "") {
  const b = document.getElementById("setup-banner");
  if (!visible) { b.classList.add("hidden"); return; }
  b.classList.remove("hidden");
  document.getElementById("setup-extra").textContent = extra;
}

// ── Page data loaders ─────────────────────────────────────────────────────────
async function loadPageData(page) {
  switch (page) {
    case "overview":       await loadOverview();       break;
    case "explorer":       await loadExplorer();       break;
    case "prediction":            loadPredictionPage(); break;
    case "eda":            await loadEDA();            break;
    case "models":         await loadModels();         break;
    case "explainability": await loadExplainability(); break;
    case "anomaly":        await loadAnomaly();        break;
    case "business":       await loadBusiness();       break;
    case "cases":          await loadCases();          break;
    case "health":         await loadHealth();         break;
  }
}

// ============================================================ OVERVIEW ========
async function loadOverview() {
  const banner = !STATE.datasetAvailable || !STATE.modelsAvailable;
  updateSetupBanner(banner,
    !STATE.datasetAvailable
      ? "Place creditcard.csv in the data/ folder, then run: python train_pipeline.py"
      : "Run: python train_pipeline.py"
  );

  // Dataset summary
  const { ok: dsOk, data: ds } = await API.datasetSummary();
  if (dsOk && ds.available) {
    STATE.datasetSummary = ds;
    document.getElementById("kpi-total").textContent     = fmtInt(ds.total_transactions);
    document.getElementById("kpi-legit").textContent     = fmtInt(ds.legitimate_count);
    document.getElementById("kpi-fraud").textContent     = fmtInt(ds.fraud_count);
    document.getElementById("kpi-fraud-pct").textContent = fmtPct(ds.fraud_percentage);
    document.getElementById("kpi-source").textContent    = ds.is_sampled ? "⚠️ SAMPLED" : "✅ Full Dataset";
  } else {
    ["kpi-total","kpi-legit","kpi-fraud"].forEach(id => {
      document.getElementById(id).textContent = "N/A";
    });
    document.getElementById("kpi-fraud-pct").textContent = "N/A";
    document.getElementById("kpi-source").textContent    = "Dataset not loaded";
  }

  // Model comparison for KPI
  const { ok: mOk, data: mc } = await API.modelsComparison();
  if (mOk && mc.available) {
    STATE.comparison = mc;
    document.getElementById("kpi-best-model").textContent = mc.best_display;
    document.getElementById("kpi-ts").textContent = mc.timestamp
      ? new Date(mc.timestamp).toLocaleString() : "—";

    // Fraud dist chart
    const cd = mc.dataset_meta;
    if (cd) {
      Charts.buildClassDistChart("chart-overview-dist", cd.fraud_count, cd.legitimate_count);
    }

    // Model perf summary
    renderOverviewModelTable(mc.supervised_results);

    // Recent fraud from test transactions
    renderRecentFlagged(mc.supervised_results);
  } else {
    document.getElementById("kpi-best-model").textContent = "Not trained";
    document.getElementById("kpi-ts").textContent = "—";
    setEmpty("overview-model-table", "🤖", "No Model Data", "Run python train_pipeline.py to train models.");
  }
}

function renderOverviewModelTable(results) {
  if (!results || !results.length) return;
  const tbody = document.getElementById("overview-model-table");
  tbody.innerHTML = results.map(r => `
    <tr>
      <td><strong>${r.display_name}</strong></td>
      <td>${fmt(r.f1)}</td>
      <td>${fmt(r.precision)}</td>
      <td>${fmt(r.recall)}</td>
      <td>${r.roc_auc ? fmt(r.roc_auc) : "—"}</td>
      <td>${r.model_type === "supervised_classifier" ? '<span class="badge badge-info">Supervised</span>' : '<span class="badge badge-warning">Unsupervised</span>'}</td>
    </tr>`).join("");
}

function renderRecentFlagged(results) {
  const best = results?.[0];
  if (!best) return;
  const container = document.getElementById("overview-flagged");
  container.innerHTML = `
    <div class="alert alert-info">
      <span>ℹ️</span>
      <span>Recent flagged transactions are shown in the Transaction Explorer after model training. Best model: <strong>${best.display_name}</strong> — TP: ${fmtInt(best.tp)}, FP: ${fmtInt(best.fp)}, FN: ${fmtInt(best.fn)}</span>
    </div>`;
}

// ============================================================ EXPLORER ========
async function loadExplorer(page = 1, filter = STATE.txFilter) {
  STATE.txFilter = filter;
  STATE.txPage   = page;

  setLoading("tx-table-body", "Loading transactions...");

  const { ok, data } = await API.transactions(page, STATE.txPerPage, filter);

  if (!ok || !data.available) {
    setEmpty("tx-table-body", "📂", "Dataset Not Loaded",
      "Please add creditcard.csv to the data/ folder.");
    return;
  }

  STATE.txTotal = data.total;

  const rows = data.transactions.map(tx => `
    <tr>
      <td>${fmt(tx.Time, 0)}s</td>
      <td>$${fmt(tx.Amount, 2)}</td>
      <td>${fmt(tx.V1, 4)}</td>
      <td>${fmt(tx.V4, 4)}</td>
      <td>${fmt(tx.V14, 4)}</td>
      <td>${fmt(tx.V17, 4)}</td>
      <td>
        <span class="badge ${tx.Class === 1 ? 'badge-fraud' : 'badge-legit'}">
          ${tx.label}
        </span>
      </td>
    </tr>`).join("") || "<tr><td colspan='7' style='text-align:center;padding:24px;color:var(--text-muted)'>No results</td></tr>";

  document.getElementById("tx-table-body").innerHTML = rows;
  renderPagination(data.page, data.total_pages);
  document.getElementById("tx-count").textContent = `${fmtInt(data.total)} transactions`;
}

function renderPagination(current, total) {
  const el = document.getElementById("tx-pagination");
  if (!el || total <= 1) { if (el) el.innerHTML = ""; return; }
  let html = `<button class="page-btn" ${current <= 1 ? "disabled" : ""} onclick="loadExplorer(${current - 1})">←</button>`;
  const start = Math.max(1, current - 2);
  const end   = Math.min(total, current + 2);
  if (start > 1) html += `<button class="page-btn" onclick="loadExplorer(1)">1</button>`;
  if (start > 2) html += `<span style="color:var(--text-muted);padding:0 4px">…</span>`;
  for (let i = start; i <= end; i++) {
    html += `<button class="page-btn ${i === current ? 'active' : ''}" onclick="loadExplorer(${i})">${i}</button>`;
  }
  if (end < total - 1) html += `<span style="color:var(--text-muted);padding:0 4px">…</span>`;
  if (end < total) html += `<button class="page-btn" onclick="loadExplorer(${total})">${total}</button>`;
  html += `<button class="page-btn" ${current >= total ? "disabled" : ""} onclick="loadExplorer(${current + 1})">→</button>`;
  el.innerHTML = html;
}

function filterTransactions(cls) {
  STATE.txFilter = cls;
  loadExplorer(1, cls);
}

// ============================================================ PREDICTION ======
function loadPredictionPage() {
  if (document.getElementById("pred-form-fields").childElementCount > 0) return;
  buildPredictionForm();
}

function buildPredictionForm() {
  const container = document.getElementById("pred-form-fields");
  const fields = ALL_FEATURES.map(f => `
    <div class="form-group">
      <label for="pred-${f}">${f}</label>
      <input type="number" step="any" id="pred-${f}" name="${f}"
             placeholder="0" aria-label="${f}" />
    </div>`).join("");
  container.innerHTML = fields;
}

async function loadTestExample() {
  const { ok, data } = await API.transactions(1, 1, 1);
  if (!ok || !data.available || !data.transactions.length) {
    toast("No transactions available to load", "warning");
    return;
  }
  const tx = data.transactions[0];
  ALL_FEATURES.forEach(f => {
    const el = document.getElementById(`pred-${f}`);
    if (el && tx[f] !== undefined) el.value = tx[f];
  });
  toast("Loaded a fraud example from the dataset (ground-truth label: Fraud)", "info");
}

async function loadLegitExample() {
  const { ok, data } = await API.transactions(1, 1, 0);
  if (!ok || !data.available || !data.transactions.length) {
    toast("No transactions available", "warning");
    return;
  }
  const tx = data.transactions[0];
  ALL_FEATURES.forEach(f => {
    const el = document.getElementById(`pred-${f}`);
    if (el && tx[f] !== undefined) el.value = tx[f];
  });
  toast("Loaded a legitimate example from the dataset", "info");
}

async function submitPrediction() {
  const body = {};
  let valid = true;
  ALL_FEATURES.forEach(f => {
    const el = document.getElementById(`pred-${f}`);
    const v  = parseFloat(el?.value);
    if (isNaN(v)) { valid = false; el?.classList.add("error"); }
    else { el?.classList.remove("error"); body[f] = v; }
  });

  if (!valid) { toast("Please fill in all feature fields with numeric values.", "warning"); return; }

  const btn = document.getElementById("pred-btn");
  btn.disabled = true; btn.textContent = "Predicting…";

  const { ok, data } = await API.predict(body);
  btn.disabled = false; btn.textContent = "Run Prediction";

  if (!ok) {
    toast(data.error || "Prediction failed", "error");
    return;
  }

  renderPredictionResult(data);

  // Auto-run explanation
  runExplanationForBody(body);
}

function renderPredictionResult(data) {
  const el = document.getElementById("pred-result");
  const isFraud = data.prediction === 1;
  el.className = `pred-result ${isFraud ? "fraud" : "legit"}`;
  el.innerHTML = `
    <div class="pred-label" style="color:${isFraud ? '#f87171' : '#34d399'}">
      ${isFraud ? "🚨 FRAUD DETECTED" : "✅ LEGITIMATE"}
    </div>
    <div class="pred-prob">
      ${data.probability != null ? `Fraud probability: <strong>${(data.probability * 100).toFixed(2)}%</strong>` : ""}
    </div>
    <span class="pred-risk-badge">${data.risk_category || ""}</span>
    <p style="margin-top:14px;font-size:0.78rem;color:var(--text-muted)">
      Model: ${data.model_display} &nbsp;|&nbsp; ${data.disclaimer}
    </p>
    <div style="margin-top:14px">
      <button class="btn btn-secondary btn-sm" onclick="escalateFromPrediction('${data.risk_category || 'High'}', '${isFraud ? 'Fraud' : 'Legitimate'}')">
        🚩 Log Investigation Case
      </button>
    </div>`;
}

async function runExplanationForBody(body) {
  const el = document.getElementById("pred-explanation");
  el.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Computing SHAP explanation…</p></div>`;

  const { ok, data } = await API.explain(body);

  if (!ok || !data.available) {
    el.innerHTML = `<div class="alert alert-warning"><span>⚠️</span><span>Explanation unavailable: ${data.error || "unknown error"}</span></div>`;
    return;
  }

  renderLocalExplanation(el, data);
}

function renderLocalExplanation(container, data) {
  const top = data.contributions.slice(0, 10);
  const maxAbs = Math.max(...top.map(c => Math.abs(c.shap_value)));

  const bars = top.map(c => {
    const pct = maxAbs > 0 ? (Math.abs(c.shap_value) / maxAbs * 100).toFixed(1) : 0;
    const cls = c.shap_value > 0 ? "pos" : "neg";
    return `
      <div class="shap-bar-row">
        <span class="shap-feature">${c.feature}</span>
        <div class="shap-bar-wrap">
          <div class="shap-bar-fill ${cls}" style="width:${pct}%"></div>
        </div>
        <span class="shap-label">${c.shap_value > 0 ? "+" : ""}${c.shap_value.toFixed(4)}</span>
      </div>`;
  }).join("");

  container.innerHTML = `
    <div style="margin-bottom:12px;font-size:0.82rem;color:var(--text-secondary)">${data.interpretation}</div>
    <div class="card-title">Top Feature Contributions</div>
    ${bars}
    <p style="margin-top:12px;font-size:0.75rem;color:var(--text-muted)">
      🔴 Red bars = increase fraud risk &nbsp;|&nbsp; 🔵 Blue bars = decrease fraud risk
    </p>`;
}

// ============================================================ EDA ============
async function loadEDA() {
  setLoading("eda-content", "Computing EDA…");

  const { ok, data } = await API.eda();

  if (!ok || !data.available) {
    setEmpty("eda-content", "📊", "Dataset Not Available",
      "Please load creditcard.csv and run the training pipeline to see EDA charts.");
    return;
  }

  const cd = data.class_distribution;
  document.getElementById("eda-content").innerHTML = `
    <div class="kpi-grid" style="margin-bottom:24px">
      <div class="kpi-card">
        <div class="kpi-label">Total Transactions</div>
        <div class="kpi-value">${fmtInt(cd.total)}</div>
      </div>
      <div class="kpi-card danger">
        <div class="kpi-label">Fraud Count</div>
        <div class="kpi-value" style="color:var(--danger)">${fmtInt(cd.fraud)}</div>
        <div class="kpi-sub">${fmtPct(cd.fraud_pct)}</div>
      </div>
      <div class="kpi-card success">
        <div class="kpi-label">Legitimate</div>
        <div class="kpi-value" style="color:var(--success)">${fmtInt(cd.legitimate)}</div>
        <div class="kpi-sub">${fmtPct(cd.legitimate_pct)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Imbalance Ratio</div>
        <div class="kpi-value" style="font-size:1.4rem">${cd.imbalance_ratio ?? "—"}:1</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="card">
        <div class="card-title"><span class="icon">🍩</span>Class Distribution</div>
        <div class="chart-container"><canvas id="eda-dist"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">💰</span>Amount Distribution (99th pct cap)</div>
        <div class="chart-container"><canvas id="eda-amount"></canvas></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">⏱️</span>Transaction Volume Over Time (Hours)</div>
      <div class="chart-container" style="height:220px"><canvas id="eda-time"></canvas></div>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">🔥</span>Feature Correlation Matrix (Top 15 vs Target)</div>
      <div style="overflow:auto">
        <canvas id="eda-corr" style="max-width:100%;height:400px"></canvas>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">📈</span>Feature Distribution by Class</div>
      <div style="margin-bottom:12px;display:flex;gap:10px;flex-wrap:wrap">
        <select id="eda-feature-select" onchange="loadFeatureDist()" style="width:auto;min-width:120px">
          ${["Amount","Time",...Array.from({length:28},(_,i)=>`V${i+1}`)].map(f => `<option value="${f}">${f}</option>`).join("")}
        </select>
        <span style="font-size:0.8rem;color:var(--text-muted);align-self:center">Select a feature to view its distribution split by class</span>
      </div>
      <div class="chart-container" style="height:250px"><canvas id="eda-feature-dist"></canvas></div>
    </div>`;

  Charts.buildClassDistChart("eda-dist", cd.fraud, cd.legitimate);
  Charts.buildAmountChart("eda-amount", data.amount_distribution);
  Charts.buildTimeChart("eda-time", data.time_distribution);
  Charts.buildCorrelationHeatmap("eda-corr", data.correlation_matrix);
  loadFeatureDist("Amount");
}

async function loadFeatureDist(feature) {
  const sel = document.getElementById("eda-feature-select");
  const f   = feature || (sel && sel.value) || "Amount";
  if (sel) sel.value = f;

  const { ok, data } = await API.edaFeature(f);
  if (ok && data.available) {
    Charts.buildFeatureDistChart("eda-feature-dist", data);
  }
}

// ============================================================ MODELS ==========
async function loadModels() {
  setLoading("models-content", "Loading model evaluation results…");

  const { ok, data } = await API.modelsComparison();

  if (!ok || !data.available) {
    document.getElementById("models-content").innerHTML = `
      <div class="empty-state">
        <div class="icon">🤖</div>
        <h3>No Model Data Available</h3>
        <p>Run the training pipeline to generate model evaluation results.<br>
        <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-family:monospace">python train_pipeline.py</code></p>
      </div>`;
    return;
  }

  STATE.comparison = data;
  const allResults = [...(data.supervised_results || []), ...(data.unsupervised_results || [])];

  document.getElementById("models-content").innerHTML = `
    <div class="alert alert-info" style="margin-bottom:20px">
      <span>🏆</span>
      <span>Best supervised model: <strong>${data.best_display}</strong> — selected by highest F1-score.</span>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">📊</span>Model Comparison</div>
      <div class="chart-container" style="height:300px">
        <canvas id="model-compare-chart"></canvas>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">📋</span>Evaluation Table</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Type</th><th>Accuracy</th><th>Precision</th>
              <th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Train Time</th>
            </tr>
          </thead>
          <tbody id="models-table-body"></tbody>
        </table>
      </div>
    </div>

    <div id="confusion-matrices" class="charts-grid"></div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">📈</span>ROC Curves (Supervised)</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
        ${data.supervised_results.map(r => `
          <button class="btn btn-secondary btn-sm"
            onclick="showRocCurve('${r.model_name}', '${r.display_name}')"
            id="roc-btn-${r.model_name}">
            ${r.display_name}
          </button>`).join("")}
      </div>
      <div class="chart-container" style="height:280px"><canvas id="roc-chart"></canvas></div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">⚡</span>Threshold Analysis — ${data.best_display}</div>
      <div class="chart-container" style="height:260px"><canvas id="threshold-chart"></canvas></div>
    </div>`;

  Charts.buildModelCompareChart("model-compare-chart", allResults);
  renderModelsTable(allResults, data.best_model);
  renderConfusionMatrices(allResults);

  if (data.supervised_results.length) {
    const first = data.supervised_results[0];
    if (first.roc_curve) showRocCurve(first.model_name, first.display_name);
  }

  loadThresholdChart();
}

function renderModelsTable(results, bestModel) {
  const tbody = document.getElementById("models-table-body");
  tbody.innerHTML = results.map(r => `
    <tr class="${r.model_name === bestModel ? 'best-row' : ''}">
      <td><strong>${r.display_name}</strong> ${r.model_name === bestModel ? '<span class="badge badge-info">Best</span>' : ''}</td>
      <td><span class="badge ${r.model_type === 'supervised_classifier' ? 'badge-info' : 'badge-warning'}">${r.model_type === 'supervised_classifier' ? 'Supervised' : 'Unsupervised'}</span></td>
      <td>${fmt(r.accuracy)}</td>
      <td>${fmt(r.precision)}</td>
      <td>${fmt(r.recall)}</td>
      <td><strong>${fmt(r.f1)}</strong></td>
      <td>${r.roc_auc ? fmt(r.roc_auc) : "—"}</td>
      <td>${r.training_time_s ? r.training_time_s + "s" : "—"}</td>
    </tr>`).join("");
}

function renderConfusionMatrices(results) {
  const el = document.getElementById("confusion-matrices");
  el.innerHTML = results.map(r => `
    <div class="card">
      <div class="card-title"><span class="icon">🔲</span>Confusion Matrix — ${r.display_name}</div>
      <div class="cm-grid">
        <div class="cm-cell cm-tn"><strong>${fmtInt(r.tn)}</strong><span class="cm-lbl">TN</span></div>
        <div class="cm-cell cm-fp"><strong>${fmtInt(r.fp)}</strong><span class="cm-lbl">FP</span></div>
        <div class="cm-cell cm-fn"><strong>${fmtInt(r.fn)}</strong><span class="cm-lbl">FN</span></div>
        <div class="cm-cell cm-tp"><strong>${fmtInt(r.tp)}</strong><span class="cm-lbl">TP</span></div>
      </div>
      ${r.note ? `<p style="font-size:0.72rem;color:var(--text-muted);margin-top:10px;line-height:1.5">${r.note}</p>` : ""}
    </div>`).join("");
}

function showRocCurve(modelName, displayName) {
  const result = STATE.comparison?.supervised_results?.find(r => r.model_name === modelName);
  if (!result?.roc_curve) { toast("No ROC curve data for " + displayName, "warning"); return; }
  Charts.buildRocChart("roc-chart", result.roc_curve, displayName);
}

async function loadThresholdChart() {
  const { ok, data } = await API.thresholdAnalysis();
  if (ok && data.available) {
    Charts.buildThresholdChart("threshold-chart", data.thresholds);
  }
}

// ============================================================ EXPLAINABILITY ==
async function loadExplainability() {
  setLoading("shap-global-content", "Computing global SHAP values… (this may take a moment)");

  const { ok, data } = await API.explainGlobal();

  if (!ok || !data.available) {
    setEmpty("shap-global-content", "🔍",
      data.error?.includes("trained") ? "Models Not Trained" : "SHAP Unavailable",
      data.error || "Run the training pipeline first.");
    return;
  }

  const container = document.getElementById("shap-global-content");
  container.innerHTML = `
    <p style="font-size:0.84rem;color:var(--text-secondary);margin-bottom:16px">
      Global feature importance computed using mean absolute SHAP values
      over <strong>${data.n_samples_used}</strong> test samples.
      Larger values indicate greater overall influence on model predictions.
    </p>
    <div class="chart-container" style="height:420px">
      <canvas id="shap-bar-chart"></canvas>
    </div>
    <div style="margin-top:20px">
      <div class="card-title">Full Feature Ranking</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th><th>Relative Importance</th></tr></thead>
          <tbody>
            ${data.feature_importance.map((d, i) => {
              const max = data.feature_importance[0].importance;
              const pct = max > 0 ? (d.importance / max * 100).toFixed(1) : 0;
              return `<tr>
                <td>${i + 1}</td>
                <td><strong>${d.feature}</strong></td>
                <td>${d.importance.toFixed(6)}</td>
                <td>
                  <div class="progress-wrap" style="width:120px">
                    <div class="progress-fill" style="width:${pct}%"></div>
                  </div>
                </td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>
    </div>`;

  Charts.buildShapBarChart("shap-bar-chart", data.feature_importance);
}

// ============================================================ ANOMALY =========
async function loadAnomaly() {
  setLoading("anomaly-content", "Loading anomaly detection results…");

  const { ok, data } = await API.anomaly();

  if (!ok || !data.available) {
    setEmpty("anomaly-content", "🌲", "Isolation Forest Not Available",
      data.message || "Run the training pipeline first.");
    return;
  }

  const s = data.score_stats;
  document.getElementById("anomaly-content").innerHTML = `
    <div class="alert alert-warning" style="margin-bottom:20px">
      <span>⚠️</span>
      <span>${data.methodology}</span>
    </div>

    <div class="kpi-grid" style="margin-bottom:24px">
      <div class="kpi-card">
        <div class="kpi-label">Scored Transactions</div>
        <div class="kpi-value">${fmtInt(data.total_scored)}</div>
      </div>
      <div class="kpi-card danger">
        <div class="kpi-label">Anomalies Detected</div>
        <div class="kpi-value" style="color:var(--danger)">${fmtInt(data.total_anomalies)}</div>
        <div class="kpi-sub">${fmtPct(data.anomaly_rate_pct)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Anomaly Score Range</div>
        <div class="kpi-value" style="font-size:1.1rem">${fmt(s.min, 3)} — ${fmt(s.max, 3)}</div>
        <div class="kpi-sub">Mean: ${fmt(s.mean, 3)}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">🎯</span>Top ${data.top_anomalies.length} Most Anomalous Transactions</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Anomaly Score</th>
              <th>Time (s)</th>
              <th>Amount</th>
              <th>Anomaly Flag</th>
              <th>Ground Truth</th>
            </tr>
          </thead>
          <tbody>
            ${data.top_anomalies.map(tx => `
              <tr>
                <td><strong>${fmt(tx.anomaly_score, 4)}</strong></td>
                <td>${fmt(tx.Time ?? tx.time ?? "—", 0)}</td>
                <td>$${fmt(tx.Amount ?? tx.amount ?? 0, 2)}</td>
                <td>
                  <span class="badge ${tx.is_anomaly ? 'badge-fraud' : 'badge-legit'}">
                    ${tx.is_anomaly ? "Anomaly" : "Normal"}
                  </span>
                </td>
                <td>
                  ${tx.Class !== undefined
                    ? `<span class="badge ${tx.Class === 1 ? 'badge-fraud' : 'badge-legit'}">${tx.Class === 1 ? "Fraud" : "Legit"}</span>`
                    : "—"}
                </td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>`;
}

// ============================================================ BUSINESS ========
async function loadBusiness() {
  setLoading("business-content", "Generating business insights…");

  const { ok, data } = await API.businessInsights();

  if (!ok || !data.available) {
    setEmpty("business-content", "💼", "No Insights Available",
      data.message || "Run the training pipeline to generate insights.");
    return;
  }

  const rankedHTML = data.model_ranking.map(r => `
    <tr>
      <td>${r.rank}</td>
      <td><strong>${r.model}</strong></td>
      <td>${fmt(r.f1)}</td>
      <td>${fmt(r.precision)}</td>
      <td>${fmt(r.recall)}</td>
      <td>${r.roc_auc ? fmt(r.roc_auc) : "—"}</td>
    </tr>`).join("");

  const insightsHTML = data.insights.map(ins => `
    <div class="insight-card">
      <div class="insight-icon">${ins.icon}</div>
      <div class="insight-body">
        <div class="insight-cat">${ins.category}</div>
        <h4>${ins.title}</h4>
        <p>${ins.body}</p>
      </div>
    </div>`).join("");

  document.getElementById("business-content").innerHTML = `
    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">🏆</span>Model Performance Ranking</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th></tr></thead>
          <tbody>${rankedHTML}</tbody>
        </table>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:14px">${insightsHTML}</div>`;
}

// ============================================================ HEALTH ==========
async function loadHealth() {
  setLoading("health-content", "Checking system health…");

  const [healthRes, statusRes] = await Promise.all([
    API.health(),
    API.modelsStatus(),
  ]);

  const h = healthRes.data;
  const s = statusRes.data;
  const ds = h.dataset || {};

  const modelRows = s.models
    ? Object.entries(s.models).map(([name, info]) => `
        <tr>
          <td>${info.display_name}</td>
          <td>
            <span class="badge ${info.available ? 'badge-legit' : 'badge-warning'}">
              ${info.available ? "✅ Available" : "⏳ Not trained"}
            </span>
          </td>
        </tr>`).join("")
    : "<tr><td colspan='2'>Status unavailable</td></tr>";

  document.getElementById("health-content").innerHTML = `
    <div class="charts-grid">
      <div class="card">
        <div class="card-title"><span class="icon">🌐</span>Backend Status</div>
        <div style="display:flex;flex-direction:column;gap:10px;font-size:0.875rem">
          <div style="display:flex;justify-content:space-between">
            <span>Service</span>
            <span>${h.service || "FraudShield AI"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Status</span>
            <span class="badge ${healthRes.ok ? 'badge-legit' : 'badge-fraud'}">${healthRes.ok ? "Online" : "Offline"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Version</span>
            <span>${h.version || "1.0.0"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Timestamp</span>
            <span style="font-size:0.8rem;color:var(--text-muted)">${h.timestamp ? new Date(h.timestamp).toLocaleString() : "—"}</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title"><span class="icon">📂</span>Dataset Status</div>
        <div style="display:flex;flex-direction:column;gap:10px;font-size:0.875rem">
          <div style="display:flex;justify-content:space-between">
            <span>Available</span>
            <span class="badge ${ds.dataset_available ? 'badge-legit' : 'badge-fraud'}">${ds.dataset_available ? "Yes" : "Not found"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Models Trained</span>
            <span class="badge ${ds.models_trained ? 'badge-legit' : 'badge-warning'}">${ds.models_trained ? "Yes" : "No"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Best Model</span>
            <span>${ds.best_model || "N/A"}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span>Last Training</span>
            <span style="font-size:0.8rem;color:var(--text-muted)">${ds.training_timestamp ? new Date(ds.training_timestamp).toLocaleString() : "—"}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <div class="card-title"><span class="icon">🤖</span>Model Artifact Status</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Model</th><th>Status</th></tr></thead>
          <tbody>${modelRows}</tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <div class="card-title"><span class="icon">📖</span>Quick Setup Guide</div>
      <div style="font-size:0.875rem;color:var(--text-secondary);line-height:2">
        <p>1. Install dependencies: <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:monospace">pip install -r requirements.txt</code></p>
        <p>2. Download dataset from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">Kaggle</a> → place <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:monospace">creditcard.csv</code> in <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:monospace">data/</code></p>
        <p>3. Train all models: <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:monospace">python train_pipeline.py</code></p>
        <p>4. Start Flask backend: <code style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:monospace">python backend/app.py</code></p>
        <p>5. Open browser at: <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>
      </div>
    </div>`;
}

// ============================================================ CASES ==========
let _caseSearchTimer = null;
function debounceCaseSearch() {
  clearTimeout(_caseSearchTimer);
  _caseSearchTimer = setTimeout(() => loadCases(), 300);
}

async function loadCases() {
  const status = document.getElementById("case-status-filter")?.value;
  const risk = document.getElementById("case-risk-filter")?.value;
  const search = document.getElementById("case-search-input")?.value?.trim();

  const tbody = document.getElementById("cases-table-body");
  if (!tbody) return;

  const { ok, data } = await API.cases(status, risk, search);
  if (!ok || !data) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:24px;color:var(--danger)">Failed to load investigation cases.</td></tr>`;
    return;
  }

  const cases = data.cases || [];
  
  // Update KPIs
  const total = cases.length;
  const openCount = cases.filter(c => c.status === "Open").length;
  const reviewCount = cases.filter(c => c.status === "Under Review" || c.status === "Escalated").length;
  const closedCount = cases.filter(c => c.status && c.status.startsWith("Closed")).length;

  const elTot = document.getElementById("case-kpi-total");
  const elOp = document.getElementById("case-kpi-open");
  const elRev = document.getElementById("case-kpi-review");
  const elCl = document.getElementById("case-kpi-closed");

  if (elTot) elTot.textContent = total;
  if (elOp) elOp.textContent = openCount;
  if (elRev) elRev.textContent = reviewCount;
  if (elCl) elCl.textContent = closedCount;

  if (cases.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:36px;color:var(--text-muted)">
      <div style="font-size:1.8rem;margin-bottom:8px">📂</div>
      No investigation cases found matching the criteria. Click "➕ New Case" or flag a transaction to create one.
    </td></tr>`;
    return;
  }

  const riskBadgeClass = {
    Critical: "badge-fraud",
    High: "badge-fraud",
    Medium: "badge-warning",
    Low: "badge-legit",
  };

  const statusBadgeClass = {
    "Open": "badge-danger",
    "Under Review": "badge-warning",
    "Escalated": "badge-danger",
    "Closed - False Positive": "badge-legit",
    "Closed - Confirmed Fraud": "badge-info",
  };

  tbody.innerHTML = cases.map(c => `
    <tr>
      <td><strong style="color:var(--accent)">${c.case_id}</strong></td>
      <td>
        <div style="font-weight:600;color:var(--text)">${c.title}</div>
        ${c.notes ? `<div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${c.notes}</div>` : ""}
      </td>
      <td>$${Number(c.amount || 0).toFixed(2)}</td>
      <td><span class="badge ${riskBadgeClass[c.risk_level] || 'badge-warning'}">${c.risk_level}</span></td>
      <td><span class="badge ${statusBadgeClass[c.status] || 'badge-info'}">${c.status}</span></td>
      <td style="font-size:0.8rem;color:var(--text-muted)">${c.flagged_by}</td>
      <td style="font-size:0.75rem;color:var(--text-muted)">${new Date(c.updated_at || c.created_at).toLocaleDateString()}</td>
      <td>
        <div style="display:flex;gap:6px">
          <button class="btn btn-secondary btn-sm" style="padding:2px 8px;font-size:0.75rem" onclick='openEditCaseModal(${JSON.stringify(c)})'>✏️ Edit</button>
          <button class="btn btn-danger btn-sm" style="padding:2px 8px;font-size:0.75rem" onclick="deleteCaseAction(${c.id})">🗑️</button>
        </div>
      </td>
    </tr>
  `).join("");
}

function openCreateCaseModal(prefill = {}) {
  document.getElementById("case-modal-title").textContent = "New Investigation Case";
  document.getElementById("case-modal-id").value = "";
  document.getElementById("case-modal-title-input").value = prefill.title || "Suspicious Activity Investigation";
  document.getElementById("case-modal-amount-input").value = prefill.amount !== undefined ? prefill.amount : "";
  document.getElementById("case-modal-risk-input").value = prefill.risk || "High";
  document.getElementById("case-modal-notes-input").value = prefill.notes || "";
  document.getElementById("case-modal-status-group").style.display = "none";
  document.getElementById("case-modal").classList.add("active");
}

function openEditCaseModal(c) {
  document.getElementById("case-modal-title").textContent = `Edit Case: ${c.case_id}`;
  document.getElementById("case-modal-id").value = c.id;
  document.getElementById("case-modal-title-input").value = c.title || "";
  document.getElementById("case-modal-amount-input").value = c.amount || 0;
  document.getElementById("case-modal-risk-input").value = c.risk_level || "High";
  document.getElementById("case-modal-status-input").value = c.status || "Open";
  document.getElementById("case-modal-notes-input").value = c.notes || "";
  document.getElementById("case-modal-status-group").style.display = "block";
  document.getElementById("case-modal").classList.add("active");
}

function closeCaseModal() {
  document.getElementById("case-modal").classList.remove("active");
}

async function saveCaseModal() {
  const id = document.getElementById("case-modal-id").value;
  const title = document.getElementById("case-modal-title-input").value.trim() || "Investigation Case";
  const amount = parseFloat(document.getElementById("case-modal-amount-input").value) || 0.0;
  const risk_level = document.getElementById("case-modal-risk-input").value;
  const status = document.getElementById("case-modal-status-input")?.value || "Open";
  const notes = document.getElementById("case-modal-notes-input").value.trim();

  if (id) {
    const { ok } = await API.updateCase(id, { title, risk_level, status, notes });
    if (ok) {
      toast("Case updated successfully", "success");
      closeCaseModal();
      loadCases();
    } else {
      toast("Failed to update case", "error");
    }
  } else {
    const { ok } = await API.createCase({ title, amount, risk_level, status: "Open", notes });
    if (ok) {
      toast("Case created successfully", "success");
      closeCaseModal();
      loadCases();
    } else {
      toast("Failed to create case", "error");
    }
  }
}

async function deleteCaseAction(id) {
  if (!confirm("Are you sure you want to delete this investigation case?")) return;
  const { ok } = await API.deleteCase(id);
  if (ok) {
    toast("Case deleted", "info");
    loadCases();
  } else {
    toast("Failed to delete case", "error");
  }
}

function escalateFromPrediction(risk, label) {
  const amt = parseFloat(document.getElementById("field-Amount")?.value || 0);
  navigateTo("cases");
  setTimeout(() => {
    openCreateCaseModal({
      title: `Escalated Flagged Transaction: ${label} ($${amt.toFixed(2)})`,
      amount: amt,
      risk: risk || "High",
      notes: `Flagged during real-time model inference as ${label} with ${risk} risk profile.`
    });
  }, 120);
}

// ── Start ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);

