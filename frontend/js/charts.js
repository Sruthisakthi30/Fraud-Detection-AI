/**
 * FraudShield AI - Chart Builders
 * All Chart.js chart construction helpers.
 */

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 600, easing: "easeOutQuart" },
  plugins: {
    legend: {
      labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } }
    },
    tooltip: {
      backgroundColor: "#1e293b",
      borderColor: "rgba(59,130,246,0.3)",
      borderWidth: 1,
      titleColor: "#e2e8f0",
      bodyColor: "#94a3b8",
      padding: 10,
    }
  },
  scales: {
    x: {
      grid: { color: "rgba(255,255,255,0.05)" },
      ticks: { color: "#64748b", font: { size: 10 } }
    },
    y: {
      grid: { color: "rgba(255,255,255,0.05)" },
      ticks: { color: "#64748b", font: { size: 10 } }
    }
  }
};

const _charts = {};

function _destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

function _ctx(id) {
  return document.getElementById(id)?.getContext("2d");
}

// ── Doughnut: Fraud vs Legit ──────────────────────────────────────────
function buildClassDistChart(id, fraud, legit) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  _charts[id] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fraudulent", "Legitimate"],
      datasets: [{
        data: [fraud, legit],
        backgroundColor: ["rgba(239,68,68,0.8)", "rgba(16,185,129,0.7)"],
        borderColor: ["#ef4444", "#10b981"],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      cutout: "70%",
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { position: "bottom", labels: { color: "#94a3b8", padding: 16, font: { size: 12 } } }
      },
      scales: {}
    }
  });
}

// ── Bar: Amount distribution ──────────────────────────────────────────
function buildAmountChart(id, amountData) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  const labels = amountData.bin_edges.slice(0, -1).map((v, i) =>
    `${v.toFixed(0)}-${amountData.bin_edges[i+1].toFixed(0)}`);
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Legitimate",
          data: amountData.legitimate,
          backgroundColor: "rgba(16,185,129,0.5)",
          borderColor: "#10b981", borderWidth: 1,
        },
        {
          label: "Fraud",
          data: amountData.fraud,
          backgroundColor: "rgba(239,68,68,0.6)",
          borderColor: "#ef4444", borderWidth: 1,
        }
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, stacked: false },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true }
      }
    }
  });
}

// ── Line: Time distribution ───────────────────────────────────────────
function buildTimeChart(id, timeData) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  const maxH = timeData.hours.length;
  const labels = timeData.hours.map(h => `${h}h`);
  _charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Legitimate",
          data: timeData.legitimate,
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.08)",
          fill: true, tension: 0.3, pointRadius: 0,
        },
        {
          label: "Fraud",
          data: timeData.fraud,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239,68,68,0.1)",
          fill: true, tension: 0.3, pointRadius: 1, borderWidth: 2,
        }
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true }
      }
    }
  });
}

// ── Bar: Model comparison ─────────────────────────────────────────────
function buildModelCompareChart(id, results) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  const labels = results.map(r => r.display_name);
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "F1", data: results.map(r => r.f1 ?? null), backgroundColor: "rgba(59,130,246,0.7)", borderColor: "#3b82f6", borderWidth: 1 },
        { label: "Precision", data: results.map(r => r.precision ?? null), backgroundColor: "rgba(139,92,246,0.7)", borderColor: "#8b5cf6", borderWidth: 1 },
        { label: "Recall", data: results.map(r => r.recall ?? null), backgroundColor: "rgba(16,185,129,0.7)", borderColor: "#10b981", borderWidth: 1 },
        { label: "ROC-AUC", data: results.map(r => r.roc_auc ?? null), backgroundColor: "rgba(245,158,11,0.7)", borderColor: "#f59e0b", borderWidth: 1 },
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true, max: 1 }
      }
    }
  });
}

// ── Line: ROC curve ───────────────────────────────────────────────────
function buildRocChart(id, rocData, modelName) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  _charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels: rocData.fpr,
      datasets: [
        {
          label: modelName,
          data: rocData.tpr,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.1)",
          fill: true, tension: 0, pointRadius: 0, borderWidth: 2,
        },
        {
          label: "Random Classifier",
          data: rocData.fpr.map(v => v),
          borderColor: "rgba(255,255,255,0.2)",
          borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5,
        }
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "bottom", labels: { color: "#94a3b8", font: { size: 11 } } } },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: "False Positive Rate", color: "#64748b" } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "True Positive Rate", color: "#64748b" }, min: 0, max: 1 }
      }
    }
  });
}

// ── Bar: SHAP feature importance ──────────────────────────────────────
function buildShapBarChart(id, importanceList, top = 15) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  const top_n = importanceList.slice(0, top);
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: top_n.map(d => d.feature),
      datasets: [{
        label: "Mean |SHAP|",
        data: top_n.map(d => d.importance),
        backgroundColor: top_n.map((_, i) => {
          const alpha = 0.9 - i * 0.04;
          return `rgba(59,130,246,${alpha})`;
        }),
        borderColor: "#3b82f6", borderWidth: 1,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: "y",
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, beginAtZero: true },
        y: { ...CHART_DEFAULTS.scales.y, ticks: { color: "#94a3b8", font: { size: 10 } } }
      }
    }
  });
}

// ── Line: Threshold analysis ──────────────────────────────────────────
function buildThresholdChart(id, thresholds) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  _charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels: thresholds.map(r => r.threshold),
      datasets: [
        { label: "Precision", data: thresholds.map(r => r.precision), borderColor: "#8b5cf6", pointRadius: 2, borderWidth: 2, tension: 0.3 },
        { label: "Recall",    data: thresholds.map(r => r.recall),    borderColor: "#10b981", pointRadius: 2, borderWidth: 2, tension: 0.3 },
        { label: "F1",        data: thresholds.map(r => r.f1),        borderColor: "#3b82f6", pointRadius: 2, borderWidth: 2, tension: 0.3 },
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: "Threshold", color: "#64748b" } },
        y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1 }
      }
    }
  });
}

// ── Bar: Feature distribution by class ───────────────────────────────
function buildFeatureDistChart(id, distData) {
  _destroyChart(id);
  const ctx = _ctx(id); if (!ctx) return;
  const labels = distData.bin_edges.slice(0, -1).map((v, i) =>
    ((v + distData.bin_edges[i+1]) / 2).toFixed(2));
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Legitimate", data: distData.legitimate, backgroundColor: "rgba(16,185,129,0.5)", borderColor: "#10b981", borderWidth: 1 },
        { label: "Fraud",      data: distData.fraud,      backgroundColor: "rgba(239,68,68,0.6)",  borderColor: "#ef4444", borderWidth: 1 },
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true }
      }
    }
  });
}

// ── Correlation heatmap (canvas drawn manually) ───────────────────────
function buildCorrelationHeatmap(id, corrData) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { features, matrix } = corrData;
  const n = features.length;
  const cellSize = Math.min(Math.floor(canvas.width / n), 32);
  const pad = 60;

  canvas.width  = cellSize * n + pad * 2;
  canvas.height = cellSize * n + pad * 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#0f1629";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = matrix[r][c];
      const norm = (v + 1) / 2;  // map [-1,1] → [0,1]
      const red  = Math.round(239 * norm);
      const blue = Math.round(59  * (1 - norm));
      ctx.fillStyle = `rgba(${red}, 80, ${blue + 130}, 0.85)`;
      ctx.fillRect(pad + c * cellSize, pad + r * cellSize, cellSize - 1, cellSize - 1);

      if (cellSize >= 20) {
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.font = `${Math.max(8, cellSize * 0.35)}px Inter`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(v.toFixed(1), pad + c * cellSize + cellSize / 2, pad + r * cellSize + cellSize / 2);
      }
    }
  }

  // Feature labels (x-axis top, y-axis left)
  ctx.fillStyle = "#94a3b8";
  ctx.font = `${Math.min(10, cellSize * 0.38)}px Inter`;
  ctx.textAlign = "center";
  features.forEach((f, i) => {
    ctx.save();
    ctx.translate(pad + i * cellSize + cellSize / 2, pad - 6);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = "left";
    ctx.fillText(f, 0, 0);
    ctx.restore();
  });
  ctx.textAlign = "right";
  features.forEach((f, i) => {
    ctx.fillText(f, pad - 6, pad + i * cellSize + cellSize / 2 + 4);
  });
}

window.Charts = {
  buildClassDistChart,
  buildAmountChart,
  buildTimeChart,
  buildModelCompareChart,
  buildRocChart,
  buildShapBarChart,
  buildThresholdChart,
  buildFeatureDistChart,
  buildCorrelationHeatmap,
  destroyAll: () => Object.keys(_charts).forEach(id => { _charts[id].destroy(); delete _charts[id]; }),
};
