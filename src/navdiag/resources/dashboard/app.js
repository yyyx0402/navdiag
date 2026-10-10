let report, trips;
const el = (id) => document.getElementById(id);
const names = {
  fixed_rules: "固定规则",
  logistic_regression: "逻辑回归",
  random_forest: "随机森林",
  gru: "GRU 时序模型",
  wls: "加权最小二乘",
  equal: "等权最小二乘",
  robust: "稳健最小二乘",
  fixed_filter: "固定方差滤波",
  adaptive_filter: "自适应方差滤波",
};
const number = (value, digits = 3) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(digits);
async function get(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "请求失败");
  return body;
}
function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}
function table(container, headers, rows) {
  container.replaceChildren();
  const t = document.createElement("table"),
    head = document.createElement("thead"),
    hr = document.createElement("tr");
  for (const header of headers) {
    const th = document.createElement("th");
    th.textContent = header;
    hr.append(th);
  }
  head.append(hr);
  t.append(head);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value) =>
      tr.append(value instanceof Node ? value : cell(value)),
    );
    body.append(tr);
  });
  t.append(body);
  container.append(t);
}
function modelTable() {
  const rows = Object.entries(report.models).map(([name, value]) => {
    const nameCell = cell(names[name] || name);
    if (name === report.selected_model) {
      const tag = document.createElement("span");
      tag.className = "selected-tag";
      tag.textContent = "验证集选择";
      nameCell.append(tag);
    }
    const scoreCell = cell(number(value.test.f1));
    const bar = document.createElement("span");
    bar.className = "bar-track";
    const inner = document.createElement("i");
    inner.style.width = `${value.test.f1 * 100}%`;
    bar.append(inner);
    scoreCell.append(bar);
    return [
      nameCell,
      number(value.test.precision),
      number(value.test.recall),
      scoreCell,
      number(value.test.average_precision),
      number(value.test.threshold, 2),
    ];
  });
  table(
    el("model-table"),
    ["方法", "Precision", "Recall", "F1", "Average Precision", "分类阈值"],
    rows,
  );
}
const NS = "http://www.w3.org/2000/svg";
function svgNode(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
  if (text !== undefined) n.textContent = text;
  return n;
}
function plot(
  container,
  series,
  xLabel,
  yLabel,
  equalAspect = false,
  riskAxis = false,
) {
  container.replaceChildren();
  const valid = series.filter((s) => s.points.length > 1);
  if (!valid.length) {
    container.textContent = "当前实验没有可用轨迹。";
    return;
  }
  const all = valid.flatMap((s) => s.points);
  const leftPoints = valid.filter((s) => !s.rightAxis).flatMap((s) => s.points);
  const width = 520,
    height = 265,
    pad = { left: 55, right: riskAxis ? 55 : 22, top: 20, bottom: 38 };
  let xmin = Math.min(...all.map((p) => p[0])),
    xmax = Math.max(...all.map((p) => p[0]));
  let ymin = Math.min(...leftPoints.map((p) => p[1])),
    ymax = Math.max(...leftPoints.map((p) => p[1]));
  if (riskAxis) {
    ymin = 0;
    ymax = Math.max(1, ymax * 1.05);
  }
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) ymax = ymin + 1;
  if (equalAspect) {
    const ratio =
      (width - pad.left - pad.right) / (height - pad.top - pad.bottom);
    const yr = Math.max(ymax - ymin, (xmax - xmin) / ratio),
      xr = yr * ratio;
    const xc = (xmin + xmax) / 2,
      yc = (ymin + ymax) / 2;
    xmin = xc - xr / 2;
    xmax = xc + xr / 2;
    ymin = yc - yr / 2;
    ymax = yc + yr / 2;
  }
  const x = (v) =>
    pad.left + ((v - xmin) / (xmax - xmin)) * (width - pad.left - pad.right);
  const y = (v) =>
    height -
    pad.bottom -
    ((v - ymin) / (ymax - ymin)) * (height - pad.top - pad.bottom);
  const riskY = (v) =>
    height - pad.bottom - v * (height - pad.top - pad.bottom);
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${xLabel} 对 ${yLabel}${riskAxis ? "，右轴风险评分 0 至 1" : ""}`,
  });
  for (let i = 0; i <= 4; i++) {
    const xv = xmin + ((xmax - xmin) * i) / 4,
      yv = ymin + ((ymax - ymin) * i) / 4;
    svg.append(
      svgNode("line", {
        x1: x(xv),
        x2: x(xv),
        y1: pad.top,
        y2: height - pad.bottom,
        class: "svg-grid",
      }),
      svgNode("line", {
        x1: pad.left,
        x2: width - pad.right,
        y1: y(yv),
        y2: y(yv),
        class: "svg-grid",
      }),
      svgNode(
        "text",
        {
          x: x(xv),
          y: height - 19,
          "text-anchor": "middle",
          class: "svg-axis",
        },
        number(xv, 0),
      ),
      svgNode(
        "text",
        {
          x: pad.left - 8,
          y: y(yv) + 3,
          "text-anchor": "end",
          class: "svg-axis",
        },
        number(yv, 0),
      ),
    );
    if (riskAxis)
      svg.append(
        svgNode(
          "text",
          { x: width - pad.right + 8, y: riskY(i / 4) + 3, class: "svg-axis" },
          number(i / 4, 2),
        ),
      );
  }
  valid.forEach((s) =>
    svg.append(
      svgNode("polyline", {
        points: s.points
          .map((p) => `${x(p[0])},${s.rightAxis ? riskY(p[1]) : y(p[1])}`)
          .join(" "),
        fill: "none",
        stroke: s.color,
        "stroke-width": s.width || 1.7,
        opacity: s.opacity || 1,
      }),
    ),
  );
  svg.append(
    svgNode(
      "text",
      {
        x: width / 2,
        y: height - 2,
        "text-anchor": "middle",
        class: "svg-axis",
      },
      xLabel,
    ),
    svgNode("text", { x: 8, y: 10, class: "svg-axis" }, yLabel),
  );
  if (riskAxis)
    svg.append(
      svgNode(
        "text",
        { x: width - 8, y: 10, "text-anchor": "end", class: "svg-axis" },
        "Risk (0–1)",
      ),
    );
  container.append(svg);
}
function points(rows, a, b) {
  return rows
    .filter(
      (r) =>
        typeof r[a] === "number" &&
        typeof r[b] === "number" &&
        Number.isFinite(r[a]) &&
        Number.isFinite(r[b]),
    )
    .map((r) => [r[a], r[b]]);
}
async function loadTrajectory() {
  try {
    const data = await get(
      `/api/trajectory?trip=${encodeURIComponent(el("trip-select").value)}&device=${encodeURIComponent(el("device-select").value)}`,
    );
    plot(
      el("trajectory-chart"),
      [
        {
          points: points(data.points, "gt_e_m", "gt_n_m"),
          color: "#235b49",
          width: 2.2,
        },
        {
          points: points(data.points, "wls_e_m", "wls_n_m"),
          color: "#6391b2",
          opacity: 0.55,
        },
        {
          points: points(
            data.points,
            "adaptive_filter_e_m",
            "adaptive_filter_n_m",
          ),
          color: "#dc9050",
          width: 2,
        },
      ],
      "East (m)",
      "North (m)",
      true,
    );
    plot(
      el("error-chart"),
      [
        {
          points: points(data.points, "time_s", "horizontal_error_m"),
          color: "#6391b2",
        },
        {
          points: points(data.points, "time_s", "predicted_risk"),
          color: "#dc9050",
          width: 1.6,
          rightAxis: true,
        },
      ],
      "Time (s)",
      "Error (m)",
      false,
      true,
    );
    el("sample-info").textContent =
      `${data.total_epochs} 历元 / 显示 ${data.sampled_epochs}`;
  } catch (error) {
    el("trajectory-chart").textContent = error.message;
  }
}
function chooseTrip() {
  const current = trips.find((t) => t.trip_id === el("trip-select").value);
  el("device-select").replaceChildren();
  current.devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = device;
    option.textContent = device;
    el("device-select").append(option);
  });
  loadTrajectory();
}
async function initialize() {
  [report, trips] = await Promise.all([get("/api/report"), get("/api/trips")]);
  const selected = report.models[report.selected_model].test;
  el("data-badge").textContent =
    report.data_kind === "synthetic"
      ? "SYNTHETIC · 模拟实验"
      : "REAL DATA · 公开数据";
  if (report.data_kind === "real") el("data-badge").classList.add("real");
  el("provenance-warning").textContent =
    report.data_kind === "synthetic"
      ? "当前使用模拟数据验证流程。所有指标只描述这个模拟实验，不代表真实城市定位性能。"
      : "当前使用公开真实观测。请结合独立行程数量、设备差异和样本长度解读指标；小样本只用于流程验证。";
  if (selected.recall === null) {
    el("provenance-warning").textContent +=
      " 当前测试集没有失准正样本，Recall 和 F1 无法评估。";
  }
  el("selected-model").textContent =
    names[report.selected_model] || report.selected_model;
  el("epoch-count").textContent = report.splits.test.rows.toLocaleString();
  el("trip-count").textContent =
    `${report.splits.test.trips.length} 个测试采集组`;
  el("recall").textContent = number(selected.recall);
  el("task-label").textContent = `水平误差 > ${report.error_limit_m} m`;
  el("position-p95").textContent = report.positioning
    ? `${number(report.positioning.methods.wls.p95_m, 2)} m`
    : "未计算";
  el("seed").textContent = report.seed;
  el("dataset-hash").textContent = report.dataset_sha256;
  el("missing-features").textContent =
    (report.dropped_features || []).join(", ") || "无";
  modelTable();
  if (report.positioning) {
    table(
      el("position-table"),
      ["方法", "历元", "Median (m)", "P95 (m)", "RMSE (m)"],
      Object.entries(report.positioning.methods).map(([name, v]) => [
        names[name] || name,
        v.epochs,
        number(v.median_m),
        number(v.p95_m),
        number(v.rmse_m),
      ]),
    );
  } else
    el("position-table").textContent = "当前实验仅包含分类特征，没有位置轨迹。";
  trips.forEach((t) => {
    const option = document.createElement("option");
    option.value = t.trip_id;
    option.textContent = t.trip_id;
    el("trip-select").append(option);
  });
  chooseTrip();
}
el("trip-select").addEventListener("change", chooseTrip);
el("device-select").addEventListener("change", loadTrajectory);
document
  .querySelectorAll(".example-questions button")
  .forEach((button) =>
    button.addEventListener(
      "click",
      () => (el("question").value = button.textContent),
    ),
  );
el("ask-button").addEventListener("click", async () => {
  const question = el("question").value.trim();
  el("agent-status").classList.remove("error");
  if (!question) {
    el("agent-status").textContent = "请先输入问题。";
    return;
  }
  el("ask-button").disabled = true;
  el("agent-status").textContent = "正在调用分析工具…";
  el("evidence-list").replaceChildren();
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        provider: el("provider").value,
        trip_id: el("trip-select").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "分析失败");
    el("answer").textContent = result.answer;
    el("answer-backend").textContent =
      result.backend === "offline" ? "离线工具基线" : result.model;
    el("agent-status").textContent =
      `${result.evidence.length} 条证据 · ${number(result.elapsed_s, 2)} 秒 · ${result.notice}`;
    result.evidence.forEach((item) => {
      const detail = document.createElement("details");
      detail.className = "evidence";
      const summary = document.createElement("summary");
      summary.textContent = `[${item.evidence_id}] ${item.tool} · ${item.data_kind}`;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(item.payload, null, 2);
      detail.append(summary, pre);
      el("evidence-list").append(detail);
    });
  } catch (error) {
    el("agent-status").textContent = error.message;
    el("agent-status").classList.add("error");
  } finally {
    el("ask-button").disabled = false;
  }
});
document.querySelectorAll(".nav-item").forEach((link) =>
  link.addEventListener("click", () => {
    document
      .querySelectorAll(".nav-item")
      .forEach((x) => x.classList.remove("active"));
    link.classList.add("active");
  }),
);
initialize().catch((error) => {
  el("provenance-warning").textContent = `实验载入失败：${error.message}`;
});
