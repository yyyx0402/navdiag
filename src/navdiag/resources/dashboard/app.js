let report, trips;
const el = (id) => document.getElementById(id);
const names = {
  fixed_rules: "固定规则",
  logistic_regression: "逻辑回归",
  random_forest: "随机森林",
  mlp: "MLP",
  cnn: "1D CNN",
  lstm: "LSTM",
  gru: "GRU",
  wls: "加权最小二乘",
  equal: "等权最小二乘",
  robust: "稳健最小二乘",
  fixed_filter: "固定方差滤波",
  adaptive_filter: "自适应方差滤波",
  cn0_rule: "C/N0 固定规则",
  transformer: "Transformer",
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
  const selected = report.models[report.selected_model].test;
  const [[tn, fp], [fn, tp]] = selected.confusion_matrix;
  el("risk-errors").textContent =
    `选用模型：正确报警 ${tp}，误报 ${fp}，漏报 ${fn}，正确排除 ${tn}。这里评估水平误差超限，不能当作遮挡识别成绩。`;
}
let activeNlosExperiment = "";
let nlosRequest = 0;
let generalizationReport, currentNlosReport;
let epochRequest = 0;
const areaNames = { TST: "尖沙咀", WP: "黄埔", MK: "旺角" };
const metricLabels = {
  f1: "F1",
  precision: "Precision",
  recall: "NLOS Recall",
};
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function comparisonBars(container, rows, labels, selected, onSelect) {
  container.replaceChildren();
  const axis = node("div", "comparison-axis");
  for (const tick of ["0", "0.5", "1"]) axis.append(node("span", "", tick));
  container.append(axis);
  rows.forEach((row) => {
    const group = node(
      "div",
      `comparison-row${row.key === selected ? " is-focused" : ""}`,
    );
    const label = node(
      onSelect ? "button" : "span",
      "comparison-label",
      row.label,
    );
    if (onSelect) {
      label.type = "button";
      label.setAttribute("aria-pressed", String(row.key === selected));
      label.setAttribute("aria-label", `查看 ${row.label} 的区域表现`);
      label.onclick = () => onSelect(row.key);
    }
    const stack = node("div", "comparison-stack");
    row.values.forEach((value, index) => {
      const line = node("div", "comparison-line");
      const track = node("div", "score-track");
      const bar = node("i", `score-fill series-${index}`);
      const valid = typeof value === "number" && Number.isFinite(value);
      bar.style.width = `${valid ? Math.max(0, Math.min(1, value)) * 100 : 0}%`;
      const description = `${row.label} · ${labels[index]} ${number(value, 4)}`;
      line.title = description;
      line.setAttribute("aria-label", description);
      track.append(bar);
      line.append(track, node("span", "score-value", number(value, 3)));
      stack.append(line);
    });
    group.append(label, stack);
    container.append(group);
  });
}
function bindMetricSwitch(id, render) {
  el(id)
    .querySelectorAll("button")
    .forEach((button) => {
      button.onclick = () => {
        el(id)
          .querySelectorAll("button")
          .forEach((item) =>
            item.setAttribute("aria-pressed", String(item === button)),
          );
        render(button.dataset.metric);
      };
    });
}
function selectedMetric(id) {
  return el(id).querySelector('[aria-pressed="true"]').dataset.metric;
}
function updateProjectOverview(dataset, scored, source) {
  el("project-badge").textContent = source;
  el("project-data-source").textContent = source;
  el("project-epochs").textContent = dataset.epochs.toLocaleString();
  el("project-rows").textContent = dataset.rows.toLocaleString();
  el("project-scored").textContent = scored.toLocaleString();
  el("project-coverage").textContent =
    `占全部观测 ${number((100 * scored) / dataset.rows, 1)}%`;
  el("project-drives").textContent = dataset.areas
    ? "三个城区 · 每区一段完整采集"
    : `${Object.keys(dataset.per_run).length} 段完整采集 · 三个区域`;
  el("project-scope").textContent = dataset.areas
    ? "UrbanNav 新增观测概览；KLT 源数据在下方单独查看。标签方法与设备存在差异，当前只验证香港同城跨城区，跨城市能力尚未验证。"
    : "KLT 相机参考标签；完整采集段隔离。三处相邻区域的分布差异有限，成绩仅代表当前实验。";
}
function renderGeneralizationCharts() {
  if (!generalizationReport) return;
  const r = generalizationReport;
  const metric = selectedMetric("generalization-metric");
  const external = r.external_transfer.models;
  const district = r.district_holdout.models;
  const selected = el("nlos-generalization-model").value;
  const choose = (name) => {
    el("nlos-generalization-model").value = name;
    el("nlos-generalization-model").onchange();
  };
  comparisonBars(
    el("generalization-comparison"),
    Object.keys(external).map((name) => ({
      key: name,
      label: names[name] || name,
      values: [
        external[name].test_macro[metric],
        district[name].test_macro[metric],
      ],
    })),
    ["冻结外部迁移", "城区互斥"],
    selected,
    choose,
  );
  const target =
    el("generalization-heatmap-protocol").value === "external"
      ? external
      : district;
  const t = node("table");
  const header = node("tr");
  for (const title of [
    "模型",
    ...r.plan.plan.target_areas.map((area) => areaNames[area]),
  ]) {
    header.append(node("th", "", title));
  }
  const head = node("thead");
  head.append(header);
  t.append(head);
  const body = node("tbody");
  Object.entries(target).forEach(([name, model]) => {
    const row = node("tr", name === selected ? "is-focused" : "");
    const label = node("th", "heatmap-label", names[name] || name);
    label.scope = "row";
    row.append(label);
    r.plan.plan.target_areas.forEach((area) => {
      const m = model.per_area[area].test_mean || model.per_area[area];
      const value = m[metric];
      const td = node("td");
      const button = node("button", "heatmap-cell", number(value, 3));
      button.style.backgroundColor = `hsl(153 24% ${96 - 40 * value}%)`;
      button.style.color = "#182f2b";
      button.title = `${names[name]} · ${areaNames[area]} · ${metricLabels[metric]} ${number(value, 4)}`;
      button.setAttribute("aria-label", button.title);
      button.onclick = () => choose(name);
      td.append(button);
      row.append(td);
    });
    body.append(row);
  });
  t.append(body);
  el("generalization-heatmap").replaceChildren(t);
  el("heatmap-metric-label").textContent = metricLabels[metric];
  el("generalization-insight").textContent =
    `当前查看 ${names[selected]} 的 ${metricLabels[metric]}。颜色越深，当前指标越高；各模型阈值由各自验证集决定，覆盖率影响成绩的适用范围。`;
  comparisonBars(
    el("generalization-area-chart"),
    r.plan.plan.target_areas.map((area) => ({
      label: areaNames[area],
      values: [
        external[selected].per_area[area][metric],
        district[selected].per_area[area].test_mean[metric],
      ],
    })),
    [`冻结外部 ${metricLabels[metric]}`, `城区互斥 ${metricLabels[metric]}`],
  );
}
function renderCoverage(r) {
  el("generalization-coverage-chart").replaceChildren();
  r.plan.plan.target_areas.forEach((area) => {
    const data = r.dataset.areas[area];
    const counts = data.label_counts;
    const group = node("div", "coverage-row");
    const heading = node("div", "coverage-heading");
    heading.append(
      node("strong", "", areaNames[area]),
      node(
        "span",
        "",
        `${number((100 * (counts.los + counts.nlos)) / data.rows, 1)}% 可评分`,
      ),
    );
    const track = node("div", "coverage-track");
    ["los", "nlos", "unknown"].forEach((label) => {
      const part = node("span", `coverage-part ${label}`);
      part.style.width = `${(100 * counts[label]) / data.rows}%`;
      part.title = `${label.toUpperCase()} ${counts[label].toLocaleString()} 条 · ${number((100 * counts[label]) / data.rows, 1)}%`;
      part.setAttribute("aria-label", part.title);
      track.append(part);
    });
    const foot = node(
      "div",
      "coverage-counts",
      `LOS ${counts.los.toLocaleString()} · NLOS ${counts.nlos.toLocaleString()} · unknown ${counts.unknown.toLocaleString()} / 共 ${data.rows.toLocaleString()} 条`,
    );
    group.append(heading, track, foot);
    el("generalization-coverage-chart").append(group);
  });
}
function renderFoldCharts(r) {
  comparisonBars(
    el("fold-model-chart"),
    Object.entries(r.models).map(([name, value]) => ({
      key: name,
      label: names[name] || name,
      values: [value.test.precision, value.test.recall, value.test.f1],
    })),
    ["Precision", "Recall", "F1"],
    r.selected_model,
  );
  const m = r.models[r.selected_model].test;
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  const grid = node("div", "confusion-grid");
  ["", "预测 LOS", "预测 NLOS"].forEach((text) =>
    grid.append(node("div", "confusion-heading", text)),
  );
  [
    ["真实 LOS", tn, fp],
    ["真实 NLOS", fn, tp],
  ].forEach((row, index) => {
    grid.append(node("div", "confusion-heading", row[0]));
    [row[1], row[2]].forEach((value, col) => {
      const correct = index === col;
      const box = node(
        "div",
        `confusion-cell ${correct ? "correct" : "incorrect"}`,
      );
      const total = row[1] + row[2];
      box.append(
        node("span", "", correct ? "判别正确" : index === 0 ? "误报" : "漏报"),
        node("strong", "", value.toLocaleString()),
        node(
          "small",
          "",
          total ? `占本类 ${number((100 * value) / total, 1)}%` : "本类无样本",
        ),
      );
      grid.append(box);
    });
  });
  el("fold-confusion").replaceChildren(grid);
  const counts = node("div", "split-bars");
  const total = Object.values(r.splits).reduce(
    (sum, item) => sum + item.epochs,
    0,
  );
  Object.entries(r.splits).forEach(([name, item], index) => {
    const label = { train: "训练", validation: "验证", test: "测试" }[name];
    const line = node("div", "split-row");
    const track = node("div", "score-track");
    const bar = node("i", `score-fill series-${index}`);
    bar.style.width = `${(100 * item.epochs) / total}%`;
    track.append(bar);
    line.append(
      node("span", "", label),
      track,
      node(
        "span",
        "",
        `${item.epochs} 历元 · ${number((100 * item.epochs) / total, 1)}%`,
      ),
    );
    counts.append(line);
  });
  el("fold-split-chart").replaceChildren(counts);
  const errorContainer = el("fold-error-chart");
  errorContainer.replaceChildren();
  r.error_analysis.cn0_bin.forEach((item) => {
    const row = node("div", "error-slice");
    const heading = node("div", "coverage-heading");
    heading.append(
      node("strong", "", item.group),
      node("span", "", `${item.rows} 条 · NLOS ${item.positives}`),
    );
    const track = node("div", "coverage-track");
    [
      [item.false_positives, "false-positive"],
      [item.false_negatives, "false-negative"],
    ].forEach(([count, name]) => {
      const segment = node("span", `coverage-part ${name}`);
      segment.style.width = `${(100 * count) / item.rows}%`;
      segment.title = `${name === "false-positive" ? "误报" : "漏报"} ${count} / ${item.rows}`;
      track.append(segment);
    });
    row.append(
      heading,
      track,
      node(
        "p",
        "chart-footnote",
        `误报 ${item.false_positives} · 漏报 ${item.false_negatives}，条带长度为占本组全部观测的比例。`,
      ),
    );
    errorContainer.append(row);
  });
}
async function initializeGeneralization() {
  const data = await get("/api/nlos/generalization");
  if (!data.available) return;
  const r = data.report;
  generalizationReport = r;
  el("nlos-content").hidden = false;
  el("nlos-load-status").textContent =
    "真实观测实验已加载 · 点击标题可展开或收起";
  const external = r.external_transfer.models;
  const district = r.district_holdout.models;
  updateProjectOverview(r.dataset, r.external_transfer.rows, "REAL · UrbanNav");
  renderCoverage(r);
  el("nlos-generalization-panel").hidden = false;
  el("nlos-generalization-note").textContent =
    `新增 ${r.dataset.epochs.toLocaleString()} 个历元、${r.dataset.rows.toLocaleString()} 条卫星观测，${r.external_transfer.rows.toLocaleString()} 条具有可评分的建筑几何参考。冻结源模型由 KLT 验证集选择：${names[r.external_transfer.selected_model]}。以下均为同一城市的跨城区测试。`;
  const coverage = external.random_forest.per_area;
  table(
    el("nlos-generalization-coverage"),
    [
      "新城区",
      "原始历元",
      "可评分历元",
      "可评分卫星观测",
      "unknown 观测",
      "标签覆盖率",
    ],
    r.plan.plan.target_areas.map((area) => [
      areaNames[area],
      r.dataset.areas[area].epochs,
      coverage[area].epochs,
      coverage[area].rows.toLocaleString(),
      coverage[area].unknown_rows.toLocaleString(),
      `${number(coverage[area].label_coverage * 100, 1)}%`,
    ]),
  );
  table(
    el("nlos-generalization-models"),
    [
      "模型",
      "冻结外部 P",
      "外部 NLOS Recall",
      "外部 F1",
      "城区互斥 P",
      "互斥 NLOS Recall",
      "互斥 F1",
    ],
    Object.keys(external).map((name) => [
      names[name] || name,
      ...["precision", "recall", "f1"].map((key) =>
        number(external[name].test_macro[key]),
      ),
      ...["precision", "recall", "f1"].map((key) =>
        number(district[name].test_macro[key]),
      ),
    ]),
  );
  Object.keys(external).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = names[name] || name;
    el("nlos-generalization-model").append(option);
  });
  el("nlos-generalization-model").value = "random_forest";
  const render = () => {
    const name = el("nlos-generalization-model").value;
    table(
      el("nlos-generalization-areas"),
      [
        "城区",
        "外部 Precision",
        "外部 NLOS Recall",
        "外部 F1",
        "互斥 Precision",
        "互斥 NLOS Recall",
        "互斥 F1",
      ],
      r.plan.plan.target_areas.map((area) => [
        areaNames[area],
        ...["precision", "recall", "f1"].map((key) =>
          number(external[name].per_area[area][key]),
        ),
        ...["precision", "recall", "f1"].map((key) =>
          number(district[name].per_area[area].test_mean[key]),
        ),
      ]),
    );
    renderGeneralizationCharts();
  };
  el("nlos-generalization-model").onchange = render;
  bindMetricSwitch("generalization-metric", renderGeneralizationCharts);
  el("generalization-heatmap-protocol").onchange = renderGeneralizationCharts;
  render();
}
async function loadNlosEpoch() {
  const request = ++epochRequest;
  const experiment = activeNlosExperiment;
  try {
    const data = await get(
      `/api/nlos/epoch?trip=${encodeURIComponent(el("nlos-trip").value)}&index=${el("nlos-epoch").value}${activeNlosExperiment ? `&experiment=${encodeURIComponent(activeNlosExperiment)}` : ""}`,
    );
    if (request !== epochRequest || experiment !== activeNlosExperiment) return;
    el("nlos-epoch").max = data.total_epochs - 1;
    el("nlos-time").textContent =
      `${data.epoch_index + 1}/${data.total_epochs} · ${number(data.time_s, 1)} s`;
    const outcomes = {
      true_positive: "正确报警",
      false_positive: "误报",
      false_negative: "漏报",
      true_negative: "正确排除",
    };
    el("nlos-satellite-chart").replaceChildren();
    data.satellites.forEach((s) => {
      const card = node("article", `satellite-card ${s.outcome}`);
      const heading = node("div", "satellite-heading");
      heading.append(
        node("strong", "", s.satellite_id),
        node("span", "", outcomes[s.outcome]),
      );
      const track = node("div", "score-track");
      const bar = node("i", "score-fill series-0");
      bar.style.width = `${100 * s.predicted_nlos_score}%`;
      track.append(bar);
      card.append(
        heading,
        node(
          "p",
          "",
          `参考 ${s.signal_label.toUpperCase()} → 预测 ${s.prediction_status.toUpperCase()}`,
        ),
        track,
        node(
          "small",
          "",
          `NLOS 分数 ${number(s.predicted_nlos_score)} · C/N0 ${number(s.cn0_dbhz, 1)} dB-Hz`,
        ),
      );
      el("nlos-satellite-chart").append(card);
    });
    table(
      el("nlos-satellites"),
      [
        "卫星",
        "C/N0 (dB-Hz)",
        "仰角 (°)",
        "相机参考标签",
        "模型预测",
        "NLOS 分数",
        "判定",
      ],
      data.satellites.map((s) => [
        s.satellite_id,
        number(s.cn0_dbhz, 1),
        number(s.elevation_deg, 1),
        s.signal_label.toUpperCase(),
        s.prediction_status.toUpperCase(),
        number(s.predicted_nlos_score),
        outcomes[s.outcome],
      ]),
    );
  } catch (error) {
    if (request === epochRequest) {
      el("nlos-satellites").textContent = error.message;
      el("nlos-satellite-chart").textContent = error.message;
    }
  }
}
async function initializeNlos(experiment = "") {
  const request = ++nlosRequest;
  const data = await get(
    `/api/nlos/report${experiment ? `?experiment=${encodeURIComponent(experiment)}` : ""}`,
  );
  if (request !== nlosRequest) return;
  if (!data.available) {
    el("nlos-fold-panel").hidden = true;
    el("nlos-badge").textContent = "尚无实验产物";
    el("nlos-load-status").textContent =
      "尚未加载卫星级遮挡实验。现有定位失准分数不能替代 LOS/NLOS 标签。";
    return;
  }
  const r = data.report,
    selected = r.models[r.selected_model].test;
  activeNlosExperiment = data.experiment_id || "";
  currentNlosReport = r;
  el("nlos-fold-panel").hidden = false;
  el("nlos-load-status").textContent =
    "真实观测实验已加载 · 点击标题可展开或收起";
  el("nlos-experiment-label").hidden = !data.suite;
  el("nlos-fold-label").textContent = data.suite
    ? `${r.protocol === "in_domain" ? "同域" : "跨区域"} · 测试 ${r.experiment.test_area}`
    : "区域隔离开发实验";
  if (data.suite) {
    if (!generalizationReport)
      updateProjectOverview(
        data.suite.dataset,
        data.suite.dataset.rows,
        "REAL · KLT",
      );
    el("nlos-protocol-panel").hidden = false;
    const protocols = data.suite.protocols;
    const labels = { in_domain: "同域", cross_region: "跨区域" };
    const renderProtocolComparison = () => {
      const metric = selectedMetric("protocol-metric");
      comparisonBars(
        el("protocol-comparison"),
        Object.keys(protocols.in_domain.models).map((name) => ({
          key: name,
          label: names[name] || name,
          values: [
            protocols.in_domain.models[name].test_macro[metric],
            protocols.cross_region.models[name].test_macro[metric],
          ],
        })),
        [`同域 ${metricLabels[metric]}`, `跨区域 ${metricLabels[metric]}`],
        el("nlos-area-model").value,
        (name) => {
          el("nlos-area-model").value = name;
          el("nlos-area-model").onchange();
        },
      );
    };
    bindMetricSwitch("protocol-metric", renderProtocolComparison);
    table(
      el("nlos-protocol-results"),
      [
        "模型",
        "同域 Precision",
        "同域 Recall",
        "同域 F1",
        "跨区域 Precision",
        "跨区域 Recall",
        "跨区域 F1",
        "F1 变化",
      ],
      Object.keys(protocols.in_domain.models).map((name) => {
        const within = protocols.in_domain.models[name].test_macro;
        const cross = protocols.cross_region.models[name].test_macro;
        const delta = cross.f1 - within.f1;
        return [
          names[name] || name,
          number(within.precision),
          number(within.recall),
          number(within.f1, 4),
          number(cross.precision),
          number(cross.recall),
          number(cross.f1, 4),
          `${delta > 0 ? "+" : ""}${number(delta, 4)}`,
        ];
      }),
    );
    const dataset = data.suite.dataset;
    el("nlos-protocol-note").textContent =
      `${dataset.epochs.toLocaleString()} 个历元，${Object.keys(dataset.per_run).length} 段，${dataset.dates.length} 个日期。固定网络参数，未追加调参；指标是三个目标区域的等权平均。每对实验使用相同验证和测试采集段，跨区域训练会移除该区域的其他采集段，训练量也随之减少。C/N0 规则固定为 <30 dB-Hz、不进行训练，因此共享测试集的两侧成绩必然相同，不能用该差值判断域偏移。三处区域样本较少，排名只代表本次观察；部分测试数据曾用于前期开发。`;
    const areaModel = el("nlos-area-model").value || "random_forest";
    el("nlos-area-model").replaceChildren();
    Object.keys(protocols.in_domain.models).forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = names[name] || name;
      el("nlos-area-model").append(option);
    });
    el("nlos-area-model").value = areaModel;
    const renderAreas = () => {
      const name = el("nlos-area-model").value;
      comparisonBars(
        el("protocol-area-chart"),
        Object.keys(protocols.in_domain.models[name].per_area).map((area) => ({
          label: area,
          values: [
            protocols.in_domain.models[name].per_area[area].f1,
            protocols.cross_region.models[name].per_area[area].f1,
          ],
        })),
        ["同域 F1", "跨区域 F1"],
      );
      renderProtocolComparison();
      table(
        el("nlos-area-results"),
        [
          "测试区域",
          "测试历元",
          "测试观测",
          "同域 Precision",
          "同域 Recall",
          "同域 F1",
          "跨区域 Precision",
          "跨区域 Recall",
          "跨区域 F1",
          "F1 变化",
        ],
        Object.keys(protocols.in_domain.models[name].per_area).map((area) => {
          const a = protocols.in_domain.models[name].per_area[area];
          const b = protocols.cross_region.models[name].per_area[area];
          const item = data.suite.experiments.find((x) => x.test_area === area);
          const delta = b.f1 - a.f1;
          return [
            area,
            item.splits.test.epochs,
            a.rows,
            number(a.precision),
            number(a.recall),
            number(a.f1, 4),
            number(b.precision),
            number(b.recall),
            number(b.f1, 4),
            `${delta > 0 ? "+" : ""}${number(delta, 4)}`,
          ];
        }),
      );
    };
    el("nlos-area-model").onchange = renderAreas;
    renderAreas();
    el("nlos-experiment").replaceChildren();
    data.suite.experiments.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${labels[item.protocol]} · 测试 ${item.test_area}`;
      el("nlos-experiment").append(option);
    });
    el("nlos-experiment").value = activeNlosExperiment;
    el("nlos-experiment").onchange = () =>
      initializeNlos(el("nlos-experiment").value);
  } else if (!generalizationReport) {
    const dataset = {
      epochs: Object.values(r.splits).reduce(
        (sum, split) => sum + split.epochs,
        0,
      ),
      rows: Object.values(r.splits).reduce((sum, split) => sum + split.rows, 0),
      per_run: Object.fromEntries(
        Object.values(r.splits)
          .flatMap((split) => split.trips)
          .map((trip) => [trip, {}]),
      ),
    };
    updateProjectOverview(dataset, dataset.rows, "REAL · KLT 历史实验");
  }
  el("nlos-badge").textContent = "REAL · KLT 相机参考";
  el("nlos-badge").classList.add("real");
  el("nlos-content").hidden = false;
  el("nlos-notice").textContent = data.suite
    ? `${r.protocol === "in_domain" ? "同域实验：训练中包含测试区域的其他完整采集段。" : "跨区域实验：测试区域完全未进入训练和验证。"}参考标签来自 KLT 鱼眼相机，模型只读取信号与几何特征。下方显示当前实验的结果。`
    : "独立任务：逐卫星预测 LOS/NLOS。参考标签来自 KLT 鱼眼相机，模型只读取信号与几何特征。当前为此前的四段区域隔离开发实验。";
  el("nlos-precision").textContent = number(selected.precision);
  el("nlos-recall").textContent = number(selected.recall);
  el("nlos-f1").textContent = number(selected.f1, 4);
  el("nlos-model").textContent =
    `${names[r.selected_model] || r.selected_model} · 遮挡实验选中`;
  el("nlos-count").textContent = r.splits.test.rows.toLocaleString();
  renderFoldCharts(r);
  const splitNames = { train: "训练", validation: "验证", test: "测试" };
  table(
    el("nlos-splits"),
    ["划分", "区域", "完整采集段", "历元", "历元占比", "卫星观测", "NLOS 比例"],
    Object.entries(r.splits).map(([name, s]) => [
      splitNames[name],
      s.scene_groups.join(", "),
      s.trips.join(", "),
      s.epochs,
      `${number((100 * s.epochs) / Object.values(r.splits).reduce((sum, split) => sum + split.epochs, 0), 1)}%`,
      s.rows,
      `${number(s.positive_rate * 100, 1)}%`,
    ]),
  );
  const neuralNames =
    r.neural_models || (r.models.transformer ? ["transformer"] : []);
  const modelRows = (entries) =>
    entries.map(([name, m]) => [
      `${names[name] || name}${name === r.selected_model ? " · 验证集选择" : name === r.best_neural_model ? " · 组内优选" : ""}`,
      number(m.test.precision),
      number(m.test.recall),
      number(m.test.f1, 4),
      number(m.test.accuracy),
      m.test.false_positives,
      m.test.false_negatives,
      number(m.test.threshold, 2),
    ]);
  const testHeaders = [
    "方法",
    "Precision",
    "NLOS 检出率 (Recall)",
    "F1",
    "总体准确率",
    "误报",
    "漏报",
    "阈值",
  ];
  el("nlos-neural-title").textContent =
    neuralNames.length === 5 ? "五种神经网络对比" : "神经网络对比";
  el("nlos-neural-panel").hidden = neuralNames.length === 0;
  el("project-models").textContent =
    `${neuralNames.length} / ${Object.keys(r.models).length - neuralNames.length}`;
  table(
    el("nlos-models"),
    testHeaders,
    modelRows(
      Object.entries(r.models).filter(([name]) => neuralNames.includes(name)),
    ),
  );
  table(
    el("nlos-reference-models"),
    testHeaders,
    modelRows(
      Object.entries(r.models).filter(([name]) => !neuralNames.includes(name)),
    ),
  );
  table(
    el("nlos-validation"),
    ["方法", "Val Precision", "Val Recall", "Val F1", "分类阈值"],
    Object.entries(r.models).map(([name, m]) => [
      `${names[name] || name}${name === r.selected_model ? " · 全体选中" : name === r.best_neural_model ? " · 组内优选" : ""}`,
      number(m.validation.precision),
      number(m.validation.recall),
      number(m.validation.f1),
      number(m.validation.threshold, 2),
    ]),
  );
  el("nlos-errors").textContent =
    `当前选用 ${names[r.selected_model] || r.selected_model}：检出 ${selected.true_positives}/${selected.positives} 条 NLOS，误报 ${selected.false_positives}、漏报 ${selected.false_negatives}。`;
  el("nlos-tuning-panel").hidden = !(
    r.baseline_comparison && r.hyperparameter_search
  );
  if (r.baseline_comparison && r.hyperparameter_search) {
    el("nlos-tuning-panel").hidden = false;
    table(
      el("nlos-tuning-results"),
      ["方法", "默认 Val F1", "调参 Val F1", "默认 Test F1", "调参 Test F1"],
      Object.entries(r.baseline_comparison.models)
        .filter(([name]) => name !== "cn0_rule")
        .map(([name, m]) => [
          names[name] || name,
          number(m.before.validation.f1, 4),
          number(m.after.validation.f1, 4),
          number(m.before.test.f1, 4),
          number(m.after.test.f1, 4),
        ]),
    );
    const baselineModel = r.baseline_comparison.selected_model_before;
    const baselineF1 =
      r.baseline_comparison.models[baselineModel].before.test.f1;
    el("nlos-tuning-note").textContent =
      `${r.hyperparameter_search.trials} 组预先固定候选，只按验证集选模型和阈值。当前 ${names[r.selected_model]} 测试 F1 ${number(selected.f1, 4)}，原 ${names[baselineModel]} ${number(baselineF1, 4)}。测试区域已在开发中查看；微小差异不代表稳定提升。`;
  }
  const tradeoffs = {};
  el("nlos-tradeoff-model").replaceChildren();
  for (const [name, m] of Object.entries(r.models)) {
    if (!m.validation_threshold_tradeoff) continue;
    tradeoffs[name] = m.validation_threshold_tradeoff.map((p) => [
      number(p.threshold, 2),
      number(p.precision),
      number(p.recall),
      number(p.f1),
      p.false_positives,
      p.false_negatives,
    ]);
    const option = document.createElement("option");
    option.value = name;
    option.textContent = names[name] || name;
    el("nlos-tradeoff-model").append(option);
  }
  if (Object.keys(tradeoffs).length) {
    el("nlos-tradeoff-panel").hidden = false;
    if (tradeoffs[r.selected_model])
      el("nlos-tradeoff-model").value = r.selected_model;
    const renderTradeoff = () => {
      const model = r.models[el("nlos-tradeoff-model").value];
      plot(
        el("nlos-tradeoff-chart"),
        ["precision", "recall", "f1"].map((key, index) => ({
          points: model.validation_threshold_tradeoff
            .filter((point) => typeof point[key] === "number")
            .map((point) => [point.threshold, point[key]]),
          color: ["#235b49", "#6391b2", "#dc9050"][index],
          label: metricLabels[key],
        })),
        "阈值",
        "验证指标 (0–1)",
        false,
        false,
        true,
      );
      table(
        el("nlos-tradeoff"),
        [
          "阈值",
          "Val Precision",
          "Val Recall",
          "Val F1",
          "验证误报",
          "验证漏报",
        ],
        tradeoffs[el("nlos-tradeoff-model").value],
      );
    };
    el("nlos-tradeoff-model").onchange = renderTradeoff;
    renderTradeoff();
  }
  table(
    el("nlos-slices"),
    ["C/N0 区间", "卫星观测", "NLOS 数", "Precision", "Recall", "误报", "漏报"],
    r.error_analysis.cn0_bin.map((s) => [
      s.group,
      s.rows,
      s.positives,
      number(s.precision),
      number(s.recall),
      s.false_positives,
      s.false_negatives,
    ]),
  );
  el("nlos-trip").replaceChildren();
  el("nlos-epoch").value = 0;
  r.splits.test.trips.forEach((trip) => {
    const option = document.createElement("option");
    option.value = trip;
    option.textContent = trip;
    el("nlos-trip").append(option);
  });
  await loadNlosEpoch();
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
  unitAxis = false,
) {
  container.replaceChildren();
  const valid = series.filter((s) => s.points.length > 1);
  if (!valid.length) {
    container.textContent = "当前实验没有可用绘图数据。";
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
  if (unitAxis) {
    ymin = 0;
    ymax = 1;
    xmin = 0;
    xmax = 1;
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
        number(xv, unitAxis ? 2 : 0),
      ),
      svgNode(
        "text",
        {
          x: pad.left - 8,
          y: y(yv) + 3,
          "text-anchor": "end",
          class: "svg-axis",
        },
        number(yv, unitAxis ? 2 : 0),
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
  valid.forEach((s) => {
    const line = svgNode("polyline", {
      points: s.points
        .map((p) => `${x(p[0])},${s.rightAxis ? riskY(p[1]) : y(p[1])}`)
        .join(" "),
      fill: "none",
      stroke: s.color,
      "stroke-width": s.width || 1.7,
      opacity: s.opacity || 1,
    });
    if (s.label) line.append(svgNode("title", {}, s.label));
    svg.append(line);
  });
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
async function initializePositioning() {
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
  el("task-label").textContent =
    `失准告警定义：误差 > ${report.error_limit_m} m`;
  el("position-p95").textContent = report.positioning
    ? `${number(report.positioning.methods.wls.p95_m, 2)} m`
    : "未计算";
  modelTable();
  if (report.positioning) {
    table(
      el("position-table"),
      ["方法", "历元", "Median (m)", "P95 (m)", "RMSE (m)", "≤2 m", "≤5 m"],
      Object.entries(report.positioning.methods).map(([name, v]) => [
        names[name] || name,
        v.epochs,
        number(v.median_m),
        number(v.p95_m),
        number(v.rmse_m),
        v.within_error_limits_m
          ? `${number(v.within_error_limits_m["2"] * 100, 1)}%`
          : "—",
        v.within_error_limits_m
          ? `${number(v.within_error_limits_m["5"] * 100, 1)}%`
          : "—",
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
el("nlos-trip").addEventListener("change", () => {
  el("nlos-epoch").value = 0;
  loadNlosEpoch();
});
el("nlos-epoch").addEventListener("change", loadNlosEpoch);

let modelConnections = [];
let editingConnectionId = "";
let connectionDirty = false;
let askingModel = false;
function connectionStatus(text, error = false) {
  el("connection-status").textContent = text;
  el("connection-status").classList.toggle("error", error);
}
function updateConnectionState() {
  const custom = el("provider").value === "connection";
  const saved = modelConnections.find(
    (item) => item.id === editingConnectionId,
  );
  el("connection-select-label").hidden = !custom;
  el("connection-summary").hidden = !custom;
  el("ask-button").disabled =
    askingModel || (custom && (!saved || connectionDirty));
  el("connection-delete").disabled = !saved;
  el("connection-summary").textContent = saved
    ? `${saved.name} · ${saved.model} · ${saved.answer_mode === "evidence" ? "证据问答" : "工具 Agent"} · ${saved.base_url}${connectionDirty ? " · 有未保存修改，请先保存并使用" : ""}`
    : "请先配置模型连接，点击“保存并使用”。";
  if (!askingModel)
    el("agent-status").textContent = custom
      ? saved && !connectionDirty
        ? "已选择自定义服务，点击分析实验后才会发起模型请求。"
        : "请先保存要使用的模型连接。"
      : el("provider").value === "offline"
        ? "离线模式采用确定性工具流程，不是大语言模型。"
        : "将读取本机 .env 中的 compatible 配置。";
}
function editConnection(identifier = "") {
  const saved = modelConnections.find((item) => item.id === identifier);
  editingConnectionId = saved?.id || "";
  el("connection-select").value = editingConnectionId;
  el("connection-name").value = saved?.name || "";
  el("connection-url").value = saved?.base_url || "";
  el("connection-model").value = saved?.model || "";
  el("connection-mode").value = saved?.answer_mode || "evidence";
  el("connection-tokens").value = saved?.max_output_tokens || 4096;
  el("connection-timeout").value = saved?.timeout_s || 90;
  el("connection-reasoning").value = saved?.reasoning_effort || "default";
  el("connection-preset").value = "manual";
  el("connection-key").value = "";
  el("connection-clear-key").checked = false;
  el("connection-key-state").textContent = saved?.has_api_key
    ? "此连接已保存密钥。留空保留原密钥；更换服务地址需重新填写或勾选清除。"
    : "未保存密钥。本地无鉴权服务可留空。";
  el("connection-models").replaceChildren();
  connectionDirty = false;
  updateConnectionState();
}
async function refreshConnections(selected = editingConnectionId) {
  const response = await fetch("/api/config");
  const config = await response.json();
  if (!response.ok) throw new Error(config.error || "连接配置读取失败");
  modelConnections = config.connections || [];
  el("connection-select").replaceChildren(new Option("新建连接", ""));
  modelConnections.forEach((item) =>
    el("connection-select").append(
      new Option(`${item.name} · ${item.model}`, item.id),
    ),
  );
  editConnection(selected);
}
function connectionPayload() {
  return {
    id: editingConnectionId || undefined,
    name: el("connection-name").value.trim(),
    base_url: el("connection-url").value.trim(),
    model: el("connection-model").value.trim(),
    api_key: el("connection-key").value.trim(),
    clear_api_key: el("connection-clear-key").checked,
    answer_mode: el("connection-mode").value,
    max_output_tokens: Number(el("connection-tokens").value),
    timeout_s: Number(el("connection-timeout").value),
    reasoning_effort: el("connection-reasoning").value,
  };
}
async function connectionRequest(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "连接操作失败");
  return result;
}
el("provider").addEventListener("change", () => {
  if (
    el("provider").value === "connection" &&
    !editingConnectionId &&
    modelConnections.length
  )
    editConnection(modelConnections[0].id);
  if (el("provider").value === "connection" && !editingConnectionId)
    el("connection-panel").open = true;
  updateConnectionState();
});
el("connection-select").addEventListener("change", () => {
  editConnection(el("connection-select").value);
  if (!editingConnectionId) el("connection-panel").open = true;
});
el("connection-form").addEventListener("input", () => {
  connectionDirty = true;
  updateConnectionState();
});
el("connection-preset").addEventListener("change", () => {
  const preset = el("connection-preset").value;
  if (preset === "manual") return;
  // Presets are new destinations, so an existing credential is explicitly cleared.
  el("connection-url").value =
    preset === "ollama"
      ? "http://127.0.0.1:11434/v1"
      : "http://127.0.0.1:8000/v1";
  el("connection-name").value =
    preset === "ollama" ? "Ollama 本地" : "vLLM 本地";
  el("connection-key").value = "";
  el("connection-clear-key").checked = true;
  el("connection-model").value = "";
  el("connection-models").replaceChildren();
  connectionDirty = true;
  updateConnectionState();
});
el("connection-key").addEventListener("input", () => {
  if (el("connection-key").value) el("connection-clear-key").checked = false;
});
el("connection-new").addEventListener("click", () => {
  editConnection();
  el("provider").value = "connection";
  el("connection-panel").open = true;
  updateConnectionState();
  connectionStatus("填写新的服务地址和模型，再保存并使用。");
});
el("connection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  el("connection-save").disabled = true;
  try {
    const result = await connectionRequest(
      "/api/connections/save",
      connectionPayload(),
    );
    el("connection-key").value = "";
    el("provider").value = "connection";
    await refreshConnections(result.connection.id);
    connectionStatus("连接已在本机保存并选中。尚未调用模型推理。");
  } catch (error) {
    connectionStatus(error.message, true);
  } finally {
    el("connection-save").disabled = false;
  }
});
el("connection-discover").addEventListener("click", async () => {
  if (!el("connection-url").value.trim()) {
    connectionStatus("请先填写 API 地址和连接名称。", true);
    return;
  }
  el("connection-discover").disabled = true;
  connectionStatus("正在获取模型列表，不进行推理…");
  try {
    const result = await connectionRequest(
      "/api/connections/models",
      connectionPayload(),
    );
    el("connection-models").replaceChildren();
    result.models.forEach((id) =>
      el("connection-models").append(new Option(id, id)),
    );
    connectionStatus(
      `发现 ${result.models.length} 个模型。点击模型 ID 输入框选择或自行填写；列表成功不代表问答与工具调用已验证。`,
    );
  } catch (error) {
    connectionStatus(`${error.message} 可手动填写模型 ID 后保存。`, true);
  } finally {
    el("connection-discover").disabled = false;
  }
});
el("connection-delete").addEventListener("click", async () => {
  el("connection-delete").disabled = true;
  try {
    await connectionRequest("/api/connections/delete", {
      id: editingConnectionId,
    });
    await refreshConnections("");
    el("provider").value = "offline";
    updateConnectionState();
    connectionStatus("该连接和其本地密钥已删除。");
  } catch (error) {
    connectionStatus(error.message, true);
    updateConnectionState();
  }
});
async function initializeConnections() {
  try {
    await refreshConnections();
    if (modelConnections.length) editConnection(modelConnections[0].id);
    connectionStatus(
      modelConnections.length
        ? `本机已保存 ${modelConnections.length} 个连接。选择自定义模型 API 后使用。`
        : "尚未配置模型服务。离线模式可直接使用。",
    );
  } catch (error) {
    connectionStatus(error.message, true);
  }
}
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
  if (
    el("provider").value === "connection" &&
    (!editingConnectionId || connectionDirty)
  ) {
    el("agent-status").textContent = "请先保存要使用的模型连接。";
    return;
  }
  askingModel = true;
  updateConnectionState();
  el("agent-status").textContent = "正在调用分析工具…";
  el("evidence-list").replaceChildren();
  el("answer").textContent = "正在检索实验指标和技术文档…";
  el("answer-backend").textContent = "处理中";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        provider: el("provider").value,
        connection_id:
          el("provider").value === "connection"
            ? editingConnectionId
            : undefined,
        task_context: el("diagnostic-task").value,
        trip_id: el("trip-select").value,
        nlos_experiment: activeNlosExperiment || undefined,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "分析失败");
    el("answer").textContent = result.answer;
    el("answer-backend").textContent = result.generation_skipped
      ? "本地证据检查"
      : result.backend === "offline"
        ? "离线工具基线"
        : result.model;
    el("agent-status").textContent = result.generation_skipped
      ? `未找到相关项目依据 · ${number(result.elapsed_s, 2)} 秒 · 未调用模型 API`
      : `${result.evidence.length} 条证据 · ${number(result.elapsed_s, 2)} 秒 · ${result.backend === "offline" ? "离线工具基线" : result.answer_mode === "evidence" ? "证据问答" : "工具 Agent"} · 引用编号已检查，事实仍需核对`;
    const stats = result.context_stats;
    if (stats?.original_evidence_characters) {
      const saved =
        1 - stats.sent_evidence_characters / stats.original_evidence_characters;
      el("agent-status").textContent +=
        ` · 发送依据减少 ${number(saved * 100, 0)}%`;
    }
    if (result.warnings?.length)
      el("agent-status").textContent += ` · ${result.warnings.join(" ")}`;
    result.evidence.forEach((item) => {
      const detail = document.createElement("details");
      detail.className = "evidence";
      const summary = document.createElement("summary");
      const provenance = {
        real: "真实观测",
        synthetic: "模拟观测",
        knowledge: "技术笔记",
      };
      summary.textContent = `[${item.evidence_id}] ${item.tool} · ${provenance[item.data_kind] || item.data_kind}`;
      detail.append(summary);
      if (item.tool === "search_knowledge") {
        const method = item.payload.embedding_model
          ? "本地语义 + 关键词"
          : "关键词";
        summary.textContent = `[${item.evidence_id}] 文档检索 · ${method} · ${item.payload.notes.length} 段依据`;
        const list = document.createElement("div");
        list.className = "retrieval-passages";
        if (item.payload.fallback_reason) {
          const hint = document.createElement("p");
          hint.className = "chart-footnote";
          hint.textContent = "本地语义模型未就绪，已使用关键词检索。";
          list.append(hint);
        }
        item.payload.notes.forEach((note) => {
          const passage = document.createElement("details");
          passage.className = "retrieval-passage";
          const title = document.createElement("summary");
          title.textContent = `${note.id} · ${note.heading || note.title}`;
          const content = document.createElement("div");
          content.className = "retrieval-content";
          const source = document.createElement("p");
          source.className = "chart-footnote";
          source.textContent = `${note.source_url}${note.start_line ? ` · 第 ${note.start_line} 行` : ""}`;
          const score = document.createElement("p");
          score.className = "chart-footnote";
          score.textContent = `排序分数 ${number(note.relevance_score, 3)}${note.retrieval_scores?.semantic_cosine != null ? ` · 语义相似度 ${number(note.retrieval_scores.semantic_cosine, 3)}` : ""} · 分数不代表回答正确率`;
          const excerpt = document.createElement("p");
          excerpt.textContent = note.content;
          content.append(source, score, excerpt);
          passage.append(title, content);
          list.append(passage);
        });
        detail.append(list);
      } else {
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(item.payload, null, 2);
        detail.append(pre);
      }
      el("evidence-list").append(detail);
    });
  } catch (error) {
    el("answer").textContent =
      "本次未生成可用回答。请根据下方提示调整后重新提问。";
    el("answer-backend").textContent = "请求未完成";
    el("agent-status").textContent = error.message;
    el("agent-status").classList.add("error");
  } finally {
    askingModel = false;
    el("ask-button").disabled =
      el("provider").value === "connection" &&
      (!editingConnectionId || connectionDirty);
  }
});
function openNavigationTarget(hash) {
  const target = document.getElementById(hash.slice(1));
  if (!target || target.hidden) return;
  let parent = target;
  while (parent && parent !== document.body) {
    if (parent.tagName === "DETAILS") parent.open = true;
    parent = parent.parentElement;
  }
  if (target.tagName === "SECTION") {
    const detail = target.querySelector("details.panel:not([hidden])");
    if (detail) detail.open = true;
  }
  if (target.id === "nlos-fold-panel") el("nlos-satellite-panel").open = true;
}
document.querySelectorAll(".nav-item").forEach((link) =>
  link.addEventListener("click", () => {
    openNavigationTarget(link.hash);
    document
      .querySelectorAll(".nav-item")
      .forEach((x) => x.classList.remove("active"));
    link.classList.add("active");
  }),
);
const disclosureIds = [
  "project-overview",
  "nlos-generalization-panel",
  "nlos-protocol-panel",
  "nlos-fold-panel",
  "diagnostics-panel",
  "positioning-panel",
  "method-panel",
];
function initializeDisclosures() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem("navdiag.disclosures.v2") || "{}");
  } catch {
    /* Browser storage may be unavailable. */
  }
  disclosureIds.forEach((id) => {
    const detail = el(id);
    if (typeof stored?.[id] === "boolean") detail.open = stored[id];
    detail.addEventListener("toggle", () => {
      try {
        const state = Object.fromEntries(
          disclosureIds.map((key) => [key, el(key).open]),
        );
        localStorage.setItem("navdiag.disclosures.v2", JSON.stringify(state));
      } catch {
        /* Collapse controls still work without persistent storage. */
      }
    });
  });
  el("expand-sections").onclick = () => {
    document
      .querySelectorAll("main details:not(.evidence)")
      .forEach((detail) => {
        if (!detail.hidden) detail.open = true;
      });
  };
  el("collapse-sections").onclick = () => {
    document
      .querySelectorAll("main details:not(.evidence)")
      .forEach((detail) => {
        detail.open = false;
      });
  };
  if (location.hash) openNavigationTarget(location.hash);
}
async function initialize() {
  initializeDisclosures();
  initializeConnections();
  const results = await Promise.allSettled([
    initializePositioning(),
    initializeNlos(),
    initializeGeneralization(),
  ]);
  if (results[0].status === "rejected")
    el("provenance-warning").textContent =
      `定位辅助实验未载入：${results[0].reason.message}`;
  const errors = results
    .slice(1)
    .filter((result) => result.status === "rejected");
  if (errors.length)
    el("nlos-load-status").textContent =
      `部分识别实验未载入：${errors.map((result) => result.reason.message).join("；")}`;
  if (!generalizationReport && !currentNlosReport) {
    el("project-badge").textContent = "尚无识别产物";
    el("project-data-source").textContent = "先生成公开观测实验";
    el("project-models").textContent = "—";
    el("project-scope").textContent =
      "尚未加载真实卫星识别报告。定位辅助实验可以单独运行，其成绩不作为 NLOS 结果。";
  }
}
initialize();
