const apiUrlInput = document.getElementById("api-url");
const apiKeyInput = document.getElementById("api-key");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const disclaimerEl = document.getElementById("disclaimer");
const FINDING_STATUSES = new Set([
  "aligned",
  "potential_gap",
  "insufficient_evidence",
  "needs_review",
]);

async function loadSettings() {
  const stored = await chrome.storage.sync.get(["apiUrl", "apiKey"]);
  apiUrlInput.value = stored.apiUrl || "http://localhost:8080";
  apiKeyInput.value = stored.apiKey || "";
}

async function saveSettings() {
  await chrome.storage.sync.set({
    apiUrl: apiUrlInput.value.trim(),
    apiKey: apiKeyInput.value.trim(),
  });
}

function setStatus(message) {
  statusEl.textContent = message;
}

function formatLabel(value) {
  return String(value ?? "").replaceAll("_", " ");
}

function appendText(parent, tag, text, className) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

function renderCitation(label, citation, cssClass) {
  const block = document.createElement("div");
  block.className = citation ? `citation ${cssClass}` : `citation missing ${cssClass}`;

  appendText(block, "strong", label);
  block.appendChild(document.createElement("br"));

  if (!citation) {
    block.appendChild(document.createTextNode("No citation found."));
    return block;
  }

  appendText(block, "span", `${citation.doc_id ?? ""} §${citation.section_path ?? ""}`);
  block.appendChild(document.createElement("br"));

  if (citation.text) {
    appendText(block, "span", citation.text);
  }

  return block;
}

function renderSummary(parent, text) {
  const block = document.createElement("div");
  block.className = "summary-block";

  const summaryText = appendText(block, "p", text, "summary-text collapsed");

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "summary-toggle";
  toggle.textContent = "Show full summary";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
    summaryText.classList.toggle("collapsed", expanded);
    toggle.textContent = expanded ? "Show full summary" : "Show less";
  });
  block.appendChild(toggle);

  parent.appendChild(block);
}

function renderFinding(finding) {
  const article = document.createElement("article");
  article.className = "finding";

  const badge = document.createElement("span");
  badge.className = "badge";
  if (FINDING_STATUSES.has(finding.status)) {
    badge.classList.add(finding.status);
  }
  badge.textContent = formatLabel(finding.status);
  article.appendChild(badge);

  appendText(article, "h2", formatLabel(finding.topic));
  renderSummary(article, finding.summary ?? "");

  article.appendChild(renderCitation("Policy", finding.policy_citation, "citation-policy"));
  article.appendChild(renderCitation("Regulation", finding.regulation_citation, "citation-regulation"));

  return article;
}

function buildStatusMessage(data) {
  return `Done. ${data.gap_count || 0} potential gaps, ${data.review_count || 0} need review, ${data.refused_count || 0} insufficient evidence.`;
}

function displayAnalysis(data, statusMessage) {
  disclaimerEl.classList.remove("hidden");
  const fragment = document.createDocumentFragment();
  for (const finding of data.findings || []) {
    fragment.appendChild(renderFinding(finding));
  }
  resultsEl.replaceChildren(fragment);
  setStatus(statusMessage || buildStatusMessage(data));
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    throw new Error("No active tab found.");
  }
  return tab;
}

async function getActiveTabPage() {
  const tab = await getActiveTab();
  const response = await chrome.tabs.sendMessage(tab.id, { type: "extract-page-text" });
  if (!response?.text) {
    throw new Error("Could not extract page text.");
  }
  return response;
}

async function storeAnalysisInPage(url, data, statusMessage) {
  const tab = await getActiveTab();
  await chrome.tabs.sendMessage(tab.id, {
    type: "store-analysis-result",
    url,
    data,
    statusMessage,
  });
}

async function restoreCachedAnalysis() {
  try {
    const tab = await getActiveTab();
    const cached = await chrome.tabs.sendMessage(tab.id, { type: "get-cached-analysis" });
    if (cached?.data) {
      displayAnalysis(cached.data, cached.statusMessage);
    }
  } catch {
    // No content script on this page (e.g. chrome:// URLs).
  }
}

analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  resultsEl.replaceChildren();
  disclaimerEl.classList.add("hidden");
  setStatus("Extracting page text...");

  try {
    await saveSettings();
    const page = await getActiveTabPage();
    const apiUrl = apiUrlInput.value.trim().replace(/\/$/, "");
    const headers = { "Content-Type": "application/json" };
    if (apiKeyInput.value.trim()) {
      headers["X-API-Key"] = apiKeyInput.value.trim();
    }

    setStatus("Analyzing policy...");
    const response = await fetch(`${apiUrl}/analyze`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        page_text: page.text,
        url: page.url,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(errorBody || `Request failed (${response.status})`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    const statusMessage = buildStatusMessage(data);
    displayAnalysis(data, statusMessage);
    await storeAnalysisInPage(page.url, data, statusMessage);
  } catch (error) {
    setStatus(error.message || "Analysis failed.");
  } finally {
    analyzeBtn.disabled = false;
  }
});

loadSettings().then(() => restoreCachedAnalysis());
