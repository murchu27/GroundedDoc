const apiUrlInput = document.getElementById("api-url");
const apiKeyInput = document.getElementById("api-key");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const disclaimerEl = document.getElementById("disclaimer");

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

function renderCitation(label, citation, cssClass) {
  if (!citation) {
    return `<div class="citation missing ${cssClass}"><strong>${label}</strong><br />No citation found.</div>`;
  }
  return `
    <div class="citation ${cssClass}">
      <strong>${label}</strong><br />
      ${citation.doc_id} §${citation.section_path}<br />
      ${citation.text || ""}
    </div>
  `;
}

function renderFinding(finding) {
  return `
    <article class="finding">
      <span class="badge ${finding.status}">${finding.status.replaceAll("_", " ")}</span>
      <h2>${finding.topic.replaceAll("_", " ")}</h2>
      <p>${finding.summary}</p>
      ${renderCitation("Policy", finding.policy_citation, "citation-policy")}
      ${renderCitation("Regulation", finding.regulation_citation, "citation-regulation")}
    </article>
  `;
}

async function getActiveTabPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    throw new Error("No active tab found.");
  }
  const response = await chrome.tabs.sendMessage(tab.id, { type: "extract-page-text" });
  if (!response?.text) {
    throw new Error("Could not extract page text.");
  }
  return response;
}

analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  resultsEl.innerHTML = "";
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

    disclaimerEl.classList.remove("hidden");
    resultsEl.innerHTML = (data.findings || []).map(renderFinding).join("");
    setStatus(
      `Done. ${data.gap_count || 0} potential gaps, ${data.review_count || 0} need review, ${data.refused_count || 0} insufficient evidence.`,
    );
  } catch (error) {
    setStatus(error.message || "Analysis failed.");
  } finally {
    analyzeBtn.disabled = false;
  }
});

loadSettings();
