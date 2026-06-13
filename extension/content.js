const CHROME_SELECTORS = [
  "nav",
  "header",
  "footer",
  "aside",
  "script",
  "style",
  "noscript",
  "[role='navigation']",
  "[role='banner']",
  "[role='contentinfo']",
  "[aria-hidden='true']",
].join(",");

function getContentRoot() {
  return (
    document.querySelector("main") ||
    document.querySelector("article") ||
    document.querySelector('[role="main"]') ||
    document.body
  );
}

function cloneContentRoot() {
  const root = getContentRoot();
  const clone = root.cloneNode(true);
  clone.querySelectorAll(CHROME_SELECTORS).forEach((element) => element.remove());
  return clone;
}

function headingLevel(tagName) {
  const match = tagName.match(/^H([1-6])$/i);
  return match ? Number(match[1]) : 0;
}

function collectSectionText(heading) {
  const parts = [];
  let sibling = heading.nextElementSibling;
  while (sibling) {
    if (headingLevel(sibling.tagName) > 0) {
      break;
    }
    const text = (sibling.innerText || sibling.textContent || "").trim();
    if (text) {
      parts.push(text);
    }
    sibling = sibling.nextElementSibling;
  }
  return parts.join("\n").trim();
}

function extractPageText() {
  const clone = cloneContentRoot();
  const headings = [...clone.querySelectorAll("h1, h2, h3, h4, h5, h6")];

  if (headings.length > 0) {
    const sections = headings
      .map((heading) => {
        const level = headingLevel(heading.tagName);
        const title = (heading.innerText || heading.textContent || "").trim();
        const body = collectSectionText(heading);
        if (!title) {
          return "";
        }
        const prefix = "#".repeat(Math.min(level, 6));
        return body ? `${prefix} ${title}\n${body}` : `${prefix} ${title}`;
      })
      .filter(Boolean);
    return sections.join("\n\n").trim();
  }

  return (clone.innerText || clone.textContent || "").trim();
}

let cachedAnalysis = null;

function clearCachedAnalysis() {
  cachedAnalysis = null;
}

function trackUrlChanges() {
  window.addEventListener("popstate", clearCachedAnalysis);
  for (const method of ["pushState", "replaceState"]) {
    const original = history[method];
    history[method] = function (...args) {
      original.apply(this, args);
      clearCachedAnalysis();
    };
  }
}

trackUrlChanges();

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "extract-page-text") {
    sendResponse({
      text: extractPageText(),
      url: window.location.href,
      title: document.title,
    });
    return true;
  }

  if (message.type === "store-analysis-result") {
    if (message.url === window.location.href) {
      cachedAnalysis = {
        url: message.url,
        data: message.data,
        statusMessage: message.statusMessage,
      };
    }
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === "get-cached-analysis") {
    if (cachedAnalysis?.url === window.location.href) {
      sendResponse(cachedAnalysis);
    } else {
      sendResponse(null);
    }
    return true;
  }

  return false;
});
