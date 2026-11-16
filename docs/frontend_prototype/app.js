"use strict";

const viewButtons = document.querySelectorAll("[data-view-target]");
const views = document.querySelectorAll("[data-view]");
const panelButtons = document.querySelectorAll("[data-panel-target]");
const panels = document.querySelectorAll("[data-panel]");
const breadcrumb = document.querySelector("[data-breadcrumb]");
const checkpoint = document.querySelector("[data-checkpoint]");
const globalNewRun = document.querySelector("[data-global-new-run]");

function setTopbar(viewName) {
  if (!breadcrumb || !checkpoint) return;
  const labels = {
    library: ["RUNS", "EVIDENCE INVENTORY"],
    "new-run": ["RUNS", "NEW EXPERIMENT", "DRAFT"],
    run: ["RUNS", "LIVE_NIHILISM_FULL", "5E67FC"],
  };
  breadcrumb.replaceChildren();
  labels[viewName].forEach((label, index) => {
    const item = document.createElement(index === labels[viewName].length - 1 ? "strong" : "span");
    item.textContent = label;
    breadcrumb.appendChild(item);
    if (index < labels[viewName].length - 1) {
      const separator = document.createElement("b");
      separator.textContent = "/";
      breadcrumb.appendChild(separator);
    }
  });
  checkpoint.hidden = viewName !== "run";
  if (globalNewRun) globalNewRun.hidden = viewName === "new-run";
}

function updateLocation(fragment) {
  if (window.location.hash !== `#${fragment}`) {
    window.history.replaceState(null, "", `#${fragment}`);
  }
}

function showView(name, updateHash = true) {
  views.forEach((view) => {
    const active = view.dataset.view === name;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  if (name !== "run") {
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.viewTarget === name && !item.dataset.panelTarget);
    });
  }
  setTopbar(name);
  if (updateHash) updateLocation(name);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showPanel(name, updateHash = true) {
  showView("run", false);
  setTopbar("run");
  panels.forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  document.querySelectorAll("[data-panel-target]").forEach((button) => {
    button.classList.toggle("active", button.dataset.panelTarget === name);
    if (button.classList.contains("nav-item")) {
      button.setAttribute("aria-current", button.dataset.panelTarget === name ? "page" : "false");
    }
  });
  if (updateHash) updateLocation(`run/${name}`);
}

viewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const panel = button.dataset.panelTarget;
    if (panel) showPanel(panel);
    else showView(button.dataset.viewTarget);
  });
});

panelButtons.forEach((button) => {
  button.addEventListener("click", () => showPanel(button.dataset.panelTarget));
});

const atlasGrid = document.querySelector("[data-atlas-grid]");
const occupied = new Set([1, 4, 6, 7, 9, 12, 13, 17, 18, 21, 22]);
const high = new Set([6, 17, 22]);
if (atlasGrid) {
  for (let index = 0; index < 24; index += 1) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "atlas-cell";
    cell.dataset.code = String(index + 1).padStart(2, "0");
    cell.setAttribute("aria-label", `Descriptor cell ${index + 1}${high.has(index) ? ", high observation" : occupied.has(index) ? ", observed" : ", empty"}`);
    if (occupied.has(index)) cell.classList.add("occupied");
    if (high.has(index)) cell.classList.add("high-cell");
    atlasGrid.appendChild(cell);
  }
}

const routeTabs = document.querySelectorAll(".route-tabs button");
routeTabs.forEach((button) => {
  button.addEventListener("click", () => {
    routeTabs.forEach((item) => {
      item.classList.remove("active");
      item.setAttribute("aria-selected", "false");
    });
    button.classList.add("active");
    button.setAttribute("aria-selected", "true");
  });
});

document.querySelectorAll(".branch-fork button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".branch-fork button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
  });
});

const wizardPages = [...document.querySelectorAll("[data-wizard-page]")];
const wizardSteps = [...document.querySelectorAll(".wizard-steps li")];
const nextButton = document.querySelector("[data-wizard-next]");
const backButton = document.querySelector("[data-wizard-back]");
const stepNumber = document.querySelector("[data-step-number]");
let wizardIndex = 0;

function renderWizard() {
  wizardPages.forEach((page, index) => {
    const active = index === wizardIndex;
    page.hidden = !active;
    page.classList.toggle("active", active);
  });
  wizardSteps.forEach((step, index) => step.classList.toggle("active", index === wizardIndex));
  stepNumber.textContent = String(wizardIndex + 1).padStart(2, "0");
  backButton.disabled = wizardIndex === 0;
  nextButton.textContent = wizardIndex === wizardPages.length - 1 ? "Launch reviewed run →" : "Continue →";
}

nextButton?.addEventListener("click", () => {
  if (wizardIndex < wizardPages.length - 1) {
    wizardIndex += 1;
    renderWizard();
    updateLocation(`new-run/${wizardIndex + 1}`);
  }
});
backButton?.addEventListener("click", () => {
  if (wizardIndex > 0) {
    wizardIndex -= 1;
    renderWizard();
    updateLocation(`new-run/${wizardIndex + 1}`);
  }
});

const jsonDisclosure = document.querySelector("[data-json-disclosure]");
const configJson = document.querySelector("[data-config-json]");
jsonDisclosure?.addEventListener("click", () => {
  const opening = configJson.hidden;
  configJson.hidden = !opening;
  jsonDisclosure.setAttribute("aria-expanded", String(opening));
  jsonDisclosure.textContent = opening ? "▾ Hide complete JSON configuration" : "▸ Inspect complete JSON configuration";
});

document.querySelectorAll(".objective-card").forEach((card) => {
  card.addEventListener("click", () => {
    document.querySelectorAll(".objective-card").forEach((item) => item.classList.remove("selected"));
    card.classList.add("selected");
  });
});

renderWizard();

const initialRoute = window.location.hash.slice(1);
if (initialRoute.startsWith("run/")) {
  showPanel(initialRoute.split("/")[1] || "overview", false);
} else if (initialRoute.startsWith("new-run/")) {
  const requestedStep = Number.parseInt(initialRoute.split("/")[1], 10);
  wizardIndex = Number.isInteger(requestedStep) ? Math.min(Math.max(requestedStep - 1, 0), wizardPages.length - 1) : 0;
  renderWizard();
  showView("new-run", false);
} else if (["library", "new-run"].includes(initialRoute)) {
  showView(initialRoute, false);
}
