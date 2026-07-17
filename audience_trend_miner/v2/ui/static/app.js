"use strict";

const form = document.querySelector("#run-form");
const asOfInput = document.querySelector("#as-of");
const startButton = document.querySelector("#start-run");
const newRunButton = document.querySelector("#new-run");
const runStatus = document.querySelector("#run-status");
const progressSection = document.querySelector("#progress");
const progressFeed = document.querySelector("#progress-feed");
const followButton = document.querySelector("#follow-progress");
const cancelButton = document.querySelector("#cancel-run");
const portfolioSection = document.querySelector("#portfolio");
const portfolioDates = document.querySelector("#portfolio-dates");
const emptyPortfolio = document.querySelector("#empty-portfolio");
const moduleTemplate = document.querySelector("#module-template");
const eventTemplate = document.querySelector("#event-template");
const cardTemplate = document.querySelector("#audience-card-template");

let activeRun = null;
let lastSequence = 0;
let socket = null;
let streamWanted = false;
let followProgress = true;
let lastProgressScrollTop = 0;
let pollGeneration = 0;
let dateSelectionGeneration = 0;
const modulePanels = new Map();
const directions = [
  {
    contractValues: ["robust_growth", "sudden_growth"],
    selector: "#growing-audiences",
    label: "Growing",
    icon: "↗",
    className: "growing",
  },
  {
    contractValues: ["robust_shrinking"],
    selector: "#shrinking-audiences",
    label: "Shrinking",
    icon: "↘",
    className: "shrinking",
  },
];

const localToday = new Date();
localToday.setMinutes(localToday.getMinutes() - localToday.getTimezoneOffset());
asOfInput.value = localToday.toISOString().slice(0, 10);

asOfInput.addEventListener("change", () => {
  loadCompletedDate(asOfInput.value, ++dateSelectionGeneration);
});

progressFeed.addEventListener(
  "scroll",
  () => {
    const distanceFromBottom =
      progressFeed.scrollHeight - progressFeed.clientHeight - progressFeed.scrollTop;
    if (distanceFromBottom < 40) {
      setFollowing(true);
    } else if (progressFeed.scrollTop < lastProgressScrollTop - 4) {
      setFollowing(false);
    }
    lastProgressScrollTop = progressFeed.scrollTop;
  },
  { passive: true },
);

followButton.addEventListener("click", () => {
  setFollowing(true);
  scrollToLatest();
});

newRunButton.addEventListener("click", () => {
  stopStreaming();
  pollGeneration += 1;
  activeRun = null;
  lastSequence = 0;
  lastProgressScrollTop = 0;
  progressFeed.replaceChildren();
  modulePanels.clear();
  progressSection.hidden = true;
  portfolioSection.hidden = true;
  newRunButton.hidden = true;
  startButton.hidden = false;
  asOfInput.value = "";
  startButton.querySelector("span").textContent = "Start new run";
  setStatus("Choose an As-of Date for the new run.");
  asOfInput.focus();
});

cancelButton.addEventListener("click", async () => {
  const runId = activeRun?.id;
  if (!runId) return;
  const confirmed = window.confirm(
    `Cancel ${runId}? Completed artifacts and progress history will be kept.`,
  );
  if (!confirmed) return;
  cancelButton.disabled = true;
  setStatus(`Cancelling ${runId}…`);
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(runId)}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmed: true }),
      },
    );
    if (!response.ok) throw new Error(await responseDetail(response));
    stopStreaming();
    pollGeneration += 1;
    finishControls(true);
    setStatus(`${runId} cancelled. Progress and completed work were retained.`);
  } catch (error) {
    cancelButton.disabled = false;
    setStatus(error.message || "The run could not be cancelled.", true);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const asOf = asOfInput.value;
  const runId = runIdForDate(asOf);
  if (activeRun?.id === runId && activeRun.status === "succeeded") return;
  const isNewRun = activeRun?.id !== runId;
  beginRun(runId, asOf, isNewRun);

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, as_of: asOf }),
    });
    if (!response.ok && response.status !== 409) {
      throw new Error(await responseDetail(response));
    }
    setStatus(
      response.status === 409
        ? `Reconnected to ${runId}. Live progress is continuing.`
        : `${runId} started. Following live progress.`,
    );
    connectEventStream(runId);
    pollRun(runId, ++pollGeneration);
  } catch (error) {
    stopStreaming();
    finishControls(true);
    setStatus(error.message || "The run could not be started.", true);
  }
});

function beginRun(runId, asOf, clearExisting) {
  stopStreaming();
  pollGeneration += 1;
  activeRun = { id: runId, asOf, status: "running" };
  if (clearExisting) {
    lastSequence = 0;
    progressFeed.replaceChildren();
    modulePanels.clear();
  }
  portfolioSection.hidden = true;
  progressSection.hidden = false;
  startButton.hidden = false;
  startButton.disabled = true;
  newRunButton.hidden = true;
  cancelButton.disabled = false;
  cancelButton.hidden = false;
  startButton.querySelector("span").textContent = "Run in progress";
  setStatus(`Starting ${runId}…`);
  setFollowing(true);
}

function finishControls(retry) {
  startButton.hidden = false;
  startButton.disabled = false;
  newRunButton.hidden = false;
  cancelButton.hidden = true;
  cancelButton.disabled = false;
  startButton.querySelector("span").textContent = retry
    ? "Retry or resume run"
    : "Start or resume run";
}

function finishCompletedControls() {
  startButton.disabled = false;
  startButton.hidden = true;
  newRunButton.hidden = false;
  cancelButton.hidden = true;
  cancelButton.disabled = false;
}

function setStatus(message, isError = false) {
  runStatus.textContent = message;
  runStatus.classList.toggle("error", isError);
}

async function responseDetail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (_) {
    // The public fallback below deliberately avoids surfacing response internals.
  }
  return "The run request was unsuccessful.";
}

function connectEventStream(runId) {
  streamWanted = true;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const openedSocket = new WebSocket(
    `${protocol}://${window.location.host}/api/runs/${encodeURIComponent(runId)}/events?after_sequence=${lastSequence}`,
  );
  socket = openedSocket;
  openedSocket.addEventListener("message", (message) => {
    const event = JSON.parse(message.data);
    lastSequence = Math.max(lastSequence, event.sequence);
    appendProgressEvent(event);
  });
  openedSocket.addEventListener("close", () => {
    if (socket !== openedSocket) return;
    socket = null;
    if (streamWanted && activeRun?.id === runId) {
      window.setTimeout(() => connectEventStream(runId), 700);
    }
  });
}

function stopStreaming() {
  streamWanted = false;
  if (socket) {
    socket.close();
    socket = null;
  }
}

async function pollRun(runId, generation) {
  while (activeRun?.id === runId && generation === pollGeneration) {
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
      if (!response.ok) throw new Error("Run status is temporarily unavailable.");
      const state = await response.json();
      if (state.status === "succeeded") {
        stopStreaming();
        activeRun.status = "succeeded";
        setStatus(`${runId} completed. Audience Portfolio is ready.`);
        finishCompletedControls();
        await loadPortfolio(runId);
        return;
      }
      if (state.status === "failed") {
        stopStreaming();
        const message = state.failure?.message || "The run ended unsuccessfully.";
        appendProgressEvent({
          module: "run",
          operation: "failed",
          level: "error",
          message,
          timestamp: new Date().toISOString(),
        });
        setStatus(`${runId} failed. ${message} Progress is retained for retry.`, true);
        finishControls(true);
        return;
      }
      if (state.status === "cancelled") {
        stopStreaming();
        setStatus(`${runId} cancelled. Progress and completed work were retained.`);
        finishControls(true);
        return;
      }
    } catch (error) {
      setStatus(error.message || "Run status is temporarily unavailable.", true);
    }
    await new Promise((resolve) => window.setTimeout(resolve, 700));
  }
}

function appendProgressEvent(event) {
  const panel = modulePanel(event.module);
  const item = eventTemplate.content.firstElementChild.cloneNode(true);
  if (Number.isInteger(event.sequence)) {
    item.dataset.sequence = String(event.sequence);
  }
  const eventLevel =
    event.level === "warning" || event.level === "error" ? event.level : "info";
  const presentation = event.operation === "retry" ? "retry" : eventLevel;
  const levelLabels = {
    info: "Update",
    retry: "Retry",
    warning: "Warning",
    error: "Error",
  };
  item.classList.add(presentation);
  item.querySelector(".event-level").textContent = levelLabels[presentation];
  item.querySelector(".event-operation").textContent = humanize(event.operation);
  item.querySelector(".event-message").textContent = event.message;

  const time = item.querySelector("time");
  time.dateTime = event.timestamp;
  const parsedTime = new Date(event.timestamp);
  time.textContent = Number.isNaN(parsedTime.valueOf())
    ? "Time unavailable"
    : parsedTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  let moduleComplete = false;
  if (event.progress) {
    const progress = item.querySelector(".event-progress");
    const meter = progress.querySelector("progress");
    progress.hidden = false;
    meter.max = event.progress.total;
    meter.value = event.progress.current;
    meter.setAttribute(
      "aria-label",
      `${humanize(event.operation)}: ${event.progress.current} of ${event.progress.total}`,
    );
    progress.querySelector("span").textContent =
      event.progress.current === event.progress.total
        ? `${event.progress.current} of ${event.progress.total} · Complete`
        : `${event.progress.current} of ${event.progress.total}`;
    moduleComplete =
      event.progress.current === event.progress.total &&
      (event.operation === "publish" || event.operation === "resume");
  }

  panel.list.append(item);
  panel.count += 1;
  panel.complete ||= moduleComplete;
  panel.element.classList.toggle("complete", panel.complete);
  panel.element.querySelector(".module-count").textContent =
    `${panel.count} ${panel.count === 1 ? "event" : "events"}` +
    (panel.complete ? " · Complete" : "");
  setStatus(
    `${humanize(event.module)}: ${levelLabels[presentation]} — ${event.message}`,
    eventLevel === "error",
  );
  if (followProgress) window.requestAnimationFrame(scrollToLatest);
}

function modulePanel(module) {
  if (modulePanels.has(module)) return modulePanels.get(module);
  const element = moduleTemplate.content.firstElementChild.cloneNode(true);
  element.querySelector("h3").textContent = humanize(module);
  progressFeed.append(element);
  const panel = {
    element,
    list: element.querySelector(".event-list"),
    count: 0,
    complete: false,
  };
  modulePanels.set(module, panel);
  return panel;
}

function humanize(value) {
  return String(value).replaceAll("-", " ");
}

function setFollowing(value) {
  followProgress = value;
  followButton.hidden = value;
}

function scrollToLatest() {
  progressFeed.scrollTop = progressFeed.scrollHeight;
  lastProgressScrollTop = progressFeed.scrollTop;
}

function runIdForDate(asOf) {
  return `run-${asOf}`;
}

async function loadCompletedDate(asOf, generation) {
  if (!asOf) return;
  const runId = runIdForDate(asOf);
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/portfolio`);
    if (generation !== dateSelectionGeneration) return;
    if (response.status === 404) {
      if (activeRun?.status === "succeeded") activeRun = null;
      portfolioSection.hidden = true;
      startButton.hidden = false;
      newRunButton.hidden = true;
      startButton.querySelector("span").textContent = "Start or resume run";
      setStatus(`No completed run found for ${asOf}. Ready to run.`);
      return;
    }
    if (!response.ok) throw new Error(await responseDetail(response));
    stopStreaming();
    pollGeneration += 1;
    activeRun = { id: runId, asOf, status: "succeeded" };
    progressSection.hidden = true;
    finishCompletedControls();
    renderPortfolio(await response.json());
    setStatus(`${runId} completed. Audience Portfolio is ready.`);
  } catch (error) {
    if (generation !== dateSelectionGeneration) return;
    setStatus(error.message || "The completed portfolio could not be loaded.", true);
  }
}

async function loadPortfolio(runId) {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/portfolio`);
    if (!response.ok) throw new Error(await responseDetail(response));
    renderPortfolio(await response.json());
  } catch (error) {
    setStatus(error.message || "The completed portfolio could not be loaded.", true);
  }
}

function renderPortfolio(portfolio) {
  const previous = portfolio.nominal_windows.previous;
  const current = portfolio.nominal_windows.current;
  portfolioDates.textContent =
    `As of ${formatDate(portfolio.as_of_date)} · ` +
    `Previous ${formatDate(previous.start)}–${formatDate(previous.end)} · ` +
    `Current ${formatDate(current.start)}–${formatDate(current.end)}`;

  for (const direction of directions) {
    renderDirection(
      direction,
      portfolio.audience_portfolio.filter(
        (audience) => direction.contractValues.includes(audience.direction),
      ),
    );
  }
  emptyPortfolio.hidden = portfolio.audience_portfolio.length !== 0;
  portfolioSection.hidden = false;
  portfolioSection.scrollIntoView({ block: "start" });
}

function renderDirection(direction, audiences) {
  const section = document.querySelector(direction.selector);
  const grid = section.querySelector(".card-grid");
  grid.replaceChildren();
  section.hidden = audiences.length === 0;
  for (const audience of audiences) {
    const card = cardTemplate.content.firstElementChild.cloneNode(true);
    card.classList.add(direction.className);
    const sudden = audience.direction === "sudden_growth";
    card.querySelector(".direction-badge").textContent = sudden
      ? "⚡ Suddenly trending"
      : `${direction.icon} ${direction.label}`;
    card.querySelector(".percentage-change").textContent = formatChange(
      audience.percentage_change,
      sudden,
    );
    card.querySelector("h4").textContent = audience.narrative.name;
    card.querySelector(".trend-summary").textContent = audience.narrative.summary;
    card.querySelector(".previous-coverage").textContent = formatCoverage(
      audience.coverage.previous,
    );
    card.querySelector(".current-coverage").textContent = formatCoverage(
      audience.coverage.current,
    );
    card.querySelector(".commercial-interpretation").textContent =
      audience.narrative.commercial_interpretation;
    card.querySelector(".buying-power-rating").textContent = humanize(
      audience.narrative.buying_power_rating,
    );
    card.querySelector(".buying-power-rationale").textContent =
      audience.narrative.buying_power_rationale;
    grid.append(card);
  }
}

function formatChange(value, suddenlyTrending = false) {
  if (suddenlyTrending) return "Suddenly trending";
  if (value === null) return "Change not available";
  return `${new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 1,
    signDisplay: "always",
  }).format(value)}% change`;
}

function formatCoverage(value) {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate(value) {
  const parsed = new Date(`${value}T00:00:00`);
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(parsed);
}
