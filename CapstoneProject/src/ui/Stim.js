// 1) CONSTANTS + DOM REFS
const API_BASE = "http://127.0.0.1:7777";
// Log object <section id="log"> in which we can write msgs
const elLog = document.getElementById("log");
// Requests for POST
const elConnection = document.getElementById("connection-status");
const elConnectionLabel = document.getElementById("connection-label");
const elRefreshLabel = document.getElementById("refresh-label");
// State store
const elSeq = document.getElementById("seq-value");
const elStimWin = document.getElementById("stim-window-value");
const elStimCaption = document.getElementById("stim-window-caption");
const elBlock = document.getElementById("block-value");
const elFreqHz = document.getElementById("freq-hz-value");
// UI Pills (rounded elements that show short pieces of info like labels/statuses)
const elFreqCodePill = document.getElementById("freq-code-pill");
// Buttons
const btnRefresh = document.getElementById("btn-refresh");
const btnTogglePoll = document.getElementById("btn-toggle-poll");

// Timer for browser requests to server
let pollInterval = null;
let pollActive = true;

// 2) LOGGING HELPER
function logLine(msg) {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = "log-entry";
  line.textContent = `[${time}] ${msg}`;
  elLog.appendChild(line);
  elLog.scrollTop = elLog.scrollHeight;
}

// 3) CONNECTION STATUS HELPER
// UI should show red/green based on C++ server connection status
function setConnectionStatus(ok) {
  if (ok) {
    elConnection.classList.add("connected");
    elConnectionLabel.textContent = "Connected";
  } else {
    elConnection.classList.remove("connected");
    elConnectionLabel.textContent = "Disconnected";
  }
}

const allowed_enums = ["stim_window", "freq_hz_e"];
// 4) INT <-> ENUM HELPER
// Must match enums in types.h
function intToLabel(enumType, integer) {
  if (!allowed_enums.includes(enumType)) {
    return "error";
  }
  switch (enumType) {
    case "stim_window":
      switch (integer) {
        case 0:
          return "StimState_Active_Run";
        case 1:
          return "StimState_Active_Calib";
        case 2:
          return "StimState_Instructions";
        case 3:
          return "StimState_None";
      }
    case "freq_hz_e":
      switch (integer) {
        case 0:
          return "TestFreq_None";
        case 1:
          return "TestFreq_8_Hz";
        case 2:
          return "TestFreq_9_Hz";
        case 3:
          return "TestFreq_10_Hz";
        case 4:
          return "TestFreq_11_Hz";
        case 5:
          return "TestFreq_12_Hz";
      }
    default:
      // handle bad entries
      return `Unknown (${integer})`;
  }
}

// 5) START POLLING (GET /state)
async function pollStateOnce() {
  let res;
  try {
    // 5.1.) use fetch() to send GET request to '${API_BASE}/state'
    res = await fetch(`${API_BASE}/state`); // 'await' = non-blocking; comes back here from other tasks when ready
    // 5.2.) check/log response ok
    setConnectionStatus(res.ok);
    if (!res.ok) {
      logLine("GET /state failed.");
      return;
    }
    // 5.3.) parse json & update dom
    const data = await res.json();
    elSeq.textContent = data.seq ?? "—";
    elBlock.textContent = data.block_id ?? "0";
    elFreqHz.textContent = data.freq_hz ?? "—";
    // Enum mappings
    stim_window_label = intToLabel("stim_window", data.stim_window);
    freq_hz_label = intToLabel("freq_hz_e", data.freq_hz_e);
    elStimWin.textContent = stim_window_label ?? "—";
    elFreqCodePill.textContent = freq_hz_label ?? "—";
    console.log("STATE:", data);
  } catch (err) {
    logLine("GET /state error: " + err);
  }
}

function startPolling() {
  const polling_period_ms = 100;
  if (pollInterval != null) {
    return false;
  }
  // repetitive polling calls (send GET requests every 100ms)
  pollInterval = setInterval(pollStateOnce, polling_period_ms);
}

function stopPolling() {
  if (pollActive == false) {
    return false;
  }
  clearInterval(pollInterval);
  pollInterval = null;
}

// 6) MONITOR REFRESH MEASUREMENT (POST /ready)
// Avg durationMs takes in period which will be used to measure refresh freq
// returned as a promise (from callback -> outer fn)
function estimateRefreshHz(durationMs = 1000) {
  return new Promise((resolve) => {
    // set refresh-label text
    elRefreshLabel.textContent = "Measuring monitor refresh rate...";
    // freq at which callback is called from requestAnimationFrame matches refresh freq
    const start_time = performance.now();
    let frames = 0;
    function onAnimFrame() {
      if (performance.now() > start_time + durationMs) {
        // done measurement - compute
        const estimated_refresh_hz = Math.round(frames / (durationMs / 1000));
        resolve(estimated_refresh_hz); // done: signal promise resolved.
        return;
      } else {
        frames += 1;
        nextFrameTime = requestAnimationFrame(onAnimFrame);
      }
    }
    // start measurement loop
    requestAnimationFrame(onAnimFrame);
  });
}

async function sendRefresh(refreshHz) {
  elRefreshLabel.textContent = `Sending monitor refresh rate: ${refreshHz} Hz`;
  // 1) build a JSON { refresh_hz: refreshHz }
  const payload = { refresh_hz: refreshHz };

  // 2) call fetch with POST method
  try {
    const res = await fetch(`${API_BASE}/ready`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      logLine(`POST /ready failed with HTTP ${res.status}`);
      elRefreshLabel.textContent = `Failed to send refresh rate (HTTP ${res.status})`;
      return;
    }
    logLine(`POST /ready ok (refresh_hz=${refreshHz})`);
    elRefreshLabel.textContent = `Monitor refresh ≈ ${refreshHz} Hz (sent to backend)`;
  } catch (err) {
    logLine(`POST /ready error: ${err}`);
    elRefreshLabel.textContent = `Failed to send refresh rate (network error)`;
  }
}

// 6) WINDOWS

// 7) INIT ON PAGE LOAD
async function init() {
  const estimated_refresh = await estimateRefreshHz();
  await sendRefresh(estimated_refresh);
  startPolling();
  // setupButtons() (TODO)
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
