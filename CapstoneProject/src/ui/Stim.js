// ================ 1) CONSTANTS + DOM REFS =================================
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
const elBlock = document.getElementById("block-value");
const elFreqHz = document.getElementById("freq-hz-value");

// UI Pills (rounded elements that show short pieces of info like labels/statuses)
const elFreqCodePill = document.getElementById("freq-code-pill");

// VIEW CONTAINERS FOR DIFF WINDOWS
const viewHome = document.getElementById("view-home");
const viewInstructions = document.getElementById("view-instructions");
const viewActiveCalib = document.getElementById("view-active-calib");
const viewActiveRun = document.getElementById("view-active-run");

// Instructions specific fields for instruction windows
const elInstrBlockId = document.getElementById("instr-block-id");
const elInstrFreqHz = document.getElementById("instr-freq-hz");
const elInstructionsText = document.getElementById("instructions-text");

// Session control buttons (TODO -> FOR LATER /event HOOKUP to tell c++)
const btnStartCalib = document.getElementById("btn-start-calib");
const btnStartRun = document.getElementById("btn-start-run");

// Timer for browser requests to server
let pollInterval = null;
let pollActive = true;

// ==================== 2) LOGGING HELPER =============================
function logLine(msg) {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = "log-entry";
  line.textContent = `[${time}] ${msg}`;
  elLog.appendChild(line);
  elLog.scrollTop = elLog.scrollHeight;
}

// ======================== 3) SHOWVIEW HELPER =========================
// show the correct stim window when it's time by removing it from 'hidden' css class
function showView(name) {
  const allViews = [viewHome, viewInstructions, viewActiveCalib, viewActiveRun];

  for (const v of allViews) {
    v.classList.add("hidden");
  }

  switch (name) {
    case "home":
      viewHome.classList.remove("hidden");
      break;
    case "instructions":
      viewInstructions.classList.remove("hidden");
      break;
    case "active_calib":
      viewActiveCalib.classList.remove("hidden");
      break;
    case "active_run":
      viewActiveRun.classList.remove("hidden");
      break;
    default:
      viewHome.classList.remove("hidden");
      break;
  }
}

// ==================== 4) CONNECTION STATUS HELPER =====================
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

// ============= 5) INT <-> ENUM HELPER FOR STIM WINDOWS ===============
const allowed_enums = ["stim_window", "freq_hz_e"];
// Must match enums in types.h
function intToLabel(enumType, integer) {
  if (!allowed_enums.includes(enumType)) {
    return "error";
  }
  switch (enumType) {
    case "stim_window":
      switch (integer) {
        case 0:
          return "UIState_Active_Run";
        case 1:
          return "UIState_Active_Calib";
        case 2:
          return "UIState_Instructions";
        case 3:
          return "UIState_Home";
        case 4:
          return "UIState_None";
        default:
          return `Unknown (${integer})`;
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
        default:
          return `Unknown (${integer})`;
      }
    default:
      // handle bad entries
      return `Unknown (${enumType})`;
  }
}

// ============= 6) MAP STIM_WINDOW FROM STATESTORE-> view + labels in UI ===============
function updateUiFromState(data) {
  // 1) Basic sidebar fields
  elSeq.textContent = data.seq ?? "—";
  elBlock.textContent = data.block_id ?? "0";
  elFreqHz.textContent = data.freq_hz ?? "—";

  // 2) Labels for enums
  const stimLabel = intToLabel("stim_window", data.stim_window);
  const freqCodeLbl = intToLabel("freq_hz_e", data.freq_hz_e);
  elStimWin.textContent = stimLabel ?? "—";
  elFreqCodePill.textContent = freqCodeLbl ?? "—";

  // 3) View routing based on stim_window value
  const stimState = data.stim_window;
  // 0 = Active_Run, 1 = Active_Calib, 2 = Instructions, 3 = Home, 4 = None
  if (stimState === 3 /* Home */ || stimState === 4 /* None */) {
    showView("home");
  } else if (stimState === 2 /* Instructions */) {
    showView("instructions");
    // Update text based on block and freq
    elInstrBlockId.textContent = data.block_id ?? "—";
    elInstrFreqHz.textContent = (data.freq_hz ?? "—") + " Hz";
    // TODO: customize elInstructionsText based on block / upcoming freq
  } else if (stimState === 1 /* Active_Calib */) {
    showView("active_calib");
    // TODO: call flicker animation for single block
  } else if (stimState === 0 /* Active_Run */) {
    showView("active_run");
    // TODO: call into dual arrow flicker animation
  }
}

// ============= 6) START POLLING FOR GET/STATE ===============
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
    updateUiFromState(data);
    console.log("STATE:", data);
  } catch (err) {
    logLine("GET /state error: " + err);
  }
}

function startPolling() {
  console.log("entered startPolling");
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

// =========== 7) MONITOR REFRESH MEASUREMENT (POST /ready) ==========================
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

// ================== 9) Flicker helpers ==========================

function startCalibFlicker(freqHz) {
  /* TODO */
}
function stopCalibFlicker() {
  /* TODO */
}

function startRunFlicker(leftFreqHz, rightFreqHz) {
  /* TODO */
}
function stopRunFlicker() {
  /* TODO */
}

// =============== 10) SEND POST EVENTS WHEN USER CLICKS BUTTONS (OR OTHER INPUTS) ===============
// Helper to send a session event to C++
async function sendSessionEvent(kind) {
  // IMPORTANT: kind is "start_calib" or "start_run" for now
  const payload = { action: kind };

  try {
    const res = await fetch(`${API_BASE}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      logLine(`POST /event failed (${res.status}) for action=${kind}`);
      return;
    }
    logLine(`POST /event ok (action=${kind})`);
  } catch (err) {
    logLine(`POST /event error for action=${kind}: ${err}`);
  }
}

// ================== 11) INIT ON PAGE LOAD ===================
async function init() {
  logLine("Initializing UI…");
  const estimated_refresh = await estimateRefreshHz();
  await sendRefresh(estimated_refresh);
  startPolling();
  // Add button event listeners
  btnStartCalib.addEventListener("click", () => {
    sendSessionEvent("start_calib");
  });
  btnStartRun.addEventListener("click", () => {
    sendSessionEvent("start_run");
  });
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
