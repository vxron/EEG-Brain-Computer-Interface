// ================ 1) CONSTANTS + DOM REFS =================================
const API_BASE = "http://127.0.0.1:7777";
// Log object <section id="log"> in which we can write msgs
const elLog = document.getElementById("log");

// Requests for POST
const elConnection = document.getElementById("connection-status");
const elConnectionLabel = document.getElementById("connection-label");
const elRefreshLabel = document.getElementById("refresh-label");

// Flicker animation DOM elements
const elCalibBlock = document.getElementById("calib-block");
const elLeftArrow = document.getElementById("left-arrow");
const elRightArrow = document.getElementById("right-arrow");

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
const viewRunOptions = document.getElementById("view-run-options");
const viewSavedSessions = document.getElementById("view-saved-sessions");

// Instructions specific fields for instruction windows (fillable by state store info)
const elInstrBlockId = document.getElementById("instr-block-id");
const elInstrFreqHz = document.getElementById("instr-freq-hz");
const elInstructionsText = document.getElementById("instructions-text");

// Run options-specific fields (fillable by state store info)
const elRunWelcomeName = document.getElementById("run-welcome-name");
const elRunLastSubject = document.getElementById("run-last-subject");
const elRunModelStatus = document.getElementById("run-model-status");
const elSessionsEmpty = document.getElementById("sessions-empty");
const elSessionsList = document.getElementById("sessions-list");

// Session control buttons (UI /event HOOKUP to tell c++)
const btnStartCalib = document.getElementById("btn-start-calib");
const btnStartRun = document.getElementById("btn-start-run");
const btnExit = document.getElementById("btn-exit");
const btnRunStartDefault = document.getElementById("btn-run-start-default");
const btnRunSavedSessions = document.getElementById("btn-run-saved-sessions");
const btnSessionsNew = document.getElementById("btn-sessions-new");
const btnSessionsBack = document.getElementById("btn-sessions-back");

// Modal (POPUP) DOM elements
const elModalBackdrop = document.getElementById("modal-backdrop");
const elModalTitle = document.getElementById("modal-title");
const elModalBody = document.getElementById("modal-body");
const btnModalOk = document.getElementById("modal-ok"); // ack btn for user to accept popup
// Track whether popup is currently visible
let modalVisible = false;

// Timer for browser requests to server
let pollInterval = null;
let pollActive = true;

// FlickerStimulus instances
let calibStimulus = null;
let leftStimulus = null;
let rightStimulus = null;
let stimAnimId = null;

// ==================== 2) LOGGING HELPER =============================
function logLine(msg) {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = "log-entry";
  line.textContent = `[${time}] ${msg}`;
  elLog.appendChild(line);
  elLog.scrollTop = elLog.scrollHeight;
}

// ======================== 3) VIEW HELPERS =========================
// (1) show the correct stim window when it's time by removing it from 'hidden' css class
function showView(name) {
  const allViews = [
    viewHome,
    viewInstructions,
    viewActiveCalib,
    viewActiveRun,
    viewRunOptions,
    viewSavedSessions,
  ];

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
    case "run_options":
      viewRunOptions.classList.remove("hidden");
      break;
    case "saved_sessions":
      viewSavedSessions.classList.remove("hidden");
      break;
    default:
      viewHome.classList.remove("hidden");
      break;
  }
}

// (2) set full screen in calib/run modes (hide side bar & log panel)
function setFullScreenMode(enabled) {
  // Toggle a class on <body> so CSS can handle layout
  document.body.classList.toggle("fullscreen-mode", enabled);

  if (btnExit) {
    if (enabled) {
      btnExit.classList.remove("hidden");
    } else {
      btnExit.classList.add("hidden");
    }
  }
}

// (3) popup handling (helpers to show and hide popup)
function showModal(title, body) {
  if (elModalTitle && title) elModalTitle.textContent = title;
  if (elModalBody && body) elModalBody.textContent = body;

  if (elModalBackdrop) {
    elModalBackdrop.classList.remove("hidden");
    modalVisible = true;
  }
}

function hideModal() {
  if (elModalBackdrop) {
    elModalBackdrop.classList.add("hidden");
  }
  modalVisible = false;
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
          return "UIState_Saved_Sessions";
        case 5:
          return "UIState_Run_Options";
        case 6:
          return "UIState_Hardware_Checks";
        case 7:
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

// lil helpers to make zeros from statestore show as dashes :,)
function fmtFreqHz(val) {
  // treat <=0 as "no frequency"
  if (val == null || val <= 0) return "—";
  return String(val);
}

function fmtFreqEnumLabel(enumType, intVal) {
  if (intVal == null) return "—";
  // for enum "None" / code 0, show dash instead of the literal label
  if (enumType === "freq_hz_e" && intVal === 0) {
    return "—";
  }
  const label = intToLabel(enumType, intVal);
  if (!label || label.startsWith("Unknown")) return "—";
  return label;
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
  // populate sidebar fields
  elSeq.textContent = data.seq ?? "-";
  elBlock.textContent = data.block_id ?? "-";
  elFreqHz.textContent = fmtFreqHz(data.freq_hz);
  elStimWin.textContent = fmtFreqEnumLabel("stim_window", data.stim_window);
  elFreqCodePill.textContent = fmtFreqEnumLabel("freq_hz_e", data.freq_hz_e);

  // run mode flag cleared by default
  document.body.classList.remove("run-mode");

  // 0 = Active_Run, 1 = Active_Calib, 2 = Instructions, 3 = Home, 7 = None
  if (stimState === 3 /* Home */ || stimState === 7 /* None */) {
    stopCalibFlicker();
    stopRunFlicker();
    setFullScreenMode(false);
    showView("home");
  } else if (stimState === 2 /* Instructions */) {
    stopCalibFlicker();
    setFullScreenMode(true);
    showView("instructions");
    // Update text based on block and freq
    elInstrBlockId.textContent = data.block_id ?? "-";
    elInstrFreqHz.textContent = fmtFreqHz(data.freq_hz) + " Hz";
    // TODO: customize elInstructionsText based on block / upcoming freq
  } else if (stimState === 1 /* Active_Calib */) {
    setFullScreenMode(true);
    showView("active_calib");
    const calibFreqHz = data.freq_hz ?? 0;
    startCalibFlicker(calibFreqHz);
  } else if (stimState === 0 /* Active_Run */) {
    setFullScreenMode(true);
    showView("active_run");
    // set run mode flag for css to max separability btwn stimuli blocks :)
    document.body.classList.add("run-mode");
    // default to freq_hz if undef right/left
    const runLeftHz = data.freq_right_hz ?? data.freq_hz ?? 0;
    const runRightHz = data.freq_left_hz ?? data.freq_hz ?? 0;
    startRunFlicker(runLeftHz, runRightHz);
  } else if (stimState === 4 /* Saved Sessions */) {
    stopCalibFlicker();
    stopRunFlicker();
    setFullScreenMode(false);
    showView("saved_sessions");

    // TODO: render session list from backend
  } else if (stimState === 5 /* Run Options */) {
    stopCalibFlicker();
    stopRunFlicker();
    setFullScreenMode(false);
    showView("run_options");

    // TODO: SET THIS UP (populating welcome info from state store)
    const subj = data.active_subject_id || "friend";
    elRunWelcomeName.textContent = subj;
    elRunLastSubject.textContent = subj;
    const modelReady = data.is_model_ready;
    elRunModelStatus.textContent = modelReady
      ? "Model ready"
      : "No trained model yet, please run calibration";
  }

  // HANDLE POPUPS
  const popupEnumIdx = data.popup ?? 0; // 0 is fallback

  // if backend says "show popup" and it's not visible, open it
  if (popupEnumIdx != 0 && !modalVisible) {
    switch (popupEnumIdx) {
      case 1: // UIPopup_MustCalibBeforeRun
        showModal(
          "No trained models found",
          "Please complete at least one calibration session before trying to start run mode."
        );
        break;
      default:
        showModal(
          "DEBUG MSG",
          "we should not reach here! check that UIPopup Enum matches JS cases"
        );
        break;
    }
  }

  // probably never need this since js handles hidemodal from ack btn press, but just in case:
  // hide popup if backend says its time to hide (e.g. state change)
  if (popupEnumIdx == 0 && modalVisible) {
    hideModal();
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

// ===================== 9) FLICKER STIMULUS CLASS ===================================
let measuredRefreshHz = 60; // will be overwritten by estimateRefreshHz() in init

class FlickerStimulus {
  constructor(el, refreshHz) {
    // el = dom element
    this.el = el;
    this.refreshHz = refreshHz;
    this.targetHz = 0;
    this.enabled = false;
    this.frameIdx = 0;
    this.framesPerCycle = 1;
    // base styles declarations so we can modulate later via filter
    if (this.el) {
      this.el.style.transition = "filter 0.0s";
    }
  }

  // methods
  setRefreshHz(refreshHz) {
    this.refreshHz = refreshHz;
    this._recomputeFramesPerCycle();
  }
  setFrequency(hz) {
    this.targetHz = hz || 0;
    this._recomputeFramesPerCycle();
  }
  _recomputeFramesPerCycle() {
    /* 
    to get flicker at f Hz: choose framesPerCycle = refreshHz / f
    */
    if (this.targetHz > 0 && this.refreshHz > 0) {
      const raw = this.refreshHz / this.targetHz;
      // minimum 2 phases per cycle (pure on/off)
      this.framesPerCycle = Math.max(2, Math.round(raw));
    } else {
      // no flicker
      this.framesPerCycle = 1;
    }
  }
  start() {
    this.enabled = true;
    this.frameIdx = 0;
    if (this.el) {
      this.el.style.visibility = "visible";
      this.el.style.filter = "brightness(1.0)";
    }
  }
  stop() {
    this.enabled = false;
    this.frameIdx = 0;
    if (this.el) {
      // Reset to neutral appearance when not flickering
      this.el.style.filter = "brightness(1.0)";
    }
  }
  // frequencymodulator
  onePeriod() {
    if (!this.enabled || !this.el || this.targetHz <= 0) return;
    this.frameIdx = (this.frameIdx + 1) % this.framesPerCycle;

    const half = this.framesPerCycle / 2;
    const on = this.frameIdx < half;

    // square wave: ON = bright, OFF = dim
    if (on) {
      this.el.style.filter = "brightness(1.6)";
    } else {
      this.el.style.filter = "brightness(0.2)";
    }
  }
}

// ================== 10) Flicker animation starters/stoppers ==========================

function stimAnimationLoop() {
  if (calibStimulus) calibStimulus.onePeriod();
  if (leftStimulus) leftStimulus.onePeriod();
  if (rightStimulus) rightStimulus.onePeriod();

  stimAnimId = requestAnimationFrame(stimAnimationLoop);
}

function startCalibFlicker(freqHz) {
  if (!calibStimulus) return;
  calibStimulus.setRefreshHz(measuredRefreshHz);
  calibStimulus.setFrequency(freqHz);
  calibStimulus.start();

  // Make sure run-mode stimuli are off
  if (leftStimulus) leftStimulus.stop();
  if (rightStimulus) rightStimulus.stop();
}
function stopCalibFlicker() {
  if (calibStimulus) calibStimulus.stop();
}

function startRunFlicker(leftFreqHz, rightFreqHz) {
  if (leftStimulus) {
    leftStimulus.setRefreshHz(measuredRefreshHz);
    leftStimulus.setFrequency(leftFreqHz);
    leftStimulus.start();
  }
  if (rightStimulus) {
    rightStimulus.setRefreshHz(measuredRefreshHz);
    rightStimulus.setFrequency(rightFreqHz);
    rightStimulus.start();
  }
  if (calibStimulus) calibStimulus.stop();
}
function stopRunFlicker() {
  if (leftStimulus) leftStimulus.stop();
  if (rightStimulus) rightStimulus.stop();
}

// =============== 11) SEND POST EVENTS WHEN USER CLICKS BUTTONS (OR OTHER INPUTS) ===============
// Helper to send a session event to C++
async function sendSessionEvent(kind) {
  // IMPORTANT: kind is defined sporadically in init(), e.g. "start_calib", "start_run"
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

// ================== 12) INIT ON PAGE LOAD ===================
async function init() {
  logLine("Initializing UI…");
  const estimated_refresh = await estimateRefreshHz();
  measuredRefreshHz = estimated_refresh; // global write
  await sendRefresh(estimated_refresh);

  // create stimulus objects now that we know refresh
  calibStimulus = new FlickerStimulus(elCalibBlock, measuredRefreshHz);
  leftStimulus = new FlickerStimulus(elLeftArrow, measuredRefreshHz);
  rightStimulus = new FlickerStimulus(elRightArrow, measuredRefreshHz);

  stimAnimationLoop();
  startPolling();

  // Add button event listeners
  btnStartCalib.addEventListener("click", () => {
    sendSessionEvent("start_calib");
  });

  btnStartRun.addEventListener("click", () => {
    sendSessionEvent("start_run");
  });

  btnExit.addEventListener("click", () => {
    sendSessionEvent("exit");
  });

  btnRunStartDefault.addEventListener("click", () => {
    // maps to UIStateEvent_UserPushesStartDefault
    sendSessionEvent("start_default");
  });

  btnRunSavedSessions.addEventListener("click", () => {
    // maps to UIStateEvent_UserPushesSessions
    sendSessionEvent("show_sessions");
  });

  btnSessionsNew.addEventListener("click", () => {
    // maps to UIStateEvent_UserSelectsNewSession
    sendSessionEvent("new_session");
  });

  btnSessionsBack.addEventListener("click", () => {
    sendSessionEvent("back_to_run_options");
  });

  if (btnModalOk) {
    // if a popup is visible, wait for user ack
    btnModalOk.addEventListener("click", () => {
      hideModal();
      // tell backend to clear popup in statestore
      sendSessionEvent("ack_popup");
    });
  }
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
