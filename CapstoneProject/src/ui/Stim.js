// ================ 1) CONSTANTS + DOM REFS =================================
const API_BASE = "http://127.0.0.1:7777";
// Log object <section id="log"> in which we can write msgs
const elLog = document.getElementById("log");

// Requests for POST
const elConnection = document.getElementById("connection-status");
const elConnectionLabel = document.getElementById("connection-label");
const elRefreshLabel = document.getElementById("refresh-label");
const elStatusUiState = document.getElementById("status-ui-state");
const elStatusActiveSubject = document.getElementById("status-active-subject");
const elStatusModel = document.getElementById("status-model");

// Flicker animation DOM elements
const elCalibBlock = document.getElementById("calib-block");
const elLeftArrow = document.getElementById("left-arrow");
const elRightArrow = document.getElementById("right-arrow");

// UI Pills (rounded elements that show short pieces of info like labels/statuses)
const elFreqCodePill = document.getElementById("freq-code-pill");

// VIEW CONTAINERS FOR DIFF WINDOWS
const viewHome = document.getElementById("view-home");
const viewInstructions = document.getElementById("view-instructions");
const viewActiveCalib = document.getElementById("view-active-calib");
const viewActiveRun = document.getElementById("view-active-run");
const viewRunOptions = document.getElementById("view-run-options");
const viewSavedSessions = document.getElementById("view-saved-sessions");
const viewHardware = document.getElementById("view-hardware-checks");

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
const btnStartHw = document.getElementById("btn-start-hw");

// Health headers (above plots, saying whether or not we overall healthy)
const elHealthBadge = document.getElementById("hw-health-badge");
const elHealthLabel = document.getElementById("hw-health-label");
const elHealthRollBad = document.getElementById("hw-roll-bad");
const elHealthOverallBad = document.getElementById("hw-overall-bad");
const elHealthRollN = document.getElementById("hw-roll-n");

// Modal (POPUP) DOM elements
const elModalBackdrop = document.getElementById("modal-backdrop");
const elModalTitle = document.getElementById("modal-title");
const elModalBody = document.getElementById("modal-body");
const btnModalOk = document.getElementById("modal-ok"); // ack btn for user to accept popup
const btnModalCancel = document.getElementById("modal-cancel"); // alternate ack btn for popups w 2 options
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

// Hardware checks DOM elements
const hwQualityRow = document.getElementById("hw-quality-row");
const hwPlotsContainer = document.getElementById("hw-plots-container");
// Hardware checks plotting configs
const HW_MAX_WINDOW_SEC = 9; // seconds visible on screen
const HW_Y_MIN = -80; // adjust to fit (scale should be uV, EEG ~10-100uV)
const HW_Y_MAX = 80;
let hwCharts = []; // one Chart per channel
let hwLabels = []; // channel names, matching backend labels (from get/eeg JSON res)
let hwActive = false;
let hwAnimId = null; // frame scheduler
let hwNChannels = 0;
let hwSamplesPerCycle = 0; // how many single eeg samples fit across the plot width
let hwSampleIdxInCycle = 0; // circular index 0... hwSamplesPerCycle-1
let hwGlobalIndex = 0; // global time idx (in samples) -> keep track of total time

// Calib Options DOM elements
const viewCalibOptions = document.getElementById("view-calib-options");
const inpCalibName = document.getElementById("calib-name");
const selEpilepsy = document.getElementById("calib-epilepsy");
const btnCalibSubmit = document.getElementById("btn-calib-submit");
const btnCalibBack = document.getElementById("btn-calib-back");

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
    viewHardware,
    viewCalibOptions,
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
    case "hardware_checks":
      viewHardware.classList.remove("hidden");
      break;
    case "calib_options":
      viewCalibOptions.classList.remove("hidden");
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

// (3) start/stop hardware mode
function startHardwareMode() {
  // Mark global mode as active so hardwareLoop does work
  hwActive = true;

  // Reset time counter so x-axis (time) starts at 0 for a new session
  hwGlobalIndex = 0;

  // Kick off hardware loop
  if (!hwAnimId) {
    hwAnimId = setTimeout(hardwareLoop, 0);
  }
}
function stopHardwareMode() {
  hwActive = false;
  // Cancel the scheduled animation frame, if any
  if (hwAnimId) {
    clearTimeout(hwAnimId);
    hwAnimId = null;
  }
}

// (4) popup handling (helpers to show and hide popup)
// displays 1 button only ('OK') by default if no opts given
function showModal(title, body, opts = {}) {
  if (elModalTitle && title) elModalTitle.textContent = title;
  if (elModalBody && body) elModalBody.textContent = body;

  // if opts are given... (to customize modal w 2 buttons)
  const okText = opts.okText ?? "OK";
  const cancelText = opts.cancelText ?? "Cancel";
  const showCancel = opts.showCancel ?? false;

  if (btnModalOk) btnModalOk.textContent = okText;

  if (btnModalCancel) {
    btnModalCancel.textContent = cancelText;
    // Only show cancel when explicitly requested
    btnModalCancel.classList.toggle("hidden", !showCancel);
  }
  // ................. end opts.............

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
          return "UIState_Calib_Options";
        case 8:
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
  // Status card summary
  const stimLabel = intToLabel("stim_window", data.stim_window);
  if (elStatusUiState) {
    elStatusUiState.textContent = stimLabel ?? "—";
  }

  if (elStatusActiveSubject) {
    const subj = data.active_subject_id || "None";
    elStatusActiveSubject.textContent = subj;
  }

  if (elStatusModel) {
    const modelReady = data.is_model_ready;
    elStatusModel.textContent = modelReady
      ? "Trained model ready"
      : "No trained model";
  }

  // run mode flag cleared by default
  document.body.classList.remove("run-mode");

  // View routing based on stim_window value
  const stimState = data.stim_window;
  // MUST MATCH UISTATE_E
  // 0 = Active_Run, 1 = Active_Calib, 2 = Instructions, 3 = Home, 4 = saved_sessions, 5 = run_options, 6 = hardware_checks, 7 = calib_options, 8 = None
  if (stimState === 3 /* Home */ || stimState === 8 /* None */) {
    stopCalibFlicker();
    stopRunFlicker();
    stopHardwareMode();
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
    const runLeftHz = data.freq_left_hz ?? data.freq_hz ?? 0;
    const runRightHz = data.freq_right_hz ?? data.freq_hz ?? 0;
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
  } else if (stimState == 6) {
    setFullScreenMode(true);
    startHardwareMode();
    showView("hardware_checks");
  } else if (stimState == 7) {
    stopCalibFlicker();
    setFullScreenMode(false);
    showView("calib_options");
  }

  // HANDLE POPUPS TRIGGERED BY BACKEND:
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
      case 3: // UIPopup_TooManyBadWindowsInRun
        showModal(
          "Too many artifactual windows detected",
          "Please check headset placement and rerun hardware checks to verify signal."
        );
        break;
      case 5: // UIPopup_ConfirmOverwriteCalib
        showModal(
          "Calibration already exists",
          `A calibration for "${
            data.pending_subject_name || "this user"
          }" already exists. Overwrite it?`,
          {
            showCancel: true,
            okText: "Overwrite",
            cancelText: "Cancel",
          }
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
    elRefreshLabel.textContent = `Monitor refresh ≈ ${refreshHz} Hz`;
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

// ================ 11) HARDWARE CHECKS MAIN RUN LOOP & PLOTTING HELPERS ===================================
// MAIN LOOP
async function hardwareLoop() {
  // if user left hw mode, do nothing
  if (!hwActive) return;
  try {
    // fetch EEG samples and quality in parallel from backend
    const [eegRes, qRes] = await Promise.all([
      fetch(`${API_BASE}/eeg`),
      fetch(`${API_BASE}/quality`),
    ]);

    // Parse JSON responses
    const eeg = await eegRes.json();
    const qJson = await qRes.json();

    // If backend says "no data yet", just try again next animation frame
    if (!eeg.ok) {
      hwAnimId = setTimeout(hardwareLoop, 200);
      return;
    }

    // extract meta from EEG json
    const fs = eeg.fs || 250;
    const units = eeg.units || "uV";
    const nChannels = eeg.n_channels || eeg.channels.length;
    const labels = Array.isArray(eeg.labels)
      ? eeg.labels
      : Array.from({ length: nChannels }, (_, i) => `Ch ${i + 1}`);
    // init charts w configs
    initHardwareCharts(nChannels, labels, fs, units);

    // extract meta from stats (getQuality GET req)
    updateHwHealthHeader(qJson.rates);
    updatePerChannelStats(qJson);

    // Each entry in eeg.channels[ch] is an array of samples for that channel
    const numSamples = eeg.channels[0].length;
    for (let s = 0; s < numSamples; s++) {
      // For each channel, read the sample and push {x, y} into its dataset
      for (let ch = 0; ch < nChannels; ch++) {
        const eeg_val = eeg.channels[ch][s];
        const chart = hwCharts[ch];
        const ds = chart.data.datasets[0];

        // use monotonically increasing sample idx for time axis
        const k = chart._nextX || 0;
        chart._nextX = k + 1;

        // SLIDING WINDOW IMPLEMENTATION
        ds.data.push({ x: 0, y: eeg_val }); // placeholder x
        // only ever keep hwSamplesPerCycle points
        if (ds.data.length > hwSamplesPerCycle) {
          ds.data.shift(); // removes first el of array (queue implementation)
        }

        // Colour line based on quality (green = good, red = bad)
        const r = qJson?.rates?.current_bad_win_rate;
        const hc = healthClassFromBadRate(r).cls;
        if (hc === "good") ds.borderColor = "#4ade80";
        else if (hc === "warn") ds.borderColor = "#facc15";
        else ds.borderColor = "#f97373";
      }

      hwGlobalIndex++;
    }
    // sample acquired.
    // SLIDE x-axis window to last HW_MAX_WINDOW_SEC
    // redraw all charts once this frame with the new data / indexes
    hwCharts.forEach((chart) => {
      const ds = chart.data.datasets[0];
      const N = ds.data.length;
      // RE-INDEX X
      for (let i = 0; i < N; i++) {
        ds.data[i].x = i; // leftmost point x=0, rightmost x=N-1
      }
      // x axis always 0 to window size (anchored to left, no offsets)
      chart.options.scales.x.min = 0;
      chart.options.scales.x.max = hwSamplesPerCycle - 1;

      chart.update("none");
    });
  } catch (err) {
    console.log("hardwareLoop error:", err);
  }
  hwAnimId = setTimeout(hardwareLoop, 200); // sched next frame in 150ms (a little more than 5Hz) (inf loop until hw mode is exited)
}

function initHardwareCharts(nChannels, labels, fs, units) {
  // (A) if we already have charts for the right num of channels -> don't need to remake
  if (hwCharts.length == nChannels) {
    hwLabels = labels;

    // update label text for each label
    hwCharts.forEach((chart, ch) => {
      if (chart._primarySpan) {
        chart._primarySpan.textContent = hwLabels[ch] || `Ch ${ch + 1}`;
      }
      // update units text if needed
      if (chart._secondarySpan) {
        chart._secondarySpan.textContent = units || "uV";
      }
      // dataset label (not visible)
      if (chart.data?.datasets?.[0]) {
        chart.data.datasets[0].label = hwLabels[ch] || `Ch ${ch + 1}`;
      }
    });
    return; // init complete
  }

  // (B) rebuild from scratch
  // Destroy any existing Chart.js instances to avoid memory leaks
  hwCharts.forEach((c) => c.destroy());
  hwCharts = [];
  hwPlotsContainer.innerHTML = ""; // clear html container

  // Update global state
  hwLabels = labels;
  hwGlobalIndex = 0;
  hwNChannels = nChannels;
  // update number of points we want across 1 sweep width
  hwSamplesPerCycle = Math.max(1, Math.floor(HW_MAX_WINDOW_SEC * fs));
  hwSampleIdxInCycle = 0; // reset

  // for each channel: build a wrapper with label + canvas
  for (let ch = 0; ch < nChannels; ch++) {
    // Outer div to hold everything
    const wrapper = document.createElement("div");
    wrapper.className = "hw-plot";

    // HEADER (title left + stats right)
    const header = document.createElement("div");
    header.className = "hw-plot-header";

    const left = document.createElement("div");
    left.className = "hw-plot-left";

    const title = document.createElement("div");
    title.className = "hw-plot-title";

    const primarySpan = document.createElement("span");
    primarySpan.textContent = hwLabels[ch] || `Ch ${ch + 1}`;

    const secondarySpan = document.createElement("span");
    secondarySpan.className = "secondary";
    secondarySpan.textContent = units || "uV";

    title.appendChild(primarySpan);
    title.appendChild(secondarySpan);
    left.appendChild(title);

    const stats = document.createElement("div");
    stats.className = "hw-plot-stats";
    stats.innerHTML = `
      <span class="stat-chip" data-k="rms">RMS <b id="hwstat-rms-${ch}">—</b></span>
      <span class="stat-chip" data-k="maxabs">MAX <b id="hwstat-maxabs-${ch}">—</b></span>
      <span class="stat-chip" data-k="step">STEP <b id="hwstat-step-${ch}">—</b></span>
      <span class="stat-chip" data-k="std">STD <b id="hwstat-std-${ch}">—</b></span>
    `;

    header.appendChild(left);
    header.appendChild(stats);
    wrapper.appendChild(header);

    // Canvas where Chart.js will draw the line plot
    const canvas = document.createElement("canvas");
    // Height in pixels; width will be controlled by CSS
    canvas.height = 110;
    wrapper.appendChild(canvas);
    hwPlotsContainer.appendChild(wrapper);
    const ctx = canvas.getContext("2d");

    // Create a new Chart.js instance for this channel
    const chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: hwLabels[ch] || `Ch ${ch + 1}`,
            data: [],
            borderWidth: 1,
            pointRadius: 0,
            parsing: false,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true, // Resize with container/viewport
        maintainAspectRatio: false, // Let CSS control
        layout: {
          padding: { top: 4, right: 6, bottom: 25, left: 6 },
        },
        scales: {
          x: {
            type: "linear",
            ticks: { display: false },
            grid: { display: false },
            offset: false,
            min: 0,
            max: hwSamplesPerCycle - 1, // fixed window 0 → N-1
          },
          y: {
            min: HW_Y_MIN,
            max: HW_Y_MAX,
          },
        },
        plugins: {
          legend: { display: false }, // No legend per plot (label is above)
          tooltip: { enabled: false },
        },
        elements: {
          line: {
            tension: 0,
          },
        },
      },
    });

    chart._primarySpan = primarySpan;
    chart._secondarySpan = secondarySpan;
    // track next x-index per channel
    chart._nextX = 0;

    // Keep chart in the global array for all channels so we can update it each frame
    hwCharts.push(chart);
  }
}

// ============================== 12) HW HEALTH HELPERS ! =============================
function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

// map current bad window rate to healthy/unhealthy/ok (green/red/yellow)
function healthClassFromBadRate(r) {
  if (r == null || Number.isNaN(r)) return { cls: "warn", label: "Measuring…" };
  if (r < 0.15) return { cls: "good", label: "OK: stable signal" };
  if (r < 0.4)
    return { cls: "warn", label: "Borderline: maybe adjust electrodes" };
  return { cls: "bad", label: "Needs work: too many artifacts" };
}

function updateHwHealthHeader(rates) {
  const r = rates?.current_bad_win_rate;
  const o = rates?.overall_bad_win_rate;
  const n = rates?.num_win_in_rolling;

  const { cls, label: txt } = healthClassFromBadRate(r);

  if (elHealthBadge) {
    elHealthBadge.classList.remove("good", "warn", "bad");
    elHealthBadge.classList.add(cls);
  }
  if (elHealthLabel) elHealthLabel.textContent = txt;

  if (elHealthRollBad) elHealthRollBad.textContent = pct(r);
  if (elHealthOverallBad) elHealthOverallBad.textContent = pct(o);
  if (elHealthRollN) elHealthRollN.textContent = n == null ? "—" : String(n);
}

function fmt1(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(1);
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function updatePerChannelStats(statsJson) {
  const roll = statsJson?.rolling;
  const n = statsJson?.n_channels || 0;
  if (!roll || n <= 0) return;

  for (let ch = 0; ch < n; ch++) {
    setText(`hwstat-rms-${ch}`, fmt1(roll.rms_uv?.[ch]));
    setText(`hwstat-maxabs-${ch}`, fmt1(roll.max_abs_uv?.[ch]));
    setText(`hwstat-step-${ch}`, fmt1(roll.max_step_uv?.[ch]));
    setText(`hwstat-std-${ch}`, fmt1(roll.std_uv?.[ch]));
  }
}

function applyChipClass(valueId, v, warnTh, badTh) {
  const el = document.getElementById(valueId);
  if (!el) return;
  const chip = el.closest(".stat-chip");
  if (!chip) return;

  chip.classList.remove("warn", "bad");

  if (v == null || Number.isNaN(v)) return;
  if (v >= badTh) chip.classList.add("bad");
  else if (v >= warnTh) chip.classList.add("warn");
}

// =============== 13) SEND POST EVENTS WHEN USER CLICKS BUTTONS (OR OTHER INPUTS) ===============
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

// special post event for calib options
// payload: { action, subject name, epilepsy risk }
async function sendCalibOptionsAndStart() {
  const name = (inpCalibName?.value || "").trim();
  const raw = selEpilepsy?.value ?? "4"; // 4 = select... (default)
  const epilepsy = parseInt(raw, 10);

  // basic UI-side validation (backend still enforces)
  if (name.length < 3) {
    showModal("Name too short", "Please enter at least 3 characters.");
    return;
  }

  // treat 4 (select..) as invalid
  if (raw === "4" || Number.isNaN(epilepsy)) {
    showModal("Missing selection", "Please select an epilepsy risk option.");
    return;
  }

  if (raw === "2" || raw === "3") {
    showModal(
      "Cannot proceed",
      "This device is not safe for use for individuals with photosensitivity."
    );
    return;
  }

  if (raw === "1") {
    showModal(
      "High frequency SSVEP decoding (>20Hz) will be attempted",
      "The final model's performance may be poor, and device functionality may be limited."
    );
    return;
  }

  const payload = {
    action: "start_calib_from_options",
    subject_name: name,
    epilepsy: epilepsy,
  };

  try {
    const res = await fetch(`${API_BASE}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      logLine(`POST /event failed (${res.status}) for calib submit`);
      return;
    }
    logLine(`Submitted calib options for ${name}`);
  } catch (err) {
    logLine(`POST /event error for calib submit: ${err}`);
  }
}

// ================== 14) INIT ON PAGE LOAD ===================
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

  if (btnStartHw) {
    btnStartHw.addEventListener("click", () => {
      // tell backend user requested hardware checks state
      sendSessionEvent("hardware_checks");
    });
  }

  if (btnCalibSubmit) {
    btnCalibSubmit.addEventListener("click", () => {
      // specialized sender because we need more than {action} back...
      // post { action, subject_name, epilepsy risk }
      sendCalibOptionsAndStart();
    });
  }

  if (btnCalibBack) {
    btnCalibBack.addEventListener("click", () => {
      sendSessionEvent("exit");
    });
  }

  if (btnModalOk) {
    // if a popup is visible, wait for user ack
    btnModalOk.addEventListener("click", () => {
      hideModal();
      // tell backend to clear popup in statestore
      sendSessionEvent("ack_popup");
    });
  }

  if (btnModalCancel) {
    btnModalCancel.addEventListener("click", () => {
      hideModal();
      // If canceling overwrite, tell backend to clear popup + stay put
      sendSessionEvent("cancel_popup"); // TODO: clear popup/handle on backend...
    });
  }
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
