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

// Refresh info
let measuredRefreshHz = 60; // updated in init()

// Flicker animation DOM elements
const elCalibBlock = document.getElementById("calib-block");
const elRunLeft = document.getElementById("run-left");
const elRunRight = document.getElementById("run-right");

// VIEW CONTAINERS FOR DIFF WINDOWS
const viewHome = document.getElementById("view-home");
const viewInstructions = document.getElementById("view-instructions");
const viewActiveCalib = document.getElementById("view-active-calib");
const viewActiveRun = document.getElementById("view-active-run");
const viewRunOptions = document.getElementById("view-run-options");
const viewSavedSessions = document.getElementById("view-saved-sessions");
const viewHardware = document.getElementById("view-hardware-checks");
const elViewTransition = document.getElementById("view-transition"); // overlay div!!
let viewTransitionInFlight = false;

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
const btnPause = document.getElementById("btn-pause");

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
let popupAckInFlight = false; // prevent races ie. give time to backend to clear (don't allow new popup to raise on state/ poll during popAckInFlight true)

// Timer for browser requests to server
let pollInterval = null;
let pollInFlight = false; // guard against pollStateOnce overlaps

// FlickerStimulus instances
let calibStimulus = null;
let leftStimulus = null;
let rightStimulus = null;
let stimAnimId = null;
let neutralLeftStimulus = null;
let neutralRightStimulus = null;

// Hardware checks DOM elements
const hwPlotsContainer = document.getElementById("hw-plots-container");
// Hardware checks plotting configs
const HW_MAX_WINDOW_SEC = 9; // seconds visible on screen
const HW_Y_MIN = -100; // adjust to fit (scale should be uV, EEG ~10-100uV)
const HW_Y_MAX = 100;
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

// Pending Training DOM elements
const elTrainingOverlay = document.getElementById("training-overlay");
const btnCancelTraining = document.getElementById("btn-cancel-training");

// Settings Page DOM elements
const viewSettings = document.getElementById("view-settings");
const btnOpenSettings = document.getElementById("btn-open-settings");
const btnSettingsBack = document.getElementById("btn-settings-back");
const btnSettingsSave = document.getElementById("btn-settings-save");
const selTrainArch = document.getElementById("set-train-arch");
const selCalibData = document.getElementById("set-calib-data");
const selWaveform = document.getElementById("set-waveform");
const selModulation = document.getElementById("set-modulation");
const elSettingsStatus = document.getElementById("settings-status");
const elFreqWarning = document.getElementById("freq-warning");
const freqInputs = Array.from(
  document.querySelectorAll('input[name="freq-select"]')
);
const FREQ_MAX_SELECT = 6;
let currentWaveform = "square"; // "square" | "sine"
let currentModulation = "flicker"; // "flicker" | "grow"
let settingsInitiallyUpdated = false;
// Slider elements
const elDurActive = document.getElementById("set-duration-active");
const elDurNone = document.getElementById("set-duration-none");
const elDurRest = document.getElementById("set-duration-rest");
const elCycleRep = document.getElementById("set-cycle-repeats");
const elDurActiveLbl = document.getElementById("set-duration-active-label");
const elDurNoneLbl = document.getElementById("set-duration-none-label");
const elDurRestLbl = document.getElementById("set-duration-rest-label");
const elCycleRepLbl = document.getElementById("set-cycle-repeats-label");

// No SSVEP Block DOM elements
const viewNoSSVEP = document.getElementById("view-neutral");
const elNeutralLeftArrow = document.getElementById("no-left-arrow");
const elNeutralRightArrow = document.getElementById("no-right-arrow");
const elNoSSVEPLeftFreq = document.getElementById("no-left-freq");
const elNoSSVEPRightFreq = document.getElementById("no-right-freq");

// Handle random frequency pairs in no_ssvep_test mode
let prevStimState = null;
// freq pair should be sticky for the duration of the Neutral state
let neutralLeftHz = 0;
let neutralRightHz = 0;
let neutralPairChosen = false;

// Paused overlay DOM elements
const elPauseOverlay = document.getElementById("pause-overlay");
const btnResume = document.getElementById("btn-resume");
const btnPauseExit = document.getElementById("btn-pause-exit");

// ===================== 2) LOGGING HELPER =============================
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
    viewSettings,
    viewNoSSVEP,
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
    case "settings":
      viewSettings.classList.remove("hidden");
      break;
    case "no_ssvep":
      viewNoSSVEP.classList.remove("hidden");
      break;
    default:
      viewHome.classList.remove("hidden");
      break;
  }
}

// (2) set full screen in calib/run modes (hide side bar & log panel)
// and then also targets mode for flickering stimuli pages, and run mode
function applyBodyMode({
  fullscreen = false,
  targets = false,
  run = false,
} = {}) {
  document.body.classList.toggle("fullscreen-mode", fullscreen);
  document.body.classList.toggle("targets-mode", targets);
  document.body.classList.toggle("run-mode", run);

  if (btnExit) {
    if (fullscreen) {
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

// (5a) pending training overlay helper
function showTrainingOverlay(show) {
  if (!elTrainingOverlay) return;

  elTrainingOverlay.classList.toggle("hidden", !show);
  document.body.classList.toggle("is-busy", show);
}

// (5b) paused overlay helper
function setPauseButtonVisible(visible) {
  if (!btnPause) return;
  btnPause.classList.toggle("hidden", !visible);
}

function showPauseOverlay(show) {
  if (!elPauseOverlay) return;

  if (show) {
    // Only stop flicker when entering pause
    stopAllStimuli();
    elPauseOverlay.classList.remove("hidden");
    document.body.classList.add("is-busy");
  } else {
    elPauseOverlay.classList.add("hidden");
    document.body.classList.remove("is-busy");
  }
}

// (6) update settings from backend when we first enter settings page (rising edge trigger)
function updateSettingsFromState(data) {
  const arch = data.settings.train_arch_setting;
  const calib = data.settings.calib_data_setting;
  const stim_mode_i = data.settings.stim_mode; // 0=flicker, 1=grow
  const waveform_i = data.settings.waveform; // 0=square, 1=sine
  const duractive_i = Number(data.settings.duration_active_s);
  const durnone_i = Number(data.settings.duration_none_s);
  const durrest_i = Number(data.settings.duration_rest_s);
  const cyclereps_i = Number(data.settings.num_times_cycle_repeats);

  if (selTrainArch && arch != null) selTrainArch.value = String(arch);
  if (selCalibData && calib != null) selCalibData.value = String(calib);
  if (selWaveform) selWaveform.value = waveform_i === 1 ? "sine" : "square";
  if (selModulation)
    selModulation.value = stim_mode_i === 1 ? "grow" : "flicker";
  currentWaveform = selWaveform?.value ?? "square";
  currentModulation = selModulation?.value ?? "flicker";

  if (elDurActive && !Number.isNaN(duractive_i))
    elDurActive.value = String(duractive_i);
  if (elDurNone && !Number.isNaN(durnone_i))
    elDurNone.value = String(durnone_i);
  if (elDurRest && !Number.isNaN(durrest_i))
    elDurRest.value = String(durrest_i);
  if (elCycleRep && !Number.isNaN(cyclereps_i))
    elCycleRep.value = String(cyclereps_i);
  if (elDurActiveLbl && elDurActive)
    elDurActiveLbl.textContent = elDurActive.value;
  if (elDurNoneLbl && elDurNone) elDurNoneLbl.textContent = elDurNone.value;
  if (elDurRestLbl && elDurRest) elDurRestLbl.textContent = elDurRest.value;
  if (elCycleRepLbl && elCycleRep) elCycleRepLbl.textContent = elCycleRep.value;

  // selected freqs from backend are ENUMS 1..15
  // grid uses checkbox.value 0..14
  const selectedEnums = Array.isArray(data.settings.selected_freqs_e)
    ? data.settings.selected_freqs_e.map((e) => Number(e))
    : [];

  const selectedZeroBased = new Set(selectedEnums.map((e) => e - 1)); // 1..15 -> 0..14

  freqInputs.forEach((inp) => {
    const v = Number(inp.value);
    inp.checked = selectedZeroBased.has(v);
  });

  updateFreqCounterUI();
  updateFreqCompatibilityIndicators();

  settingsInitiallyUpdated = true; // rising edge
  if (elSettingsStatus) elSettingsStatus.textContent = "";
}

// (7) handle transitions during calib protocol so it's smoother using overlay div
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
async function transitionToView({
  viewName,
  stopStimuli = () => {},
  startStimuli = () => {},
  fadeOutMs = 90, // hadeel if u wanna play with these bs numbers at some point i would love that bcuz i am going insane crazy :)
  holdMs = 35,
  fadeInDelayMs = 25,
} = {}) {
  if (viewTransitionInFlight) return;
  viewTransitionInFlight = true;
  try {
    document.body.classList.add("stim-swap");
    stopStimuli(); // stop any flicker BEFORE swapping

    // show overlay + fade in
    if (elViewTransition) {
      elViewTransition.classList.remove("hidden");
      requestAnimationFrame(() => elViewTransition.classList.add("on"));
    }

    await sleep(fadeOutMs);

    // swap view underneath
    showView(viewName);

    await sleep(holdMs);

    // start stimuli AFTER the UI is stable
    await sleep(fadeInDelayMs);
    startStimuli();

    // fade overlay out
    if (elViewTransition) {
      elViewTransition.classList.remove("on");
      await sleep(fadeOutMs);
      elViewTransition.classList.add("hidden");
    }
  } finally {
    // restore regular transitions
    document.body.classList.remove("stim-swap");
    viewTransitionInFlight = false;
  }
}

// (8) FREQUENCY GRID HELPERS (on settings page for selecting freqs) -> live counter
function getSelectedHzFromGrid() {
  return freqInputs
    .filter((i) => i.checked)
    .map((i) => Number(i.dataset.hz))
    .filter((x) => Number.isFinite(x) && x > 0);
}

function setSelectedHzToGrid(selectedHz) {
  const set = new Set((selectedHz || []).map(Number));
  freqInputs.forEach((i) => {
    const hz = Number(i.dataset.hz);
    i.checked = set.has(hz);
  });
  updateFreqCounterUI();
}
function updateFreqCounterUI() {
  if (!elFreqWarning) return;
  const n = getSelectedHzFromGrid().length;

  // text
  elFreqWarning.textContent = n === 1 ? "1 selected" : `${n} selected`;

  // color semantic
  elFreqWarning.style.color =
    n === 0
      ? "rgba(250, 204, 21, 0.95)"
      : n <= FREQ_MAX_SELECT
      ? "var(--success)"
      : "var(--danger)";
}
function attachFreqGridHandlers() {
  freqInputs.forEach((inp) => {
    inp.addEventListener("change", (e) => {
      const n = getSelectedHzFromGrid().length;

      // enforce max select
      if (n > FREQ_MAX_SELECT) {
        // undo this click
        inp.checked = false;
        updateFreqCounterUI();
        showModal(
          "Too many frequencies selected",
          `Please select up to ${FREQ_MAX_SELECT} frequencies.`,
          { okText: "OK" }
        );
        return;
      }

      updateFreqCounterUI();
    });
  });
  updateFreqCounterUI();
}

// (9) SLIDER HELPERS FOR SETTINGS PAGE
function bindSliderValue(sliderEl, labelEl) {
  if (!sliderEl || !labelEl) return;
  const update = () => (labelEl.textContent = String(sliderEl.value));
  sliderEl.addEventListener("input", update);
  update();
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
    case "stim_window": // must match UIState_E
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
          return "UIState_Pending_Training";
        case 9:
          return "UIState_Settings";
        case 10:
          return "UIState_NoSSVEP_Test";
        case 11:
          return "UIState_Paused";
        case 12:
          return "UIState_None";
        default:
          return `Unknown (${integer})`;
      }
    case "freq_hz_e": // must match TestFreq_E
      switch (integer) {
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
        case 6:
          return "TestFreq_13_Hz";
        case 7:
          return "TestFreq_14_Hz";
        case 8:
          return "TestFreq_15_Hz";
        case 9:
          return "TestFreq_16_Hz";
        case 10:
          return "TestFreq_17_Hz";
        case 11:
          return "TestFreq_18_Hz";
        case 12:
          return "TestFreq_20_Hz";
        case 13:
          return "TestFreq_25_Hz";
        case 14:
          return "TestFreq_30_Hz";
        case 15:
          return "TestFreq_35_Hz";
        case 99:
          return "TestFreq_NoSSVEP";
        case 0:
          return "TestFreq_None";
        default:
          return 0;
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

// HELPER FOR CHOOSING freq pair randomly in NEUTRAL (NO_SSVEP)
// Allowed TestFreq enums for randomly selecting neutral targets
// (exclude 0=None and 99=NoSSVEP because those are not flicker freqs)
const NEUTRAL_TESTFREQ_ENUMS = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

function testFreqEnumToHz(e) {
  switch (e) {
    case 1:
      return 8;
    case 2:
      return 10;
    case 3:
      return 11;
    case 4:
      return 12;
    case 5:
      return 9;
    case 6:
      return 13;
    case 7:
      return 14;
    case 8:
      return 15;
    case 9:
      return 16;
    case 10:
      return 17;
    case 11:
      return 18;
    case 12:
      return 20;
    case 13:
      return 25;
    case 14:
      return 30;
    case 15:
      return 35;
    case 99:
      return -1; // NoSSVEP CSV marker
    case 0:
      return 0; // None
    default:
      return 0;
  }
}

// Pick one random element from the array
function pickOne(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}
// Pick two distinct TestFreq enums and return Hz values
function pickRandomNeutralHzPair() {
  const leftEnum = pickOne(NEUTRAL_TESTFREQ_ENUMS);
  let rightEnum = pickOne(NEUTRAL_TESTFREQ_ENUMS);
  // Ensure distinct
  while (rightEnum === leftEnum && NEUTRAL_TESTFREQ_ENUMS.length > 1) {
    rightEnum = pickOne(NEUTRAL_TESTFREQ_ENUMS);
  }
  return {
    leftHz: testFreqEnumToHz(leftEnum),
    rightHz: testFreqEnumToHz(rightEnum),
  };
}

// Map settings strings -> backend ints (from SettingWaveform_E and SettingStimMode_E)
function waveformToInt(w) {
  // backend: 0=square, 1=sine
  return w === "sine" ? 1 : 0;
}
function stimModeToInt(m) {
  // backend: 0=flicker, 1=grow
  return m === "grow" ? 1 : 0;
}

// Convert checkbox value="0..14" -> backend enum e="1..15"
function getSelectedFreqEnumsFromGrid() {
  // HTML checkbox inputs have:
  // - value: 0..14
  // - data-hz: 8,9,10... etc (for display only)
  const picked = freqInputs
    .filter((i) => i.checked)
    .map((i) => Number(i.value) + 1) // backend expects 1..15
    .filter((e) => Number.isInteger(e));
  picked.sort((a, b) => a - b);
  return Array.from(new Set(picked));
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

  // View routing based on stim_window value (MUST MATCH UISTATE_E)
  const stimState = data.stim_window;

  // capture no_ssvep_test state transitions so we only randomize freq pair ONCE per entry
  const enteringNeutral = stimState === 10 && prevStimState !== 10;
  const leavingNeutral = prevStimState === 10 && stimState !== 10;
  // on falling edge (exit), reset
  if (leavingNeutral) {
    stopNeutralFlicker();
    neutralPairChosen = false;
  }

  // general detection of rising edges for any state
  const stateChanged = prevStimState !== stimState;

  const pauseVisible =
    stimState === 0 || // Active_Run
    stimState === 1 || // Active_Calib
    stimState === 2 || // Instructions
    stimState === 10; // NoSSVEP_Test
  setPauseButtonVisible(pauseVisible);

  // 0 = Active_Run, 1 = Active_Calib, 2 = Instructions, 3 = Home, 4 = saved_sessions, 5 = run_options, 6 = hardware_checks, 7 = calib_options, 8 = pending_training, 9 = settings, 10 = no_ssvep, 11 = paused, 12 = None
  if (stimState === 3 /* Home */ || stimState === 12 /* None */) {
    stopAllStimuli();
    stopHardwareMode();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    settingsInitiallyUpdated = false; // reset flag
    showView("home");
  } else if (stimState === 2 /* Instructions */) {
    stopAllStimuli();
    applyBodyMode({ fullscreen: true, targets: false, run: false });
    // Update text based on block and freq
    elInstrBlockId.textContent = data.block_id ?? "-";
    elInstrFreqHz.textContent = fmtFreqHz(data.freq_hz) + " Hz";
    if (stateChanged) {
      transitionToView({
        viewName: "instructions",
        stopStimuli: stopAllStimuli,
        startStimuli: () => {},
      });
    } else {
      showView("instructions");
    }
  }
  // TODO: customize elInstructionsText based on block / upcoming freq
  else if (stimState === 1 /* Active_Calib */) {
    applyBodyMode({ fullscreen: true, targets: true, run: false });
    const calibFreqHz = data.freq_hz ?? 0;
    if (stateChanged) {
      transitionToView({
        viewName: "active_calib",
        stopStimuli: stopAllStimuli,
        startStimuli: () => startCalibFlicker(calibFreqHz),
      });
    } else {
      showView("active_calib");
      startCalibFlicker(calibFreqHz); // keep it running
    }
  } else if (stimState === 0 /* Active_Run */) {
    applyBodyMode({ fullscreen: true, targets: true, run: true });
    showView("active_run");
    // default to freq_hz if undef right/left
    const runLeftHz = data.freq_left_hz ?? data.freq_hz ?? 0;
    const runRightHz = data.freq_right_hz ?? data.freq_hz ?? 0;
    startRunFlicker(runLeftHz, runRightHz);
  } else if (stimState === 4 /* Saved Sessions */) {
    stopAllStimuli();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    showView("saved_sessions");

    // TODO: render session list from backend
  } else if (stimState === 5 /* Run Options */) {
    stopAllStimuli();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
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
    applyBodyMode({ fullscreen: true, targets: false, run: false });
    startHardwareMode();
    showView("hardware_checks");
  } else if (stimState == 7) {
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    showView("calib_options");
  } else if (stimState == 8) {
    stopAllStimuli();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
  } else if (stimState == 9) {
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    if (!settingsInitiallyUpdated) {
      updateSettingsFromState(data);
    }
    showView("settings");
  } else if (stimState == 10) {
    // on rising edge, need to choose new freq pair
    stopCalibFlicker();
    if (enteringNeutral || !neutralPairChosen) {
      const pair = pickRandomNeutralHzPair();
      neutralLeftHz = pair.leftHz;
      neutralRightHz = pair.rightHz;
      neutralPairChosen = true;
    }

    applyBodyMode({ fullscreen: true, targets: true, run: false });

    if (elNoSSVEPLeftFreq)
      elNoSSVEPLeftFreq.textContent = `${neutralLeftHz} Hz`;
    if (elNoSSVEPRightFreq)
      elNoSSVEPRightFreq.textContent = `${neutralRightHz} Hz`;

    if (stateChanged) {
      transitionToView({
        viewName: "no_ssvep",
        stopStimuli: stopAllStimuli,
        startStimuli: () => startNeutralFlicker(neutralLeftHz, neutralRightHz),
      });
    } else {
      showView("no_ssvep");
      startNeutralFlicker(neutralLeftHz, neutralRightHz);
    }
  }

  // pending training overlay driven purely by state
  showTrainingOverlay(stimState === 8); // uistate_pending_training
  showPauseOverlay(stimState == 11); // uistate_paused

  // HANDLE POPUPS TRIGGERED BY BACKEND:
  const popupEnumIdx = data.popup ?? 0; // 0 is fallback (popup NONE)

  // track popup to clear in-flight when backend clears it
  if (popupEnumIdx === 0) {
    popupAckInFlight = false;
  }

  // if popup ack is in flight and it's not none, backend hasn't cleared yet, so we don't want to retrigger
  if (popupEnumIdx !== 0 && popupAckInFlight) {
    // keep modal hidden
    prevStimState = stimState;
    return;
  }

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
      case 6: // UIPopup_ConfirmHighFreqOk
        showModal(
          "High frequency SSVEP decoding (>20Hz) will be attempted",
          "The final model's performance may be poor, and device functionality may be limited."
        );
        break;
      case 7: // UIPopup_TrainJobFailed
        showModal(
          "Training job failed",
          "There was an internal error. Please try calibration again."
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
  // update state
  prevStimState = stimState;
}

// ============= 6) START POLLING FOR GET/STATE ===============
async function pollStateOnce() {
  // Guard: don't start a new poll if the previous one hasn't finished yet
  // (prevents overlapping fetches, out-of-order UI updates)
  if (pollInFlight) return;
  pollInFlight = true;

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
  } finally {
    // Always clear the in-flight flag, even if we early-returned or threw
    pollInFlight = false;
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

// =========== 7) MONITOR REFRESH MEASUREMENT (POST /ready) ==========================
// Helper: check if freq is integer divisor of refresh
function isFreqOptimalForRefresh(freqHz, refreshHz) {
  if (!refreshHz || !freqHz) return true; // unknown = assume ok
  return refreshHz % freqHz === 0;
}

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

// Call this when settings page loads AND when refresh is measured
function updateFreqCompatibilityIndicators() {
  const refresh = measuredRefreshHz;

  freqInputs.forEach((inp) => {
    const hz = Number(inp.dataset.hz);
    const checkbox = inp.closest(".freq-checkbox");
    const isOptimal = isFreqOptimalForRefresh(hz, refresh);

    // Add/remove visual indicator
    if (isOptimal) {
      checkbox.classList.add("freq-optimal");
      checkbox.classList.remove("freq-suboptimal");
    } else {
      checkbox.classList.remove("freq-optimal");
      checkbox.classList.add("freq-suboptimal");
    }
  });
  // Update instruction text to mention refresh rate
  const freqInstruction = document.querySelector(".freq-instruction");
  if (freqInstruction) {
    const optimal = [];
    freqInputs.forEach((inp) => {
      const hz = Number(inp.dataset.hz);
      if (isFreqOptimalForRefresh(hz, refresh)) optimal.push(hz);
    });

    if (optimal.length < freqInputs.length) {
      freqInstruction.innerHTML = `
        Select <strong>up to 6 frequencies</strong> for calibration.
        <br><small style="opacity: 0.7">
          Your ${refresh} Hz display works best with: ${optimal.join(", ")} Hz
          <span style="color: var(--accent)">●</span>
        </small>
      `;
    }
  }
}

// ===================== 9) FLICKER STIMULUS CLASS ===================================
class FlickerStimulus {
  constructor(el, refreshHz) {
    // el = dom element
    this.el = el;
    this.refreshHz = refreshHz;
    this.targetHz = 0;
    this.enabled = false;
    this.frameIdx = 0;
    this.framesPerCycle = 1;
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
    if (this.enabled) return; // don't reset phase every poll
    this.enabled = true;
    this.frameIdx = 0;
    if (this.el) {
      this.el.style.visibility = "visible";
      this.el.style.setProperty("--stim-brightness", "1.0");
      this.el.style.setProperty("--stim-scale", "1");
    }
  }
  stop() {
    this.enabled = false;
    this.frameIdx = 0;
    if (this.el) {
      // Reset to neutral appearance when not flickering or transforming
      this.el.style.setProperty("--stim-brightness", "1.0");
      this.el.style.setProperty("--stim-scale", "1");
    }
  }
  // frequencymodulator
  onePeriod() {
    if (!this.enabled || !this.el || this.targetHz <= 0) return;

    this.frameIdx = (this.frameIdx + 1) % this.framesPerCycle;
    const phase = (2 * Math.PI * this.frameIdx) / this.framesPerCycle;

    // Use global settings chosen in Settings page (kept synced from /state)
    const waveform = currentWaveform; // "square" | "sine"
    const modulation = currentModulation; // "flicker" | "grow"

    // ======= modulation: FLICKER (brightness) =======
    if (modulation === "flicker") {
      let b;

      if (waveform === "sine") {
        // Sine brightness oscillation: map sin(-1..1) -> [0.4..1.3]
        const s = (Math.sin(phase) + 1) / 2; // 0..1
        b = 0.4 + s * (1.3 - 0.4);
      } else {
        // Square wave brightness: half cycle ON/OFF
        const half = this.framesPerCycle / 2;
        const on = this.frameIdx < half;
        b = on ? 1.3 : 0.4;
      }

      this.el.style.setProperty("--stim-brightness", String(b));
      // keep scale neutral
      this.el.style.setProperty("--stim-scale", "1");
      return;
    }

    // ======= modulation: GROW/SHRINK (scale) =======
    if (modulation === "grow") {
      let s;

      if (waveform === "sine") {
        // Sine scale oscillation: map sin -> [0.85..1.15]
        const x = (Math.sin(phase) + 1) / 2; // 0..1
        s = 0.85 + x * (1.15 - 0.85);
      } else {
        // Square scale toggle: [0.85, 1.15]
        const half = this.framesPerCycle / 2;
        const on = this.frameIdx < half;
        s = on ? 1.15 : 0.85;
      }

      this.el.style.setProperty("--stim-scale", s.toFixed(3));
      // keep brightness neutral
      this.el.style.setProperty("--stim-brightness", "1.0");
      return;
    }
  }
}

function stopAllStimuli() {
  stopCalibFlicker();
  stopRunFlicker();
  stopNeutralFlicker();
}

// ================== 10) Flicker animation starters/stoppers ==========================

function stimAnimationLoop() {
  if (calibStimulus) calibStimulus.onePeriod();
  if (leftStimulus) leftStimulus.onePeriod();
  if (rightStimulus) rightStimulus.onePeriod();
  if (neutralLeftStimulus) neutralLeftStimulus.onePeriod();
  if (neutralRightStimulus) neutralRightStimulus.onePeriod();

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

function startNeutralFlicker(leftHz, rightHz) {
  if (neutralLeftStimulus) {
    neutralLeftStimulus.setRefreshHz(measuredRefreshHz);
    neutralLeftStimulus.setFrequency(leftHz);
    neutralLeftStimulus.start();
  }
  if (neutralRightStimulus) {
    neutralRightStimulus.setRefreshHz(measuredRefreshHz);
    neutralRightStimulus.setFrequency(rightHz);
    neutralRightStimulus.start();
  }
}

function stopNeutralFlicker() {
  if (neutralLeftStimulus) neutralLeftStimulus.stop();
  if (neutralRightStimulus) neutralRightStimulus.stop();
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

// special post event for settings save
// payload: { action, train_arch, calib_data, stim_mode, waveform, selected_freqs_e[] }
async function sendSettingsAndSave() {
  const trainArchRaw = selTrainArch?.value ?? "";
  const calibDataRaw = selCalibData?.value ?? "";

  const trainArch = parseInt(trainArchRaw, 10);
  const calibData = parseInt(calibDataRaw, 10);
  const waveformStr = selWaveform?.value ?? "square"; // "square" | "sine"
  const modStr = selModulation?.value ?? "flicker"; // "flicker" | "grow"
  const selected_freqs_e = getSelectedFreqEnumsFromGrid();

  // basic UI-side validation (backend still enforces)
  if (Number.isNaN(trainArch) || Number.isNaN(calibData)) {
    showModal(
      "Missing selection",
      "Please select both training architecture and calibration data options."
    );
    return;
  }
  if (selected_freqs_e.length < 1) {
    showModal("Missing frequencies", "Please select at least 1 frequency.");
    return;
  }

  const payload = {
    action: "set_settings",
    train_arch_setting: trainArch,
    calib_data_setting: calibData,
    stim_mode: stimModeToInt(modStr),
    waveform: waveformToInt(waveformStr),
    duration_active_s: Number(elDurActive?.value ?? 11), // otherwise default
    duration_none_s: Number(elDurNone?.value ?? 10),
    duration_rest_s: Number(elDurRest?.value ?? 8),
    num_times_cycle_repeats: Number(elCycleRep?.value ?? 3),
    selected_freqs_e, // int enums 1..15
  };
  console.log("SET_SETTINGS payload:", payload);
  logLine(
    `Saving timers: active=${payload.duration_active_s}s none=${payload.duration_none_s}s rest=${payload.duration_rest_s}s reps=${payload.num_times_cycle_repeats}`
  );

  try {
    const res = await fetch(`${API_BASE}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      logLine(`POST /event failed (${res.status}) for set_settings`);
      if (elSettingsStatus)
        elSettingsStatus.textContent = `Save failed (HTTP ${res.status})`;
      return;
    }

    // Apply locally right away (so flicker mode changes instantly)
    currentWaveform = waveformStr;
    currentModulation = modStr;

    logLine(
      `Settings saved (train_arch=${trainArch}, calib_data=${calibData})`
    );
    if (elSettingsStatus) {
      elSettingsStatus.textContent = "Saved ✓";
      setTimeout(() => {
        if (elSettingsStatus.textContent === "Saved ✓")
          elSettingsStatus.textContent = "";
      }, 2500);
    }
  } catch (err) {
    logLine(`POST /event error for set_settings: ${err}`);
    if (elSettingsStatus)
      elSettingsStatus.textContent = "Save failed (network)";
  }
}

// ================== 14) INIT ON PAGE LOAD ===================
async function init() {
  logLine("Initializing UI…");
  const estimated_refresh = await estimateRefreshHz();
  measuredRefreshHz = estimated_refresh; // global write
  await sendRefresh(estimated_refresh);
  // Update compatibility indicators now that we know refresh
  if (viewSettings && !viewSettings.classList.contains("hidden")) {
    updateFreqCompatibilityIndicators();
  }

  // create stimulus objects now that we know refresh
  calibStimulus = new FlickerStimulus(elCalibBlock, measuredRefreshHz);
  leftStimulus = new FlickerStimulus(elRunLeft, measuredRefreshHz);
  rightStimulus = new FlickerStimulus(elRunRight, measuredRefreshHz);
  neutralLeftStimulus = new FlickerStimulus(
    elNeutralLeftArrow,
    measuredRefreshHz
  );
  neutralRightStimulus = new FlickerStimulus(
    elNeutralRightArrow,
    measuredRefreshHz
  );

  stimAnimationLoop();
  startPolling();
  attachFreqGridHandlers();

  // add event listeners for all the sliders (settings page)
  bindSliderValue(elDurActive, elDurActiveLbl);
  bindSliderValue(elDurNone, elDurNoneLbl);
  bindSliderValue(elDurRest, elDurRestLbl);
  bindSliderValue(elCycleRep, elCycleRepLbl);

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

  if (btnPause) {
    btnPause.addEventListener("click", () => {
      sendSessionEvent("pause");
    });
  }

  if (btnResume) {
    btnResume.addEventListener("click", () => {
      sendSessionEvent("resume_after_pause");
    });
  }

  if (btnPauseExit) {
    btnPauseExit.addEventListener("click", () => {
      sendSessionEvent("exit");
    });
  }

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

  if (btnSettingsSave) {
    btnSettingsSave.addEventListener("click", () => {
      sendSettingsAndSave();
    });
  }

  if (btnOpenSettings) {
    btnOpenSettings.addEventListener("click", () => {
      sendSessionEvent("open_settings");
    });
  }

  if (btnSettingsBack) {
    btnSettingsBack.addEventListener("click", () => {
      sendSessionEvent("exit");
    });
  }

  if (btnCancelTraining) {
    btnCancelTraining.addEventListener("click", () => {
      // Cancel python job + return home
      sendSessionEvent("exit");
    });
  }

  if (btnModalOk) {
    // if a popup is visible, wait for user ack
    btnModalOk.addEventListener("click", () => {
      hideModal();
      // Prevent the same popup from re-opening while we're waiting for backend to clear it
      popupAckInFlight = true;
      // tell backend to clear popup in statestore
      sendSessionEvent("ack_popup");
    });
  }

  if (btnModalCancel) {
    btnModalCancel.addEventListener("click", () => {
      hideModal();
      popupAckInFlight = true;
      // If canceling overwrite, tell backend to clear popup + stay put
      sendSessionEvent("cancel_popup");
    });
  }
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
