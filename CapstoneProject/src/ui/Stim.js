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

// Home page training results DOM elements
const elHomeTrainSummary = document.getElementById("home-train-summary");
const btnDismissTrainSummary = document.getElementById(
  "btn-dismiss-train-summary",
);
const btnReopenTrainSummary = document.getElementById(
  "btn-reopen-train-summary",
);
// Status pill
const elTrainStatusPill = document.getElementById("train-status-pill");
const elTrainStatusLabel = document.getElementById("train-status-label");
const elTrainSummarySub = document.getElementById("train-summary-sub");
// KPIs
const elTrainKpiUser = document.getElementById("train-kpi-user");
const elTrainKpiSession = document.getElementById("train-kpi-session");
const elTrainKpiArch = document.getElementById("train-kpi-arch");
const elTrainKpiHoldout = document.getElementById("train-kpi-holdout");
const elTrainKpiTrain = document.getElementById("train-kpi-train");
const elTrainKpiHoldoutOk = document.getElementById("train-kpi-holdout-ok");
const elTrainKpiHoldoutWarn = document.getElementById("train-kpi-holdout-warn");
// Best frequency pair
const elTrainBestLeftHz = document.getElementById("train-best-left-hz");
const elTrainBestRightHz = document.getElementById("train-best-right-hz");
const elTrainBestLeftE = document.getElementById("train-best-left-e");
const elTrainBestRightE = document.getElementById("train-best-right-e");
// Consistency checks
const elCTrainOk = document.getElementById("c-check-train-ok");
const elCCvOk = document.getElementById("c-check-cv-ok");
const elCOnnxOk = document.getElementById("c-check-onnx-ok");
const elCHoldoutOk = document.getElementById("c-check-holdout-ok");
const elCTrainMsg = document.getElementById("c-train-msg");
const elCCvMsg = document.getElementById("c-cv-msg");
const elCOnnxMsg = document.getElementById("c-onnx-msg");
const elCHoldoutMsg = document.getElementById("c-holdout-msg");
// Data insufficiency table
const elTrainSuffEmpty = document.getElementById("train-suff-empty");
const elTrainSuffTableWrap = document.getElementById("train-suff-table-wrap");
const elTrainSuffTbody = document.getElementById("train-suff-tbody");

// TODO: functionality to DELETE SAVED SESSIONS
// Saved Sessions Page DOM elements
const elSessionsGrid = document.getElementById("sessions-grid");
const elSessionsEmptyState = document.getElementById("sessions-empty-state");
const elSessionsMetaStrip = document.getElementById("sessions-meta-strip");
let sessionsLoadInFlight = false; // guard against overlapping fetches from user or backend
let sessionsPageConsumedRefresh = false; // should fetch and render sessions on rising edges of entering page

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
  document.querySelectorAll('input[name="freq-select"]'),
);
const FREQ_MAX_SELECT = 6;
let currentWaveform = "square"; // "square" | "sine"
let currentModulation = "flicker"; // "flicker" | "grow"
let currentHparam = 0; // 0=Off, 2=Quick, 1=Full (match backend Types.h)
let currentDemoMode = false;
let currentDebugMode = false;
let InferenceSnapLoadInFlight = false;
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
// hparam elements
const elHparamCard = document.getElementById("hparam-card");
const elHparamBadge = document.getElementById("hparam-badge");
const elHparamMeter = document.getElementById("hparam-meter");
const elHparamHint = document.getElementById("hparam-hint");
const elHparamCnnOnly = document.getElementById("hparam-cnn-only");
const hparamBtns = Array.from(document.querySelectorAll(".hparam-seg-btn"));

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

// Demo & Debug Mode DOM elements
const viewDemoRun = document.getElementById("view-demo-run");
const elDebugHwBadge = document.getElementById("debug-hw-badge");
const elDebugFakeBadge = document.getElementById("debug-fake-badge");
const elDemoDisabledOverlay = document.getElementById("demo-disabled-overlay");
const elDemoModeToggle = document.getElementById("toggle-demo-mode");
const elOnnxOverlayToggle = document.getElementById("toggle-onnx-overlay");
const elDemoModeStatusBadge = document.getElementById("demo-mode-status-badge");
const elOnnxOverlayStatusBadge = document.getElementById(
  "onnx-overlay-status-badge",
);
const elDemoSessionLabel = document.getElementById("demo-session-label");
let consumedDemoMode = false;
let consumedDebugMode = false;
let inferenceSnapInterval = null;
let lastStateData = null; // needed for get/inference snap argument

let currIdx = 0;
let prevIdx = 0;
let prevTrainFailMsg = "";
let homeConsumedIdxChange = false;
let homeConsumedTrainChange = false;
let trainDismissed = false;
try {
  // remember dismiss across reloads
  trainDismissed = localStorage.getItem("trainDismissed") === "1";
} catch {}

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
    viewDemoRun,
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
    case "demo_run":
      viewDemoRun.classList.remove("hidden");
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

// (5c) onnx loading overlay helper
function showOnnxLoadingOverlay(show) {
  const el = document.getElementById("onnx-loading-overlay");
  if (!el) return;
  el.classList.toggle("hidden", !show);
  el.setAttribute("aria-hidden", show ? "false" : "true");
}

// (6) update settings from backend when we first enter settings page (rising edge trigger)
function updateSettingsFromState(data) {
  const arch = data.settings.train_arch_setting;
  const calib = data.settings.calib_data_setting;
  const stim_mode_i = data.settings.stim_mode; // 0=flicker, 1=grow
  const waveform_i = data.settings.waveform; // 0=square, 1=sine
  const hparam_i = data.settings.hparam;
  const debug_mode_i = data.settings.debug_mode;
  const demo_mode_i = data.settings.demo_mode;
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
  const trainArchNow = Number(selTrainArch?.value ?? arch ?? 0);
  renderHparamUI(hparam_i ?? 0, trainArchNow); // dep on train arch...
  currentDemoMode = demo_mode_i === true;
  currentDebugMode = debug_mode_i === true;
  if (elDemoModeToggle) elDemoModeToggle.checked = currentDemoMode;
  if (elOnnxOverlayToggle) elOnnxOverlayToggle.checked = currentDebugMode;
  if (elDemoModeStatusBadge) {
    elDemoModeStatusBadge.textContent = currentDemoMode ? "ON" : "OFF";
    elDemoModeStatusBadge.classList.toggle("on", currentDemoMode);
    elDemoModeStatusBadge.classList.toggle("off", !currentDemoMode);
  }
  if (elOnnxOverlayStatusBadge) {
    elOnnxOverlayStatusBadge.textContent = currentDebugMode ? "ON" : "OFF";
    elOnnxOverlayStatusBadge.classList.toggle("on", currentDebugMode);
    elOnnxOverlayStatusBadge.classList.toggle("off", !currentDebugMode);
  }

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
          { okText: "OK" },
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

// (10) Hparam helpers for settings page
function getHparamBtn(val) {
  return hparamBtns.find((b) => Number(b.dataset.hp) === Number(val)) || null;
}
function renderHparamUI(hpVal, trainArchVal) {
  // trainArchVal: 0=CNN, 1=SVM (based on <select>)
  const isCnn = Number(trainArchVal) === 0;

  // If not CNN, force Off
  const effectiveHp = isCnn ? Number(hpVal) : 0;

  // CNN-only note visibility
  if (elHparamCnnOnly) elHparamCnnOnly.classList.toggle("hidden", isCnn);

  // Disable buttons when SVM
  hparamBtns.forEach((b) => {
    b.disabled = !isCnn;
    b.classList.toggle("is-disabled", !isCnn); // if you want a CSS hook
  });

  // Active button styling
  hparamBtns.forEach((b) =>
    b.classList.toggle("is-active", Number(b.dataset.hp) === effectiveHp),
  );

  // Badge text + classes
  if (elHparamBadge) {
    elHparamBadge.classList.remove("off", "quick", "full");
    if (effectiveHp === 2) {
      elHparamBadge.textContent = "Quick";
      elHparamBadge.classList.add("quick");
    } else if (effectiveHp === 1) {
      elHparamBadge.textContent = "Full";
      elHparamBadge.classList.add("full");
    } else {
      elHparamBadge.textContent = "Off";
      elHparamBadge.classList.add("off");
    }
  }

  // Meter fill class
  if (elHparamMeter) {
    elHparamMeter.classList.remove("off", "quick", "full");
    elHparamMeter.classList.add(
      effectiveHp === 2 ? "quick" : effectiveHp === 1 ? "full" : "off",
    );
  }

  // Hint text
  if (elHparamHint) {
    if (!isCnn) {
      elHparamHint.textContent =
        "Hyperparameter tuning is only available for CNN training.";
    } else if (effectiveHp === 0) {
      elHparamHint.textContent = "Fastest training. Uses default settings.";
    } else if (effectiveHp === 2) {
      elHparamHint.textContent =
        "Good tradeoff: small search to improve generalization.";
    } else {
      elHparamHint.textContent = "Best potential model. Can take a long time.";
    }
  }
  currentHparam = effectiveHp;
}
function attachHparamHandlers() {
  hparamBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const hp = Number(btn.dataset.hp);

      const trainArch = Number(selTrainArch?.value ?? 0); // 0=CNN, 1=SVM
      // If SVM, ignore clicks (or show a modal)
      if (trainArch !== 0) {
        showModal(
          "CNN only",
          "Hyperparameter tuning currently applies to CNN training only.",
        );
        return;
      }

      renderHparamUI(hp, trainArch);
      // NOTE: this does NOT save yet — it just updates UI state like waveform selection does
    });
  });

  // Also re-render if user flips Model Architecture (CNN <-> SVM)
  if (selTrainArch) {
    selTrainArch.addEventListener("change", () => {
      const trainArch = Number(selTrainArch.value);
      renderHparamUI(currentHparam, trainArch);
    });
  }
}

/* contract with backend: expect state.train_fail_msg to contain:
- empty string("") for successful training
- error msg str for failed training
*/
// (11) rendering training result

// (11a - subfunc) to render training result card upon FAILURE
function renderTrainingFailure(state, failMsg) {
  // Show the training summary card in failure state
  if (elHomeTrainSummary) elHomeTrainSummary.classList.remove("hidden");
  if (btnReopenTrainSummary) btnReopenTrainSummary.classList.add("hidden");

  // Set status pill to fail
  if (elTrainStatusPill) {
    elTrainStatusPill.classList.remove("success", "warn");
    elTrainStatusPill.classList.add("fail");
  }
  setTextEl(elTrainStatusLabel, "Failed");
  setTextEl(elTrainSummarySub, failMsg || "Training job encountered an error.");

  // Fill basic KPIs that we have
  setTextEl(elTrainKpiUser, state.active_subject_id);
  if (elTrainKpiArch) {
    const archE = state.settings ? state.settings.train_arch_setting : null;
    const archLbl = archE === 0 ? "CNN" : archE === 1 ? "SVM" : "—";
    elTrainKpiArch.textContent = archLbl;
  }

  // Clear metrics that don't apply
  setTextEl(elTrainKpiHoldout, "—");
  setTextEl(elTrainKpiTrain, "—");
  if (elTrainKpiHoldoutOk) elTrainKpiHoldoutOk.classList.add("hidden");
  if (elTrainKpiHoldoutWarn) elTrainKpiHoldoutWarn.classList.add("hidden");

  // Clear best frequency pair
  setTextEl(elTrainBestLeftHz, "—");
  setTextEl(elTrainBestRightHz, "—");
  setTextEl(elTrainBestLeftE, "—");
  setTextEl(elTrainBestRightE, "—");

  // Set all consistency checks to fail
  if (elCTrainOk) {
    elCTrainOk.classList.remove("ok", "warn");
    elCTrainOk.classList.add("fail");
  }
  if (elCCvOk) {
    elCCvOk.classList.remove("ok", "warn");
    elCCvOk.classList.add("fail");
  }
  if (elCOnnxOk) {
    elCOnnxOk.classList.remove("ok", "warn");
    elCOnnxOk.classList.add("fail");
  }
  if (elCHoldoutOk) {
    elCHoldoutOk.classList.remove("ok", "warn");
    elCHoldoutOk.classList.add("fail");
  }
  setTextEl(elCTrainMsg, "Failed");
  setTextEl(elCCvMsg, "Failed");
  setTextEl(elCOnnxMsg, "Failed");
  setTextEl(elCHoldoutMsg, "Failed");

  const issues = state.train_fail_issues || [];

  if (issues.length > 0) {
    // Show the table wrapper
    if (elTrainSuffEmpty) elTrainSuffEmpty.classList.add("hidden");
    if (elTrainSuffTableWrap) elTrainSuffTableWrap.classList.remove("hidden");
    clearTbody(elTrainSuffTbody);

    for (const issue of issues) {
      const tr = document.createElement("tr");
      tr.classList.add("row-bad"); // Red highlight

      const td = (v) => {
        const c = document.createElement("td");
        c.textContent =
          v === undefined || v === null || v === "" ? "—" : String(v);
        return c;
      };

      tr.appendChild(td(issue.stage));
      tr.appendChild(td("")); // Metric column (not applicable)
      tr.appendChild(td("")); // Actual column (not applicable)
      tr.appendChild(td("")); // Required column (not applicable)

      // Frequency column: show candidate freqs if available
      if (issue.details?.cand_freqs?.length > 0) {
        const freqStr = issue.details.cand_freqs.join(", ") + " Hz";
        tr.appendChild(td(freqStr));
      } else {
        tr.appendChild(td("—"));
      }

      // Message column: main diagnostic
      const msgCell = td(issue.message);
      msgCell.classList.add("msg"); // Use existing msg styling
      tr.appendChild(msgCell);

      elTrainSuffTbody.appendChild(tr);
    }
  } else {
    // No detailed issues, show generic empty state
    if (elTrainSuffEmpty) {
      elTrainSuffEmpty.classList.remove("hidden");
      const titleEl = elTrainSuffEmpty.querySelector(".train-empty-title");
      const subEl = elTrainSuffEmpty.querySelector(".train-empty-sub");
      if (titleEl) titleEl.textContent = "Training Failed";
      if (subEl)
        subEl.textContent =
          "No diagnostic data available. Please try recalibrating.";
    }
    if (elTrainSuffTableWrap) elTrainSuffTableWrap.classList.add("hidden");
    clearTbody(elTrainSuffTbody);
  }
}

function renderTrainingResult(state) {
  if (!state) return;
  if (trainDismissed) return;

  const failMsg = state.train_fail_msg || "";
  const hasFailed = failMsg.trim() != ""; // removes whitespace for safety

  // prioritize fail status if there is one in train_fail_msg (rising edge)
  if (hasFailed) {
    renderTrainingFailure(state, failMsg);
    return;
  }

  // else -> SUCCESS CARD
  // format from backend:
  // state.data_insuff = [{ metric, required, actual, frequency_hz, stage, message }, ...]
  const rows = state.data_insuff;
  const hasRows = Array.isArray(rows) && rows.length > 0;
  // Show the training summary card if we have either training numbers or issues.
  const hasTrainingNumbers =
    typeof state.final_holdout_acc === "number" ||
    typeof state.final_train_acc === "number" ||
    typeof state.freq_left_hz === "number" ||
    typeof state.freq_right_hz === "number";

  if (!hasRows && !hasTrainingNumbers) {
    // Nothing to show at all
    if (elHomeTrainSummary) elHomeTrainSummary.classList.add("hidden");
    // hide reopen too
    if (btnReopenTrainSummary) btnReopenTrainSummary.classList.add("hidden");
    return;
  }

  // Fill KPIs
  setTextEl(elTrainKpiUser, state.active_subject_id);
  if (elTrainKpiArch) {
    const archE = state.settings ? state.settings.train_arch_setting : null;
    // backend: TrainArch_CNN=0, TrainArch_SVM=1
    const archLbl = archE === 0 ? "CNN" : archE === 1 ? "SVM" : "—";
    elTrainKpiArch.textContent = archLbl;
  }
  setTextEl(elTrainKpiHoldout, fmtPct(state.final_holdout_acc));
  setTextEl(elTrainKpiTrain, fmtPct(state.final_train_acc));

  // Holdout badge
  const h = state.final_holdout_acc;
  let holdoutPct = null;
  if (typeof h === "number" && isFinite(h)) holdoutPct = h <= 1.2 ? h * 100 : h;

  const ok = holdoutPct !== null && holdoutPct >= 70.0;
  if (ok) {
    if (elTrainKpiHoldoutOk) elTrainKpiHoldoutOk.classList.remove("hidden");
    if (elTrainKpiHoldoutWarn) elTrainKpiHoldoutWarn.classList.add("hidden");
  } else {
    if (elTrainKpiHoldoutWarn) elTrainKpiHoldoutWarn.classList.remove("hidden");
    if (elTrainKpiHoldoutOk) elTrainKpiHoldoutOk.classList.add("hidden");
  }

  // Best frequency pair
  setTextEl(elTrainBestLeftHz, state.freq_left_hz);
  setTextEl(elTrainBestRightHz, state.freq_right_hz);
  setTextEl(elTrainBestLeftE, state.freq_left_hz_e);
  setTextEl(elTrainBestRightE, state.freq_right_hz_e);

  // Status pill + subtitle
  // If insuff rows exist -> warn; else success
  if (elTrainStatusPill) {
    elTrainStatusPill.classList.remove("success", "warn", "fail");
    elTrainStatusPill.classList.add(hasRows ? "warn" : "success");
  }
  setTextEl(elTrainStatusLabel, hasRows ? "Warnings" : "Success");
  setTextEl(
    elTrainSummarySub,
    hasRows
      ? "Some sufficiency checks failed. Collect more calibration data."
      : "Model artifacts exported and ready for Run Mode.",
  );

  // Table render
  if (!hasRows) {
    if (elTrainSuffEmpty) elTrainSuffEmpty.classList.remove("hidden");
    if (elTrainSuffTableWrap) elTrainSuffTableWrap.classList.add("hidden");
    clearTbody(elTrainSuffTbody);
    return;
  }

  if (elTrainSuffEmpty) elTrainSuffEmpty.classList.add("hidden");
  if (elTrainSuffTableWrap) elTrainSuffTableWrap.classList.remove("hidden");
  clearTbody(elTrainSuffTbody);

  for (const di of rows) {
    const tr = document.createElement("tr");

    const td = (v) => {
      const c = document.createElement("td");
      c.textContent =
        v === undefined || v === null || v === "" ? "—" : String(v);
      return c;
    };

    tr.appendChild(td(di.stage));
    tr.appendChild(td(di.metric));
    tr.appendChild(td(di.actual));
    tr.appendChild(td(di.required));

    // backend always emits numeric frequency_hz
    tr.appendChild(td(`${di.frequency_hz} Hz`));

    tr.appendChild(td(di.message));

    elTrainSuffTbody.appendChild(tr);
  }
}

// (12) toggle home "welcome" containers visibility
function setHomeWelcomeVisible(visible) {
  const welcomeElements = [
    document.querySelector("#view-home > h2"),
    document.querySelector("#view-home > .view-lead"),
    document.querySelector("#view-home > .home-actions"),
    document.querySelector("#view-home > .home-tip"),
  ];
  welcomeElements.forEach((el) => {
    if (el) {
      if (visible) {
        el.classList.remove("hidden");
      } else {
        el.classList.add("hidden");
      }
    }
  });
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

// lil helper to make decimals -> percents
// Accepts 0..1 OR 0..100. Returns string like "84.2%"
function fmtPct(x) {
  if (typeof x !== "number" || !isFinite(x)) return "—%";
  const pct = x <= 1.2 ? x * 100.0 : x;
  return `${pct.toFixed(1)}%`;
}

function fmt1(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(1);
}

// SAVED SESSION FORMATTING
// Formatters for holdout accuracy
function holdoutAccClass(acc) {
  if (acc == null || acc <= 0) return "acc-none";
  const pct = acc < 1.0 ? acc * 100 : acc;
  if (pct >= 77) return "acc-good";
  if (pct >= 63) return "acc-warn";
  return "acc-fail";
}
function fmtAcc(acc) {
  if (acc == null || acc <= 0) return "—";
  const pct = acc < 1.0 ? acc * 100 : acc;
  return `${pct.toFixed(1)}%`;
}
// Formatter for readable session_id timestamp
// e.g. "2026-02-05_12-03-10" -> "Feb 5, 2026 · 12:03"
function fmtSessionId(id) {
  if (!id) return "—";
  // Match date portion wherever it appears: YYYY-MM-DD_HH-MM
  const m = id.match(/(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})/);
  if (!m) return id;
  const [, yr, mo, dy, hr, mn] = m;
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const mLabel = months[parseInt(mo, 10) - 1] ?? mo;
  return `${mLabel} ${parseInt(dy, 10)}, ${yr} · ${hr}:${mn}`;
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function setTextEl(el, v) {
  if (!el) return;
  el.textContent = v === undefined || v === null || v === "" ? "—" : String(v);
}

function clearTbody(tbody) {
  if (!tbody) return;
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
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
      return 9;
    case 3:
      return 10;
    case 4:
      return 11;
    case 5:
      return 12;
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

// Model arch enum -> label
function archEnumToLabel(arch) {
  if (typeof arch == "string" && arch.trim()) return arch.trim().toUpperCase();
  if (arch == 0) return "CNN";
  if (arch == 1) return "SVM";
  return "Unknown";
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
  const currIdx = data.curr_idx;
  const currTrainFailMsg = data.train_fail_msg;

  // capture no_ssvep_test state transitions so we only randomize freq pair ONCE per entry
  const enteringNeutral = stimState === 10 && prevStimState !== 10;
  const leavingNeutral = prevStimState === 10 && stimState !== 10;
  // on falling edge (exit), reset
  if (leavingNeutral) {
    stopNeutralFlicker();
    neutralPairChosen = false;
  }

  // detect when we're leaving saved sessions page to reset consumption flag
  const leavingSavedSessions = prevStimState === 4 && stimState !== 4;
  if (leavingSavedSessions) {
    sessionsPageConsumedRefresh = false;
  }

  // detection of rising edges for different params
  const stateChanged = prevStimState !== stimState;
  const sessionIdxChanged = currIdx != prevIdx;
  const trainStatusChanged = currTrainFailMsg != prevTrainFailMsg;
  if (sessionIdxChanged) {
    homeConsumedIdxChange = false;
  }
  if (trainStatusChanged) {
    homeConsumedTrainChange = false;
  }

  // on sessIdx change -> clear training summary (as long as its not from pending_training state)
  if (sessionIdxChanged && prevStimState !== 8) {
    if (elHomeTrainSummary) elHomeTrainSummary.classList.add("hidden");
    if (btnReopenTrainSummary) btnReopenTrainSummary.classList.add("hidden");
    trainDismissed = false; // for when we go back into training next time
    homeConsumedIdxChange = false; // in case we're coming in here instead of above if
    homeConsumedTrainChange = true; // dont need to consume train change again
    setHomeWelcomeVisible(true); // Show welcome content instead of training status
  }

  // detect when leaving run mode
  if (prevStimState == 0 && stimState != 0) {
    stopInferenceSnapPolling();
    consumedDemoMode = false;
  }

  const pauseVisible =
    stimState === 0 || // Active_Run
    stimState === 1 || // Active_Calib
    stimState === 2 || // Instructions
    stimState === 10; // NoSSVEP_Test
  setPauseButtonVisible(pauseVisible);

  {
    // checking compile time flag
    const isFake = data.is_fake_acq === true; // backend sends string "true"/"false"
    if (elDebugHwBadge) elDebugHwBadge.classList.toggle("hidden", isFake);
    if (elDebugFakeBadge) elDebugFakeBadge.classList.toggle("hidden", !isFake);
    // Only allow demo mode toggle in FAKE builds
    if (elDemoModeToggle) elDemoModeToggle.disabled = !isFake;
    if (elDemoDisabledOverlay)
      elDemoDisabledOverlay.classList.toggle("hidden", isFake);
  }

  // 0 = Active_Run, 1 = Active_Calib, 2 = Instructions, 3 = Home, 4 = saved_sessions, 5 = run_options, 6 = hardware_checks, 7 = calib_options, 8 = pending_training, 9 = settings, 10 = no_ssvep, 11 = paused, 12 = None
  if (stimState === 3 /* Home */ || stimState === 12 /* None */) {
    stopAllStimuli();
    stopHardwareMode();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    settingsInitiallyUpdated = false; // reset flag
    if (
      prevStimState !== 4 /*Saved Sessions*/ &&
      ((sessionIdxChanged && !homeConsumedIdxChange) ||
        (trainStatusChanged && !homeConsumedTrainChange))
    ) {
      // new result arrived -> open
      // if we just changed session from saved sessions -> don't open (not available anymore)
      renderTrainingResult(data);
      if (elHomeTrainSummary) elHomeTrainSummary.classList.remove("hidden");
      if (btnReopenTrainSummary) btnReopenTrainSummary.classList.add("hidden");
      setHomeWelcomeVisible(false); // hide home content
      homeConsumedIdxChange = true;
      homeConsumedTrainChange = true;
      trainDismissed = false;
      try {
        localStorage.setItem("trainDismissed", "0");
      } catch {}
    } else if (
      trainDismissed == true ||
      (sessionIdxChanged && !homeConsumedIdxChange && prevStimState == 4)
    ) {
      // hide training summary
      setHomeWelcomeVisible(true); // recover home content
      if (elHomeTrainSummary) elHomeTrainSummary.classList.add("hidden");
      if (btnReopenTrainSummary)
        btnReopenTrainSummary.classList.remove("hidden");
    }
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
    const isDemoMode = data.settings.demo_mode === true;
    if (isDemoMode) {
      // show inference dashboard instead of arrows
      applyBodyMode({ fullscreen: true, targets: false, run: false });
      stopAllStimuli();
      showView("demo_run");
      if (!consumedDemoMode) {
        startInferenceSnapPolling();
        if (elDemoSessionLabel) {
          const subj = data.active_subject_id || "unknown";
          const sess = data.active_session_id || "";
          elDemoSessionLabel.textContent = sess
            ? `${subj} · ${sess.slice(0, 16)}`
            : `${subj} · live dataset stream`;
        }
        consumedDemoMode = true; // rising edge complete
      }
    } else {
      stopInferenceSnapPolling();
      applyBodyMode({ fullscreen: true, targets: true, run: true });
      showView("active_run");
      const runLeftHz = data.freq_left_hz ?? data.freq_hz ?? 0;
      const runRightHz = data.freq_right_hz ?? data.freq_hz ?? 0;
      startRunFlicker(runLeftHz, runRightHz);
    }
  } else if (stimState === 4 /* Saved Sessions */) {
    stopAllStimuli();
    applyBodyMode({ fullscreen: false, targets: false, run: false });
    showView("saved_sessions");
    if (!sessionsPageConsumedRefresh) {
      fetchAndRenderSessions(); /// fetch fresh on each entry to saved sess
      sessionsPageConsumedRefresh = true;
    }
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
    showView("home");
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
  showOnnxLoadingOverlay(data.is_onnx_reloading === true);

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
    prevIdx = currIdx;
    prevTrainFailMsg = currTrainFailMsg;
    return;
  }

  // if backend says "show popup" and it's not visible, open it
  if (popupEnumIdx != 0 && !modalVisible) {
    switch (popupEnumIdx) {
      case 1: // UIPopup_MustCalibBeforeRun
        showModal(
          "No trained models found",
          "Please complete at least one calibration session before trying to start run mode.",
        );
        break;
      case 3: // UIPopup_TooManyBadWindowsInRun
        showModal(
          "Too many artifactual windows detected",
          "Please check headset placement and rerun hardware checks to verify signal.",
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
          },
        );
        break;
      case 6: // UIPopup_ConfirmHighFreqOk
        showModal(
          "High frequency SSVEP decoding (>20Hz) will be attempted",
          "The final model's performance may be poor, and device functionality may be limited.",
        );
        break;
      case 7: // UIPopup_TrainJobFailed
        showModal(
          "Training job failed",
          "There was an internal error. Please try calibration again.",
        );
        break;
      default:
        showModal(
          "DEBUG MSG",
          "we should not reach here! check that UIPopup Enum matches JS cases",
        );
        break;
    }
  }
  // update state
  prevStimState = stimState;
  prevIdx = currIdx;
  prevTrainFailMsg = currTrainFailMsg;
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
    lastStateData = data;
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

// ================================ 11) SAVED SESSIONS RUNTIME BUILDER ==========================================

function buildSessionCard(sess, activeSessionId) {
  // diagnostics & info from sess (http server GET/SESSIONS)
  const isActive = sess.session_id == activeSessionId;
  const isBroken = !sess.onnx_model_found || !sess.json_meta_found;
  const hasInsuff = sess.data_insufficiency == true;
  const accClass = holdoutAccClass(sess.model_holdout_acc);
  const accText = fmtAcc(sess.model_holdout_acc);
  const leftHz = sess.freq_left_hz > 0 ? sess.freq_left_hz : null;
  const rightHz = sess.freq_right_hz > 0 ? sess.freq_right_hz : null;

  // main card element <3
  const card = document.createElement("div");
  card.className = "session-card";
  if (isActive) card.classList.add("is-active");
  if (isBroken) card.classList.add("is-broken");
  card.dataset.sessIdx = sess.sess_idx;

  // badges for da cards, we'll make them spans
  const activeBadge = isActive
    ? `<span class="sess-badge active"><span class="bdot"></span>Active</span>`
    : "";
  const archLabel = archEnumToLabel(sess.model_arch);
  const archBadge =
    archLabel !== "—"
      ? `<span class="sess-badge arch">${archLabel}</span>`
      : "";
  const qualBadge = isBroken
    ? `<span class="sess-badge fail">Incomplete</span>`
    : hasInsuff
      ? `<span class="sess-badge warn">Low data</span>`
      : "";
  const freqPairHtml =
    leftHz || rightHz
      ? `<div class="session-freq-pair">
        ${leftHz ? `<span class="freq-chip fc-left">${leftHz}<span class="fc-unit">Hz</span></span>` : ""}
        ${leftHz && rightHz ? `<span class="freq-pair-arrow">→</span>` : ""}
        ${rightHz ? `<span class="freq-chip fc-right">${rightHz}<span class="fc-unit">Hz</span></span>` : ""}
       </div>`
      : `<span style="font-size:0.78rem;color:rgba(148,163,184,0.55)">No freq data</span>`;
  const onnxChip = sess.onnx_model_found
    ? `<span class="artifact-chip found"><span class="ac-icon">✓</span>ONNX</span>`
    : `<span class="artifact-chip missing"><span class="ac-icon">✗</span>ONNX</span>`;

  const metaChip = sess.json_meta_found
    ? `<span class="artifact-chip found"><span class="ac-icon">✓</span>Meta</span>`
    : `<span class="artifact-chip missing"><span class="ac-icon">✗</span>Meta</span>`;

  const insuffChip = hasInsuff
    ? `<span class="artifact-chip warn"><span class="ac-icon">⚠</span>Low data</span>`
    : "";

  // Build card HTML
  card.innerHTML = `
    <div class="session-card-head">
      <div class="session-card-identity">
        <div class="session-card-subject">${sess.subject || "Unknown user"}</div>
        <div class="session-card-id">${fmtSessionId(sess.session_id)}</div>
      </div>
      <div class="session-card-badges">
        ${activeBadge}${archBadge}${qualBadge}
      </div>
    </div>

    <div class="session-card-divider"></div>

    <div class="session-card-metrics">
      <div class="sess-metric">
        <div class="sess-metric-k">Holdout Acc</div>
        <div class="sess-metric-v ${accClass}">${accText}</div>
      </div>
      <div class="sess-metric">
        <div class="sess-metric-k">Freq Pair</div>
        ${freqPairHtml}
      </div>
    </div>

    <div class="session-card-artifacts">
      ${onnxChip}${metaChip}${insuffChip}
    </div>

    <!-- Loading indicator shown on click -->
    <div class="session-card-loading">
      <div class="loading-spinner"></div>
    </div>
  `;

  // Select handler
  if (!isBroken) {
    card.addEventListener("click", () => {
      selectSession(sess.sess_idx, card); // select by idx
    });
  } else {
    // no clicks allowed !
    card.title =
      "Session artifacts are corrupt or incomplete, cannot load this model.";
  }

  return card;
}

// render meta strip chips
function renderSessionsMetaStrip(data) {
  const el = document.getElementById("sessions-meta-strip");
  if (!el) return;
  // info from GET/sessions
  const total = data.num_saved_sessions ?? 0;
  const ok = data.num_saved_sessions_ok ?? 0;
  const bad = total - ok; // the ones with missing artifacts

  el.innerHTML = `
    <span class="sessions-meta-chip ok">
      <span class="chip-dot"></span>${total} session${total !== 1 ? "s" : ""}
    </span>
    <span class="sessions-meta-chip ok">
      <span class="chip-dot"></span>${ok} ready
    </span>
    ${bad > 0 ? `<span class="sessions-meta-chip fail"><span class="chip-dot"></span>${bad} incomplete</span>` : ""}
  `;
}

// Show skeleton cards while loading data upon entering page
function showSessionSkeletons(n = 3) {
  const grid = document.getElementById("sessions-grid");
  if (!grid) return;
  grid.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const sk = document.createElement("div");
    sk.className = "session-card-skeleton";
    sk.innerHTML = `
      <div class="skeleton-line" style="width:55%;height:13px;margin-bottom:10px"></div>
      <div class="skeleton-line" style="width:38%;height:9px;margin-bottom:14px"></div>
      <div class="skeleton-line" style="width:100%;height:1px;opacity:0.4;margin-bottom:12px"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
        <div class="skeleton-line" style="height:44px;border-radius:10px"></div>
        <div class="skeleton-line" style="height:44px;border-radius:10px"></div>
      </div>
      <div style="display:flex;gap:5px">
        <div class="skeleton-line" style="width:56px;height:20px;border-radius:6px"></div>
        <div class="skeleton-line" style="width:56px;height:20px;border-radius:6px"></div>
      </div>
    `;
    grid.appendChild(sk);
  }
}

// Main fetch + render function for entering the saved sessions page
async function fetchAndRenderSessions() {
  if (
    typeof sessionsLoadInFlight != "undefined" &&
    sessionsLoadInFlight == true
  )
    return; // don't render again if we're already trying to render
  sessionsLoadInFlight = true;

  const grid = document.getElementById("sessions-grid");
  const emptyState = document.getElementById("sessions-empty-state");
  if (!grid || !emptyState) {
    sessionsLoadInFlight = false;
    return;
  }

  // show skeleton while fetching
  showSessionSkeletons(3);
  emptyState.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/sessions`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    grid.innerHTML = "";

    const sessions = Array.isArray(data.sessions) ? data.sessions : [];

    if (sessions.length === 0) {
      emptyState.classList.remove("hidden");
      const strip = document.getElementById("sessions-meta-strip");
      if (strip) strip.innerHTML = "";
      console.log("pathway 1 for remove emptystate hidden");
    } else {
      console.log("pathway 2 for remove emptystate hidden"); // HERE
      renderSessionsMetaStrip(data);
      emptyState.classList.add("hidden");

      // Sort: active session first, then by sess_idx descending (newest first!!)
      sessions.sort((a, b) => {
        if (a.session_id === data.active_session_id) return -1;
        if (b.session_id === data.active_session_id) return 1;
        return (b.sess_idx ?? 0) - (a.sess_idx ?? 0);
      });
      // Build all the cards
      sessions.forEach((sess) => {
        const card = buildSessionCard(sess, data.active_session_id);
        grid.appendChild(card);
      });
    }
    logLine(`Loaded ${sessions.length} saved session(s)`);
  } catch (err) {
    grid.innerHTML = "";
    emptyState.classList.remove("hidden");
    console.log("GET /sessions error:", err);
    logLine("GET /sessions error: " + err);
  } finally {
    sessionsLoadInFlight = false; // always reset before exiting this func...
  }
}

// Select a new session (post req to backend)
async function selectSession(sessIdx, cardEl) {
  if (cardEl) cardEl.classList.add("is-loading"); // for when backend is consuming the req

  const payload = {
    action: "select_session",
    sess_idx: sessIdx,
  };

  try {
    const res = await fetch(`${API_BASE}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      logLine(
        `POST /event failed (${res.status}) for select_session idx=${sessIdx}`,
      );
      if (cardEl) cardEl.classList.remove("is-loading");
      return;
    }

    logLine(`Selected session idx=${sessIdx}`);
  } catch (err) {
    logLine(`POST /event error for select_session: ${err}`);
    if (cardEl) cardEl.classList.remove("is-loading");
  }
}

// ==================== 12) DEMO MODE RUNTIME BUILDER ==========================================================
// Main fetch + render function for entering the saved sessions page
async function fetchAndRenderInferenceSnap(state) {
  if (
    typeof InferenceSnapLoadInFlight != "undefined" &&
    InferenceSnapLoadInFlight == true
  )
    return; // don't render again if we're already trying to render
  InferenceSnapLoadInFlight = true;

  try {
    const freq_left_hz = state.freq_left_hz;
    const freq_right_hz = state.freq_right_hz;

    const res = await fetch(`${API_BASE}/runtime_inference_snapshot`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    updateDemoRunView(data, freq_left_hz, freq_right_hz);
  } catch (err) {
    console.log("GET /runtime_inference_snapshot error:", err);
  } finally {
    InferenceSnapLoadInFlight = false; // always reset before exiting this func...
  }
}

// demo mode view for dataset streamer
// state = get/state
// data = get/runtime_inference_snapshot
function updateDemoRunView(data, freq_left_hz, freq_right_hz) {
  const d = data.onnx_inference;
  if (!d) return;

  // 1) Prediction panel
  const predEl = document.getElementById("demo-pred-state");
  const predIcon = document.getElementById("demo-pred-icon");
  const predLabel = document.getElementById("demo-pred-label");
  const predSub = document.getElementById("demo-pred-sub");
  const artBadge = document.getElementById("demo-artifact-badge");
  const actFlash = document.getElementById("demo-actuation-flash");

  // Remove all state classes
  if (predEl)
    predEl.classList.remove("left", "right", "none", "unknown", "artifact");

  if (d.is_artifactual) {
    if (predEl) predEl.classList.add("artifact");
    if (predIcon) predIcon.textContent = "⚡";
    if (predLabel) predLabel.textContent = "ARTIFACT";
    if (predSub) predSub.textContent = "Window rejected — signal quality check";
    if (artBadge) artBadge.classList.remove("hidden");
  } else {
    if (artBadge) artBadge.classList.add("hidden");
    const ps = d.predicted_state; // 0=L 1=R 2=None 3=Unknown
    const stateMap = {
      0: { cls: "left", icon: "←", lbl: "LEFT", sub: "Looking left detected" },
      1: {
        cls: "right",
        icon: "→",
        lbl: "RIGHT",
        sub: "Looking right detected",
      },
      2: { cls: "none", icon: "·", lbl: "NONE", sub: "No SSVEP detected" },
      3: {
        cls: "unknown",
        icon: "?",
        lbl: "UNKNOWN",
        sub: "Classifier returned unknown",
      },
    };
    const info = stateMap[ps] ?? stateMap[3];
    if (predEl) predEl.classList.add(info.cls);
    if (predIcon) predIcon.textContent = info.icon;
    if (predLabel) predLabel.textContent = info.lbl;
    if (predSub) predSub.textContent = info.sub;
  }

  // 2) Debounce bar
  const sc = d.stable_count ?? 0;
  const st = d.stable_target ?? 10;
  const pct = st > 0 ? Math.min(100, Math.round((sc / st) * 100)) : 0;
  const debounceBar = document.getElementById("demo-debounce-bar");
  const debounceFrac = document.getElementById("demo-debounce-fraction");
  if (debounceBar) debounceBar.style.width = pct + "%";
  if (debounceFrac) debounceFrac.textContent = `${sc} / ${st}`;

  // Actuation flash: show briefly when actuation_count increments
  if (actFlash) {
    if (!actFlash._lastCount) actFlash._lastCount = 0;
    if (d.actuation_count > actFlash._lastCount) {
      actFlash._lastCount = d.actuation_count;
      actFlash.classList.remove("hidden");
      clearTimeout(actFlash._timer);
      actFlash._timer = setTimeout(
        () => actFlash.classList.add("hidden"),
        1200,
      );
    }
  }

  // 3) Softmax confidence bars
  const sm = Array.isArray(d.softmax) ? d.softmax : [0.333, 0.333, 0.334];
  const pcts = sm.map((v) => Math.round(v * 100));
  ["left", "right", "none"].forEach((side, i) => {
    const bar = document.getElementById(`demo-bar-${side}`);
    const pctEl = document.getElementById(`demo-pct-${side}`);
    if (bar) bar.style.width = pcts[i] + "%";
    if (pctEl) pctEl.textContent = pcts[i] + "%";
  });

  // 4) Raw logits
  const lg = Array.isArray(d.logits) ? d.logits : [0, 0, 0];
  ["left", "right", "none"].forEach((side, i) => {
    const el = document.getElementById(`demo-logit-${side}`);
    if (el) el.textContent = typeof lg[i] === "number" ? lg[i].toFixed(3) : "—";
  });

  // 5) Streamer panel
  const str = data.streamer ?? {};
  const gtHzEl = document.getElementById("demo-gt-hz");
  const gtLblEl = document.getElementById("demo-gt-label");
  const tgtIdxEl = document.getElementById("demo-target-idx");
  const trialIdxEl = document.getElementById("demo-trial-idx");
  const blockIdxEl = document.getElementById("demo-block-idx");
  const cyclingEl = document.getElementById("demo-cycling-badge");

  const hz = str.active_target_hz ?? 0;
  if (gtHzEl) gtHzEl.textContent = hz > 0 ? `${hz} Hz` : "—";

  // Map hz → LEFT/RIGHT/NONE using session's freq pair
  let gtLabel = "NO SSVEP";
  if (hz > 0) {
    const leftHz = freq_left_hz ?? -1;
    const rightHz = freq_right_hz ?? -1;
    if (Math.abs(hz - leftHz) < 0.5) gtLabel = "LEFT";
    else if (Math.abs(hz - rightHz) < 0.5) gtLabel = "RIGHT";
    else gtLabel = `OTHER (${hz} Hz)`;
  }
  if (gtLblEl) gtLblEl.textContent = gtLabel;

  if (tgtIdxEl)
    tgtIdxEl.textContent =
      str.active_target_idx >= 0 ? str.active_target_idx : "—";
  if (trialIdxEl)
    trialIdxEl.textContent =
      str.active_trial_idx >= 0 ? str.active_trial_idx : "—";
  if (blockIdxEl)
    blockIdxEl.textContent = str.block_idx >= 0 ? str.block_idx : "—";
  if (cyclingEl) cyclingEl.classList.toggle("hidden", !str.is_cycling);

  // 6) Verdict
  const verdictEl = document.getElementById("demo-verdict");
  const verdictIcon = document.getElementById("demo-verdict-icon");
  const verdictLabel = document.getElementById("demo-verdict-label");
  const verdictSub = document.getElementById("demo-verdict-sub");

  if (verdictEl)
    verdictEl.classList.remove(
      "verdict-correct",
      "verdict-incorrect",
      "verdict-unknown",
    );

  if (d.is_artifactual) {
    if (verdictEl) verdictEl.classList.add("verdict-unknown");
    if (verdictIcon) verdictIcon.textContent = "—";
    if (verdictLabel) verdictLabel.textContent = "N/A";
    if (verdictSub) verdictSub.textContent = "Artifact window";
  } else {
    // Map predicted_state (int) to direction string
    const predDir =
      d.predicted_state === 0
        ? "LEFT"
        : d.predicted_state === 1
          ? "RIGHT"
          : d.predicted_state === 2
            ? "NONE"
            : "UNKNOWN";

    if (gtLabel === "LEFT" || gtLabel === "RIGHT" || gtLabel === "NO SSVEP") {
      const gtDir = gtLabel === "NO SSVEP" ? "NONE" : gtLabel;
      if (predDir === "UNKNOWN") {
        if (verdictEl) verdictEl.classList.add("verdict-unknown");
        if (verdictIcon) verdictIcon.textContent = "?";
        if (verdictLabel) verdictLabel.textContent = "UNKNOWN";
        if (verdictSub) verdictSub.textContent = `GT: ${gtDir}`;
      } else if (predDir === gtDir) {
        if (verdictEl) verdictEl.classList.add("verdict-correct");
        if (verdictIcon) verdictIcon.textContent = "✓";
        if (verdictLabel) verdictLabel.textContent = "CORRECT";
        if (verdictSub) verdictSub.textContent = `${predDir} = ${gtDir}`;
      } else {
        if (verdictEl) verdictEl.classList.add("verdict-incorrect");
        if (verdictIcon) verdictIcon.textContent = "✗";
        if (verdictLabel) verdictLabel.textContent = "INCORRECT";
        if (verdictSub)
          verdictSub.textContent = `Got ${predDir}, expected ${gtDir}`;
      }
    } else {
      if (verdictEl) verdictEl.classList.add("verdict-unknown");
      if (verdictIcon) verdictIcon.textContent = "—";
      if (verdictLabel) verdictLabel.textContent = "UNKNOWN";
      if (verdictSub) verdictSub.textContent = "GT not determined";
    }
  }

  // 7) Window counters
  const winCount = document.getElementById("demo-win-count");
  const artCount = document.getElementById("demo-art-count");
  const actCount = document.getElementById("demo-act-count");
  if (winCount) winCount.textContent = d.total_windows ?? 0;
  if (artCount) artCount.textContent = d.artifactual_windows ?? 0;
  if (actCount) actCount.textContent = d.actuation_count ?? 0;
}

// separate timer and polling function just for inference snapshots (demo/debug modes), independent of state poll
function startInferenceSnapPolling() {
  if (inferenceSnapInterval) return;
  inferenceSnapInterval = setInterval(() => {
    if (!InferenceSnapLoadInFlight && lastStateData)
      fetchAndRenderInferenceSnap(lastStateData);
  }, 100);
}
function stopInferenceSnapPolling() {
  if (inferenceSnapInterval) {
    clearInterval(inferenceSnapInterval);
    if (elDemoSessionLabel)
      elDemoSessionLabel.textContent = "Live inference · dataset stream";
    inferenceSnapInterval = null;
  }
}

// ================ 13) HARDWARE CHECKS MAIN RUN LOOP & PLOTTING HELPERS ===================================
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

// =============== 14) SEND POST EVENTS WHEN USER CLICKS BUTTONS (OR OTHER INPUTS) ===============
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
      "This device is not safe for use for individuals with photosensitivity.",
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
      "Please select both training architecture and calibration data options.",
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
    hparam: trainArch === 0 ? currentHparam : 0, // force off for SVM
    demo_mode: currentDemoMode,
    debug_mode: currentDebugMode,
    duration_active_s: Number(elDurActive?.value ?? 11), // otherwise default
    duration_none_s: Number(elDurNone?.value ?? 10),
    duration_rest_s: Number(elDurRest?.value ?? 8),
    num_times_cycle_repeats: Number(elCycleRep?.value ?? 3),
    selected_freqs_e, // int enums 1..15
  };
  console.log("SET_SETTINGS payload:", payload);
  logLine(
    `Saving timers: active=${payload.duration_active_s}s none=${payload.duration_none_s}s rest=${payload.duration_rest_s}s reps=${payload.num_times_cycle_repeats}`,
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
      `Settings saved (train_arch=${trainArch}, calib_data=${calibData} hparam=${currentHparam})`,
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

// ================== 15) INIT ON PAGE LOAD ===================
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
    measuredRefreshHz,
  );
  neutralRightStimulus = new FlickerStimulus(
    elNeutralRightArrow,
    measuredRefreshHz,
  );

  stimAnimationLoop();
  startPolling();
  attachFreqGridHandlers();
  attachHparamHandlers();

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

  if (btnDismissTrainSummary) {
    btnDismissTrainSummary.addEventListener("click", () => {
      if (btnReopenTrainSummary)
        btnReopenTrainSummary.classList.remove("hidden");
      if (elHomeTrainSummary) elHomeTrainSummary.classList.add("hidden");
      trainDismissed = true;
      setHomeWelcomeVisible(true);
      try {
        localStorage.setItem("trainDismissed", "1");
      } catch {}
    });
  }
  if (btnReopenTrainSummary) {
    btnReopenTrainSummary.addEventListener("click", () => {
      if (elHomeTrainSummary) elHomeTrainSummary.classList.remove("hidden");
      if (btnReopenTrainSummary) btnReopenTrainSummary.classList.add("hidden");
      trainDismissed = false;
      setHomeWelcomeVisible(false);
      try {
        localStorage.setItem("trainDismissed", "0");
      } catch {}
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

  if (elDemoModeToggle) {
    elDemoModeToggle.addEventListener("change", () => {
      currentDemoMode = elDemoModeToggle.checked;
      // Constraint: demo mode requires debug mode
      if (currentDemoMode) {
        currentDebugMode = true;
        if (elOnnxOverlayToggle) elOnnxOverlayToggle.checked = true;
        if (elOnnxOverlayStatusBadge) {
          elOnnxOverlayStatusBadge.textContent = "ON";
          elOnnxOverlayStatusBadge.classList.add("on");
          elOnnxOverlayStatusBadge.classList.remove("off");
        }
      }
      // Update demo badge
      if (elDemoModeStatusBadge) {
        elDemoModeStatusBadge.textContent = currentDemoMode ? "ON" : "OFF";
        elDemoModeStatusBadge.classList.toggle("on", currentDemoMode);
        elDemoModeStatusBadge.classList.toggle("off", !currentDemoMode);
      }
    });
  }

  if (elOnnxOverlayToggle) {
    elOnnxOverlayToggle.addEventListener("change", () => {
      currentDebugMode = elOnnxOverlayToggle.checked;
      // Constraint: turning off debug mode must also turn off demo mode
      if (!currentDebugMode && currentDemoMode) {
        currentDemoMode = false;
        if (elDemoModeToggle) elDemoModeToggle.checked = false;
        if (elDemoModeStatusBadge) {
          elDemoModeStatusBadge.textContent = "OFF";
          elDemoModeStatusBadge.classList.remove("on");
          elDemoModeStatusBadge.classList.add("off");
        }
      }
      // Update debug badge
      if (elOnnxOverlayStatusBadge) {
        elOnnxOverlayStatusBadge.textContent = currentDebugMode ? "ON" : "OFF";
        elOnnxOverlayStatusBadge.classList.toggle("on", currentDebugMode);
        elOnnxOverlayStatusBadge.classList.toggle("off", !currentDebugMode);
      }
    });
  }
}
// Init as soon as page loads
window.addEventListener("DOMContentLoaded", () => {
  init();
});
