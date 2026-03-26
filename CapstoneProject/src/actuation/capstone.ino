#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

/*
 * Serial protocol (9600 baud, \r\n line endings from println):
 *
 *  BOOT SEQUENCE
 *  ─────────────
 *  Arduino → host:  READY          (repeated every 400 ms until P received from host)
 *  Host    → Arduino: P            (ping to confirm connection)
 *  Arduino → host:  STATUS:CONNECTED
 *
 *  COMMANDS (host → Arduino, single char)
 *  ──────────────────────────────────────
 *  F / f  — flip forward
 *  B / b  — flip backward
 *  H      — heartbeat ping  → Arduino replies: HB
 *  S      — status request  → Arduino replies: full STATUS dump
 *
 *  RESPONSES (Arduino → host)
 *  ──────────────────────────
 *  STATUS:<key>=<value>   — structured status events (see STATUS_* below)
 *  HB                     — heartbeat ack
 *  DONE                   — command completed successfully
 *  NOCALIB                — command rejected: not calibrated yet
 *  ESTOP                  — command rejected: e-stop is active
 *  DBG:<message>          — verbose debug line (only sent when DBG_VERBOSE=1)
 *
 *  STATUS keys emitted by Arduino
 *  ───────────────────────────────
 *  STATUS:BOOT            — setup() entered
 *  STATUS:CONNECTED       — handshake P received
 *  STATUS:CALIB_START     — calibration stage 0 → 1
 *  STATUS:CALIB_RIGHT=<n> — right edge set (step count)
 *  STATUS:CALIB_LEFT=<n>  — left edge set (step count)
 *  STATUS:CALIB_DONE      — calibration complete, bookWidth known
 *  STATUS:ESTOP_ON        — e-stop engaged
 *  STATUS:ESTOP_OFF       — e-stop cleared
 *  STATUS:CMD_F           — forward flip command received
 *  STATUS:CMD_B           — backward flip command received
 *  STATUS:MODE=<IDLE|CALIB|FUNCTION>
 */

// ─── Mechanical team debug override & Other debug controls ──────────────────────────────────────────
#define DBG_VERBOSE 1 // Set to 1 to enable DBG: lines (e.g. when monitoring via Arduino IDE Serial Monitor)
// MECH TEAM: Set to 1 to skip the BCI handshake AND the calibration requirement (just straight debug B/F)
#define MECH_DEBUG_MODE 0
#define DBG(msg)  do { if (DBG_VERBOSE) { Serial.print("DBG:"); Serial.println(msg); } } while(0)
#define STATUS(kv) do { Serial.print("STATUS:"); Serial.println(kv); } while(0)

// ─── PWM / Servo ─────────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
/* SERVO DEFINITIONS */
#define CLIP2_MIN  150
#define CLIP2_MAX  590
#define CLIP1_MIN  500
#define CLIP1_MAX  150
#define FLIP_MIN   120
#define FLIP_MAX   600
#define SLID_MIN   275
#define SLID_MAX   600

int flip   = 8;
int clip2  = 0;
int clip1  = 12;
int slider = 4;

// ─── System mode ─────────────────────────────────────────────────────────────
enum SystemMode { IDLE, CALIBRATION, FUNCTION };
SystemMode currentMode = IDLE;
// ─── Handshake state ─────────────────────────────────────────────────────────
bool hostConnected = false;

// ─── Buttons ─────────────────────────────────────────────────────────────────
const int buttonPin = 6;
bool lastButtonState = HIGH;

const int eStopPin = 7;
bool eStopActive = false;
bool lastEStopState = HIGH;

const int calibPin = 5;
bool lastCalibState = HIGH;
unsigned long lastCalibrationPress = 0;
const unsigned long debounceDelay = 200;

// ─── Stepper ─────────────────────────────────────────────────────────────────
const int dirPin    = 2;
const int stepPin   = 3;
const int enablePin = 10;

const int stepsToMove = 300;
const int stepDelay   = 800;

long mechanicalZero = 0;
long rightEdge = 0;
long leftEdge  = 0;

unsigned long lastStepTime = 0;
const unsigned long calibrationInterval = 10;
long calibrationCounter = 0;
long bookWidth = 0;

bool calibrating      = false;
int  calibrationStage = 0;
bool clipsOpen = false;   // start assuming system boots closed

// ─── Servo timeout & limits ───────────────────────────────────────────────
unsigned long clip1StartTime = 0;
unsigned long clip2StartTime = 0;
bool clip1Active = false;
bool clip2Active = false;
const unsigned long servoTimeout = 100; // ms (tune this)

/* STEPPER MOTOR*/
void stepMotor(int direction) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(stepDelay);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(stepDelay);

  calibrationCounter += direction;
}

void setClip1(int pulse) {
  pwm.setPWM(clip1, 0, pulse);
  clip1StartTime = millis();
  clip1Active = true;
}

void setClip2(int pulse) {
  pwm.setPWM(clip2, 0, pulse);
  clip2StartTime = millis();
  clip2Active = true;
}

/* HOME POSITION */
void zeroPosition() {
  setClip1(map(180, 0, 180, CLIP1_MIN, CLIP1_MAX));
  setClip2(map(180, 0, 180, CLIP2_MIN, CLIP2_MAX));
  pwm.setPWM(flip,  0, map(180, 0, 180, FLIP_MIN, FLIP_MAX));
  pwm.setPWM(slider, 0, map(0, 0, 180, SLID_MIN, SLID_MAX));
  delay(1000);
}

void moveToMechanicalZero() {

  digitalWrite(enablePin, LOW);

  if (calibrationCounter > mechanicalZero) {
    digitalWrite(dirPin, LOW);  // move right
  } else {
    digitalWrite(dirPin, HIGH); // move left
  }

  while (calibrationCounter != mechanicalZero) {
    if (eStopActive) return;

    if (calibrationCounter > mechanicalZero) {
      digitalWrite(dirPin, LOW);   // RIGHT
      stepMotor(-1);
    } 
    else {
      digitalWrite(dirPin, HIGH);  // LEFT
      stepMotor(+1);
    }
  }
  digitalWrite(enablePin, HIGH);
}

/* E-STOP FUNCTION */
void hardStop() {
  // Disable stepper immediately
  digitalWrite(enablePin, HIGH);

  // Cut PWM to all servos
  setClip1(0);
  setClip2(0);
  pwm.setPWM(flip, 0, 0);
  pwm.setPWM(slider, 0, 0);

  Serial.println("!!! ALL MOTION STOPPED !!!");
}

bool checkEStop() {
  bool currentState = digitalRead(eStopPin);

  if (lastEStopState == HIGH && currentState == LOW) {
    eStopActive = !eStopActive;

    if (eStopActive) {
      STATUS("ESTOP_ON");
      hardStop();
    } 
    else {
      STATUS("ESTOP_OFF");
      if (currentMode == CALIBRATION) {
        DBG("E-stop cleared in CALIB, returning to zero");
        moveToMechanicalZero();
      }
      else if (currentMode == FUNCTION) {
        DBG("E-stop cleared in FUNCTION, resetting");
        zeroPosition();
        moveToMechanicalZero();
      }

      // Full reset state
      bookWidth = 0;
      calibrationStage = 0;
      calibrating = false;
      clipsOpen = false;
      currentMode = IDLE;
      STATUS("MODE=IDLE");
      Serial.println("System reset. Please recalibrate using button.");

      return true;  // SIGNAL RESET OCCURRED
    }
  }
  lastEStopState = currentState;
  return false;
}

bool stopIfEStop() {
  if (checkEStop()) return true;   // reset just happened
  if (eStopActive) {
    hardStop();
    return true;
  }

  return false;
}

// ─── Calibration motion ───────────────────────────────────────────────────────
void calibrationMoveLeft() {
  unsigned long currentTime = millis();

  if (currentTime - lastStepTime >= calibrationInterval) {
    lastStepTime = currentTime;

    digitalWrite(enablePin, LOW);
    digitalWrite(dirPin, HIGH);   // LEFT

    stepMotor(+1);
  }

 if (calibrationCounter > 1250) {
    Serial.println("Calibration limit reached. Returning to HOME...");

    calibrating = false;

    // Move back to mechanical zero
    digitalWrite(dirPin, LOW);   // move RIGHT toward home
    digitalWrite(enablePin, LOW);

    while (calibrationCounter > 0) {
      stepMotor(-1);   // moving right
    }

    digitalWrite(enablePin, HIGH);  // disable motor

    calibrationStage = 0;
    Serial.println("Returned to HOME. Calibration reset.");
  }
}

void moveToLeftEdge() {
  long stepsNeeded = leftEdge - calibrationCounter;

  if (stepsNeeded <= 0) return;   // already there or past

  digitalWrite(enablePin, LOW);
  digitalWrite(dirPin, HIGH);     // LEFT direction

  for (long i = 0; i < stepsNeeded; i++) {
    stepMotor(+1);   // moving left
  }
  digitalWrite(enablePin, HIGH);
}

void moveToRightEdge() {
  long stepsNeeded = calibrationCounter - rightEdge;

  if (stepsNeeded <= 0) return;   // already there or past

  digitalWrite(enablePin, LOW);
  digitalWrite(dirPin, LOW);      // RIGHT direction

  for (long i = 0; i < stepsNeeded; i++) {
    stepMotor(-1);   // moving right
  }
  digitalWrite(enablePin, HIGH);
}

// ─── Clip open ────────────────────────────────────────────────────────────────
void openClips() {
  Serial.println("Opening clips...\n");

  for (int pos = 180; pos >= 90; pos--) {
    setClip1(map(pos, 0, 180, CLIP1_MIN, CLIP1_MAX));
    setClip2(map(pos, 0, 180, CLIP2_MIN, CLIP2_MAX));
    delay(2);
  }

  delay(50);

  setClip1(0);
  setClip2(0);
}

// ─── Calibration button handler ──────────────────────────────────────────────
void handleCalibrationButton() {

  bool currentState = digitalRead(calibPin);

    if (currentState == LOW && lastCalibState == HIGH) {

      unsigned long currentTime = millis();

      if (currentTime - lastCalibrationPress > debounceDelay) {

        lastCalibrationPress = currentTime;

        if (calibrationStage == 0) {
          currentMode = CALIBRATION;
          STATUS("CALIB_START");
          STATUS("MODE=CALIB");
          calibrationCounter = 0;
          mechanicalZero = 0;

          calibrating = true;
          calibrationStage = 1;

          DBG("Stage 1: moving left, press at RIGHT edge");
        }

        else if (calibrationStage == 1) {
          rightEdge = calibrationCounter;

          Serial.print("STATUS:CALIB_RIGHT="); Serial.println(rightEdge);

          digitalWrite(enablePin, HIGH); 
          delay(1000);
          digitalWrite(enablePin, LOW);   

          calibrationStage = 2;

          DBG("Stage 2: continue left, press at LEFT edge");
        }

        else if (calibrationStage == 2) {

          leftEdge = calibrationCounter;

          Serial.print("STATUS:CALIB_LEFT="); Serial.println(leftEdge);

          digitalWrite(enablePin, HIGH); 
          delay(1000);
          digitalWrite(enablePin, LOW);  

          bookWidth = leftEdge - rightEdge;

          Serial.print("Book width: ");
          Serial.println(bookWidth);

          calibrating = false;

          long returnSteps = leftEdge - rightEdge;

          digitalWrite(enablePin, LOW);
          digitalWrite(dirPin, LOW);

          for (long i = 0; i < returnSteps; i++) {
            stepMotor(-1);
          }

          digitalWrite(enablePin, HIGH);

          calibrationCounter = rightEdge;
          calibrationStage = 0;
          currentMode = IDLE;

          Serial.print("STATUS:CALIB_DONE bookWidth="); Serial.println(bookWidth);
          STATUS("MODE=IDLE");
        }
      }
    }
  lastCalibState = currentState;
}

/*  LOOP  */

void loop() {
// -------- E-STOP TOGGLE --------
  if (checkEStop()) return;
  if (eStopActive) return;

  handleCalibrationButton();

  if (calibrating && calibrationStage > 0) {
    calibrationMoveLeft();
  }

 // -------- SETUP BUTTON TOGGLE CONTROL --------
bool buttonState = digitalRead(buttonPin);

// Detect button press (HIGH -> LOW transition)
  if (lastButtonState == HIGH && buttonState == LOW) {

    if (clipsOpen) {
      zeroPosition();        // close clips
      clipsOpen = false;
    } 
    else {
      openClips();           // open clips
      clipsOpen = true;
    }

    delay(50);  // small debounce
  }

  lastButtonState = buttonState;

  // CLIP SERVO TIMEOUT ---
  if (clip1Active && millis() - clip1StartTime > servoTimeout) {
    setClip1(0);
    clip1Active = false;
  }

  if (clip2Active && millis() - clip2StartTime > servoTimeout) {
    setClip2(0);
    clip2Active = false;
  }

  // -------- SERIAL COMMAND READ --------
  if (!Serial.available()) return;
  char input = Serial.read();
  DBG(String("RX char=") + String((int)input) + " '" + String(input) + "'");

  // Heartbeat — respond immediately
  if (input == 'H' || input == 'h') {
      Serial.println("HB");
      return;
  }
  // Status dump request
  if (input == 'S' || input == 's') {
      emitFullStatus();
      return;
  }

  /*  FORWARD PAGE TURN */
  if (input == 'f' || input == 'F') {
    STATUS("CMD_F");
    if (eStopActive) {
      Serial.println("ESTOP");
      DBG("CMD_F rejected: e-stop active");
      return;
    }

#if !MECH_DEBUG_MODE
    if (bookWidth == 0) {
      Serial.println("NOCALIB");
      DBG("CMD_F rejected: not calibrated");
      return;
    }
#endif
    DBG("Flipping page forward...");
    currentMode = FUNCTION;
    STATUS("MODE=FUNCTION");
    
    /* 1. Motor already at home */

    /* 2. Slider pitches inward */
    for (int pos = 0; pos <= 90; pos++) {
       
      pwm.setPWM(slider, 0, map(pos, 0, 180, SLID_MIN, SLID_MAX));
      delay(5);
    }
    delay(300);
    
    /* 3. Motor moves left ~25 mm WHILE clip1 raises */
    digitalWrite(enablePin, LOW);
    digitalWrite(dirPin, HIGH);

    for (int i = 0; i < stepsToMove; i++) {
      if (stopIfEStop()) return;
      stepMotor(+1);   // moving left
  
      int clipPos = map(i, 0, stepsToMove, 180, 60);
      setClip1(map(clipPos, 0, 180, CLIP1_MIN, CLIP1_MAX));
    }

    digitalWrite(enablePin, HIGH);

    /* 4. Right clip snaps back */
    setClip1(map(180, 0, 180, CLIP1_MIN, CLIP1_MAX));
    for (int t = 0; t < 200; t += 10) {
       
      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* 5. Flipper rotates */
    for (int pos = 180; pos >= 90; pos--) {
       
      pwm.setPWM(flip, 0, map(pos, 0, 180, FLIP_MIN, FLIP_MAX));
      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* 6. Slider pitches outward */
    pwm.setPWM(slider, 0, map(0, 0, 180, SLID_MIN, SLID_MAX));
    delay(5);
    
    /* 7. Flipper completes turn + clip2 raises */
    for (int step = 0; step <= 90; step++) {
         
        pwm.setPWM(flip, 0, map(90 - step, 0, 180, FLIP_MIN, FLIP_MAX));
        setClip2(map(180 - step * 0.75, 0, 180, CLIP2_MIN, CLIP2_MAX));
       for (int d = 0; d < 10; d++) {
         if (stopIfEStop()) return;
          delay(1);
        }
    }

    /* 8. Clip2 snaps back */
    setClip2(map(180, 0, 180, CLIP2_MIN, CLIP2_MAX));
    delay(200);

    /* 9. Motor returns to home */
    digitalWrite(enablePin, LOW);
    digitalWrite(dirPin, LOW);   // RIGHT direction

    for (int i = 0; i < stepsToMove; i++) {

      if (stopIfEStop()) return;
      stepMotor(-1);   // moving right
    }

    digitalWrite(enablePin, HIGH);

    /* 10. Reset */
    zeroPosition();
    currentMode = IDLE;
    clipsOpen = false;
    STATUS("MODE=IDLE");
    DBG("doFlipForward complete");
    Serial.println("DONE");
    }
  
  /*  BACKWARD PAGE TURN */
  else if (input == 'b' || input == 'B') {
    STATUS("CMD_B");
      if (eStopActive) {
        Serial.println("ESTOP");
        DBG("CMD_B rejected: e-stop active");
        return;
      }

#if !MECH_DEBUG_MODE
    if (bookWidth == 0) {
      Serial.println("NOCALIB");
      DBG("CMD_F rejected: not calibrated");
      return;
    }
#endif
    DBG("Flipping page backward...");
    currentMode = FUNCTION;
    STATUS("MODE=FUNCTION");

    /* 1. Flipper rotates clockwise (0 to 180) */
   pwm.setPWM(flip, 0, map(0, 0, 180, FLIP_MIN, FLIP_MAX));
    for (int t = 0; t < 200; t += 10) {
      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* 2. Motor aligns with LEFT edge */
    moveToLeftEdge();

    /* 3. Slider pitches inward (controlled + compensated) */
    int sliderTargetDeg = 95;  // 🔥 compensate for drift (tune 93–100 if needed)

    for (int pos = 180; pos >= sliderTargetDeg; pos--) {
      pwm.setPWM(slider, 0, map(pos, 0, 180, SLID_MIN, SLID_MAX));

      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* Lock final position and let mechanism settle */
    int sliderHold = map(sliderTargetDeg, 0, 180, SLID_MIN, SLID_MAX);
    pwm.setPWM(slider, 0, sliderHold);
    delay(80);  // critical for stability under load


    /* 4. Motor moves RIGHT ~25mm while clip2 raises */
    digitalWrite(enablePin, LOW);
    digitalWrite(dirPin, LOW);   // RIGHT direction

    for (int i = 0; i < stepsToMove; i++) {
      if (stopIfEStop()) return;

      stepMotor(-1);   // moving right

      int clipPos = map(i, 0, stepsToMove, 180, 60);
      setClip2(map(clipPos, 0, 180, CLIP2_MIN, CLIP2_MAX));
    }

    digitalWrite(enablePin, HIGH);

    /* 5. Clip2 snaps back */
    setClip2(map(180, 0, 180, CLIP2_MIN, CLIP2_MAX));
    delay(200);

    /* 6. Flipper rotates anticlockwise (180 to 90) */
    for (int pos = 0; pos <= 90; pos++) {
       
      pwm.setPWM(flip, 0, map(pos, 0, 180, FLIP_MIN, FLIP_MAX));
      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* 7. At 90 deg, slider pitches outward */
    pwm.setPWM(slider, 0, map(0, 0, 180, SLID_MIN, SLID_MAX));
    for (int t = 0; t < 200; t += 10) {
       
      for (int d = 0; d < 10; d++) {
        if (stopIfEStop()) return;
        delay(1);
      }
    }

    /* 8. Flipper completes turn (90 to 0) while clip1 (right clip) raises */
    for (int step = 0; step <= 90; step++) {
         
        int flipPos = 90 + step;           
        int clip1Pos = 180 - step * 0.5;   
        pwm.setPWM(flip, 0, map(flipPos, 0, 180, FLIP_MIN, FLIP_MAX));
        setClip1(map(clip1Pos, 0, 180, CLIP1_MIN, CLIP1_MAX));
        for (int d = 0; d < 10; d++) {
          if (stopIfEStop()) return;
          delay(1);
        }
      }

    /* 9. Clip1 snaps back */
    setClip1(map(180, 0, 180, CLIP1_MIN, CLIP1_MAX));
    delay(200);

    /* 10. Motor returns to RIGHT edge (home)*/
    moveToRightEdge();

    zeroPosition();  
    currentMode = IDLE;
    clipsOpen   = false;
    STATUS("MODE=IDLE");
    DBG("doFlipBackward complete");
    Serial.println("DONE");

  }
  // Unknown byte (log it so we can see noise/garbage on the line)
  else if (input >= 0x20 && input < 0x7F) {
      DBG(String("Unknown cmd: '") + String(input) + "'");
  }
}

// ═════════════════════════════════════════════════════════════════════════════
//  SETUP
// ═════════════════════════════════════════════════════════════════════════════
void emitFullStatus() {
    Serial.print("STATUS:CONNECTED=");   Serial.println(hostConnected ? 1 : 0);
    Serial.print("STATUS:CALIBRATED=");  Serial.println(bookWidth > 0 ? 1 : 0);
    Serial.print("STATUS:BOOKWIDTH=");   Serial.println(bookWidth);
    Serial.print("STATUS:ESTOP=");       Serial.println(eStopActive ? 1 : 0);
    Serial.print("STATUS:MODE=");
    switch(currentMode){
        case IDLE:        Serial.println("IDLE");     break;
        case CALIBRATION: Serial.println("CALIB");    break;
        case FUNCTION:    Serial.println("FUNCTION"); break;
    }
}

void setup() {
  Serial.begin(9600);

  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(eStopPin, INPUT_PULLUP);
  pinMode(calibPin, INPUT_PULLUP);

  digitalWrite(enablePin, HIGH); // motor OFF at startup

  zeroPosition();
  clipsOpen = false;

#if MECH_DEBUG_MODE
// skip handshake entirely - go straight to loop so F/B can work immediately
// without backend
// bookWidth check is also bypassed
  hostConnected = true;
  Serial.println("MECH_DEBUG_MODE: skipping handshake, ready for F/B commands");
#else
  DBG ("entering handshake loop");
  /* NON-BLOCKING READY PING: wastes power but wtv too late to fix */
  // send READY every 400 ms; handle buttons/e-stop while waiting for P back
  // break as soon as 'P' arrives.
  unsigned long lastReadyTx = 0;
  const unsigned long readyInterval = 400; // ms

  while(true){
      unsigned long now = millis();
      // send ready ping
      if(now - lastReadyTx >= readyInterval){
        Serial.println("READY");
        lastReadyTx = now;
      }

      // Check for 'P' ping from host
      if(Serial.available()){
        char c = Serial.read();
        if (c == 'P' || c == 'p'){
          hostConnected = true;
          STATUS("CONNECTED");
          DBG("handshake complete");
          emitFullStatus();
          break;
        }
      }
      checkEStop(); // still service estop during handshake
  }
#endif
}