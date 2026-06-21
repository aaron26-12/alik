
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Servo.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

const int JOY_X  = A0;
const int JOY_Y  = A1;
const int JOY_SW = 2;
const int pinLO  = 3;
const int pinLI  = 5;
const int pinRI  = 6;
const int pinRO  = 9;
const int LED    = 13;

Servo cejaLO, cejaLI, cejaRI, cejaRO;

#define CH_LR  0
#define CH_UD  1
#define CH_TR  2
#define CH_BR  3
#define CH_TL  4
#define CH_BL  5
#define CH_JAW 6
#define CH_LUL 7
#define CH_LUR 8
#define CH_LLL 9
#define CH_LLR 10
#define CH_CUL 11
#define CH_CUR 12
#define CH_CLL 13
#define CH_CLR 14
#define CH_TON 15

#define SERVO_FREQ    50
#define EYE_MIN      110
#define EYE_MAX      490
#define FACE_MIN     150
#define FACE_MAX     600
#define BOCA_CERRADA 148
#define BOCA_ABIERTA 130
#define LENGUA_UP     65
#define LENGUA_DOWN    0

const bool INV[16] = {
  false, true, true, true, false, false,
  true, false, false, false, false,
  false, false, false, false, false
};

float curPCA[16], tgtPCA[16], velPCA[16];
float browCur[4] = {90,90,90,90};
float browTgt[4] = {90,90,90,90};
float browVel[4] = {0,0,0,0};

float targetLR = 90, targetUD = 90;
float eyeVelLR = 0, eyeVelUD = 0;
float gazeContactPct = 0.6f;
bool manualOverride = false;
unsigned long lastJoyMs = 0;
const unsigned long JOY_TIMEOUT = 8000;

bool btnStable  = HIGH;
bool btnLast    = HIGH;
unsigned long btnChangeMs = 0;
unsigned long btnPressMs  = 0;
bool longDone = false;
uint8_t pendingClicks = 0;
bool waitClick = false;
unsigned long clickDeadline = 0;

const unsigned long DEBOUNCE_MS = 25;
const unsigned long DCLICK_MS   = 320;
const unsigned long LONG_MS     = 900;
const unsigned long VLONG_MS    = 3000;

enum Emotion : uint8_t {
  EMO_NEUTRAL=0, EMO_HAPPY, EMO_SAD, EMO_ANGRY,
  EMO_SURPRISE, EMO_SLEEPY, EMO_COQUETO, EMO_AFRAID,
  EMO_CONFUSED, EMO_PROUD, EMO_LAUGH, EMO_COUNT
};
Emotion currentEmo = EMO_NEUTRAL;
Emotion blendEmo   = EMO_NEUTRAL;
float blendAmt   = 0.0f;

bool blinking = false;
unsigned long nextBlinkMs = 0;
unsigned long blinkOpenMs = 0;

bool isListening = false;
bool isSpeaking  = false;
unsigned long speakStartMs = 0;

struct EmoPose {
  float brow[4];
  float jaw, lul, lur, lll, llr;
  float cul, cur_, cll, clr;
  float tongue;
};

const EmoPose EMO_POSES[EMO_COUNT] PROGMEM = {
  {{90,90,90,90}, 148, 90,90,90,90, 90,90,90,90, 65},
  {{80,60,120,100}, 142, 80,100,100,80, 70,110,110,70, 60},
  {{60,120,60,120}, 148, 110,70,70,110, 100,80,80,100, 70},
  {{120,100,80,60}, 148, 110,70,70,110, 90,90,90,90, 65},
  {{120,60,120,60}, 130, 110,70,110,70, 65,105,120,65, 0},
  {{85,85,85,85}, 150, 120,80,80,120, 115,75,75,115, 72},
  {{110,80,100,70}, 144, 95,95,95,95, 80,100,100,80, 62},
  {{125,120,125,120}, 128, 115,65,115,65, 60,120,120,60, 0},
  {{115,85,75,105}, 146, 100,90,90,100, 85,95,95,85, 66},
  {{100,70,110,80}, 146, 85,100,100,85, 75,105,105,75, 60},
  {{82,78,95,92}, 125, 100,92,92,100, 80,100,100,80, 0},
};

float clampf(float v, float mn, float mx) {
  return v < mn ? mn : (v > mx ? mx : v);
}

float lerpf(float a, float b, float t) {
  return a + (b - a) * clampf(t, 0, 1);
}

float mapf(float x, float a, float b, float c, float d) {
  if (b == a) return c;
  return (x - a) * (d - c) / (b - a) + c;
}

void accelerateTo(float &cur, float tgt, float &vel, float acc, float dec, float maxSpd) {
  float want = clampf((tgt - cur) * dec, -maxSpd, maxSpd);
  if (vel < want) {
    vel += acc;
    if (vel > want) vel = want;
  } else {
    vel -= acc;
    if (vel < want) vel = want;
  }
  cur += vel;
}

void writePCA(uint8_t ch, float angle) {
  int a = constrain((int)(INV[ch] ? 180 - angle : angle), 0, 180);
  bool isFace = (ch >= CH_JAW);
  int pulse = map(a, 0, 180, isFace ? FACE_MIN : EYE_MIN,
                              isFace ? FACE_MAX : EYE_MAX);
  pca.setPWM(ch, 0, pulse);
}

float normalizeJoy(int raw) {
  const int dz = 45;
  int d = raw - 512;
  if (abs(d) <= dz) return 0.0f;
  float out = (d > 0) ? (float)(d - dz) / (512.0f - dz)
                      : (float)(d + dz) / (512.0f - dz);
  return clampf(out, -1.0f, 1.0f);
}

void applySafePose() {
  for (int i = 0; i < 16; i++) curPCA[i] = tgtPCA[i] = 90, velPCA[i] = 0;
  curPCA[CH_JAW] = tgtPCA[CH_JAW] = BOCA_CERRADA;
  curPCA[CH_TON] = tgtPCA[CH_TON] = LENGUA_UP;
  for (int i = 0; i < 4; i++) browCur[i] = browTgt[i] = browVel[i] = 90;
  for (int i = 0; i < 16; i++) writePCA(i, curPCA[i]);
  cejaLO.write(90); cejaLI.write(90); cejaRI.write(90); cejaRO.write(90);
}

EmoPose getEmoPose(Emotion e) {
  EmoPose p;
  memcpy_P(&p, &EMO_POSES[e], sizeof(EmoPose));
  return p;
}

void setEmotionTargets() {
  EmoPose base = getEmoPose(currentEmo);
  EmoPose blend;
  bool doBlend = (blendAmt > 0.01f && blendEmo != currentEmo);
  if (doBlend) blend = getEmoPose(blendEmo);

  for (int i = 0; i < 4; i++) {
    browTgt[i] = doBlend ? lerpf(base.brow[i], blend.brow[i], blendAmt)
                         : base.brow[i];
  }

  #define SETBLEND(CH, field) \
    tgtPCA[CH] = doBlend ? lerpf(base.field, blend.field, blendAmt) : base.field

  SETBLEND(CH_JAW, jaw);
  SETBLEND(CH_LUL, lul);
  SETBLEND(CH_LUR, lur);
  SETBLEND(CH_LLL, lll);
  SETBLEND(CH_LLR, llr);
  SETBLEND(CH_CUL, cul);
  SETBLEND(CH_CUR, cur_);
  SETBLEND(CH_CLL, cll);
  SETBLEND(CH_CLR, clr);
  SETBLEND(CH_TON, tongue);
}

void updateLids() {
  float yOff = mapf(curPCA[CH_UD], 0, 180, -12, 12);
  float xOff = mapf(curPCA[CH_LR], 0, 180, -8,  8);

  float open = 0.58f;
  switch (currentEmo) {
    case EMO_SURPRISE:
    case EMO_AFRAID: open = 0.90f; break;
    case EMO_ANGRY:
    case EMO_SLEEPY: open = 0.28f; break;
    case EMO_SAD: open = 0.32f; break;
    case EMO_HAPPY: open = 0.62f; break;
    case EMO_LAUGH: open = 0.42f; break;
    case EMO_COQUETO: open = 0.48f; break;
    default: break;
  }
  if (blinking) open = 0.0f;

  float bias = 0;
  if (currentEmo == EMO_SLEEPY) bias = -10;
  if (currentEmo == EMO_SURPRISE || currentEmo == EMO_AFRAID) bias = 6;
  if (currentEmo == EMO_ANGRY) bias = -4;

  tgtPCA[CH_TL] = 92 + (155-92)*open + yOff + xOff*0.12f + bias;
  tgtPCA[CH_TR] = 92 + (25 -92)*open + yOff - xOff*0.12f + bias;
  tgtPCA[CH_BL] = 92 + (38 -92)*open + yOff + xOff*0.12f + bias;
  tgtPCA[CH_BR] = 92 + (145-92)*open + yOff - xOff*0.12f + bias;
}

void updateBlink() {
  unsigned long now = millis();
  if (!blinking && now >= nextBlinkMs) {
    blinking = true;
    blinkOpenMs = now + 75;
  }
  if (blinking && now >= blinkOpenMs) {
    blinking = false;
    nextBlinkMs = now + random(isSpeaking ? 1800 : 2500, isSpeaking ? 3500 : 5500);
  }
}

void updateSocialGaze() {
  unsigned long now = millis();
  if (manualOverride) return;
  if (now < nextGazeMs) return;

  bool makeContact = (random(100) < (int)(gazeContactPct * 100));
  if (makeContact) {
    targetLR = 90 + random(-8, 8);
    targetUD = 90 + random(-5, 5);
    nextGazeMs = now + random(800, 2200);
  } else {
    int dir = random(4);
    if (dir == 0)      { targetLR = random(55, 72); targetUD = 90 + random(-8,8); }
    else if (dir == 1) { targetLR = random(108,125); targetUD = 90 + random(-8,8); }
    else if (dir == 2) { targetLR = 90 + random(-8,8); targetUD = random(70,80); }
    else               { targetLR = 90 + random(-8,8); targetUD = random(100,110); }
    nextGazeMs = now + random(400, 1200);
  }
}

void updateJoystick() {
  int rawX = analogRead(JOY_X);
  int rawY = analogRead(JOY_Y);
  float x = normalizeJoy(rawX);
  float y = normalizeJoy(rawY);

  if (abs(x) > 0.06f || abs(y) > 0.06f) {
    manualOverride = true;
    lastJoyMs = millis();
  }
  if (manualOverride && millis() - lastJoyMs > JOY_TIMEOUT) {
    manualOverride = false;
  }
  if (manualOverride) {
    targetLR = 90 + x * 38.0f;
    targetUD = 90 + y * 30.0f;
  }
}

void handleSingleClick() {
  currentEmo = (Emotion)((currentEmo + 1) % EMO_COUNT);
  Serial.print(F("EMO_CHANGED:"));
  Serial.println((int)currentEmo);
}

void handleDoubleClick() {
  manualOverride = !manualOverride;
  lastJoyMs = millis();
  Serial.println(manualOverride ? F("MANUAL:1") : F("MANUAL:0"));
}

void handleLongPress() {
  currentEmo = EMO_NEUTRAL;
  manualOverride = false;
  isSpeaking = false;
  applySafePose();
  Serial.println(F("RESET:1"));
}

void updateButton() {
  bool reading = digitalRead(JOY_SW);
  unsigned long now = millis();

  if (reading != btnLast) { btnChangeMs = now; btnLast = reading; }

  if (now - btnChangeMs > DEBOUNCE_MS && reading != btnStable) {
    btnStable = reading;
    if (btnStable == LOW) {
      btnPressMs = now;
      longDone = false;
    } else {
      if (!longDone) {
        pendingClicks++;
        waitClick = true;
        clickDeadline = now + DCLICK_MS;
      }
    }
  }

  if (btnStable == LOW) {
    unsigned long dur = now - btnPressMs;
    if (!longDone && dur >= VLONG_MS) { longDone = true; handleLongPress(); return; }
    if (!longDone && dur >= LONG_MS) { }
  }

  if (waitClick && now >= clickDeadline) {
    if (pendingClicks == 1) handleSingleClick();
    else if (pendingClicks >= 2) handleDoubleClick();
    pendingClicks = 0;
    waitClick = false;
  }
}

void parseSerial(char *line) {
  if (strncmp(line, "EYE:", 4) == 0 && !manualOverride) {
    char *comma = strchr(line + 4, ',');
    if (comma) {
      targetLR = clampf(atof(line + 4), 0, 180);
      targetUD = clampf(atof(comma + 1), 0, 180);
    }
  } else if (strncmp(line, "EMO:", 4) == 0) {
    int e = atoi(line + 4);
    if (e >= 0 && e < EMO_COUNT) currentEmo = (Emotion)e;
  } else if (strncmp(line, "BLEND:", 6) == 0) {
    char *comma = strchr(line + 6, ',');
    if (comma) {
      int e = atoi(line + 6);
      if (e >= 0 && e < EMO_COUNT) blendEmo = (Emotion)e;
      blendAmt = clampf(atof(comma + 1) / 100.0f, 0, 1);
    }
  } else if (strncmp(line, "STATE:", 6) == 0) {
    if (strncmp(line + 6, "listen", 6) == 0) {
      isListening = true; isSpeaking = false;
    } else if (strncmp(line + 6, "speak", 5) == 0) {
      isListening = false; isSpeaking = true; speakStartMs = millis();
    } else {
      isListening = false; isSpeaking = false;
    }
  } else if (strncmp(line, "GAZE:", 5) == 0) {
    gazeContactPct = clampf(atof(line + 5) / 100.0f, 0, 1);
  } else if (strncmp(line, "FACE:", 5) == 0) {
    // FACE:lr,ud,conf — reservado para futuras mejoras
  } else if (strncmp(line, "EXP:", 4) == 0) {
    int e = atoi(line + 4);
    if (e >= 0 && e < EMO_COUNT) currentEmo = (Emotion)e;
  } else if (strncmp(line, "STATUS?", 7) == 0) {
    Serial.print(F("STATUS:emo="));
    Serial.print((int)currentEmo);
    Serial.print(F(",manual="));
    Serial.print(manualOverride ? 1 : 0);
    Serial.print(F(",lr="));
    Serial.print((int)curPCA[CH_LR]);
    Serial.print(F(",ud="));
    Serial.print((int)curPCA[CH_UD]);
    Serial.print(F(",blink="));
    Serial.println(blinking ? 1 : 0);
  }
}

void readSerial() {
  static char buf[80];
  static uint8_t idx = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (idx > 0) {
        buf[idx] = '\0';
        parseSerial(buf);
        idx = 0;
      }
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(SERVO_FREQ);

  pinMode(LED, OUTPUT);
  pinMode(JOY_SW, INPUT_PULLUP);

  cejaLO.attach(pinLO);
  cejaLI.attach(pinLI);
  cejaRI.attach(pinRI);
  cejaRO.attach(pinRO);

  randomSeed(analogRead(A7));
  applySafePose();

  unsigned long now = millis();
  nextBlinkMs = now + random(2000, 4000);
  nextGazeMs  = now + 1000;

  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  updateButton();
  updateJoystick();
  updateBlink();
  updateSocialGaze();

  setEmotionTargets();
  updateLids();

  digitalWrite(LED, manualOverride ? HIGH : LOW);

  float eyeAcc = isSpeaking ? 0.35f : 0.42f;
  accelerateTo(curPCA[CH_LR], targetLR, eyeVelLR, eyeAcc, 0.18f, 7.0f);
  accelerateTo(curPCA[CH_UD], targetUD, eyeVelUD, eyeAcc, 0.18f, 5.5f);

  float talkPulse = isSpeaking ? (sin(millis() * 0.012f) * 0.5f + 0.5f) : 0.0f;
  float jawSpeak = BOCA_CERRADA - talkPulse * 16.0f;
  tgtPCA[CH_JAW] = lerpf(tgtPCA[CH_JAW], jawSpeak, isSpeaking ? 0.08f : 0.02f);

  accelerateTo(curPCA[CH_TL], tgtPCA[CH_TL], velPCA[CH_TL], blinking ? 1.8f : 0.6f, 0.28f, blinking ? 18.0f : 9.0f);
  accelerateTo(curPCA[CH_TR], tgtPCA[CH_TR], velPCA[CH_TR], blinking ? 1.8f : 0.6f, 0.28f, blinking ? 18.0f : 9.0f);
  accelerateTo(curPCA[CH_BL], tgtPCA[CH_BL], velPCA[CH_BL], blinking ? 1.8f : 0.6f, 0.28f, blinking ? 18.0f : 9.0f);
  accelerateTo(curPCA[CH_BR], tgtPCA[CH_BR], velPCA[CH_BR], blinking ? 1.8f : 0.6f, 0.28f, blinking ? 18.0f : 9.0f);

  for (int i = 6; i < 16; i++) {
    float acc = 0.4f, dec = 0.18f, spd = 5.0f;
    if (i == CH_JAW) { acc = 0.60f; dec = 0.30f; spd = 10.0f; }
    else if (i == CH_TON) { acc = 0.22f; dec = 0.16f; spd = 5.0f; }
    else { acc = 0.12f; dec = 0.12f; spd = 4.0f; }
    accelerateTo(curPCA[i], tgtPCA[i], velPCA[i], acc, dec, spd);
  }

  for (int i = 0; i < 4; i++) {
    accelerateTo(browCur[i], browTgt[i], browVel[i], 0.38f, 0.22f, 6.0f);
  }

  for (int i = 0; i < 16; i++) writePCA(i, curPCA[i]);

  cejaLO.write(constrain((int)browCur[0], 0, 180));
  cejaLI.write(constrain((int)browCur[1], 0, 180));
  cejaRI.write(constrain((int)browCur[2], 0, 180));
  cejaRO.write(constrain((int)browCur[3], 0, 180));

  delay(16);
}
