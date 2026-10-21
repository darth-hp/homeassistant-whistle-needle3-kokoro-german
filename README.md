# Local Voice Assistant Pipeline for Home Assistant based on Cactus Computing Whistle and Needle 3 and Kokoro

A fully local, privacy-focused, and resource-efficient Voice Assistant stack running in a Proxmox LXC container (Debian/Ubuntu-based).

The stack provides three independent components:
1. **Speech-to-Text (STT):** Cactus Whistle via the Wyoming Protocol (Port `10300`).
2. **Intent & Tool Calling:** Needle 3 OpenAI-compatible API bridge with dynamic Home Assistant REST dispatching and fuzzy device matching (Port `10400`).
3. **Text-to-Speech (TTS):** Kokoro-82M ONNX (German Martin fine-tune) via the Wyoming Protocol (Port `10200`).

---

## 1. System Prerequisites & Environment Setup

### 1.1 Hardware & Container Sizing
* **Container:** Proxmox LXC (Debian 12 / Ubuntu 22.04+)
* **Cores:** 2–4 vCPUs
* **RAM:** Minimum 1024 MB RAM + 512 MB Swap (prevents OOM kills during ONNX runtime tensor allocation)
* **Disk:** 2GB
* **Base Path:** `/opt/whistle-wyoming`

### 1.2 System Packages & Global Virtual Environment
Execute inside the container:

```bash
timedatectl set-timezone Europe/Berlin
apt update && apt install -y \
  git \
  curl \
  ffmpeg \
  espeak-ng \
  python3 \
  python3-pip \
  python3-venv

mkdir -p /opt/whistle-wyoming
cd /opt/whistle-wyoming

# Create shared virtual environment
python3 -m venv venv
./venv/bin/pip install --upgrade pip setuptools wheel
```

### 1.3 Python Dependencies
Install all required libraries across STT, Intent Bridge, and TTS:

```bash
cd /opt/whistle-wyoming
./venv/bin/pip install \
  wyoming \
  numpy \
  soundfile \
  rapidfuzz \
  text2num \
  fastapi \
  uvicorn \
  pydantic \
  onnxruntime \
  phonemizer \
  kokoro-onnx
```

### 1.4 Central Environment Configuration (`.env`)
Create `/opt/whistle-wyoming/.env` with strict permissions. Set your desired default language (`de` or `en`) and local timezone:

```bash
cat << 'EOF' > /opt/whistle-wyoming/.env
# Language & Timezone Configuration
VOICE_LANG=de
TZ=Europe/Berlin

# Runtime Performance Flags
NEEDLE_TELEMETRY=0
HF_HUB_ENABLE_HF_TRANSFER=0
OMP_NUM_THREADS=2
OMP_PROC_BIND=false
ONNXRUNTIME_DISABLE_THREAD_AFFINITY=1

# Home Assistant Connection Details
HA_URL=http://<YOUR_HA_IP>:8123
HA_TOKEN=<YOUR_LONG_LIVED_ACCESS_TOKEN>

# Huggingface Token
HF_TOKEN=<YOUR_HUGGINFACE_TOKEN>
EOF

chmod 600 /opt/whistle-wyoming/.env
```

---

## 2. Speech-to-Text Service: Whistle (`whistle-stt-wyoming`)

Whistle runs as a Wyoming STT provider listening on TCP port `10300`.

### 2.1 Keyword Biasing Setup
Create `/opt/whistle-wyoming/whistle_stt_keywords.txt` to boost recognition accuracy for home automation vocabulary:

```text
Schalte
Setze
Stelle
Mach
Bürorollo
Bürolicht
Rollo
Licht
Prozent
```

### 2.2 Systemd Service Setup
Create `/opt/whistle-wyoming/whistle-stt-wyoming.service`:

```ini
[Unit]
Description=Cactus Whistle Wyoming STT Service (Port 10300)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/whistle-wyoming
EnvironmentFile=/opt/whistle-wyoming/.env
ExecStart=/opt/whistle-wyoming/venv/bin/python3 /opt/whistle-wyoming/whistle_stt_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
ln -sf /opt/whistle-wyoming/whistle-stt-wyoming.service /etc/systemd/system/whistle-stt-wyoming.service
systemctl daemon-reload
systemctl enable --now whistle-stt-wyoming.service
```

---

## 3. Intent & Tool Calling: Needle 3 Bridge (`needle-bridge`)

Needle 3 acts as an OpenAI-compatible conversation agent listening on port `10400`. It extracts targets, states, and unified percentage values, resolving entities dynamically via `ha_client.py`.

### 3.1 Configuration Files

Create `/opt/whistle-wyoming/ha_client_config.json`:

```json
{
  "matching": {
    "strict_threshold": 82,
    "scorer": "token_set_ratio",
    "generic_nouns": [
      "rollo",
      "rolladen",
      "licht",
      "lampe",
      "led",
      "blind",
      "shutter",
      "light",
      "climate",
      "heizung"
    ],
    "all_trigger_words": [
      "alle",
      "all",
      "jeder",
      "every"
    ],
    "group_entity_patterns": [
      "alle_",
      "all_"
    ],
    "group_name_patterns": [
      "alle ",
      "all "
    ]
  }
}
```

Create `/opt/whistle-wyoming/ha_client_responses.json`:

```json
{
  "de": {
    "not_understood": "Das habe ich leider nicht verstanden.",
    "device_not_found": "Gerät '{device}' wurde nicht gefunden.",
    "time_reply": "Es ist {hours} Uhr {minutes}.",
    "time_reply_exact": "Es ist genau {hours} Uhr.",
    "action_completed": "Aktion ausgeführt.",
    "light_brightness": "{name} auf {brightness}% gestellt.",
    "light_state": "{name} wurde {state}.",
    "light_on": "eingeschaltet",
    "light_off": "ausgeschaltet",
    "cover_position": "{name} auf {position}% gefahren.",
    "cover_opened": "{name} wird geöffnet.",
    "cover_closed": "{name} wird geschlossen.",
    "cover_stopped": "{name} angehalten.",
    "climate_temp": "{name} auf {temp}°C gestellt.",
    "climate_mode": "{name} Modus {mode} aktiviert."
  },
  "en": {
    "not_understood": "Sorry, I could not understand that request.",
    "device_not_found": "Device '{device}' was not found.",
    "time_reply": "It is {hours}:{minutes}.",
    "time_reply_exact": "It is {hours} o'clock.",
    "action_completed": "Action completed.",
    "light_brightness": "{name} set to {brightness}%.",
    "light_state": "{name} turned {state}.",
    "light_on": "on",
    "light_off": "off",
    "cover_position": "{name} set to position {position}%.",
    "cover_opened": "{name} opened.",
    "cover_closed": "{name} closed.",
    "cover_stopped": "{name} stopped.",
    "climate_temp": "{name} set to {temp}°C.",
    "climate_mode": "{name} mode set to {mode}."
  }
}
```

### 3.2 Service Implementation Notes
* `ha_client.py` implements boot-resilience with a 5-attempt retry loop to handle delayed DHCP network assignments (`[Errno 101] Network is unreachable`).
* `ha_client.py` initializes `logging.basicConfig` immediately upon import so device-count logs appear reliably at boot time.
* `needle_bridge.py` merges split compound words (`clean_input_text`), retrieves the local timezone (`TZ`) from `.env`, and routes both lighting and covers through a unified `percentage` parameter.

### 3.3 Systemd Service Setup
Create `/opt/whistle-wyoming/needle-bridge.service`:

```ini
[Unit]
Description=Cactus Needle Intent Bridge Service (Port 10400)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/whistle-wyoming
EnvironmentFile=/opt/whistle-wyoming/.env
ExecStart=/opt/whistle-wyoming/venv/bin/python3 /opt/whistle-wyoming/needle_bridge.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
ln -sf /opt/whistle-wyoming/needle-bridge.service /etc/systemd/system/needle-bridge.service
systemctl daemon-reload
systemctl enable --now needle-bridge.service
```

---

## 4. Text-to-Speech Service: Kokoro ONNX (`kokoro-tts-wyoming`)

Kokoro-82M ONNX runs as a Wyoming TTS provider listening on port `10200`. It only loads the engine defined in `VOICE_LANG` into RAM to minimize memory overhead.

### 4.1 Model Assets Download
```bash
mkdir -p /opt/whistle-wyoming/models/kokoro
cd /opt/whistle-wyoming/models/kokoro

# 1. German Martin Fine-tune (Active when VOICE_LANG=de)
curl -L -o kokoro-martin.onnx https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin/resolve/main/kokoro-martin.onnx
curl -L -o voices-martin.npz https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin/resolve/main/voices-martin.npz

# 2. English Base Model Assets (Active when VOICE_LANG=en)
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 3. German G2P phonemization frontend
cd /opt/whistle-wyoming
git clone https://github.com/Thomcle/kokoro_german.git
```

### 4.2 Configuration File (`kokoro_tts_config.json`)
Create `/opt/whistle-wyoming/kokoro_tts_config.json`:

```json
{
  "server": {
    "port": 10200,
    "default_language": "de"
  },
  "languages": {
    "de": {
      "model_path": "models/kokoro/kokoro-martin.onnx",
      "voices_path": "models/kokoro/voices-martin.npz",
      "default_voice": "martin",
      "use_german_g2p": true,
      "kokoro_lang_code": "de",
      "ha_voice_description": "German Martin Voice"
    },
    "en": {
      "model_path": "models/kokoro/kokoro-v1.0.onnx",
      "voices_path": "models/kokoro/voices-v1.0.bin",
      "default_voice": "af_bella",
      "use_german_g2p": false,
      "kokoro_lang_code": "en-us",
      "ha_voice_description": "English Bella Voice"
    }
  }
}
```

### 4.3 Systemd Service Setup
Create `/opt/whistle-wyoming/kokoro-tts-wyoming.service`:

```ini
[Unit]
Description=Kokoro ONNX Wyoming TTS Service (Port 10200)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/whistle-wyoming
EnvironmentFile=/opt/whistle-wyoming/.env
ExecStart=/opt/whistle-wyoming/venv/bin/python3 /opt/whistle-wyoming/kokoro_tts_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
ln -sf /opt/whistle-wyoming/kokoro-tts-wyoming.service /etc/systemd/system/kokoro-tts-wyoming.service
systemctl daemon-reload
systemctl enable --now kokoro-tts-wyoming.service
```

---

## 5. Home Assistant Integration Steps

### 5.1 Register STT (Whistle)
1. In Home Assistant, navigate to **Settings** $\rightarrow$ **Devices & Services** $\rightarrow$ **Add Integration**.
2. Select **Wyoming Protocol**.
3. Enter:
   * **Host:** `<CONTAINER_IP>`
   * **Port:** `10300`
4. Confirm with **Submit**. Home Assistant registers `Cactus Whistle STT`.

### 5.2 Register Intent Agent (Needle 3)
1. Install **Extended OpenAI Conversation** via HACS.
2. Restart Home Assistant.
3. Go to **Settings** $\rightarrow$ **Devices & Services** $\rightarrow$ **Add Integration** $\rightarrow$ **Extended OpenAI Conversation**.
4. Set the initial connection fields:
   * **API Key:** `needle-agent` (dummy key)
   * **Base Url:** `http://<CONTAINER_IP>:10400/v1`
   * Check **Skip authentication**.
   * Click **Submit**.
5. Configure the integration options:
   * **chat_model:** `needle3`
   * **Prompt Template:** Delete the default prompt completely (keep it empty or set a single line: `You are a smart home voice assistant. Keep answers short.`).
   * **Functions / Tool Calling (YAML box):** Clear completely (empty).
   * **Use Tools:** **Disabled (toggle OFF)**.
   * **Context Threshold:** Set to `200`.
   * **Truncation strategy:** `Clear All Messages`.
   * Save configuration.

### 5.3 Register TTS (Kokoro Martin)
1. Go to **Settings** $\rightarrow$ **Devices & Services** $\rightarrow$ **Add Integration**.
2. Select **Wyoming Protocol**.
3. Enter:
   * **Host:** `<CONTAINER_IP>`
   * **Port:** `10200`
4. Confirm with **Submit**. Home Assistant registers `Kokoro-TTS`.

### 5.4 Configure Voice Assistant Pipeline
1. Navigate to **Settings** $\rightarrow$ **Voice Assistants**.
2. Select your primary pipeline.
3. Configure the parameters:
   * **Conversation Agent:** `Extended OpenAI Conversation (needle3)`
   * **Prefer local command handling:** **Disabled (toggle OFF)** *(ensures all queries, including time requests, pass through Needle)*
   * **Speech-to-Text:** `Cactus Whistle STT`
   * **Text-to-Speech:** `Kokoro-TTS` (Voice: `martin`)
4. Save the pipeline.

---

## 6. Live Monitoring & Troubleshooting

To monitor all three services simultaneously in a unified log stream:

```bash
journalctl -u whistle-stt-wyoming.service -u needle-bridge.service -u kokoro-tts-wyoming.service -n 50 -f
```
