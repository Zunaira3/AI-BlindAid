# AI-BlindAid (AIBA)

> An intelligent assistive wearable system for visually impaired individuals — combining real-time object detection, spatial depth sensing, OCR, voice interaction, and live information retrieval on edge hardware.

---

## What It Does

AIBA is a voice-controlled assistive system designed to run on a **Luxonis OAK-D camera + Jetson Nano** edge setup. A visually impaired user speaks a command — the system responds with audio feedback in real time.

**Supported voice commands:**
- `"detect objects"` — identifies surrounding obstacles and speaks their name and distance in meters
- `"read"` — reads visible text aloud using OCR
- `"tell weather"` — fetches and speaks today's and tomorrow's weather for any city
- `"get news"` — retrieves and reads BBC News headlines aloud
- `"exit"` — shuts the system down

---

## Key Features

### 1. Spatial Object Detection
- Runs **YOLOv4-Tiny** on-device via the **DepthAI pipeline** on the OAK-D camera
- Detects 80 COCO object classes in real time
- Uses **stereo depth** to calculate the **exact X/Y/Z distance** of each detected object in meters
- Announces detected objects and distances via **pyttsx3 TTS**
- Achieves **15 FPS** on edge hardware

### 2. OCR — Text Reading
- Two-stage neural network pipeline:
  - **EAST text detection** (`east_text_detection_256x256`) — locates text regions in frame
  - **text-recognition-0012** — reads the actual characters from each region
- Uses **CTC decoder** with greedy decoding for character sequence prediction
- Speaks recognized text aloud instantly via TTS
- Only reads text with confidence above **0.8** threshold to reduce errors

### 3. Weather Updates
- Takes city name via voice input
- Calls **OpenWeatherMap API** for current and next-day forecast
- Speaks temperature (°C) and weather description for both today and tomorrow

### 4. News Headlines
- Scrapes **BBC News** headlines using BeautifulSoup
- Reads headlines one by one with voice confirmation between each
- User can say "yes" or "no" to continue or stop

### 5. Fully Voice-Controlled Interface
- Built on **SpeechRecognition** + Google Speech API
- All interaction is hands-free — no screen, no keyboard required
- Designed for real-world use by visually impaired individuals

---

## System Architecture

```
User Voice Input
      │
      ▼
SpeechRecognition (Google API)
      │
      ▼
Command Router (wrapper_function)
      │
      ├── "detect objects" ──► DepthAI Pipeline
      │                              │
      │                         OAK-D Camera (RGB + Stereo)
      │                              │
      │                         YOLOv4-Tiny (80 classes)
      │                              │
      │                         Spatial Coordinates (X/Y/Z)
      │                              │
      │                         pyttsx3 TTS ──► Audio Output
      │
      ├── "read" ──────────────► DepthAI OCR Pipeline
      │                              │
      │                         EAST Text Detection (256x256)
      │                              │
      │                         text-recognition-0012
      │                              │
      │                         CTC Decoder ──► pyttsx3 TTS
      │
      ├── "tell weather" ──────► OpenWeatherMap API ──► pyttsx3 TTS
      │
      └── "get news" ──────────► BBC News Scraper ──► pyttsx3 TTS
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Edge Hardware | Luxonis OAK-D Camera, Jetson Nano |
| Object Detection | YOLOv4-Tiny (OpenVINO blob, DepthAI) |
| Depth Sensing | DepthAI Stereo Pipeline (StereoDepth node) |
| OCR Stage 1 | EAST Text Detection (256x256) |
| OCR Stage 2 | text-recognition-0012 (CTC decoder) |
| Computer Vision | OpenCV |
| Speech Input | SpeechRecognition + Google Speech API |
| Speech Output | pyttsx3 (offline TTS) |
| Weather | OpenWeatherMap API |
| News | BeautifulSoup + BBC News |
| Language | Python |

---

## Performance

| Metric | Value |
|---|---|
| Object detection FPS | 15 FPS on edge |
| Supported object classes | 80 (COCO dataset) |
| Depth range | 100mm – 5000mm |
| OCR confidence threshold | 0.8 |
| Distance output | X, Y, Z in meters |

---

## Project Structure

```
├── main.py              # Core system — all modules and voice command router
├── east.py              # EAST text detection decoder and NMS utilities
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Getting Started

### Hardware Required
- Luxonis OAK-D Camera
- Jetson Nano (or any device that supports DepthAI)
- Microphone
- Speaker or headphones

### 1. Clone the repository

```bash
git clone https://github.com/Zunaira3/AI-BlindAid.git
cd AI-BlindAid
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your OpenWeatherMap API key

In `main.py`, find the `get_weather` function and replace:
```python
api_key = "ENTER_YOUR_API_KEY_HERE"
```
with your key from [openweathermap.org](https://openweathermap.org/api)

### 4. Run the system

```bash
python main.py
```

AIBA will greet you and wait for a voice command.

---

## Dependencies

```
depthai
opencv-python
pyttsx3
SpeechRecognition
requests
beautifulsoup4
numpy
blobconverter
```

---

## Authors

**Zunaira Khalid**

---

[LinkedIn](https://www.linkedin.com/in/zunaira-khalid) · [GitHub](https://github.com/Zunaira3)
