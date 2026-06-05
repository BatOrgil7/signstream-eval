# Few-Shot Sign Lab

Honors capstone prototype for real-time isolated sign recognition with signer personalization.

Live demo: https://asl-translation-capstone.vercel.app

The research problem is not "translate all ASL." The focused problem is:

> Can a lightweight landmark-based recognizer adapt to a new signer after only a few examples per sign?

This repository contains two prototype tracks:

- **Web prototype**: a Vercel-ready browser app that uses the webcam, MediaPipe hand landmarks, and a local few-shot k-nearest-neighbor classifier.
- **Python prototype**: early OpenCV/MediaPipe/TensorFlow scripts for collecting hand landmarks, training a model, and running live desktop inference.

## Quick Test For Professors And Reviewers

1. Open https://asl-translation-capstone.vercel.app in Chrome, Edge, or another modern browser.
2. Click **Start Camera** and allow webcam access.
3. Keep the default labels or replace them with a small set of isolated signs/glosses.
4. Select one label, perform that sign, and click **Capture One** several times.
5. Repeat for the other labels.
6. Try the signs again and watch the live prediction panel.
7. Use **Export JSON** or **Export CSV** if you want to inspect the collected landmark samples.

For a more structured test session, use [docs/USER_TESTING_GUIDE.md](docs/USER_TESTING_GUIDE.md).

## Live Web Prototype

The web app runs fully in the browser. Camera frames and calibration samples stay local unless the tester exports JSON or CSV.

### Local setup

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, start the camera, pick a label, and capture a few examples per sign.

### Production build

```bash
npm run build
npm run preview
```

### Deploy

```bash
npm run build
vercel deploy
```

For production:

```bash
vercel deploy --prod
```

## Research Framing

Many sign-recognition demos report high accuracy when training and testing data are randomly mixed. That can hide the harder problem: a deployed recognizer must work for a new signer whose hand shape, speed, angle, and signing style were not in the training set.

This prototype supports a smaller, testable research question:

1. How well does landmark-based sign recognition work for a new user?
2. How much does performance improve after 1, 3, 5, or 10 examples per sign from that user?
3. Can this run in real time on normal consumer hardware?

## Suggested Evaluation Plan

Use the web prototype for early pilot testing, then move to a controlled benchmark:

1. Choose a public multi-signer dataset.
2. Train on a set of signers and hold out unseen signers.
3. Measure baseline unseen-signer accuracy.
4. Add 1, 3, 5, and 10 calibration examples per sign from the held-out signer.
5. Report accuracy, macro F1, confusion matrix, latency, model size, and failure cases.

## Web Prototype Controls

- **Labels**: comma-separated sign labels or glosses.
- **Capture One**: saves one hand-landmark example for the active label.
- **Record Stream**: saves examples continuously for the active label.
- **Neighbors**: controls the k value in k-nearest neighbors.
- **Confidence threshold**: controls when the app shows a label instead of "Unsure."
- **Export JSON / CSV**: exports the local calibration session for analysis.

## Project Structure

```text
.
|-- index.html                  # Web app entry point
|-- src/
|   |-- app.js                  # MediaPipe, calibration, kNN, export logic
|   `-- styles.css              # Responsive app styling
|-- public/
|   |-- models/                 # Local MediaPipe hand landmarker model
|   `-- wasm/                   # Local MediaPipe WebAssembly runtime
|-- docs/
|   `-- USER_TESTING_GUIDE.md   # Suggested tester protocol
|-- collect_data.py             # Early desktop data-collection prototype
|-- train_model.py              # Early TensorFlow training prototype
|-- recognize_signs.py          # Early desktop inference prototype
|-- python-requirements.txt     # Python-only dependencies
|-- package.json                # Web app dependencies and scripts
|-- vercel.json                 # Vercel deployment config
`-- README.md
```

## Python Prototype

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r python-requirements.txt
```

Collect landmark samples:

```bash
python collect_data.py
```

Train a simple neural model:

```bash
python train_model.py
```

Run desktop inference:

```bash
python recognize_signs.py
```

The Python scripts are early local prototypes. The web app is the shareable professor/testing prototype.

## Current Limitations

- The web classifier is a transparent few-shot baseline, not a final deep-learning model.
- The app recognizes isolated signs only, not continuous ASL grammar.
- It uses one detected hand and does not currently model facial expression or body pose.
- Browser performance depends on device, lighting, and camera angle.

## Troubleshooting

- If the app says **Hand tracker failed to load**, refresh once and confirm the browser is online.
- If the camera does not start, check browser camera permissions and use HTTPS or localhost.
- If predictions stay **Unsure**, capture more examples per label and keep the hand centered.
- If one sign is confused with another, clear that label's samples and recapture slower, cleaner examples.
- If testing on a phone, use good lighting and keep the hand inside the camera frame.

## Privacy Note

The web prototype runs in the browser. It does not upload video frames or samples to a server. Exported JSON/CSV files are created only when the tester clicks an export button.

## Thesis Direction

Working title:

**Few-Shot Personalization for Real-Time Sign Recognition in Human-Robot Interaction**

One-sentence summary:

> This honors thesis studies whether lightweight hand-landmark models can recognize isolated signs from unseen signers in real time, and how much few-shot personalization improves performance.
