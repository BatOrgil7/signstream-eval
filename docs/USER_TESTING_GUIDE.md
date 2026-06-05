# User Testing Guide

This guide is for pilot testing the Few-Shot Sign Lab web prototype.

## Goal

The goal is to observe whether a new signer can calibrate the recognizer with only a few examples per sign and then get stable real-time predictions.

This is a prototype test, not a claim that the system translates full ASL.

## Recommended Setup

- Use Chrome or Edge on a laptop if possible.
- Sit in a well-lit room.
- Keep one hand visible and centered in the webcam.
- Avoid busy backgrounds when possible.
- Test isolated signs or short glosses, not full sentences.

## Suggested Test Session

1. Open https://asl-translation-capstone.vercel.app.
2. Click **Start Camera** and allow webcam access.
3. Use 3 to 6 labels for the first test session.
4. For each label:
   - Select the label button.
   - Perform the sign clearly.
   - Click **Capture One** 5 times, or use **Record Stream** for about 2 seconds.
5. After collecting samples for all labels, repeat each sign without pressing capture.
6. Record whether the prediction is correct, wrong, or unsure.
7. Export JSON or CSV only if the tester agrees to share the landmark samples.

## What To Record

For each test session, write down:

- Browser and device used.
- Number of labels tested.
- Number of examples captured per label.
- Which signs worked well.
- Which signs were confused.
- Whether the system felt real time.
- Any lighting, camera, hand-angle, or distance issues.

## Suggested Metrics

For informal pilot testing:

- Accuracy after 1, 3, 5, and 10 examples per sign.
- Number of **Unsure** predictions.
- Signs that are frequently confused.
- Tester feedback on usability.

For thesis-level evaluation:

- Accuracy.
- Macro F1.
- Confusion matrix.
- Average inference latency.
- Model/runtime size.
- Comparison between no personalization and few-shot personalization.

## Ethics And Privacy

The web app does not upload camera frames or samples by itself. If exported JSON/CSV files are collected from other people, treat them as research data and ask the thesis advisor whether IRB review or consent language is needed.

Avoid claiming that the prototype understands or translates ASL grammar. The current prototype recognizes isolated signs from hand landmarks.
