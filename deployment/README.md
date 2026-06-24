# 🚀 Live Deployment

This folder contains the **cloud-deployable version** of the Traffic Violation Detection System, adapted to run on [Hugging Face Spaces](https://huggingface.co/spaces/yashgupta11122004/traffic-violation-detection) (CPU-only, free tier).

**🔗 Live Demo:** https://huggingface.co/spaces/yashgupta11122004/traffic-violation-detection

---

## What's different from the main project?

The root of this repository contains the original **college major project** — modular training pipeline (`detection/`, `tracking/`, `violation_engine/`), GPU-based training scripts, and the full-featured local Flask app (`flask_app.py`).

This `deployment/` folder is a **single-file, cloud-ready adaptation** of that same detection engine, rebuilt with a different frontend:

| Aspect | Main Project (root) | Deployment (this folder) |
|---|---|---|
| Structure | Modular (multiple files/folders) | Single `app.py` |
| Models | Trained locally, loaded from disk | Auto-downloaded from [HF Model Hub](https://huggingface.co/yashgupta11122004/traffic-violation-models) |
| Hardware | GPU (RTX 3050) | CPU-only (free tier) |
| Detection logic | Same EasyOCR + YOLOv8 pipeline | Same EasyOCR + YOLOv8 pipeline |
| UI | Original light dashboard | Rebuilt "control room" interface (see below) |
| Purpose | Academic submission, training | Public live demo, portfolio |

---

## 🎨 UI Redesign — "Control Room" Interface

The frontend was rebuilt from scratch around a different concept: instead of an upload box sitting next to a results box, **the image or video itself becomes the interface**, similar to camera surveillance / detection software. Below is what was added and why.

### Layout concept
- **Full-bleed detection stage** — the uploaded image/video fills the entire viewport area (like a camera feed), with corner targeting brackets framing it, instead of living inside a small card.
- **Floating HUD panels** — detection results (vehicle count, person count, plate number, violations, challan confirmation) appear as semi-transparent glass panels *overlaid directly on the image*, not in a separate sidebar.
- **Command-center dashboard** — the Dashboard tab leads with one large "Outstanding Fine" hero panel (with a live sparkline graph) instead of seven equal-sized stat boxes, so the most important number is the most visually dominant.

### Visual system
- **Number-plate motif** — vehicle numbers, badges, and key data tags are styled like physical number plates (dark background, monospace, bordered) throughout the UI for thematic consistency.
- **Tactical HUD color coding** — every color has one specific meaning instead of a single dominant accent color:
  - 🩵 Cyan — active scanning / live data state
  - 🟣 Violet — number-plate / key-identifier highlight
  - 🟠 Amber — primary actions and violation alerts only
  - 🟢 Green — system-online / success / paid status
  - 🔴 Red — violations / unpaid challans
- **Animated radar (canvas)** — a real animated sweep drawn on `<canvas>` in the header, indicating the detection engine is active.
- **Scanline sweep** — while an image/video is being processed, an animated light line sweeps across the stage to visually represent "scanning in progress."
- **Sparkline mini-charts** — dashboard metric cards draw a small trend-line graph (canvas) showing recent history, not just a static number.
- **Animated counters** — dashboard numbers count up from 0 to their target value instead of appearing instantly.
- **Film-grain texture overlay** — a very subtle noise texture across the whole page so flat color areas don't look like a flat PNG.

### Theming
- **Light/Dark mode toggle** — a switch in the header toggles between a warm "daylight ops" light theme and a near-black "night ops" dark theme. Preference is remembered via `localStorage`.
- Both themes share the same color *meaning* (cyan = scan, amber = alert, etc.) so the interface behaves consistently regardless of theme.

### Interaction details
- **Drag-and-drop upload** directly onto the stage (not just click-to-browse).
- **Toast notifications** for every action (detection complete, challan generated, record updated/deleted).
- **Auto-refreshing dashboard** — polls every 30 seconds while the Dashboard tab is open.
- All existing functionality from the original dashboard (search, filter, status update modal, CSV export, PDF download, charts) is preserved — only the visual presentation changed.

> None of this affects the underlying detection logic — `check_violations()`, the per-vehicle proximity matching, and the fine rules are identical to the main project described in [TECHNICAL.md](../TECHNICAL.md).

---

## Run this version locally

```bash
cd deployment
pip install -r requirements.txt --break-system-packages
python app.py
```

Open `http://localhost:7860`

## Deploy your own copy

1. Create a Hugging Face Space (SDK: Docker)
2. Upload your trained model weights to a HF Model Hub repo
3. Update the `repo_id` in `app.py` to point to your model repo
4. Push the contents of this folder to your Space repo