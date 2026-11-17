# AI Video Ad Engine 🎬⚡

An autonomous, multi-scene marketing ad production system that transforms raw client product photos or videos into high-converting 9:16 vertical video ads.

---

## 🏗️ Architecture Layout

```
ai-video-ad-engine/
├── config/                 # Environment & path settings
├── src/
│   ├── core/               # Pydantic schemas (Manifest, Scenes, Transitions)
│   ├── deconstruction/     # Stage 1: Dense Multimodal Asset Audit & Keyframes
│   ├── intelligence/       # Stage 2: Mixture-of-Experts (MoE) Creative Brain
│   ├── generation/         # Stage 3: FLUX staging & Video API adapters
│   ├── transitions/        # Scene-to-Scene Continuity & Handshake Protocol
│   ├── compositor/         # Remotion / FFmpeg Programmatic Video Assembly
│   └── verifier/           # Stage 4: Multimodal QA & Hallucination Auditor
├── storage/                # Local asset cache, brand vaults, & outputs
└── ui/                     # Interactive testing dashboard & prompt workbench
```

---

## 🚀 Quick Setup

1. Copy `.env.example` to `.env` and insert your API keys:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Workbench UI:
   ```bash
   python -m ui.server
   ```

