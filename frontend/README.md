# SnapClass Frontend

This directory contains the React/Vite frontend for SnapClass. It provides an Apple Design–inspired tactile interface for webcam capture, image file uploads, bulk reference archive ingestion, and real-time visual classification.

## Prerequisites

- [Bun](https://bun.sh/) 1.0+ (recommended for high-speed builds, testing, and package management) or Node.js 20+
- The SnapClass backend running locally on `http://localhost:8000`

The frontend communicates with `http://localhost:8000` by default (configured in `src/api/snapclass.js`).

## Getting Started with Bun

Install dependencies:

```bash
bun install
```

Start the local development server:

```bash
bun run dev
```

Vite will serve the application at `http://localhost:5173`.

## Available Scripts

- **`bun run dev`**: Starts the Vite development server with Hot Module Replacement (HMR).
- **`bun run test`** (or **`bun test test/`**): Runs the automated UI test suite with `bun test`, `@testing-library/react`, and `happy-dom`.
- **`bun run lint`**: Runs ESLint across all components and hooks.
- **`bun run build`**: Compiles and bundles production assets into `dist/`.
- **`bun run preview`**: Serves the compiled production build locally for verification.

> **Note on npm**: `package-lock.json` is also maintained for npm environments (`npm install`, `npm run dev`, `npm test`).

## UI Testing Architecture

UI component tests are located in `test/`:
- `test/dom-setup.js` & `test/cleanup-setup.js`: Preloads the DOM environment (`happy-dom`) and registers `@testing-library/react` cleanup hooks.
- `test/Header.test.jsx`: Header bar, branding, connection status pill, and light/dark theme switcher.
- `test/ClassifyPanel.test.jsx`: Segmented mode controls (Camera, Upload, Catalog) and classify triggers.
- `test/CameraBox.test.jsx`: Optical viewfinder reticles, live camera controls, and dropzone preview.
- `test/MatchCard.test.jsx`: Top match card scores, runner-up progress bars, and confidence formatting.
- `test/ResultPanel.test.jsx`: Empty placeholder state, embedding skeleton loaders, and candidate lists.
