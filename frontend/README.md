# SnapClass Frontend

This directory contains the React/Vite frontend for SnapClass. It provides a simple interface for starting the camera, capturing an image, uploading files, and sending requests to the backend API.

## Prerequisites

- Node.js 20+
- npm
- The SnapClass backend running locally on `http://localhost:8000`

The frontend currently uses a hardcoded API base URL in `src/api/snapclass.js`, so the backend should be available at that address during development.

## Development

```bash
npm install
npm run dev
```

Vite will print the local development URL, usually `http://localhost:5173`.

## Available Scripts

- `npm run dev` starts the Vite dev server.
- `npm run build` creates a production build.
- `npm run preview` previews the production build locally.
- `npm run lint` runs ESLint.
