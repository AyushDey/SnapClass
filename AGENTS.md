# AGENTS.md — SnapClass AI Agent & Contributor Guidelines

This document provides system instructions for any AI assistant (Antigravity, Cursor, Windsurf, Claude Code, GitHub Copilot, Codex, etc.) or human developer working on the **SnapClass** codebase.

---

## 1. Project Overview & Architecture

SnapClass is an offline-first few-shot image classification engine with high-performance vector similarity search:

- **Backend**: Python 3.13+, FastAPI, SQLAlchemy 2.1+, PostgreSQL 14+ with `pgvector` and HNSW indexing (`vector_cosine_ops`), ResNet18 feature extraction. Managed via [`uv`](https://docs.astral.sh/uv/).
- **Frontend**: React 19, Vite 8, Bun runtime & package manager. Fast testing via `bun test`, `@testing-library/react`, and `happy-dom`.
- **Architecture**:
  - Image embeddings (512-dim) are stored directly in PostgreSQL with an HNSW index for logarithmic nearest-neighbor retrieval.
  - SQLite fallback is maintained for fast in-memory testing.
  - The frontend communicates with the backend via `http://localhost:8000`.

---

## 2. Design System & Frontend Principles

SnapClass uses a **tactile optical laboratory instrument aesthetic** inspired by **Apple Design** and **Emil Kowalski's Design Engineering** principles. The complete design skill definitions are preserved in `.agents/skills/`.

### Strict Aesthetic Rules:
1. **No "Generic AI" Look**:
   - **STRICTLY FORBIDDEN**: Multi-color gradients, neon glow effects, and purple/blue accent combinations commonly associated with template AI interfaces.
   - **PALETTE**: Neutral basalt (`#0d0d0f`) and obsidian (`#141416`) surfaces, hairline precision borders (`rgba(255, 255, 255, 0.08)` / `#2e2e34`), physical vermilion primary accents (`#ea580c`), and amber confidence meters.
2. **Unobstructed Viewfinder**:
   - The live camera feed and captured/uploaded preview images must **never** be obstructed or covered by match result cards, floating pills, or modal overlays. Results are strictly rendered in the adjacent `ResultPanel`.
3. **Typography & Numbers**:
   - Use San Francisco font stack with tabular numbers (`font-variant-numeric: tabular-nums;`) on all scores, percentages, and metrics to prevent layout jitter during real-time updates.
   - Optical letter spacing (`letter-spacing: -0.02em` on headings, `-0.01em` on UI labels).
4. **Motion & Interaction Physics**:
   - Use spring physics transitions (`cubic-bezier(0.16, 1, 0.3, 1)`).
   - Ensure all gesture controls (such as the Light/Dark theme toggle) support bidirectional drag tracking with physical boundary clamping and tactile snap haptics.
   - Respect user accessibility preferences (`prefers-reduced-motion` and `prefers-contrast`).

---

## 3. Commit Message Style & Automated Tagging

The GitHub Actions workflow (`.github/workflows/test.yml`) automatically creates git tags and release notes based on commit messages pushed to `main`.

### Commit Types:
- `feat`: New features or capabilities.
- `fix`: Bug fixes.
- `chore`: Dependency updates, tooling, or maintenance.
- `docs`: Documentation updates.
- `refactor`: Code reorganization without behavioral change.

### Automated Release Tag Trigger Format:
When ready to create a new version tag, format the commit header with the version tag in parentheses:

```text
feat(vX.Y.Z): concise summary of the release

- Bullet point detailing feature 1
- Bullet point detailing feature 2
- Bullet point detailing bug fix or optimization
```

**Examples:**
- `feat(v0.3.2): tactile haptic feedback and zoom controls`
- `fix(v0.3.3): prevent viewfinder layout shift on mobile viewports`
- `chore(v0.4.0): upgrade to postgresql 17 and pgvector 0.6`

#### How CI Handles This:
1. When pushed to `main`, the test workflow validates all backend and frontend tests.
2. If tests pass, CI inspects the commit header for the pattern `^(feat|fix|release|chore)\(v[0-9]+\.[0-9]+\.[0-9]+.*\)`.
3. If matched, it extracts `vX.Y.Z`, verifies the tag does not already exist, and creates an **annotated git tag** containing the complete commit message as the release notes.
4. You can then draft or publish a GitHub Release from that tag with a single click.

*Commits without a version tag (e.g. `feat: improve button focus styles`) will run the full test suite without generating a tag.*

---

## 4. Mandatory Quality Gates

Before committing any change, ensure all quality gates pass:

### Backend:
```bash
cd backend
uv run pytest -q tests --cov=. --cov-report=term-missing --cov-fail-under=100
```
- **100% statement and branch coverage** is enforced as a hard gate. Every new branch, exception handler, and fallback path must have test coverage.

### Frontend:
```bash
cd frontend
bun run lint      # Must report 0 errors and 0 warnings (eslint .)
bun run test      # Must pass all UI test cases (bun test test/)
bun run build     # Must compile clean in production mode (vite build)
```

---

## 5. Agent Skills Reference

Detailed guidance is checked into the repository under `.agents/skills/`:
- [`.agents/skills/apple-design/SKILL.md`](.agents/skills/apple-design/SKILL.md): Fluid physical motion, spring animation tables, and design foundations.
- [`.agents/skills/emil-design-eng/SKILL.md`](.agents/skills/emil-design-eng/SKILL.md): UI polish, invisible details, micro-interactions, and contrast ratios.
- [`.agents/skills/frontend-design/SKILL.md`](.agents/skills/frontend-design/SKILL.md): Aesthetic direction, restrained typography, and non-templated layouts.
