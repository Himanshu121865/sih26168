# Contributing to SIH26168

Two tracks, one `main`. Android crew owns `android/`; AI crew owns `python/` +
` sih26168_colab.ipynb`. Interface files (`model.tflite`, `scaler.json`,
`docs/adr/`, CSV log schema) need **cross-track review** — see
`docs/INTERFACE_CONTRACTS.md`.

## Setup

**Android track** — Android Studio Hedgehog+, JDK 17. Open the `android/`
folder, let Gradle sync, run on a real device (emulator IMUs are too clean).
No `gradlew` wrapper is checked in; use Studio or system Gradle 8.7:
`gradle :app:assembleDebug --project-dir android`.

**AI track** — Colab T4 GPU. Open
`https://colab.research.google.com/github/Himanshu121865/sih26168/blob/main/sih26168_colab.ipynb`,
Factory reset runtime, Run all. Local Python is for lint/compile only
(no torch/numpy here); heavy runs always happen on Colab.

## Branching

- Never push to `main` directly. Branch from `main`: `android/<short-name>` or
  `model/<short-name>` (e.g. `android/map-snap-blend`, `model/stratified-split`).
- One concern per PR. Small diffs get reviewed in hours, big ones in days.
- Rebase onto `main` before requesting review; force-push your own branch only.

## PR checklist (enforced by CI + reviewer)

- [ ] CI green (python workflow and/or android workflow, whichever you touched).
- [ ] Tests: new behavior has a test (`tests/test_*.py` or `*Test.kt`).
  Mirror rule — Python harness change? Update the Kotlin test expectation too.
- [ ] Docs: behavior change updates `docs/`, `android/README.md`, or an ADR.
  New decision with trade-offs? Add `docs/adr/NNNN-slug.md` + index row.
- [ ] Interface change (`model.tflite` I/O, `scaler.json`, CSV schema, drift
  metrics)? Request a reviewer from the **other** track (see CODEOWNERS).
- [ ] No secrets, no datasets, no `data/` dumps in the diff.

## Review rules (mixed levels — be explicit)

- Reviewers: check out the branch and run the affected command, don't just read.
- Authors: paste test output into the PR body (Colab cell output or
  `./gradlew` log tail). "Works on my machine" without logs = request changes.
- Interface PRs need one approval per track. Docs-only PRs need one approval.

## Field test / data collection

Rider + holder + route needed. Coordinate in the tracking issue before riding:
device model, route, scenarios (potholes, hard-brake, lean turns, underpass).
Raw logs go to shared drive, never into git — only processed summaries land in
`reports/`.
