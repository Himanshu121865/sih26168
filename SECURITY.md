# Security Policy

## Scope

This is a hackathon research project (ISRO SIH26168, deadline 20 Sep 2026), not a
production service. There are no servers, accounts, or backends.

## Data handling

- The app processes IMU + GNSS on-device only. Trajectories stay on the phone
  (local CSV log); nothing is uploaded anywhere — there is no network code path
  for user data (map tiles are fetched once for offline use, then cached).
- No camera, microphone, or contacts access. Permissions: location only.
- Training data (IO-VNBD, public) and checkpoints contain no PII.

## Reporting a vulnerability

Open a GitHub issue titled `[security] …` or contact the repo owner directly.
Hackathon timelines apply: expect triage within days, not hours.

## Out of scope

DoS via crafted CSV inputs to the offline training scripts is not treated as a
security issue (trusted researcher input). The debug APK in `releases/` is
unsigned and for field testing only — install at your own risk until the first
signed CI release.
