# Scheduled daily run (SPEC-008)

- `run_daily.sh` — the weekday-morning runner. The *installed* copy lives at
  `~/Library/Application Support/wheel-rlbot/run_daily.sh` (launchd cannot read
  program scripts from ~/Documents due to macOS TCC); after editing the repo
  copy, re-copy it there.
- `com.rahul.wheel-rlbot-daily.plist` — LaunchAgent, installed at
  `~/Library/LaunchAgents/`, fires weekdays 06:00 local (America/Chicago).
  Missed runs (Mac asleep) fire once on wake.

Reinstall after changes:
  cp scripts/run_daily.sh "$HOME/Library/Application Support/wheel-rlbot/"
  cp scripts/com.rahul.wheel-rlbot-daily.plist ~/Library/LaunchAgents/
  launchctl bootout gui/$(id -u)/com.rahul.wheel-rlbot-daily 2>/dev/null
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.rahul.wheel-rlbot-daily.plist

Manual trigger:  launchctl kickstart gui/$(id -u)/com.rahul.wheel-rlbot-daily
Logs:            data_local/live/daily_run.log (app) and launchd.log (spawn errors)
