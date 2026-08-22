#!/bin/zsh
# Weekday-morning wheel brief (SPEC-008). Installed as LaunchAgent
# com.rahul.wheel-rlbot-daily — see scripts/com.rahul.wheel-rlbot-daily.plist.
set -u
cd /Users/rahul/Documents/MyFiles/Areas/Claude-Workspace/Projects/wheel-strategy-rlbot
LOG=data_local/live/daily_run.log
echo "=== run started $(date '+%Y-%m-%d %H:%M:%S %Z') ===" >> "$LOG"
/Users/rahul/opt/anaconda3/envs/wheel-rlbot/bin/python -m rlbot.assistant.daily --download >> "$LOG" 2>&1
STATUS=$?
echo "=== run finished status=$STATUS $(date '+%H:%M:%S') ===" >> "$LOG"
# keep the log from growing unbounded (~last 5000 lines)
tail -n 5000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
exit $STATUS
