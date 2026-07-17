#!/usr/bin/env bash
# Double-click this file in Finder to build & run the app.
# Build/run output is saved to logs/app-run.log so Claude can read it directly.
cd "$(dirname "$0")/MeetingTranscriber"
mkdir -p ../logs
echo "==> swift run started: $(date)" > ../logs/app-run.log
swift run 2>&1 | tee -a ../logs/app-run.log
echo ""
echo "Log saved to logs/app-run.log — tell Claude 'app finished' to have it read the log."
read -r -p "Press Enter to close..."
