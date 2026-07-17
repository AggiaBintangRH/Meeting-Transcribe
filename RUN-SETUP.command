#!/usr/bin/env bash
# Double-click this file in Finder to run the model setup.
# All output is saved to logs/setup-run.log so Claude can read it directly.
cd "$(dirname "$0")"
./download-best-models.sh
echo ""
echo "=============================================="
echo "Done. Full log saved to logs/setup-run.log"
echo "Tell Claude: 'setup finished' — it will read the log itself."
echo "=============================================="
read -r -p "Press Enter to close..."
