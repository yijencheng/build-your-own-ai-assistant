#!/bin/bash
set -e

source .env

WEBHOOK_URL="https://ivanleomk--koroku-server-chat.modal.run"
COMMANDS_JSON='[
  {"command":"clear","description":"Clear state and stop run"},
  {"command":"stop","description":"Stop current run"}
]'

echo "Setting webhook..."
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=${WEBHOOK_URL}"
echo

echo "Setting bot commands..."
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setMyCommands" \
  --data-urlencode "commands=${COMMANDS_JSON}"
echo
