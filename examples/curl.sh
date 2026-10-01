#!/usr/bin/env bash
# 最小调用示例。用法：
#   ./examples/curl.sh                      # 英文示例
#   ./examples/curl.sh request_zh.json      # 中文示例
#   BASE_URL=http://127.0.0.1:8077 API_KEY=xxx ./examples/curl.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8077}"
PAYLOAD="${1:-request.json}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

auth=()
if [[ -n "${API_KEY:-}" ]]; then
  auth=(-H "Authorization: Bearer ${API_KEY}")
fi

echo "# 可用模型"
curl -sS "${BASE_URL}/v1/models" | python3 -m json.tool

echo
echo "# POST /v1/systemone < ${PAYLOAD}"
curl -sS -X POST "${BASE_URL}/v1/systemone" \
  -H 'Content-Type: application/json' \
  "${auth[@]}" \
  --data-binary @"${DIR}/${PAYLOAD}" | python3 -m json.tool
