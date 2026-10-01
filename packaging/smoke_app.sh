#!/usr/bin/env bash
# 冒烟：把 .app 起起来，确认它真的能用，然后停掉。
#
# 分两层，因为「起得来」和「能干活」是两件不同的事：
#
#   默认              --backend echo，不加载任何权重。只验证「双击之后能连上」这条
#                     路径：进程起得来、HTML 能拿到、管理接口能读、能自己停下来。
#                     不需要权重，也不会因为没网而失败。
#
#   --mlx             真加载权重、真跑一次推理。**打包回归只有这一层能抓到** ——
#                     打包最容易出的问题就是 MLX 的原生件（core.so / libmlx.dylib /
#                     181MB 的 mlx.metallib）没被正确收进产物，而那种情况下服务照样
#                     起得来、/healthz 也可能通，只有真跑一次推理才暴露。
#
# 用法：
#   packaging/smoke_app.sh                  # 快速冒烟（几秒）
#   packaging/smoke_app.sh --mlx            # 含真实推理（需要本地已有权重）
#   SMOKE_PORT=8099 packaging/smoke_app.sh --mlx
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WANT_MLX=0
APP=""
for arg in "$@"; do
  case "$arg" in
    --mlx) WANT_MLX=1 ;;
    *) APP="$arg" ;;
  esac
done
APP="${APP:-$REPO/dist/LayaServer.app}"
PORT="${SMOKE_PORT:-8077}"
BASE="http://127.0.0.1:$PORT"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/laya-smoke.XXXXXX")"

#: 热态推理的上限。GPU 上实测约 120ms；MLX 找不到 Metal 内核会**静默**退回 CPU，
#: 那时是秒级。留到 1 秒既能挡住 CPU 回退，又不会被机器忙时的抖动误伤。
MLX_BUDGET_MS=1000

[[ -d "$APP" ]] || { echo "找不到 $APP" >&2; exit 1; }

BACKEND="echo"
[[ "$WANT_MLX" == "1" ]] && BACKEND="mlx"
echo "▸ 启动 $APP --port $PORT --no-browser --backend $BACKEND"
"$APP/Contents/MacOS/laya-console" --port "$PORT" --no-browser --backend "$BACKEND" \
  > "$WORK/out.log" 2>&1 &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "▸ 等 /healthz"
for _ in $(seq 1 120); do
  curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "$BASE/healthz" | head -c 400; echo

echo "▸ 控制台是不是 HTML"
ctype="$(curl -fsS -o /dev/null -w '%{content_type}' "$BASE/")"
echo "  content-type: $ctype"
[[ "$ctype" == text/html* ]] || { echo "根路径不是 HTML，控制台没被打进去" >&2; exit 1; }

echo "▸ 管理接口"
curl -fsS "$BASE/admin/overview" | head -c 300; echo
curl -fsS "$BASE/admin/stats" | head -c 200; echo
# 资源采集在打包后容易因为缺东西而整块不可用，顺手看一眼。
curl -fsS "$BASE/admin/system" | head -c 200; echo

if [[ "$WANT_MLX" == "1" ]]; then
  echo "▸ 等权重就绪（auto_load=local，只加载本地已有的）"
  loaded=0
  for _ in $(seq 1 300); do
    loaded="$(curl -fsS "$BASE/healthz" 2>/dev/null \
      | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("loaded") or []))' 2>/dev/null || echo 0)"
    [[ "${loaded:-0}" -gt 0 ]] && break
    sleep 1
  done
  if [[ "${loaded:-0}" -eq 0 ]]; then
    echo "本地一份权重都没有，没法做真实推理。" >&2
    echo "先在有网的机器上跑一次服务把权重下下来，或者用 --backend mlx 手动加载。" >&2
    exit 1
  fi
  echo "  已驻留 $loaded 份"

  echo "▸ 真跑三次推理（首发放着不算，看热态）"
  BEST=999999
  for i in 1 2 3; do
    start="$(python3 -c 'import time; print(time.time())')"
    code="$(curl -sS -o "$WORK/resp-$i.json" -w '%{http_code}' -X POST "$BASE/v1/systemone" \
      -H 'Content-Type: application/json' --data-binary "@$REPO/examples/request.json" || echo 000)"
    elapsed="$(python3 -c "import time; print(round((time.time()-$start)*1000))")"
    echo "  #$i HTTP $code  ${elapsed}ms"
    [[ "$code" == "200" ]] || {
      echo "推理失败：$(head -c 400 "$WORK/resp-$i.json" 2>/dev/null)" >&2
      exit 1
    }
    [[ "$elapsed" -lt "$BEST" ]] && BEST="$elapsed"
  done

  python3 - "$WORK/resp-3.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
answers = body.get("answers") or {}
assert answers, f"响应里没有 answers：{body}"
print("  模型:", body.get("model"))
print("  答案字段:", sorted(answers))
PY

  if [[ "$BEST" -gt "$MLX_BUDGET_MS" ]]; then
    echo "最快一发也要 ${BEST}ms（上限 ${MLX_BUDGET_MS}ms）—— 慢得不像在用 GPU。" >&2
    echo "MLX 很可能静默退到了 CPU：检查 mlx/lib/mlx.metallib 和 libmlx.dylib 有没有" >&2
    echo "落在 Contents/Resources/runtime/mlx/ 下（原因见 packaging/laya_server.spec 顶部）。" >&2
    exit 1
  fi
  echo "  最快 ${BEST}ms —— 在 GPU 的合理区间内"
else
  echo "▸ 跳过真实推理（加 --mlx 可以跑；它能挡住「MLX 原生件没收进产物」这类回归）"
fi

echo "▸ 停机"
curl -fsS -X POST "$BASE/admin/shutdown" | head -c 120; echo
sleep 2
if kill -0 "$PID" 2>/dev/null; then
  echo "进程还活着，停机没生效" >&2
  exit 1
fi

echo "✓ 冒烟通过"
