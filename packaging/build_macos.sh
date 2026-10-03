#!/usr/bin/env bash
# 打出可分发的 LayaServer.app（不依赖本机 Python）。
#
# ⚠️ 未在本机验证。原因写在 README 的「打包」一节：这台机器连不上 Hugging Face，
#    .app 能验证「起得来」，验证不了「答得对」。请在有网、有权重的机器上跑完整流程，
#    并用 packaging/smoke_app.sh 做一次冒烟。
#
# 用法：
#   packaging/build_macos.sh              # 出 .app
#   packaging/build_macos.sh --dmg        # 顺带出 .dmg
#   MAC_SIGN_ID="Developer ID Application: ..." packaging/build_macos.sh --dmg
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WANT_DMG=0
[[ "${1:-}" == "--dmg" ]] && WANT_DMG=1

VENV="${VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY。先建 venv 并安装：pip install -e '.[dev,packaging]'" >&2
  exit 1
fi

echo "▸ 确认打包依赖"
"$PY" -c "import PyInstaller" 2>/dev/null || "$VENV/bin/pip" install "pyinstaller>=6.6.0"

echo "▸ 清理旧产物"
rm -rf "$REPO/build" "$REPO/dist"

echo "▸ 生成应用图标"
# 从 assets/icon 的母版重建 packaging/LayaServer.icns（中间产物落在 build/ 下）。
# 仓库里提交了一份生成好的，这里再跑一次是为了保证它与母版同步 —— 改了母版却忘了
# 重跑 make_icon.py，图标会静默地留在旧版本，而且没有任何报错。
"$PY" packaging/make_icon.py

echo "▸ 跑测试（打包前先确认代码本身是好的）"
"$PY" -m pytest -q

echo "▸ PyInstaller 打包"
"$VENV/bin/pyinstaller" packaging/laya_server.spec --noconfirm --distpath "$REPO/dist" --workpath "$REPO/build"

APP="$REPO/dist/LayaServer.app"
[[ -d "$APP" ]] || { echo "没找到 $APP，打包失败" >&2; exit 1; }

echo "▸ 附带 mlx（原样拷贝，不经 PyInstaller）"
# 为什么不让 PyInstaller 收它：见 packaging/laya_server.spec 顶部那段。
# 一句话版本 —— PyInstaller 会把 core.so 的 rpath 从 `@loader_path/lib` 改写成
# `@loader_path/..`、把 dylib 和 181MB 的 metallib 分到两个目录去，结果 import mlx
# 直接失败。原样拷贝不动任何一个字节，绕开全部重链接。
MLX_SRC="$(ls -d "$VENV"/lib/python*/site-packages/mlx 2>/dev/null | head -1)"
[[ -d "$MLX_SRC" ]] || { echo "找不到 mlx 包（venv=$VENV），先 pip install -e '.[dev]'" >&2; exit 1; }
RUNTIME="$APP/Contents/Resources/runtime"
rm -rf "$RUNTIME/mlx"
mkdir -p "$RUNTIME"
cp -R "$MLX_SRC" "$RUNTIME/mlx"
# __pycache__ 只占体积，和加载无关（首次 import 会重建）。
find "$RUNTIME/mlx" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 校验：少一个都是「起得来但答不了」，而且报错会离原因很远，所以这里就拦住。
for required in mlx/lib/libmlx.dylib mlx/lib/mlx.metallib; do
  [[ -e "$RUNTIME/$required" ]] || { echo "mlx 拷进产物后缺少 $required" >&2; exit 1; }
done
ls "$RUNTIME"/mlx/core.*.so >/dev/null 2>&1 \
  || { echo "mlx 拷进产物后缺少 Python 扩展（core.*.so）" >&2; exit 1; }
echo "  → $RUNTIME/mlx（$(du -sh "$RUNTIME/mlx" | cut -f1)）"

echo "▸ 去掉隔离属性"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

# 签名分两档。ad-hoc 只能本机跑；要给别人的机器用，必须 Developer ID + 公证，
# 否则 Gatekeeper 会直接拦住并说「已损坏」——那其实是没签名，不是真的坏了。
if [[ -n "${MAC_SIGN_ID:-}" ]]; then
  echo "▸ 用 $MAC_SIGN_ID 签名"
  codesign --force --deep --options runtime --timestamp --sign "$MAC_SIGN_ID" "$APP"
  if [[ -n "${NOTARY_PROFILE:-}" ]]; then
    echo "▸ 提交公证（几分钟）"
    ditto -c -k --keepParent "$APP" "$REPO/dist/notarize.zip"
    xcrun notarytool submit "$REPO/dist/notarize.zip" \
      --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP"
    rm -f "$REPO/dist/notarize.zip"
  else
    echo "  （设 NOTARY_PROFILE=xxx 可以顺带做公证）"
  fi
else
  echo "▸ 没有 MAC_SIGN_ID，做 ad-hoc 签名（只能本机用）"
  codesign --force --deep --sign - "$APP"
fi

if [[ "$WANT_DMG" == "1" ]]; then
  echo "▸ 生成 dmg"
  VERSION="${VERSION:-0.1.0}"
  ARCH="$(uname -m)"
  DMG="$REPO/dist/LayaServer-macos-$ARCH.dmg"
  rm -f "$DMG"
  hdiutil create -volname "LayaServer" -srcfolder "$APP" -ov -format UDZO "$DMG"
  echo "✓ $DMG"
fi

echo "✓ $APP"
echo
echo "冒烟测试："
echo "  packaging/smoke_app.sh"
