#!/usr/bin/env bash
# 轻量 .app 包装：双击启动 laya-server 控制台，不需要 PyInstaller。
#
# 它做的只是「把脚本包成 App」—— .app 里的可执行文件是个 shell 脚本，指向本仓库的
# .venv。所以：
#   * 秒级完成，零额外依赖；
#   * 但**不能分发**：.app 里没有 Python，换台机器就废；
#   * 仓库目录挪走要重新生成（路径是生成时写进去的）。
#
# 要能发给别人的独立包，用 build_macos.sh（PyInstaller）。
#
# 用法：
#   packaging/make_macos_app.sh                 # 生成到 ~/Applications
#   OUT_DIR=/tmp packaging/make_macos_app.sh    # 换输出目录
#   额外参数原样传给 laya-console（如 --port 9000 --no-native）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$HOME/Applications}"
APP_NAME="${APP_NAME:-LayaServer}"
APP="$OUT_DIR/$APP_NAME.app"
# 默认 --native：双击出来的应该是真正的 macOS 原生窗口（系统 WebKit），
# 而不是借浏览器。浏览器模式的问题是「所有调用都返回成功，只是没人看得到」——
# 用户点完图标什么都没发生，就以为 App 坏了。
# 没装 pyobjc 时 --native 会自动退回浏览器路径，不会因此起不来。
MODES=("$@")
[[ ${#MODES[@]} -eq 0 ]] && MODES=("--native")

VENV_PY="$REPO/.venv/bin/laya-console"
if [[ ! -x "$VENV_PY" ]]; then
  echo "找不到 $VENV_PY。" >&2
  echo "先在仓库根跑：python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

# 原生壳要 pyobjc。缺了它 --native 会退回浏览器路径 —— 不算错，但用户装这个
# .app 图的就是「不依赖浏览器」，所以这里提醒一句。
if ! "$REPO/.venv/bin/python" -c "import AppKit, WebKit" 2>/dev/null; then
  echo "! 没装 pyobjc，原生窗口用不了，会退回浏览器。装一下：" >&2
  echo "    .venv/bin/pip install 'laya-server[native]'" >&2
fi

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# 图标：直接用仓库里提交好的 .icns，不在这里重建。这个脚本的卖点是「零额外依赖、
# 秒级完成」，而重建图标要拉 Pillow —— 为了一个图标破掉这个承诺不划算。
# 改了母版就手动跑一次：.venv/bin/python packaging/make_icon.py
ICNS="$REPO/packaging/LayaServer.icns"
if [[ -f "$ICNS" ]]; then
  cp "$ICNS" "$APP/Contents/Resources/LayaServer.icns"
else
  echo "! 没找到 $ICNS，生成的应用会用系统默认图标。" >&2
  echo "   重建：.venv/bin/python packaging/make_icon.py" >&2
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>Laya Server</string>
  <key>CFBundleIdentifier</key><string>ai.laya.server.console</string>
  <key>CFBundleExecutable</key><string>launch</string>
  <key>CFBundleIconFile</key><string>LayaServer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# 用 exec 让 Python 进程直接接管这个 pid：这样在 Dock 上退出、或者发 SIGTERM，
# 都会打到真正跑服务的那个进程上，而不是留在外壳脚本里。
#
# 输出重定向到日志文件是必须的：双击启动时进程是 launchd 的子进程，stdout/stderr
# 没有任何地方能看到。没有日志的话，「双击了没反应」就真的无从查起 —— 而这个
# .app 的整个设计目标就是「不碰命令行」，用户不会想到去终端里手动跑一遍看报错。
#
# 顺序也很重要：**日志先于一切**。之前写成先 `cd` 再设日志，于是在没有这个仓库
# 目录的机器上（也就是别人拿到这个 .app 时），cd 失败、脚本退出，而失败发生在
# 重定向之前 —— 表现是双击「一声不响什么都没有」，连一行日志都留不下来。
{
  echo '#!/bin/bash'
  echo "# 由 packaging/make_macos_app.sh 生成，指向 $REPO"
  echo 'LOG_DIR="$HOME/.laya-server"'
  echo 'mkdir -p "$LOG_DIR"'
  # 只留一份、超过 2MB 就轮转 —— 这是个控制台应用的日志，不是审计日志。
  echo 'LOG="$LOG_DIR/app.log"'
  echo 'if [ -f "$LOG" ] && [ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 2097152 ]; then mv "$LOG" "$LOG.1"; fi'
  echo 'printf "\n===== %s 启动 =====\n" "$(date "+%Y-%m-%d %H:%M:%S")" >> "$LOG"'
  echo ''
  echo '# 先确认壳指向的东西还在，再 cd。这个 .app 只有 20KB：它**不含** Python、'
  echo '# 不含依赖库、不含模型权重，只是指向这个仓库的一个快捷方式。'
  printf 'if [ ! -x "%s" ]; then\n' "$VENV_PY"
  echo '  echo "找不到启动器。这个 App 只是一个外壳，不含 Python、依赖库和模型权重，" >> "$LOG"'
  printf '  echo "必须配合仓库目录使用（生成时指向 %s）。" >> "$LOG"\n' "$REPO"
  echo '  echo "要把它给别人用，请对方自己从仓库安装，或改用 packaging/build_macos.sh 打独立包。" >> "$LOG"'
  echo '  osascript -e "display alert \"LayaServer 启动失败\" message \"找不到启动器。这个 App 只是一个外壳，必须配合它生成时指向的仓库目录使用。详情见 ~/.laya-server/app.log\" as critical" >/dev/null 2>&1 || true'
  echo '  exit 1'
  echo 'fi'
  printf 'cd "%s" || exit 1\n' "$REPO"
  printf 'exec "%s"' "$VENV_PY"
  for mode in "${MODES[@]}"; do printf ' "%s"' "$mode"; done
  echo ' "$@" >> "$LOG" 2>&1'
} > "$APP/Contents/MacOS/launch"
chmod +x "$APP/Contents/MacOS/launch"

# 未签名的 .app 在别的机器上会被 Gatekeeper 拦；本机自用只需要去掉隔离属性。
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP" 2>/dev/null \
  && echo "✓ 已做 ad-hoc 签名" \
  || echo "! ad-hoc 签名失败（不影响本机使用）"

echo "✓ 生成 $APP"
echo "  双击即可启动（参数：${MODES[*]}）。"
echo "  改参数就重新生成，例如：packaging/make_macos_app.sh --backend echo --port 9000"
