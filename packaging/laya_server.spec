# PyInstaller 打包配置：把 laya-server 打成一个不依赖本机 Python 的 .app。
#
# 运行：.venv/bin/pyinstaller packaging/laya_server.spec --noconfirm
# 或直接用 packaging/build_macos.sh（它会先装打包依赖、调用这里、再补 mlx）。
#
# 需要额外关照两件事：
#   1. tokenizers 是 Rust 扩展，PyInstaller 的官方 hook 能处理，但要保证它被收集到。
#   2. 控制台 HTML 是普通数据文件，必须显式放进 laya_server/static，
#      因为 app.py 是相对 __file__ 找它的。
#
# ── 关于 MLX：它**刻意不走** PyInstaller ───────────────────────────────
#
# `mlx` 由 build_macos.sh 把整个 `site-packages/mlx` 目录**原样**拷进
# `LayaServer.app/Contents/Resources/runtime/mlx`，运行时由 entry.py 加进 sys.path。
# 下面既不 collect_dynamic_libs("mlx") 也不 collect_data_files("mlx")，并把它列进
# excludes —— 这是刻意的，不是漏了。
#
# 为什么：
#
#   * `mlx/core.cpython-313-darwin.so` 的 rpath 是 `@loader_path/lib`；
#   * `mlx/lib/mlx.metallib`（181MB，Metal 着色器库）由 C++ 层**按 dylib 所在的
#     相对目录**查找，Python 层没有任何接口能改（`mx.metal` 里没有 set_metallib_path）。
#
#   实测交给 PyInstaller 的后果：它把 .so 和 .dylib 放进 `Contents/Frameworks`、
#   把 .metallib 放进 `Contents/Resources`（再互相插符号链接兜底），并把 core.so 的
#   rpath 改写成 `@loader_path/..`。结果是 import 直接失败：
#
#       ImportError: Encountered an error while initializing the extension.
#
#   注意这个报错**不是**「找不到 metallib」—— 那种情况的文案是
#   `Failed to load the default metallib`（把 metallib 挪走可以复现）。两者要分开，
#   否则会朝错误的方向修。
#
# 原样拷贝则完全绕开重链接：rpath、相对路径、数据文件位置一个字节都不动。
# 代价是产物里多一个目录，且签名时要能覆盖 Resources 里的 Mach-O（`codesign --deep`
# 会走到；换成显式逐个签名见 build_macos.sh 的注释）。
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


REPO = Path.cwd()
assert (REPO / "laya_server" / "app.py").is_file(), "请在仓库根目录执行 pyinstaller"

datas = [
    (str(REPO / "laya_server" / "static"), "laya_server/static"),
]
datas += collect_data_files("laya_mlx")
datas += collect_data_files("tokenizers")

binaries = []

hiddenimports = (
    collect_submodules("laya_mlx")
    + collect_submodules("uvicorn")
    + ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
       "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]
    + ["tokenizers", "huggingface_hub", "anyio._backends._asyncio"]
    # 原生壳：pyobjc 的框架模块是运行时才 import 的（native.py 里刻意延迟到
    # available() 之后），静态分析看不到，必须显式列出来。
    #   AppKit/Foundation 来自 pyobjc-framework-Cocoa，WebKit 来自 -WebKit，
    #   PyObjCTools.AppHelper 是事件循环的入口，objc 是运行时。
    # collect_submodules("objc") 而不是只写 "objc"：objc 包里有子模块
    # （_machsignals 这类）在特定路径上才会被导入。
    + collect_submodules("objc")
    + ["AppKit", "Foundation", "WebKit", "PyObjCTools", "PyObjCTools.AppHelper"]
)

# 权重一律不打包：它们是运行时从 Hugging Face 拉的，体积几个 GB，
# 而且换 checkpoint 不该要求重新打包。
a = Analysis(
    [str(REPO / "packaging" / "entry.py")],
    pathex=[str(REPO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # mlx 由 build_macos.sh 单独拷贝（原因见文件头）。列进 excludes 而不是放着不管：
    # 不列的话 PyInstaller 会去找它、找不到就记一堆 missing module 警告，噪音会淹掉
    # 真正的问题。
    excludes=["mlx", "tkinter", "matplotlib", "PyQt5", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="laya-console",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # 双击不该弹终端
    # 出异常时弹一个窗口。默认（True）是静默退出 —— 一个不弹终端、也不弹错误框
    # 的 app，用户能提供的唯一信息就是「点了没反应」。
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="laya-console",
)

app = BUNDLE(
    coll,
    name="LayaServer.app",
    icon=None,  # 有图标就写路径进来，换成 .icns
    bundle_identifier="ai.laya.server.console",
    version="0.1.0",
    info_plist={
        "CFBundleName": "LayaServer",
        "CFBundleDisplayName": "Laya Server",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        # 关掉「未签名应用」的警告需要走签名+公证，见 build_macos.sh 的注释。
    },
)
