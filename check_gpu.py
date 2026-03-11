"""
check_gpu.py — Quick GPU diagnostic for MTF Dashboard
Run: python check_gpu.py
"""

import sys

print("=" * 55)
print("MTF Dashboard — GPU Diagnostic")
print("=" * 55)

# ── 1. CUDA / GPU hardware ────────────────────────────────────
print("\n📦 1. CUDA Hardware")
try:
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            print(f"  ✅ GPU Found : {parts[0]}")
            print(f"     Driver   : {parts[1]}")
            print(f"     VRAM     : {parts[2]}")
            print(f"     Usage    : {parts[3]}")
    else:
        print("  ❌ nvidia-smi not found — no NVIDIA GPU or driver not installed")
except FileNotFoundError:
    print("  ❌ nvidia-smi not found — NVIDIA driver may not be installed")
except Exception as e:
    print(f"  ⚠️  nvidia-smi error: {e}")

# ── 2. XGBoost GPU test ───────────────────────────────────────
print("\n🤖 2. XGBoost GPU")
try:
    import xgboost as xgb
    import numpy as np
    print(f"  ✅ XGBoost version: {xgb.__version__}")

    # Try CUDA device
    X = np.random.rand(200, 10).astype("float32")
    y = (X[:, 0] > 0.5).astype(int)

    model = xgb.XGBClassifier(
        device="cuda",
        tree_method="hist",
        n_estimators=10,
        verbosity=0,
    )
    try:
        model.fit(X, y)
        prob = model.predict_proba(X[:1])[0][1]
        print(f"  ✅ XGBoost CUDA: WORKING (test prob={prob:.3f})")
        print(f"  🟢 GPU WILL BE USED for ML training")
    except Exception as cuda_err:
        print(f"  ❌ XGBoost CUDA failed: {cuda_err}")
        print(f"  🔵 Falling back to CPU")

        # Try CPU
        model_cpu = xgb.XGBClassifier(
            tree_method="hist",
            n_estimators=10,
            verbosity=0,
        )
        model_cpu.fit(X, y)
        print(f"  ✅ XGBoost CPU: working fine")

except ImportError:
    print("  ❌ XGBoost not installed — run: pip install xgboost")
except Exception as e:
    print(f"  ❌ XGBoost error: {e}")

# ── 3. CUDA toolkit (PyTorch / cupy check) ───────────────────
print("\n🔧 3. CUDA Toolkit")
try:
    import torch
    print(f"  ✅ PyTorch    : {torch.__version__}")
    print(f"  ✅ CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA ver   : {torch.version.cuda}")
        print(f"  ✅ GPU name   : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✅ VRAM       : {mem:.1f} GB")
except ImportError:
    print("  ℹ️  PyTorch not installed (not required — just for CUDA version info)")

# ── 4. Summary ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("📊 Summary & Recommendation")
print("=" * 55)

xgb_gpu = False
try:
    import xgboost as xgb
    import numpy as np
    X = np.random.rand(50, 5).astype("float32")
    y = (X[:, 0] > 0.5).astype(int)
    m = xgb.XGBClassifier(device="cuda", n_estimators=5, verbosity=0)
    m.fit(X, y)
    xgb_gpu = True
except Exception:
    pass

if xgb_gpu:
    print("\n  🟢 GPU is ACTIVE — XGBoost will use CUDA")
    print("  Expected ML time: ~2-3 sec per ticker")
    print("  Expected Nifty 50 full load: ~3-4 min")
else:
    print("\n  🔵 GPU NOT active — running on CPU")
    print("\n  To enable GPU, you need:")
    print("  1. NVIDIA GPU (GeForce/RTX/Quadro)")
    print("  2. NVIDIA driver: https://www.nvidia.com/drivers")
    print("  3. CUDA Toolkit 11.8+: https://developer.nvidia.com/cuda-downloads")
    print("  4. GPU-enabled XGBoost:")
    print("     pip install xgboost --upgrade")
    print("\n  On CPU, ML still works — just slower (~8-10 sec/ticker)")
    print("  Tip: Turn off 🤖 ML toggle for fast scanning,")
    print("       enable only when you've filtered to <10 stocks")

print("\n  Dashboard works fine either way — GPU just speeds up ML.")
print("=" * 55)
