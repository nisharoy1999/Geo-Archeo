# verify_setup.py
# Run this first to confirm everything works.
# python verify_setup.py

import sys

print("Checking all required packages...\n")

checks = [
    ("torch",            "2.2.2",  True),
    ("transformers",     "5.8.1",  True),
    ("rasterio",         "1.4.3",  True),
    ("sklearn",          "1.7.1",  True),
    ("mapie",            "1.4.0",  True),
    ("geopandas",        "1.1.3",  True),
    ("pandas",           "2.3.1",  True),
    ("wandb",            "0.27.0", True),
    ("numpy",            None,     True),
    ("tqdm",             None,     True),
    ("yaml",             None,     True),
    ("peft",             None,     True),
    ("scallopy",         None,     False),
]

all_ok      = True
scallop_ok  = False

for pkg, version, required in checks:

    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg:<20} {ver}")

        if pkg == "scallopy":
            scallop_ok = True

    except ImportError:
        if required:
            print(f"  ✗ {pkg:<20} MISSING — install required")
            all_ok = False
        else:
            print(f"  ⚠ {pkg:<20} not found "
                  f"(optional — fallback will be used)")

print()

if all_ok:
    print("✓ All required packages present.")
else:
    print("✗ Missing packages. Run:")
    print("  pip install peft")
    print()

if scallop_ok:
    print("✓ scallopy found — full Scallop reasoning enabled.")
else:
    print("⚠ scallopy not found — fallback symbolic.py will be used.")
    print("  To enable: pip install scallopy")

print()

# Check config loads correctly
try:
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    print("✓ configs/config.yaml loads correctly.")
    print(f"  Site types: {cfg['sites']['classes']}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  LoRA rank:  {cfg['lora']['rank']}")
except FileNotFoundError:
    print("✗ configs/config.yaml not found.")
    print("  Create it before running the pipeline.")

print()
print("Setup check complete.")
