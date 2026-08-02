# check_imports.py
# Run this first before anything else
# python check_imports.py
import sys

print(sys.executable)
print(sys.version)
print("Interpreter:", sys.executable)
print("Checking imports...")

try:
    import torch
    print(f"  torch        {torch.__version__}")
except ImportError as e:
    print(f"torch MISSING {e}")

try:
    import transformers
    print(f"  transformers {transformers.__version__}")
except ImportError as e:
    print(f"  transformers MISSING {e}")

try:
    import peft
    print(f"  peft         {peft.__version__}")
except ImportError as e:
    print(f"  peft         MISSING — run: pip install peft")

try:
    import yaml
    print(f"  yaml         OK")
except ImportError as e:
    print(f"  yaml         MISSING — run: pip install pyyaml")

try:
    import rasterio
    print(f"  rasterio     {rasterio.__version__}")
except ImportError as e:
    print(f"  rasterio     MISSING {e}")

try:
    import numpy
    print(f"  numpy        {numpy.__version__}")
except ImportError as e:
    print(f"  numpy        MISSING {e}")

try:
    import mapie
    print(f"  mapie        {mapie.__version__}")
except ImportError as e:
    print(f"  mapie        MISSING {e}")

try:
    import sklearn
    print(f"  sklearn      {sklearn.__version__}")
except ImportError as e:
    print(f"  sklearn      MISSING {e}")

try:
    import wandb
    print(f"  wandb        {wandb.__version__}")
except ImportError as e:
    print(f"  wandb        MISSING {e}")

# Check config loads
try:
    import yaml
    with open("GeoArcheo/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"\n  config.yaml  OK")
    print(f"  image size:  {cfg['image']['size']}")
    print(f"  bands:       {cfg['image']['bands']}")
    print(f"  lora rank:   {cfg['lora']['rank']}")
    print(f"  site types:  {cfg['sites']['classes']}")
except FileNotFoundError:
    print(f"\n  config.yaml  NOT FOUND")
    print(f"  Create configs/config.yaml first")

print("\nDone.")
