from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
    filename="config.json"
)

print(path)

with open(path) as f:
    model_cfg = json.load(f)

print(model_cfg.get("num_labels"))
print(model_cfg)