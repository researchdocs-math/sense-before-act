import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--destination',default='models/qwen2.5-1.5b-4bit');a=p.parse_args()
os.environ.setdefault('HF_HOME',str(ROOT/'models/cache'))
from huggingface_hub import snapshot_download
cfg=json.loads((ROOT/'configs/model.json').read_text())
snapshot_download(cfg['repository'],revision=cfg['revision'],local_dir=a.destination,allow_patterns=['*.json','*.safetensors','*.txt','*.model','README.md','LICENSE'],max_workers=4)
