"""Record actual runtime capabilities; CPU checks are not MRI training."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scipy', 'scikit-learn'):
        packages[name] = importlib.metadata.version(name)
    receipt = {'python': platform.python_version(), 'platform': platform.platform(),
               'packages': packages, 'cuda_available': torch.cuda.is_available(),
               'cuda_devices': torch.cuda.device_count(),
               'purpose': 'CPU source, numerical, and synthetic optimizer tests; no MRI dataset training'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
