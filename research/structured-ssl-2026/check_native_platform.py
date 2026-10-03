"""Record a full Metal validation reference for the immutable native checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sspredrnet.model import SSPredRNet
from sspredrnet.evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.baseline, map_location='cpu', weights_only=False)
    model = SSPredRNet().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to('mps')
    with torch.inference_mode():
        measured = score(model, args.dataset_root, 'val', torch.device('mps'), batch_size=128, workers=2)
    report = {'split':'val','test_opened':False,'torch':str(torch.__version__),
              'anchor_sha256':hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
              'expected_prior_validation_correct':9979,'mps':measured,
              'new_training_updates':0,'full_raven_adaptation_result':False,
              'cpu_full_validation_replayed':False}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
