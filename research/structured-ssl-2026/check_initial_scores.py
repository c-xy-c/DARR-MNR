"""Full validation compares the zero-gate structured score to its own anchor."""
import argparse
import json
from pathlib import Path

import torch

from sspredrnet.data import loader
from structured_ssl.data import ObjectRaven, to_device
from structured_ssl.model import StructuredCompletion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    model = StructuredCompletion().to('mps').eval()
    model.load_anchor(torch.load(args.anchor, map_location='mps', weights_only=False)['model'])
    correct, anchor_correct, samples, nonfinite, changed, maximum = 0, 0, 0, [], [], 0.
    with torch.inference_mode():
        for pack, answers, configs in loader(ObjectRaven(args.dataset_root, 'val'), 128, 2,
                                             torch.Generator().manual_seed(12346)):
            pack = to_device(pack, torch.device('mps'))
            scores, reference = model(pack).cpu(), model.anchor(pack['views']).cpu()
            for index in (~scores.isfinite().all(1)).nonzero().flatten().tolist():
                nonfinite.append({'validation_index':samples+index,'configuration':int(configs[index])})
            for index in (scores.argmin(1)!=reference.argmin(1)).nonzero().flatten().tolist():
                changed.append({'validation_index':samples+index,'configuration':int(configs[index])})
            correct += int((scores.argmin(1)==answers).sum())
            anchor_correct += int((reference.argmin(1)==answers).sum())
            finite = scores.isfinite() & reference.isfinite()
            maximum=max(maximum,float((scores-reference)[finite].abs().max()))
            samples+=len(answers)
            if samples%1280==0:
                print(json.dumps({'validation_seen':samples,'structured_correct':correct,
                                  'anchor_correct':anchor_correct,'nonfinite':len(nonfinite),
                                  'changed':len(changed)}),flush=True)
    assert samples==14000
    report={'split':'val','total':samples,'gate':float(model.gate),'seed':12345,
            'structured_correct':correct,'anchor_correct':anchor_correct,
            'nonfinite_records':nonfinite,'changed_records':changed,'max_finite_score_difference':maximum,
            'test_opened':False,'device':'mps','full_raven_adaptation_result':False}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
