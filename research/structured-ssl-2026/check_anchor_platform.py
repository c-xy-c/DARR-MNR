"""Paired full validation of the immutable anchor on CPU and Metal."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from attention_ssl.model import AttentionCompletion
from sspredrnet.data import Raven, CONFIGS, loader, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.anchor, map_location='cpu', weights_only=False)['model']
    cpu, metal = AttentionCompletion().eval(), AttentionCompletion().to('mps').eval()
    cpu.load_state_dict(state, strict=True)
    metal.load_state_dict(state, strict=True)
    totals, correct_cpu, correct_metal = [0]*7, [0]*7, [0]*7
    changes, score_error, input_error, samples = [], 0., 0., 0
    with torch.inference_mode():
        for images, answers, configs in loader(Raven(args.dataset_root, 'val'), 128, 2,
                                              torch.Generator().manual_seed(12346)):
            pixels_cpu, pixels_metal = normalize(images), normalize(images.to('mps'))
            cpu_scores, metal_scores = cpu(pixels_cpu), metal(pixels_metal).cpu()
            input_error = max(input_error, float((pixels_cpu - pixels_metal.cpu()).abs().max()))
            score_error = max(score_error, float((cpu_scores - metal_scores).abs().max()))
            predicted_cpu, predicted_metal = cpu_scores.argmin(1), metal_scores.argmin(1)
            for index in range(7):
                mask = configs == index
                totals[index] += int(mask.sum())
                correct_cpu[index] += int(((predicted_cpu == answers) & mask).sum())
                correct_metal[index] += int(((predicted_metal == answers) & mask).sum())
            for index in (predicted_cpu != predicted_metal).nonzero().flatten().tolist():
                changes.append({'validation_index': samples+index, 'configuration': CONFIGS[int(configs[index])],
                    'answer': int(answers[index]), 'cpu_prediction': int(predicted_cpu[index]),
                    'metal_prediction': int(predicted_metal[index]),
                    'cpu_sorted_scores': cpu_scores[index].sort().values.tolist(),
                    'metal_sorted_scores': metal_scores[index].sort().values.tolist(),
                    'max_score_difference': float((cpu_scores[index]-metal_scores[index]).abs().max())})
            samples += len(images)
            if samples % 1280 == 0:
                print(json.dumps({'validation_seen':samples,'cpu_correct':sum(correct_cpu),
                                  'metal_correct':sum(correct_metal),'changed':len(changes)}), flush=True)
    assert totals == [2000]*7
    metadata = json.loads(Path(args.anchor).with_name('val_evaluation.json').read_text())
    expected = metadata['checkpoints']['best']['full']['correct']
    def result(counts):
        return {'correct':sum(counts),'total':samples,'accuracy_macro':100*sum(counts)/samples,
                'by_configuration':{c:100*n/2000 for c,n in zip(CONFIGS,counts)}}
    report = {'split':'val','test_opened':False,'torch':str(torch.__version__),
        'anchor_sha256':hashlib.sha256(Path(args.anchor).read_bytes()).hexdigest(),
        'expected_prior_validation_correct':expected,'cpu_replays_prior_correct_count':sum(correct_cpu)==expected,
        'cpu':result(correct_cpu),'mps':result(correct_metal),'changed_predictions':changes,
        'max_score_difference':score_error,'max_normalized_input_difference':input_error,
        'different_hardware_rounding_is_not_an_accuracy_gain':True}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='changed_predictions'}), flush=True)
    if sum(correct_cpu) != expected:
        raise RuntimeError('CPU full validation also failed the historical anchor count')


if __name__ == '__main__':
    main()
