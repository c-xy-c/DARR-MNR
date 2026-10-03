"""Export two isolated contribution controls; no production mode switches."""
import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = ('no_object_masking', 'static_parameters')


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f'expected one source anchor: {before!r}')
    return source.replace(before, after, 1)


def export(repository, destination, variant):
    destination.mkdir(parents=True, exist_ok=False)
    original, exported = {}, {}
    for package in ('structured_ssl', 'attention_ssl', 'sspredrnet'):
        for path in sorted((repository / package).glob('*.py')):
            relative = str(path.relative_to(repository))
            source = path.read_text()
            original[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if relative == 'structured_ssl/objectives.py' and variant == 'no_object_masking':
                source = replace_once(source, 'loss = retrieval + .1 * completion + .1 * masked',
                                      'loss = retrieval + .1 * completion + 0. * masked')
            if relative == 'structured_ssl/pav.py' and variant == 'static_parameters':
                source = replace_once(source,
                    '        factors = self.compiler(support, support_valid)',
                    '        support = torch.zeros_like(support)\n'
                    '        support_valid = torch.ones_like(support_valid)\n'
                    '        support_target = torch.zeros_like(support_target)\n'
                    '        support_objects = torch.zeros_like(support_objects)\n'
                    '        support_object_valid = torch.zeros_like(support_object_valid)\n'
                    '        factors = self.compiler(support, support_valid)')
                source = replace_once(source,
                    '        n = len(support)\n        padding = torch.cat',
                    '        support = torch.zeros_like(support)\n'
                    '        valid = torch.ones_like(valid)\n'
                    '        n = len(support)\n        padding = torch.cat')
            if relative == 'structured_ssl/check_contracts.py':
                if variant == 'no_object_masking':
                    source = replace_once(source, 'assert all(value > 0 for value in gradients.values())',
                                          "assert all(value > 0 for key, value in gradients.items() if key != 'masked_object')\n"
                                          "    assert gradients['masked_object'] == 0")
                else:
                    source = replace_once(source, 'assert swap_effect > 0', 'assert swap_effect == 0')
            if relative == 'structured_ssl/train.py':
                source = replace_once(source, "'method': 'structured-object-support-pav',",
                                      f"'method': 'structured-object-support-pav', 'experiment_variant': '{variant}',")
                if variant == 'no_object_masking':
                    source = replace_once(source, "'object_masking': .1", "'object_masking': 0.")
                source = replace_once(source,
                    "seal = {'schema_version': 1, 'method': config['method'],",
                    "seal = {'schema_version': 1, 'method': config['method'], 'experiment_variant': config['experiment_variant'],")
            if relative == 'structured_ssl/evaluate.py':
                # Preregistered controls report declines too. They must have
                # sealed, complete validation; they need not pass main acceptance.
                source = replace_once(source,
                    "validation['seal'] != seal or not validation_allows_test(validation['checkpoints']['best'])",
                    "validation['seal'] != seal or validation['checkpoints']['best']['full']['total'] != 14000")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            exported[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {'variant': variant, 'original_source_sha256': original, 'exported_source_sha256': exported,
                'total_trainable_capacity_preserved': True,
                'zero_weight_mask_head_retained_only_in_control': variant == 'no_object_masking',
                'unchanged_answer_anchor_support_is_real': True,
                'control_test_requires_sealed_full_validation_not_main_nonregression_gate': True,
                'full_raven_result_claimed': False}
    (destination / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    export(repository, args.output_root, args.variant)
    print(json.dumps({'variant': args.variant, 'source': str(args.output_root)}))


if __name__ == '__main__':
    main()
