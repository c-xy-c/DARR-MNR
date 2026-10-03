"""Export isolated, sealed-source studies without adding runtime mode switches.

Each study changes exactly one mechanism relative to attention_ssl. The exported
trainer/evaluator use the same full-data selection and test boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = ('completion_energy_training', 'random_negatives', 'tokenwise_mlp', 'discrete_support')
PACKAGES = ('attention_ssl', 'sspredrnet')


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'expected one source anchor: {before!r}')
    return source.replace(before, after, 1)


def export(repository, destination, variant):
    destination.mkdir(parents=True, exist_ok=False)
    original_hashes, exported_hashes = {}, {}
    for package in PACKAGES:
        for path in sorted((repository / package).glob('*.py')):
            relative = path.relative_to(repository)
            source = path.read_text()
            original_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if str(relative) == 'attention_ssl/model.py':
                if variant == 'completion_energy_training':
                    source = replace_once(source,
                        'logits = (-scores / self.temperature).masked_fill(~valid, -torch.inf)',
                        'logits = (-energy / self.temperature).masked_fill(~valid, -torch.inf)')
                elif variant == 'random_negatives':
                    source = replace_once(source,
                        'distance = operator_energy(F.relu(target)[:, None], target)[:, 0]',
                        'distance = torch.rand(n, n, device=device)')
                elif variant == 'tokenwise_mlp':
                    source = replace_once(source,
                        'self.attention = nn.MultiheadAttention(64, 4, batch_first=True)',
                        'self.attention = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))')
                    source = replace_once(source,
                        'attended, _ = self.attention(normed, normed, normed, need_weights=False)',
                        'attended = self.attention(normed)')
                elif variant == 'discrete_support':
                    source = replace_once(source,
                        'from sspredrnet.energy import operator_energy',
                        'from sspredrnet.energy import operator_energy\nfrom .discrete_support import DiscreteSupport')
                    source = replace_once(source,
                        'self.completion = SupportCompletion()',
                        'self.completion = DiscreteSupport()')
                    start = source.index('    def completion_energy(self, support, prefix, targets):')
                    end = source.index('    def row_scores(', start)
                    source = (source[:start] + '    def completion_energy(self, support, prefix, targets):\n'
                              '        return self.completion.energies(support, prefix, targets)\n\n' + source[end:])
                source = f'"""Research ablation: {variant}; see ablation_manifest.json."""\n' + source
            if str(relative) == 'attention_ssl/check_contracts.py' and variant == 'tokenwise_mlp':
                source = replace_once(source,
                    'assert model.completion.keys.attention.in_proj_weight.grad.norm() > 0',
                    'assert sum(float(p.grad.norm()) for p in model.completion.keys.attention.parameters()) > 0')
                source = source.replace('attention_and_gate_have_nonzero_gradients_after_updates',
                                        'tokenwise_mlp_and_gate_have_nonzero_gradients_after_updates')
            if str(relative) == 'attention_ssl/train.py':
                source = replace_once(source,
                    "'schema_version': 1, 'method': 'attention-support-completion',",
                    f"'schema_version': 1, 'method': 'attention-support-completion', 'experiment_variant': '{variant}',")
                source = replace_once(source,
                    "'negative_mining': 'up_to_seven_same_layout_component_known_targets',",
                    "'negative_mining': 'up_to_seven_same_layout_component_" +
                    ('random' if variant == 'random_negatives' else 'nearest') + "_known_targets',")
                source = replace_once(source,
                    "seal = {'schema_version': 1, 'method': config['method'],",
                    "seal = {'schema_version': 1, 'method': config['method'], 'experiment_variant': config['experiment_variant'],")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            exported_hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    if variant == 'discrete_support':
        relative = 'attention_ssl/discrete_support.py'
        source = Path(__file__).with_name('discrete_support.py').read_text()
        target = destination / relative
        target.write_text(source)
        exported_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        trainer = destination / 'attention_ssl/provenance.py'
        trainer.write_text(replace_once(trainer.read_text(),
            "SOURCE_FILES = (\n",
            "SOURCE_FILES = (\n    'attention_ssl/discrete_support.py',\n"))
        exported_hashes['attention_ssl/provenance.py'] = hashlib.sha256(trainer.read_bytes()).hexdigest()
        relative = 'attention_ssl/check_contracts.py'
        target = destination / relative
        target.write_text(Path(__file__).with_name('check_discrete.py').read_text())
        exported_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {'variant': variant, 'primary_source': str(repository),
                'baseline_frozen': True, 'candidate_free_adaptation': True,
                'original_source_sha256': original_hashes,
                'exported_source_sha256': exported_hashes,
                'full_raven_result_claimed': False}
    (destination / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    manifest = export(repository, args.output_root, args.variant)
    print(json.dumps({'variant': manifest['variant'], 'source': str(args.output_root)}))


if __name__ == '__main__':
    main()
