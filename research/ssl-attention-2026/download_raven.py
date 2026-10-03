"""Acquire official RAVEN-10000; extract images without inspecting test labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import gdown


URL = 'https://drive.google.com/file/d/1fUSmWZpCsoP6sLsmqrxbnD_RO2o1zj1S/view?usp=sharing'
CONFIGURATIONS = ('center_single', 'distribute_four', 'distribute_nine',
                  'in_center_single_out_center_single', 'in_distribute_four_out_center_single',
                  'left_center_single_right_center_single', 'up_center_single_down_center_single')


def archive_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists() or not zipfile.is_zipfile(args.archive):
        print(json.dumps({'phase': 'download', 'official_url': URL}), flush=True)
        downloaded = gdown.download(id='1fUSmWZpCsoP6sLsmqrxbnD_RO2o1zj1S', output=str(args.archive),
                                    resume=True, use_cookies=False, timeout=(20, 60), retries=2)
        if downloaded is None or not zipfile.is_zipfile(args.archive):
            raise RuntimeError('official download did not produce a complete ZIP archive')
    expected = {(c, split): size for c in CONFIGURATIONS
                for split, size in (('train', 6000), ('val', 2000), ('test', 2000))}
    members, counts, destinations = [], Counter(), set()
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.suffix != '.npz':
                continue
            if path.is_absolute() or '..' in path.parts or len(path.parts) < 2:
                raise RuntimeError('unsafe image archive path')
            configuration, name = path.parent.name, path.name
            if configuration not in CONFIGURATIONS or not name.startswith('RAVEN_'):
                raise RuntimeError(f'unexpected image entry: {entry.filename}')
            split = name.removesuffix('.npz').rsplit('_', 1)[-1]
            if (configuration, split) not in expected or (configuration, name) in destinations:
                raise RuntimeError('unexpected split or duplicated image filename')
            destinations.add((configuration, name))
            counts[configuration, split] += 1
            members.append((entry, configuration, name))
        if dict(counts) != expected:
            raise RuntimeError(f'official archive split counts differ: {counts}')
        print(json.dumps({'phase': 'extract', 'images': len(members),
                          'archive_bytes': args.archive.stat().st_size}), flush=True)
        for number, (entry, configuration, name) in enumerate(members, 1):
            directory = args.dataset_root / configuration
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / name
            if not target.exists() or target.stat().st_size != entry.file_size:
                temporary = target.with_suffix('.partial')
                with archive.open(entry) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                temporary.replace(target)
            if number % 5000 == 0:
                print(json.dumps({'phase': 'extract', 'images_completed': number}), flush=True)
    actual = {(c, split): len(list((args.dataset_root / c).glob(f'RAVEN_*_{split}.npz')))
              for c in CONFIGURATIONS for split in ('train', 'val', 'test')}
    if actual != expected:
        raise RuntimeError('extracted split counts differ')
    report = {'official_project': 'https://wellyzhang.github.io/project/raven.html',
              'official_download_url': URL, 'archive_sha256': archive_digest(args.archive),
              'archive_bytes': args.archive.stat().st_size,
              'dataset_root': str(args.dataset_root.resolve()), 'images': len(members),
              'counts': {c: {s: actual[c, s] for s in ('train', 'val', 'test')} for c in CONFIGURATIONS},
              'npz_contents_opened_for_selection': False, 'xml_used': False}
    args.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'phase': 'complete', **report}), flush=True)


if __name__ == '__main__':
    main()
