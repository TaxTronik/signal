"""Freeze and score the full existing corpus for the unchanged lexical 9B path.

Historical reference labels remain historical. No model, expert annotation,
ranking fit, or production pipeline is changed by this extension.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import statistics
import time
import unicodedata

from experiments.grounded_signal import learning, runner
from experiments.grounded_signal.lexical_probe import check_files, check_inventory, inventory, snapshot
from experiments.grounded_signal.sources import SourceCorpus, render_package
from tools.quality_research_agent import validate_reference

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'grounded_signal_full_inputs/1'
REFERENCE_SHA = 'ad7223bb2609df6a0f962d3d167ab566cc79336e1d5e95bb54b8a2d4e660ec6d'
PRIMARY_SHA = '14fbd69e9305cefd3105db4571d746a5e7f8df351fa0645fec45d56a49f1de50'
BASELINE_SHA = 'f4b47009d542d514e91e6d3524823b8d12db948f2ec3702779c4d4e28fcaa9bf'
MODEL = {'alias': 'signal-research-qwen35-9b-q6',
         'gguf_sha256': '91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59'}


def require(value, message):
    if not value:
        raise ValueError(message)


def norm_ids(suggestions):
    return sorted({norm['id'] for group in suggestions for norm in group['normen']})


def build_cases(reference, diagnostics, baseline):
    historical = validate_reference(reference)
    require(len(diagnostics) == 16 and len(baseline['rows']) == 16, 'Incomplete diagnostic or baseline inventory')
    old = {r['case_id']: r for r in baseline['rows']}
    require(len(old) == 16 and set(old) == {c['case_id'] for c in diagnostics}, 'Diagnostic baseline identities differ')
    cases = []
    for row in historical:
        cases.append({'case_id': row['case_id'], 'text': row['query'],
                      'suite': 'historical_' + row['dataset'], 'family_id': row['group'],
                      'labels': {'targets': list(row['targets']), 'negative': row['negative'],
                                 'context_only': False, 'no_hint_norm_ids': [], 'allowed_hint_norm_ids': [],
                                 'insufficient_context_norm_ids': [], 'provenance': 'historical_development_reference'},
                      'historical_metadata': {k: row[k] for k in ('dataset', 'group', 'category', 'uncertain')},
                      'baseline_norm_ids': norm_ids(row['treffer']['suggestions'])})
    for case in diagnostics:
        labels = case['diagnostic_labels']
        context = bool(labels['insufficient_context_norm_ids']) and not labels['hint_expected']
        row = old[case['case_id']]
        require(row['status'] == 'complete', 'Diagnostic baseline row is not complete')
        cases.append({'case_id': case['case_id'], 'text': case['text'],
                      'suite': 'grounded_' + case['split'], 'family_id': case['family_id'],
                      'labels': {'targets': list(labels['target_norm_ids']),
                                 'negative': not labels['hint_expected'] and not context, 'context_only': context,
                                 **{k: deepcopy(labels[k]) for k in ('no_hint_norm_ids', 'allowed_hint_norm_ids',
                                                                   'insufficient_context_norm_ids')},
                                 'provenance': labels['annotation_status']},
                      'historical_metadata': None,
                      'baseline_norm_ids': norm_ids(row['output']['suggestions'])})
    require(len(cases) == 178 and len({c['case_id'] for c in cases}) == 178, 'Full case plan must contain178 unique IDs')
    normalized = [''.join(ch for ch in unicodedata.normalize('NFKC', c['text']).casefold() if ch.isalnum()) for c in cases]
    require(len(set(normalized)) == 178, 'Text duplicates in full plan')
    require(sum(bool(c['labels']['targets']) for c in cases) == 129 and
            sum(len(c['labels']['targets']) for c in cases) == 217 and
            sum(c['labels']['negative'] for c in cases) == 48 and
            sum(c['labels']['context_only'] for c in cases) == 1, 'Full diagnostic denominators differ')
    return cases


def score_group(cases, outputs, statuses):
    counts: Counter[str] = Counter()
    for case in cases:
        cid, labels = case['case_id'], case['labels']
        status = statuses.get(cid, 'not_run')
        complete = status == 'complete'
        output = outputs.get(cid, set()) if complete else set()
        targets = set(labels['targets'])
        known = targets | set(labels['no_hint_norm_ids']) | set(labels['allowed_hint_norm_ids']) | set(labels['insufficient_context_norm_ids'])
        counts['planned_cases'] += 1
        counts['complete_cases'] += complete
        counts['pending_cases'] += status == 'not_run'
        counts['failed_cases'] += status not in ('complete', 'not_run')
        counts['positive_cases'] += bool(targets)
        counts['positive_case_hits'] += bool(targets & output)
        counts['positive_cases_unknown'] += bool(targets) and not complete
        counts['target_norm_pairs'] += len(targets)
        counts['target_norm_hits'] += len(targets & output)
        counts['target_norm_pairs_on_technically_unknown_cases'] += len(targets) if not complete else 0
        counts['output_norm_pairs'] += len(output)
        counts['unadjudicated_output_pairs'] += len(output - known)
        counts['explicit_no_hint_norm_pairs'] += len(labels['no_hint_norm_ids'])
        counts['explicit_no_hint_output_pairs'] += len(set(labels['no_hint_norm_ids']) & output)
        for kind in ('negative', 'context_only'):
            if labels[kind]:
                counts[kind + '_cases'] += 1
                counts[kind + '_cases_with_output'] += bool(output)
                counts[kind + '_output_norm_pairs'] += len(output)
                counts[kind + '_complete_without_output'] += complete and not output
                counts[kind + '_cases_unknown'] += not complete
    for kind in ('negative', 'context_only'):
        for suffix in ('cases', 'cases_with_output', 'output_norm_pairs', 'complete_without_output', 'cases_unknown'):
            counts.setdefault(kind + '_' + suffix, 0)
    return dict(counts)


def aggregate(cases, rows, packages, protocol):
    planned = {c['case_id'] for c in cases}
    require(len(planned) == len(cases) and len({r['case_id'] for r in rows}) == len(rows), 'Duplicate case/result ID')
    require({r['case_id'] for r in rows} <= planned, 'Unplanned result')
    statuses = {r['case_id']: r['status'] for r in rows}
    outputs = {}
    for row in rows:
        if row['status'] == 'complete':
            require(isinstance(row.get('answer'), dict), 'Complete result lacks an answer')
            outputs[row['case_id']] = {h['norm_id'] for h in row['answer']['hints'] if h['status'] == 'hint'}
    baseline = {c['case_id']: set(c['baseline_norm_ids']) for c in cases}
    baseline_status = {c['case_id']: 'complete' for c in cases}
    groups = {'all': cases, 'grounded': [c for c in cases if c['suite'].startswith('grounded_')],
              **{s: [c for c in cases if c['suite'] == s] for s in sorted({c['suite'] for c in cases})}}
    observed = {r['case_id']: r for r in rows}
    grouped = {}
    for name, selected in groups.items():
        stages: Counter[str] = Counter()
        for case in selected:
            package = packages[protocol['case_packages'][case['case_id']]]
            available = set(package['candidate_norm_ids']) & set(case['labels']['targets'])
            stages['target_pairs_in_top3'] += len(available)
            stages['target_pairs_in_ready_package'] += len(available) if package['status'] == 'ready' else 0
            stages['target_pairs_offered_by_ready_package_on_incomplete_cases'] += (
                len(available) if package['status'] == 'ready' and statuses.get(case['case_id']) != 'complete' else 0)
            stages['package_' + package['status'] + '_cases'] += 1
            stages['target_pairs_upper_bound_from_three_output_slots'] += min(3, len(case['labels']['targets']))
        decisions: Counter[str] = Counter()
        for case in selected:
            row = observed.get(case['case_id'], {})
            if row.get('status') == 'complete':
                hints = row['answer']['hints']
                for hint in hints:
                    decisions['norm_status_' + hint['status']] += 1
                positive = any(h['status'] == 'hint' for h in hints)
                uncertain = any(h['status'] == 'insufficient_context' for h in hints)
                decisions['complete_with_any_uncertain_norm'] += uncertain
                decisions['complete_without_hint_with_uncertain_norm'] += not positive and uncertain
                decisions['complete_all_norms_no_hint'] += bool(hints) and all(h['status'] == 'no_hint' for h in hints)
        grouped[name] = {'observed': score_group(selected, outputs, statuses),
                         'historical_semantic_baseline': score_group(selected, baseline, baseline_status),
                         'retrieval': dict(stages), 'decision_statuses': dict(decisions)}
    seconds = [r['timing']['elapsed_seconds'] for r in rows if 'elapsed_seconds' in r.get('timing', {})]
    return {'groups': grouped, 'status_counts': dict(Counter(statuses.values())),
            'timing': {'recorded_cases': len(seconds), 'total_case_seconds': sum(seconds),
                       'median_case_seconds': statistics.median(seconds) if seconds else None},
            'output_precision': None, 'semantic_truth_verified': False,
            'baseline_scope': 'Previously measured semantic suggestions on exactly these texts; no new baseline inference in full probe.',
            'label_scope': 'Historical targets/no-suggestion labels and synthetic diagnostic labels remain separate; context-only is not negative. Historical uncertain=false was a runner default, not expert approval.'}


def load_prepared(directory):
    manifest = runner.read(directory / 'prepared-manifest.json')
    require(manifest['format'] == FORMAT, 'Unknown full input manifest')
    check_inventory(directory, manifest['artifacts'], exclude=('prepared-manifest.json',))
    protocol = runner.read(directory / 'protocol.json')
    require(protocol['format'] == FORMAT and protocol['source_hashes'] == runner.source_hashes(), 'Full source protocol changed')
    check_files(protocol['original_inputs'])
    cases = runner.read(directory / 'cases.json')
    require([c['case_id'] for c in cases] == protocol['case_ids'] and len(cases) == 178, 'Full plan changed')
    return manifest, protocol, cases, runner.read(directory / 'packages.json'), runner.read(directory / 'source-units.json')


def prepare(args):
    out, primary = args.out.resolve(), args.primary.resolve()
    require(not out.exists(), 'Use a fresh full input directory')
    require(runner.sha_file(args.reference) == REFERENCE_SHA, 'Historical reference changed')
    require(runner.sha_file(args.baseline) == BASELINE_SHA, 'Diagnostic baseline changed')
    require(runner.sha_file(primary / 'input-manifest.json') == PRIMARY_SHA, 'Original lexical corpus receipt changed')
    original_manifest = runner.read(primary / 'input-manifest.json')
    check_inventory(primary, original_manifest['artifacts'], exclude=('input-manifest.json',))
    for name in ('experiments/grounded_signal/runtime.py', 'experiments/grounded_signal/learning.py',
                 'experiments/grounded_signal/sources.py', 'risk_layer/research_agent.py',
                 'tools/quality_research_agent.py'):
        require(runner.sha_file(ROOT / name) == original_manifest['artifacts']['source/' + name],
                'The measured runtime or lexical implementation changed')
    out.mkdir(parents=True)
    bindings: dict[str, str] = {}
    for source, name in ((args.reference, 'historical-reference.json'), (args.baseline, 'diagnostic-baseline.json'),
                         (primary / 'cases.json', 'diagnostic-cases.json'), (primary / 'graph.sqlite', 'graph.sqlite'),
                         (primary / 'learning-prototype.json', 'lexical-prototype.json'),
                         (primary / 'input-manifest.json', 'original-primary-manifest.json')):
        snapshot(source, out / name, bindings)
    for name, expected in (('historical-reference.json', REFERENCE_SHA), ('diagnostic-baseline.json', BASELINE_SHA),
                           ('original-primary-manifest.json', PRIMARY_SHA)):
        require(runner.sha_file(out / name) == expected, 'Pinned source changed before snapshot')
    for copied, original in (('diagnostic-cases.json', 'cases.json'), ('graph.sqlite', 'graph.sqlite'),
                             ('lexical-prototype.json', 'learning-prototype.json')):
        require(runner.sha_file(out / copied) == original_manifest['artifacts'][original], 'Copied primary artifact differs')
    source_hashes = runner.freeze_code(out)
    cases = build_cases(runner.read(out / 'historical-reference.json'), runner.read(out / 'diagnostic-cases.json'),
                        runner.read(out / 'diagnostic-baseline.json'))
    prototype = runner.read(out / 'lexical-prototype.json')
    learning._validate_prototype(prototype)
    packages, plans, rankings = {}, {}, []
    with SourceCorpus(out / 'graph.sqlite') as corpus:
        for norm, text in prototype['norm_documents'].items():
            require(''.join(u.text for u in corpus.units(norm)) == text, 'Lexical document differs from original norm')
        for index, case in enumerate(cases, 1):
            started = time.monotonic()
            ranking = learning.retrieve(case['text'], prototype, k=3, adapted=False)
            package = render_package(corpus.package([r['norm_id'] for r in ranking], max_chars=32000, reference_depth=0))
            packages[package['package_id']] = package
            plans[case['case_id']] = package['package_id']
            rankings.append({'case_id': case['case_id'], 'ranking': ranking,
                             'elapsed_seconds': time.monotonic() - started})
            if index % 20 == 0:
                print(f'Prepared {index}/{len(cases)} lexical packages', flush=True)
    units = {u['unit_id']: u for package in packages.values() for u in package['units']}
    protocol = {'format': FORMAT, 'case_ids': [c['case_id'] for c in cases], 'case_packages': plans,
                'source_hashes': source_hashes, 'original_inputs': bindings, 'expected_model': MODEL,
                'top_k': 3, 'max_package_chars': 32000, 'reference_depth': 0, 'adapted': False,
                'context_tokens': 16384, 'reasoning_budget_tokens': 768, 'max_output_tokens': 3072,
                'timeout_seconds': 300, 'retries': 0, 'fact_reuse': False,
                'interface': 'unchanged fixed_norm_slots/2',
                'baseline': 'historically measured semantic suggestions; no new baseline calls',
                'plan': 'all162 historical norm-labelled cases plus all16 grounded diagnostics, no outcome selection'}
    for name, value in (('cases.json', cases), ('protocol.json', protocol), ('packages.json', packages),
                        ('source-units.json', units), ('rankings.json', rankings),
                        ('preflight-evaluation.json', aggregate(cases, [], packages, protocol))):
        runner.save(out / name, value)
    check_files(bindings)
    check_inventory(primary, original_manifest['artifacts'], exclude=('input-manifest.json',))
    require(runner.source_hashes() == source_hashes, 'Source code changed during preparation')
    runner.save(out / 'prepared-manifest.json', {'format': FORMAT, 'artifacts': inventory(out)})
    print({'status': 'prepared', 'cases': len(cases), 'package_statuses': dict(Counter(packages[plans[c['case_id']]]['status'] for c in cases))}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=ROOT / 'out/grounded-signal-inputs-20260908-r2')
    parser.add_argument('--reference', type=Path, default=ROOT / 'out/quality-recall-live/report.json')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'out/grounded-signal-public-baseline-20260908-r3/report.json')
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
