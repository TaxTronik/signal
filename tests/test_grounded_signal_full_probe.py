"""Meaningful denominator and label-boundary checks for the full extension."""
from copy import deepcopy

import pytest

from experiments.grounded_signal import full_probe as probe


def case(cid, *, targets=(), negative=False, context=False, suite='historical_main'):
    return {'case_id': cid, 'suite': suite, 'labels': {'targets': list(targets),
            'negative': negative, 'context_only': context, 'no_hint_norm_ids': [],
            'allowed_hint_norm_ids': [], 'insufficient_context_norm_ids': []}, 'baseline_norm_ids': []}


def row(cid, norms=(), status='complete'):
    return {'case_id': cid, 'status': status, 'answer': {
        'hints': [{'norm_id': n, 'status': 'hint'} for n in norms]}, 'timing': {'elapsed_seconds': 2.}}


def inputs(cases):
    return {'p': {'status': 'ready', 'candidate_norm_ids': ['norm:A:1', 'norm:A:2', 'norm:A:3']}}, {
        'case_packages': {c['case_id']: 'p' for c in cases}}


def test_failed_and_pending_negatives_are_not_correct_abstentions():
    cases = [case('failed', negative=True), case('pending', negative=True), case('quiet', negative=True)]
    packages, protocol = inputs(cases)
    report = probe.aggregate(cases, [row('failed', status='timeout'), row('quiet')], packages, protocol)
    counts = report['groups']['all']['observed']
    assert counts['negative_cases'] == 3
    assert counts['negative_complete_without_output'] == 1
    assert counts['negative_cases_unknown'] == 2
    assert (counts['failed_cases'], counts['pending_cases'], counts['complete_cases']) == (1, 1, 1)


def test_context_only_and_unannotated_extras_do_not_become_negative_labels():
    cases = [case('p', targets=['norm:A:1']), case('context', context=True, suite='grounded_evaluation')]
    packages, protocol = inputs(cases)
    report = probe.aggregate(cases, [row('p', ['norm:A:1', 'norm:A:2']), row('context')], packages, protocol)
    counts = report['groups']['all']['observed']
    assert (counts['target_norm_hits'], counts['unadjudicated_output_pairs']) == (1, 1)
    assert (counts['negative_cases'], counts['context_only_cases']) == (0, 1)
    assert report['output_precision'] is None


def test_overflow_and_output_cap_keep_all_target_denominators():
    cases = [case('p', targets=[f'norm:A:{i}' for i in range(1, 6)])]
    packages, protocol = inputs(cases)
    packages['p']['status'] = 'overflow'
    report = probe.aggregate(cases, [row('p', status='source_package_overflow')], packages, protocol)
    group = report['groups']['all']
    assert group['observed']['target_norm_pairs'] == 5
    assert group['observed']['target_norm_hits'] == 0
    assert group['retrieval']['target_pairs_in_top3'] == 3
    assert group['retrieval']['target_pairs_in_ready_package'] == 0
    assert group['retrieval']['target_pairs_upper_bound_from_three_output_slots'] == 3


def test_uncertainty_is_visible_separately_from_all_no_hint():
    cases = [case('n', negative=True)]
    packages, protocol = inputs(cases)
    result = row('n')
    result['answer']['hints'] = [{'norm_id': 'norm:A:1', 'status': 'insufficient_context'}]
    group = probe.aggregate(cases, [result], packages, protocol)['groups']['all']
    assert group['decision_statuses']['complete_without_hint_with_uncertain_norm'] == 1
    assert group['decision_statuses']['complete_all_norms_no_hint'] == 0


def test_explicit_negative_norms_separate_from_any_negative_output():
    cases = [case('n', negative=True)]
    cases[0]['labels']['no_hint_norm_ids'] = ['norm:A:1']
    packages, protocol = inputs(cases)
    group = probe.aggregate(cases, [row('n', ['norm:A:2'])], packages, protocol)['groups']['all']['observed']
    assert group['explicit_no_hint_output_pairs'] == 0
    assert group['negative_cases_with_output'] == 1
    assert group['negative_output_norm_pairs'] == 1


@pytest.mark.parametrize('results', [[row('x')], [row('p'), row('p')]])
def test_unplanned_or_duplicate_results_fail(results):
    cases = [case('p')]
    packages, protocol = inputs(cases)
    with pytest.raises(ValueError):
        probe.aggregate(cases, results, packages, protocol)


def source_data():
    historical = []
    for dataset, positive, negative, double in [('main', 102, 30, 74), ('additional', 15, 15, 12)]:
        for i in range(positive + negative):
            targets = (['norm:A:1', 'norm:A:2'] if i < double else ['norm:A:1']) if i < positive else []
            historical.append({'case_id': f'{dataset}-{i}', 'query': f'Historical {dataset} case {i}',
                               'dataset': dataset, 'group': dataset, 'targets': targets, 'negative': i >= positive,
                               'category': None, 'uncertain': False, 'treffer': {'suggestions': []}})
    diagnostics = []
    baseline = {'rows': []}
    for i in range(16):
        cid = f'grounded-{i}'
        diagnostics.append({'case_id': cid, 'text': f'Grounded case {i}', 'split': 'development' if i < 8 else 'evaluation',
                            'family_id': f'f{i // 2}', 'diagnostic_labels': {
                                'hint_expected': i < 12, 'target_norm_ids': (['norm:A:1', 'norm:A:2'] if i < 2 else ['norm:A:1']) if i < 12 else [],
                                'no_hint_norm_ids': [], 'allowed_hint_norm_ids': [],
                                'insufficient_context_norm_ids': ['norm:A:1'] if i == 15 else [],
                                'annotation_status': 'assistant_reviewed_synthetic'}})
        baseline['rows'].append({'case_id': cid, 'status': 'complete', 'output': {'suggestions': []}})
    return historical, diagnostics, baseline


def test_full_plan_preserves_label_provenance_and_context(monkeypatch):
    historical, diagnostics, baseline = source_data()
    monkeypatch.setattr(probe, 'validate_reference', lambda reference: reference)
    cases = probe.build_cases(historical, diagnostics, baseline)
    assert len(cases) == 178
    assert sum(c['labels']['negative'] for c in cases) == 48
    assert sum(c['labels']['context_only'] for c in cases) == 1
    assert cases[0]['labels']['provenance'] == 'historical_development_reference'
    assert cases[-1]['labels']['provenance'] == 'assistant_reviewed_synthetic'
    assert historical == source_data()[0]


def test_normalized_duplicate_texts_fail_before_model_use(monkeypatch):
    historical, diagnostics, baseline = source_data()
    monkeypatch.setattr(probe, 'validate_reference', lambda reference: reference)
    diagnostics = deepcopy(diagnostics)
    diagnostics[0]['text'] = historical[0]['query'].upper() + '!!'
    with pytest.raises(ValueError, match='duplicates'):
        probe.build_cases(historical, diagnostics, baseline)
