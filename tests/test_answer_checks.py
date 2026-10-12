from navdiag.answer_checks import check_answer_metrics


def evidence():
    return [{'evidence_id': 'E1', 'tool': 'compare_nlos', 'payload': {
        'new_district_generalization': {
            'external_transfer': {'models': {
                'random_forest': {'test_macro': {'f1': .7884, 'precision': .8107}},
                'transformer': {'test_macro': {'f1': .7852, 'precision': .7567}}}},
            'district_holdout': {'models': {'random_forest': {'test_macro': {'f1': .7852}}}}}}}]


def test_rounded_and_percent_values_match_but_wrong_values_are_exposed():
    result = check_answer_metrics('外部迁移 RF：F1=0.788，Precision=81.07%。 [E1]\nRF F1=0.999。 [E1]', evidence())
    assert result['matched'] == 2 and result['needs_review'] == 1


def test_table_checker_keeps_model_and_protocol_separate():
    text = '外部迁移 [E1]\n| 模型 | F1 |\n|---|---|\n| RF | 0.7852 |\n| Transformer | 0.7852 |'
    result = check_answer_metrics(text, evidence())
    assert result['needs_review'] == 1 and result['matched'] == 1


def test_paper_results_and_hypothetical_targets_are_not_project_claims():
    result = check_answer_metrics('论文作者 F1=0.80 [E1]\n例如假设 RF F1=0.95。', evidence())
    assert not result['claims']


def test_missing_measurement_is_not_reported_as_verified():
    result = check_answer_metrics('RMSE=1.2 [E1]', evidence())
    assert result['not_checked'] == 1 and result['matched'] == 0


def test_other_evidence_cannot_validate_a_wrong_cited_tool():
    result = check_answer_metrics('RF F1=0.7884 [E2]', evidence())
    assert result['not_checked'] == 1 and not result['matched']


def test_frozen_source_validation_threshold_does_not_turn_test_scores_into_validation():
    text = '外部迁移（源=跨区域协议） [E1]\n| 模型 | F1 |\n|---|---|\n| RF（冻结源验证阈值） | 0.7884 |'
    result = check_answer_metrics(text, evidence())
    assert result['matched'] == 1 and result['needs_review'] == 0


def test_primary_drive_tail_improvement_claim_gets_warning_but_other_drive_does_not():
    rows = [{'tool': 'compare_positioning', 'evidence_id': 'E1', 'payload': {'baseline_comparison': {
        'metrics': {'p95_m': {'doppler_improved': False}, 'rmse_m': {'doppler_improved': True}}}}}]
    assert check_answer_metrics('在旺角Doppler的RMSE和P95略优。', rows)['comparison_warnings']
    assert not check_answer_metrics('在黄埔Doppler的P95略优。', rows)['comparison_warnings']
    assert not check_answer_metrics('在旺角Doppler的P95未改善。', rows)['comparison_warnings']
