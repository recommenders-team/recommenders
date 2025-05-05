# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

# NOTE:
# The times on GPU environment have been calculated on an Azure STANDARD_NC6S_V2
# with 6 vCPUs, 112 GB memory, 1 NVIDIA Tesla P100 GPU.
# The times on CPU and Spark environments have been calculated on an Azure
# Standard_A8m_v2 with 8 vCPUs and 64 GiB memory.

# IMPORTANT NOTE:
# FOR NIGHTLY, NO GROUP SHOULD SURPASS 45MIN = 2700s !!!
# FOR PR GATE, NO GROUP SHOULD SURPASS 15MIN = 900s !!!

global nightly_test_groups, pr_gate_test_groups

nightly_test_groups = {
    "group_cpu_001": [  # Total group time: 1883s
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_download_and_extract_movielens",  # 0.45s + 0.61s + 3.47s + 8.28s
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_item_df",  # 0.47s + 0.59s + 3.59s + 8.44s
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_pandas_df",  # 16.87s + 37.33s + 352.99s + 673.61s
        #
        "tests/data_validation/recommenders/datasets/test_mind.py::test_mind_url",  # 0.38s
        "tests/data_validation/recommenders/datasets/test_mind.py::test_download_mind_demo",
        "tests/data_validation/recommenders/datasets/test_mind.py::test_extract_mind_demo",
        "tests/data_validation/recommenders/datasets/test_mind.py::test_download_mind_small",
        "tests/data_validation/recommenders/datasets/test_mind.py::test_extract_mind_small",
        "tests/data_validation/recommenders/datasets/test_mind.py::test_download_mind_large",
        "tests/data_validation/recommenders/datasets/test_mind.py::test_extract_mind_large",
        "tests/data_validation/examples/test_mind.py::test_mind_utils_runs",  # 219.77s
        "tests/data_validation/examples/test_mind.py::test_mind_utils_values",  # 219.26s
        #
        "tests/data_validation/examples/test_wikidata.py::test_wikidata_runs",
        "tests/data_validation/examples/test_wikidata.py::test_wikidata_values",
        #
        "tests/smoke/examples/test_notebooks_python.py::test_lightgbm_quickstart_smoke",  # 46.42s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_cornac_bpr_smoke",  # 16.62s
        "tests/functional/examples/test_notebooks_python.py::test_cornac_bpr_functional",  # 165.72s
    ],
    "group_cpu_002": [  # Total group time: 1801s
        "tests/smoke/examples/test_notebooks_python.py::test_baseline_deep_dive_smoke",  # 15.98s
        "tests/functional/examples/test_notebooks_python.py::test_baseline_deep_dive_functional",  # 170.73s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_surprise_svd_smoke",  # 45.88s
        "tests/functional/examples/test_notebooks_python.py::test_surprise_svd_functional",  # 503.54s
        #
        "tests/functional/examples/test_notebooks_python.py::test_geoimc_functional",  # 1006.19s
        #
        "tests/functional/examples/test_notebooks_python.py::test_benchmark_movielens_cpu",  # 58s
    ],
    "group_cpu_003": [  # Total group time: 2253s
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_download_criteo_sample",  # 1.05s
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_extract_criteo_sample",  # 1.22s
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_criteo_load_pandas_df_sample",  # 1.73s
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_criteo_load_pandas_df_full",  # 1368.63s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_sar_single_node_smoke",  # 12.58s
        "tests/functional/examples/test_notebooks_python.py::test_sar_single_node_functional",  # 57.67s + 808.83s
        "tests/functional/examples/test_notebooks_python.py::test_xlearn_fm_functional",  # 255.73s
        "tests/smoke/examples/test_notebooks_python.py::test_vw_deep_dive_smoke",
        "tests/functional/examples/test_notebooks_python.py::test_vw_deep_dive_functional",
        "tests/functional/examples/test_notebooks_python.py::test_nni_tuning_svd",
    ],
    "group_gpu_001": [  # Total group time: 1937.01s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/smoke/recommenders/models/test_deeprec_utils.py",  # 2.91
        "tests/smoke/recommenders/models/test_deeprec_model.py::test_FFM_iterator",  # 0.74s
        "tests/smoke/recommenders/models/test_newsrec_utils.py::test_news_iterator",  # 3.04s
        #
        "tests/smoke/recommenders/models/test_deeprec_model.py::test_model_lightgcn",  # 6.03s
        "tests/functional/examples/test_notebooks_gpu.py::test_lightgcn_deep_dive_functional",  # 19.45s
        #
        # "tests/smoke/recommenders/models/test_deeprec_model.py::test_model_sum",  # 27.23s  # FIXME: Disabled due to the issue with TF version > 2.10.1 See #2018
        #
        "tests/smoke/recommenders/models/test_deeprec_model.py::test_model_dkn",  # 187.20s
        "tests/functional/examples/test_notebooks_gpu.py::test_dkn_quickstart_functional",  # 1167.93s
        #
        "tests/functional/examples/test_notebooks_gpu.py::test_slirec_quickstart_functional",  # 175.00s
        "tests/smoke/recommenders/models/test_deeprec_model.py::test_model_slirec",  # 346.72s
    ],
    "group_gpu_002": [  # Total group time: 1896.76s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/smoke/recommenders/models/test_deeprec_model.py::test_model_xdeepfm",  # 3.10s
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/smoke/examples/test_notebooks_gpu.py::test_xdeepfm_smoke",  # 77.93s
        "tests/functional/examples/test_notebooks_gpu.py::test_xdeepfm_functional",
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_cornac_bivae_smoke",  # 67.84s
        "tests/functional/examples/test_notebooks_gpu.py::test_cornac_bivae_functional",  # 453.21s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_wide_deep_smoke",  # 122.71s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_fastai_smoke",  # 33.22s
        "tests/functional/examples/test_notebooks_gpu.py::test_fastai_functional",  # 667.88s
    ],
    "group_gpu_003": [  # Total group time: 2072.15s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/smoke/examples/test_notebooks_gpu.py::test_ncf_smoke",  # 114.39s
        "tests/functional/examples/test_notebooks_gpu.py::test_ncf_functional",  # 1046.97s
        "tests/smoke/examples/test_notebooks_gpu.py::test_ncf_deep_dive_smoke",  # 102.71s
        "tests/functional/examples/test_notebooks_gpu.py::test_ncf_deep_dive_functional",  # 351.17s
        #
        "tests/smoke/recommenders/models/test_newsrec_utils.py::test_naml_iterator",  # 5.50s
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/smoke/recommenders/models/test_newsrec_model.py::test_model_naml",  # 450.65s
    ],
    "group_gpu_004": [  # Total group time: 2103.34s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/smoke/examples/test_notebooks_gpu.py::test_nrms_smoke",  # 232.55s
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/functional/examples/test_notebooks_gpu.py::test_nrms_quickstart_functional",  # 857.05s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_lstur_smoke",  # 246.46s
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/functional/examples/test_notebooks_gpu.py::test_lstur_quickstart_functional",  # 766.52s
    ],
    "group_gpu_005": [  # Total group time: 1844.05s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/functional/examples/test_notebooks_gpu.py::test_wide_deep_functional",  # 1843.29s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_npa_smoke",  # 366.22s
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/functional/examples/test_notebooks_gpu.py::test_npa_quickstart_functional",  # 810.92s
    ],
    "group_gpu_006": [  # Total group time: 1763.99s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/smoke/recommenders/models/test_newsrec_model.py::test_model_npa",  # 202.61s
        "tests/smoke/recommenders/models/test_newsrec_model.py::test_model_nrms",  # 188.60s
    ],
    "group_gpu_007": [  # Total group time: 846.89s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        # FIXME: https://github.com/microsoft/recommenders/issues/1883
        # "tests/smoke/examples/test_notebooks_gpu.py::test_naml_smoke",  # 620.13s
        #
        "tests/functional/examples/test_notebooks_gpu.py::test_benchmark_movielens_gpu",  # 226s
        # FIXME: Reduce test time https://github.com/microsoft/recommenders/issues/1731
        # "tests/functional/examples/test_notebooks_gpu.py::test_naml_quickstart_functional",  # 2033.85s
        # FIXME: https://github.com/microsoft/recommenders/issues/1716
        # "tests/functional/examples/test_notebooks_gpu.py::test_sasrec_quickstart_functional",  # 448.06s + 614.69s
        "tests/smoke/recommenders/models/test_newsrec_model.py::test_model_lstur",  # 194.88s
    ],
    "group_spark_001": [  # Total group time: 987.16s
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_spark_df",  # 4.33s+ 25.58s + 101.99s + 139.23s
        #
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_criteo_load_spark_df_sample",  # 6.83s
        "tests/data_validation/recommenders/datasets/test_criteo.py::test_criteo_load_spark_df_full",  # 374.64s
        #
        "tests/smoke/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_smoke",  # 32.45s
        "tests/functional/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_functional",
        #
        "tests/smoke/examples/test_notebooks_pyspark.py::test_als_pyspark_smoke",  # 49.53s
        "tests/functional/examples/test_notebooks_pyspark.py::test_als_pyspark_functional",  # 110.58s
        #
        "tests/functional/examples/test_notebooks_pyspark.py::test_benchmark_movielens_pyspark",  # 142s
    ],
}

pr_gate_test_groups = {
    "group_cpu_001": [  # Total group time: 525.96s
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__has_default_col_names",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_df_remove_default_col__return_success",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_df_invalid_param__return_failure",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_df__return_success",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_data__no_name_collision",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_pandas_df_mock_100__with_default_param__succeed",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_pandas_df_mock_100__with_custom_param__succeed",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_find_wikidata_id_correct",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_find_wikidata_id_incorrect",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_query_entity_links",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_read_linked_entities",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_query_entity_description",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_search_wikidata_correct",
        "tests/data_validation/recommenders/datasets/test_wikidata.py::test_search_wikidata_incorrect",
        "tests/unit/recommenders/datasets/test_download_utils.py::test_maybe_download",
        "tests/unit/recommenders/datasets/test_download_utils.py::test_maybe_download_wrong_bytes",
        "tests/unit/recommenders/datasets/test_download_utils.py::test_maybe_download_maybe",
        "tests/unit/recommenders/datasets/test_download_utils.py::test_maybe_download_retry",
        "tests/unit/recommenders/datasets/test_download_utils.py::test_download_path",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_negative_feedback_sampler",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_filter_by",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_csv_to_libffm",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_has_columns",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_has_same_base_dtype",
        "tests/unit/recommenders/datasets/test_pandas_df_utils.py::test_lru_cache_df",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_split_pandas_data",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_min_rating_filter",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_random_splitter",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_chrono_splitter",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_stratified_splitter",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_int_numpy_stratified_splitter",
        "tests/unit/recommenders/datasets/test_python_splitter.py::test_float_numpy_stratified_splitter",
        "tests/unit/recommenders/datasets/test_sparse.py::test_df_to_sparse",
        "tests/unit/recommenders/datasets/test_sparse.py::test_sparse_to_df",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_column_dtypes_match",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_merge_rating",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_merge_ranking",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_rmse",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_mae",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_rsquared",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_exp_var",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_get_top_k_items",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_get_top_k_items_largek",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_ndcg_at_k",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_map",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_map_at_k",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_precision_at_k",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_recall_at_k",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_auc",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_logloss",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_python_errors",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_catalog_coverage",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_distributional_coverage",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_item_novelty",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_novelty",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_diversity",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_diversity",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_item_serendipity",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_serendipity",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_serendipity",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_diversity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_diversity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_item_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_user_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_python_evaluation.py::test_serendipity_item_feature_vector",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_init",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_clean_dataframe",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_fit",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_tokenize_text",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_get_tokens",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_get_stop_words",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_recommend_top_k_items",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_get_top_k_recommendations",
        "tests/unit/recommenders/models/test_cornac_utils.py::test_predict",
        "tests/unit/recommenders/models/test_cornac_utils.py::test_recommend_k_items",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_init",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_fit",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_predict",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_predict_all_items",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_sar_item_similarity",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_user_affinity",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_recommend_k_items",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_get_item_based_topk",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_get_popularity_based_topk",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_get_normalized_scores",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_match_similarity_type_from_json_file",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_dataset_with_duplicates",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_get_topk_most_similar_users",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_item_frequencies",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_user_frequencies",
        "tests/unit/recommenders/models/test_surprise_utils.py::test_predict",
        "tests/unit/recommenders/models/test_surprise_utils.py::test_recommend_k_items",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_vw_init_del",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_to_vw_cmd",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_parse_train_cmd",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_parse_test_cmd",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_to_vw_file",
        "tests/unit/recommenders/models/test_vowpal_wabbit.py::test_fit_and_predict",
        "tests/unit/recommenders/tuning/test_ncf_utils.py::test_compute_test_results__return_success",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_get_experiment_status",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_experiment_status_done",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_experiment_status_tuner_no_more_trial",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_experiment_status_running",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_experiment_status_no_more_trial",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_experiment_status_failed",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_stopped_timeout",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_stopped",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_metrics_written",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_check_metrics_written_timeout",
        "tests/unit/recommenders/tuning/test_nni_utils.py::test_get_trials",
        "tests/unit/recommenders/tuning/test_sweep.py::test_param_sweep",
        "tests/unit/recommenders/utils/test_general_utils.py::test_invert_dictionary",
        "tests/unit/recommenders/utils/test_general_utils.py::test_get_number_processors",
        "tests/unit/recommenders/utils/test_plot.py::test_line_graph",
        "tests/unit/recommenders/utils/test_python_utils.py::test_python_jaccard",
        "tests/unit/recommenders/utils/test_python_utils.py::test_python_lift",
        "tests/unit/recommenders/utils/test_python_utils.py::test_exponential_decay",
        "tests/unit/recommenders/utils/test_python_utils.py::test_get_top_k_scored_items",
        "tests/unit/recommenders/utils/test_python_utils.py::test_binarize",
        "tests/unit/recommenders/utils/test_python_utils.py::test_rescale",
        "tests/unit/recommenders/utils/test_timer.py::test_no_time",
        "tests/unit/recommenders/utils/test_timer.py::test_stop_before_start",
        "tests/unit/recommenders/utils/test_timer.py::test_interval_before_stop",
        "tests/unit/recommenders/utils/test_timer.py::test_timer",
        "tests/unit/recommenders/utils/test_timer.py::test_timer_format",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_merge_rating",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_merge_ranking",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_rmse",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_mae",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_rsquared",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_exp_var",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_get_top_k_items",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_get_top_k_items_largek",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_ndcg_at_k",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_map_at_k",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_precision",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_recall",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_auc",
        "tests/performance/recommenders/evaluation/test_python_evaluation_time_performance.py::test_python_logloss",
        "tests/security/test_dependency_security.py::test_requests",
        "tests/security/test_dependency_security.py::test_numpy",
        "tests/security/test_dependency_security.py::test_pandas",
        "tests/responsible_ai/recommenders/datasets/test_criteo_privacy.py",
        "tests/responsible_ai/recommenders/datasets/test_movielens_privacy.py",
        "tests/integration/recommenders/utils/test_k8s_utils.py",
    ],
    "group_notebooks_cpu_001": [  # Total group time: 226.42s
        "tests/unit/examples/test_notebooks_python.py::test_sar_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_baseline_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_template_runs",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_is_jupyter",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_update_parameters",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_notebook_execution",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_notebook_execution_with_parameters",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_notebook_execution_value_error_fails",
        "tests/unit/examples/test_notebooks_python.py::test_surprise_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_lightgbm",
        "tests/unit/examples/test_notebooks_python.py::test_cornac_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_sar_single_node_runs",
        "tests/unit/examples/test_notebooks_python.py::test_vw_deep_dive_runs",
    ],
    "group_spark_001": [  # Total group time: 270.41s
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_init_spark",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__return_success",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__store_tmp_file",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__data_serialization_default_param",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_spark_df_mock_100__with_default_param__succeed",
        "tests/data_validation/recommenders/datasets/test_movielens.py::test_load_spark_df_mock_100__with_custom_param__succeed",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_stratified_splitter",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_chrono_splitter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_diversity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_python_match",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_exp_var",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_diversity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_item_serendipity",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_min_rating_filter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_serendipity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_serendipity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_diversity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_diversity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_item_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_novelty",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_timestamp_splitter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_ndcg_at_k",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_recall_at_k",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_precision_at_k",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_map_at_k",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_map",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_rmse",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_item_novelty",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_mae",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_rsquared",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_random_splitter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_init_spark_rating_eval",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_catalog_coverage",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_distributional_coverage",
    ],
    "group_notebooks_spark_001": [  # Total group time: 794s
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_is_databricks",
        "tests/unit/examples/test_notebooks_pyspark.py::test_als_deep_dive_runs",  # 287.70s
        "tests/unit/examples/test_notebooks_pyspark.py::test_als_pyspark_runs",  # 374.15s
        "tests/unit/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_runs",  # 132.09s
    ],
    "group_notebooks_spark_002": [  # Total group time: 466s
        "tests/unit/examples/test_notebooks_pyspark.py::test_data_split_runs",  # 43.66s
        "tests/unit/examples/test_notebooks_pyspark.py::test_evaluation_runs",  # 45.24s
        "tests/unit/examples/test_notebooks_pyspark.py::test_evaluation_diversity_runs",  # 376.61s
        # TODO: This is a flaky test, skip for now, to be fixed in future iterations.
        # Refer to the issue: https://github.com/microsoft/recommenders/issues/1770
        # "tests/unit/examples/test_notebooks_pyspark.py::test_spark_tuning",  # 212.29s+190.02s+180.13s+164.09s=746.53s (flaky test, it rerun several times)
    ],
    "group_gpu_001": [  # Total group time: 492.62s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/unit/recommenders/models/test_rbm.py::test_class_init",
        "tests/unit/recommenders/models/test_rbm.py::test_sampling_funct",
        "tests/unit/recommenders/models/test_rbm.py::test_train_param_init",
        "tests/unit/recommenders/models/test_rbm.py::test_save_load",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_init",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_train_loader",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_test_loader",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_init_unsorted",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_init_empty",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_missing_column",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_negative_sampler",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_init",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_fit",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_neumf_save_load",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_regular_save_load",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_predict",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_wide_model",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_deep_model",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_wide_deep_model",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_naml_component_definition",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_lstur_component_definition",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_nrms_component_definition",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_npa_component_definition",
        "tests/unit/recommenders/models/test_newsrec_utils.py::test_prepare_hparams",
        "tests/unit/recommenders/models/test_newsrec_utils.py::test_load_yaml_file",
        # "tests/unit/recommenders/models/test_sasrec_model.py::test_prepare_data", # FIXME: it takes too long to run
        # "tests/unit/recommenders/models/test_sasrec_model.py::test_sampler", # FIXME: it takes too long to run
        # "tests/unit/recommenders/models/test_sasrec_model.py::test_sasrec", # FIXME: it takes too long to run
        # "tests/unit/recommenders/models/test_sasrec_model.py::test_ssept", # FIXME: it takes too long to run
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_get_gpu_info",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_get_number_gpus",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_clear_memory_all_gpus",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_get_cuda_version",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_get_cudnn_version",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_cudnn_enabled",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_tensorflow_gpu",
        "tests/unit/recommenders/utils/test_gpu_utils.py::test_pytorch_gpu",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_evaluation_log_hook",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_pandas_input_fn",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_pandas_input_fn_for_saved_model",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_build_optimizer",
    ],
    "group_gpu_002": [  # Total group time:
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        # "tests/unit/recommenders/models/test_deeprec_model.py::test_xdeepfm_component_definition",  # FIXME: Disabled due to the issue with TF version > 2.10.1 See #2018
        "tests/unit/recommenders/models/test_deeprec_model.py::test_dkn_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_dkn_item2item_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_slirec_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_nextitnet_component_definition",
        # "tests/unit/recommenders/models/test_deeprec_model.py::test_sum_component_definition",  # FIXME: Disabled due to the issue with TF version > 2.10.1 See #2018
        "tests/unit/recommenders/models/test_deeprec_model.py::test_lightgcn_component_definition",
        "tests/unit/recommenders/models/test_deeprec_utils.py::test_prepare_hparams",
        "tests/unit/recommenders/models/test_deeprec_utils.py::test_load_yaml_file",
        "tests/security/test_dependency_security.py::test_tensorflow",
        "tests/security/test_dependency_security.py::test_torch",
        "tests/regression/test_compatibility_tf.py",
    ],
    "group_notebooks_gpu_001": [  # Total group time: 563.35s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/unit/examples/test_notebooks_gpu.py::test_dkn_quickstart",
        "tests/unit/examples/test_notebooks_gpu.py::test_ncf",
        "tests/unit/examples/test_notebooks_gpu.py::test_ncf_deep_dive",
        "tests/unit/examples/test_notebooks_gpu.py::test_fastai",
    ],
    "group_notebooks_gpu_002": [  # Total group time: 241.15s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s (Always the first test to check the GPU works)
        "tests/unit/examples/test_notebooks_gpu.py::test_wide_deep",
        "tests/unit/examples/test_notebooks_gpu.py::test_xdeepfm",
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",
    ],
}

# Experimental are additional test groups that require to install extra dependencies: pip install .[experimental]
experimental_test_groups = {
    "group_cpu_001": [
        "tests/unit/recommenders/models/test_lightfm_utils.py::test_interactions",
        "tests/unit/recommenders/models/test_lightfm_utils.py::test_fitting",
        "tests/unit/recommenders/models/test_lightfm_utils.py::test_sim_users",
        "tests/unit/recommenders/models/test_lightfm_utils.py::test_sim_items",
        "tests/functional/examples/test_notebooks_python.py::test_lightfm_functional",
        "tests/unit/recommenders/models/test_geoimc.py::test_imcproblem",
        "tests/unit/recommenders/models/test_geoimc.py::test_dataptr",
        "tests/unit/recommenders/models/test_geoimc.py::test_length_normalize",
        "tests/unit/recommenders/models/test_geoimc.py::test_mean_center",
        "tests/unit/recommenders/models/test_geoimc.py::test_reduce_dims",
        "tests/unit/recommenders/models/test_geoimc.py::test_imcproblem",
        "tests/unit/recommenders/models/test_geoimc.py::test_inferer_init",
        "tests/unit/recommenders/models/test_geoimc.py::test_inferer_infer",
        "tests/unit/examples/test_notebooks_python.py::test_rlrmc_quickstart_runs",
    ]
}
