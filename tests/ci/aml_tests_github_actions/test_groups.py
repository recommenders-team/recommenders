# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: All the times are being calculated on an Azure STANDARD_NC6S_V2
# with 6 vcpus, 112 GiB memory, 1 NVIDIA Tesla P100 GPU.

# IMPORTANT NOTE: NO GROUP SHOULD SURPASS 30MIN = 1800s !!!

global nightly_test_groups, unit_test_groups

nightly_test_groups = {
    "group_cpu_001": [  # Total group time: 1020.68s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_download_and_extract_movielens",  # 0.45s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_item_df",  # 0.47s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_pandas_df",  # 2.45s
        #
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_pandas_df",  # 16.87s
        "tests/integration/recommenders/datasets/test_movielens.py::test_download_and_extract_movielens",  # 0.61s + 3.47s + 8.28s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_item_df",  # 0.59s + 3.59s + 8.44s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_pandas_df",  # 155.18s + 302.65s
        #
        "tests/smoke/recommenders/dataset/test_criteo.py::test_download_criteo",  # 1.05s
        "tests/smoke/recommenders/dataset/test_criteo.py::test_extract_criteo",  # 1.22s
        "tests/smoke/recommenders/dataset/test_criteo.py::test_criteo_load_pandas_df",  # 1.73s
        "tests/integration/recommenders/datasets/test_criteo.py::test_criteo_load_pandas_df",  # 427.89s
        #
        "tests/smoke/recommenders/dataset/test_mind.py::test_mind_url",  # 0.38s
        "tests/smoke/recommenders/dataset/test_mind.py::test_extract_mind",  # 10.23s
        "tests/smoke/examples/test_notebooks_python.py::test_mind_utils",  # 219.77s
        "tests/integration/recommenders/datasets/test_mind.py::test_download_mind",  # 37.63s
        "tests/integration/recommenders/datasets/test_mind.py::test_extract_mind",  # 56.30s
        "tests/integration/recommenders/datasets/test_mind.py::test_mind_utils_integration",  # 219.26s
    ],
    "group_cpu_002": [  # Total group time: 1444,31s
        "tests/smoke/examples/test_notebooks_python.py::test_lightgbm_quickstart_smoke",  # 46.42s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_baseline_deep_dive_smoke",  # 15.98s
        "tests/integration/examples/test_notebooks_python.py::test_baseline_deep_dive_integration",  # 170.73s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_surprise_svd_smoke",  # 45.88s
        "tests/integration/examples/test_notebooks_python.py::test_surprise_svd_integration",  # 503.54s
        #
        "tests/smoke/examples/test_notebooks_python.py::test_sar_single_node_smoke",  # 12.58s
        "tests/integration/examples/test_notebooks_python.py::test_sar_single_node_integration",  # 49.89s + 599.29s
    ],
    "group_cpu_003": [  # Total group time: 1444.26
        "tests/smoke/examples/test_notebooks_python.py::test_cornac_bpr_smoke",  # 16.62s
        "tests/integration/examples/test_notebooks_python.py::test_cornac_bpr_integration",  # 165.72s
        #
        "tests/integration/examples/test_notebooks_python.py::test_geoimc_integration",  # 1006.19s
        #
        # "tests/integration/examples/test_notebooks_python.py::test_xlearn_fm_integration",  # 255.73s
    ],
    "group_gpu_001": [  # Total group time: 1415.29s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/recommenders/recommender/test_deeprec_utils.py",  # 2.91
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_FFM_iterator",  # 0.74s
        "tests/smoke/recommenders/recommender/test_newsrec_utils.py::test_news_iterator",  # 3.04s
        #
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_lightgcn",  # 6.03s
        "tests/integration/examples/test_notebooks_gpu.py::test_lightgcn_deep_dive_integration",  # 19.45s
        #
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_sum",  # 27.23s
        #
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_dkn",  # 187.20s
        "tests/integration/examples/test_notebooks_gpu.py::test_dkn_quickstart_integration",  # 1167.93s
    ],
    "group_gpu_002": [  # Total group time: 1594.67s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_xdeepfm",  # 3.10s
        "tests/smoke/examples/test_notebooks_gpu.py::test_xdeepfm_smoke",  # 77.93s
        #
        "tests/integration/examples/test_notebooks_gpu.py::test_xdeepfm_integration",  # 470.11s
        #
        "tests/integration/examples/test_notebooks_gpu.py::test_slirec_quickstart_integration",  # 175.00s
        "tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_slirec",  # 346.72s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_cornac_bivae_smoke",  # 67.84s
        "tests/integration/examples/test_notebooks_gpu.py::test_cornac_bivae_integration",  # 453.21s
    ],
    "group_gpu_003": [  # Total group time: 1616s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/examples/test_notebooks_gpu.py::test_ncf_deep_dive_smoke",  # 102.71s
        "tests/smoke/examples/test_notebooks_gpu.py::test_ncf_smoke",  # 114.39s
        "tests/integration/examples/test_notebooks_gpu.py::test_ncf_deep_dive_integration",  # 351.17s
        "tests/integration/examples/test_notebooks_gpu.py::test_ncf_integration",  # 1046.97s
    ],
    "group_gpu_004": [  # Total group time: 1778.14s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/recommenders/recommender/test_newsrec_utils.py::test_naml_iterator",  # 5.50s
        "tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_naml",  # 450.65s
        "tests/smoke/examples/test_notebooks_gpu.py::test_naml_smoke",  # 620.13s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_fastai_smoke",  # 33.22s
        "tests/integration/examples/test_notebooks_gpu.py::test_fastai_integration",  # 667.88s
    ],
    "group_gpu_005": [  # Total group time: 1503.22s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_npa",  # 202.61s
        "tests/smoke/examples/test_notebooks_gpu.py::test_npa_smoke",  # 366.22s
        "tests/integration/examples/test_notebooks_gpu.py::test_npa_quickstart_integration",  # 810.92s
        #
        "tests/smoke/examples/test_notebooks_gpu.py::test_wide_deep_smoke",  # 122.71s
    ],
    "group_gpu_006": [  # Total group time: 1844.05s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/integration/examples/test_notebooks_gpu.py::test_wide_deep_integration",  # 1843.29s
    ],
    "group_gpu_007": [  # Total group time: 1278.96s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_nrms",  # 188.60s
        "tests/smoke/examples/test_notebooks_gpu.py::test_nrms_smoke",  # 232.55s
        "tests/integration/examples/test_notebooks_gpu.py::test_nrms_quickstart_integration",  # 857.05s
    ],
    "group_gpu_008": [  # Total group time: 1208.62s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/smoke/examples/test_notebooks_gpu.py::test_lstur_smoke",  # 246.46s
        "tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_lstur",  # 194.88s
        "tests/integration/examples/test_notebooks_gpu.py::test_lstur_quickstart_integration",  # 766.52s
    ],
    "group_gpu_009": [  # Total group time: 1063.51s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/integration/examples/test_notebooks_gpu.py::test_sasrec_quickstart_integration",  # 448.06s + 614.69s
    ],
    "group_gpu_010": [  # Total group time: 2034.61s
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",  # 0.76s
        "tests/integration/examples/test_notebooks_gpu.py::test_naml_quickstart_integration",  # 2033.85s
    ],
    "group_spark_001": [  # Total group time: 845.16s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_spark_df",  # 4.33s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_spark_df",  # 25.58s + 101.99s + 139.23s
        #
        "tests/smoke/recommenders/dataset/test_criteo.py::test_criteo_load_spark_df",  # 6.83s
        "tests/smoke/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_smoke",  # 32.45s
        "tests/integration/recommenders/datasets/test_criteo.py::test_criteo_load_spark_df",  # 374.64s
        #
        "tests/smoke/examples/test_notebooks_pyspark.py::test_als_pyspark_smoke",  # 49.53s
        "tests/integration/examples/test_notebooks_pyspark.py::test_als_pyspark_integration",  # 110.58s
    ],
}

unit_test_groups = {
    "group_pyspark_001": [
        "tests/unit/recommenders/datasets/test_movielens.py::test_load_spark_df_mock_100__with_custom_param__succeed",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__return_success",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_stratified_splitter",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_chrono_splitter",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__data_serialization_default_param",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_diversity_item_feature_vector",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_spark_df__store_tmp_file",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_python_match",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_precision",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_exp_var",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_diversity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_item_serendipity",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_min_rating_filter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_serendipity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_serendipity",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_diversity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_diversity",

    ],
    "group_pyspark_002": [
        "tests/unit/recommenders/datasets/test_movielens.py::test_load_spark_df_mock_100__with_default_param__succeed",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_user_item_serendipity_item_feature_vector",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_novelty",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_timestamp_splitter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_ndcg",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_recall",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_map",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_rmse",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_item_novelty",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_mae",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_spark_rsquared",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_random_splitter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_init_spark_rating_eval",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_catalog_coverage",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_distributional_coverage",
        "tests/unit/recommenders/datasets/test_spark_splitter.py::test_min_rating_filter",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_catalog_coverage",
        "tests/unit/recommenders/evaluation/test_spark_evaluation.py::test_init_spark_rating_eval",
    ],
    "group_notebooks_pyspark_001": [
        "tests/unit/examples/test_notebooks_pyspark.py::test_spark_tuning",
        "tests/unit/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_runs",
    ],
    "group_notebooks_pyspark_002": [
        "tests/unit/examples/test_notebooks_pyspark.py::test_als_deep_dive_runs",
        "tests/unit/examples/test_notebooks_pyspark.py::test_data_split_runs",
        "tests/unit/examples/test_notebooks_pyspark.py::test_evaluation_runs",
    ],
    "group_notebooks_pyspark_003": [
        "tests/unit/examples/test_notebooks_pyspark.py::test_als_pyspark_runs",
        "tests/unit/examples/test_notebooks_pyspark.py::test_evaluation_diversity_runs",
    ],
    "group_gpu_001": [
        "tests/unit/recommenders/models/test_deeprec_model.py::test_xdeepfm_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_dkn_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_dkn_item2item_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_slirec_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_nextitnet_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_sum_component_definition",
        "tests/unit/recommenders/models/test_deeprec_model.py::test_lightgcn_component_definition",
    ],
    "group_gpu_002": [
        "tests/unit/recommenders/models/test_rbm.py::test_sampling_funct",
        "tests/unit/recommenders/models/test_rbm.py::test_train_param_init",
        "tests/unit/recommenders/models/test_rbm.py::test_save_load",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_wide_model",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_neumf_save_load",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_regular_save_load",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_evaluation_log_hook",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_pandas_input_fn_for_saved_model",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_wide_deep_model",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_naml_component_definition",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_lstur_component_definition",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_nrms_component_definition",
        "tests/unit/recommenders/models/test_wide_deep_utils.py::test_deep_model",
        "tests/unit/recommenders/models/test_newsrec_model.py::test_npa_component_definition",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_fit",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_init",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_test_loader",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_init",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_train_loader",
        "tests/unit/recommenders/models/test_rbm.py::test_class_init",
        "tests/unit/recommenders/utils/test_tf_utils.py::test_pandas_input_fn",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_init_unsorted",
        "tests/unit/recommenders/models/test_ncf_singlenode.py::test_predict",
        "tests/unit/recommenders/models/test_ncf_dataset.py::test_datafile_missing_column",
    ],
    "group_gpu_003": [
        "tests/unit/recommenders/models/test_sasrec_model.py::test_prepare_data",
        "tests/unit/recommenders/models/test_sasrec_model.py::test_sampler",
        "tests/unit/recommenders/models/test_sasrec_model.py::test_sasrec",
        "tests/unit/recommenders/models/test_sasrec_model.py::test_ssept",
    ],
    "group_notebooks_gpu_001": [
        "tests/unit/examples/test_notebooks_gpu.py::test_dkn_quickstart",  
    ],
    "group_notebooks_gpu_002": [
        "tests/unit/examples/test_notebooks_gpu.py::test_ncf",
        "tests/unit/examples/test_notebooks_gpu.py::test_ncf_deep_dive",
        "tests/unit/examples/test_notebooks_gpu.py::test_fastai",
    ],
    "group_notebooks_gpu_003": [
        "tests/unit/examples/test_notebooks_gpu.py::test_wide_deep",
        "tests/unit/examples/test_notebooks_gpu.py::test_xdeepfm",
        "tests/unit/examples/test_notebooks_gpu.py::test_gpu_vm",
    ],
    "group_cpu_001": [
        "tests/unit/recommenders/datasets/test_movielens.py::test_load_pandas_df_mock_100__with_default_param__succeed",
        "tests/unit/recommenders/datasets/test_dataset.py::test_maybe_download_wrong_bytes",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__has_default_col_names",
        "tests/unit/recommenders/datasets/test_movielens.py::test_load_pandas_df_mock_100__with_custom_param__succeed",
        "tests/unit/recommenders/datasets/test_dataset.py::test_maybe_download_retry",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_df__return_success",
        "tests/unit/recommenders/utils/test_timer.py::test_timer",
        "tests/unit/recommenders/tuning/test_ncf_utils.py::test_compute_test_results__return_success",
        "tests/unit/recommenders/datasets/test_movielens.py::test_mock_movielens_schema__get_df_remove_default_col__return_success",
        "tests/unit/recommenders/models/test_geoimc.py::test_imcproblem",
        "tests/unit/recommenders/datasets/test_wikidata.py::test_find_wikidata_id",
        "tests/unit/recommenders/models/test_sar_singlenode.py::test_sar_item_similarity",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_tokenize_text",
        "tests/unit/recommenders/models/test_tfidf_utils.py::test_get_tokens",
        "tests/unit/recommenders/models/test_cornac_utils.py::test_recommend_k_items",
    ],
    "group_notebooks_cpu_001": [
        "tests/unit/examples/test_notebooks_python.py::test_rlrmc_quickstart_runs",
        "tests/unit/examples/test_notebooks_python.py::test_sar_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_baseline_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_template_runs",
        "tests/unit/recommenders/utils/test_notebook_utils.py::test_is_jupyter",
    ],
    "group_notebooks_cpu_002": [
        "tests/unit/examples/test_notebooks_python.py::test_surprise_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_lightgbm",
        "tests/unit/examples/test_notebooks_python.py::test_cornac_deep_dive_runs",
        "tests/unit/examples/test_notebooks_python.py::test_sar_single_node_runs",
    ],
}