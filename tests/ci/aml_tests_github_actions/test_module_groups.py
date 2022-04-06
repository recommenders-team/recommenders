# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: All the times are being calculated on an Azure STANDARD_NC6S_V2
# with 6 vcpus, 112 GiB memory, 1 NVIDIA Tesla P100 GPU.

# IMPORTANT NOTE: NO GROUP SHOULD SURPASS 30MIN = 1800s !!!

groups = {
    "group_cpu_001": [ # Total group time: 1020.68s
        # Movielens dataset
        "tests/smoke/recommenders/dataset/test_movielens.py::test_download_and_extract_movielens", # 0.45s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_item_df", # 0.47s
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_pandas_df", # 2.45s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_pandas_df", # 16.87s
        "tests/integration/recommenders/datasets/test_movielens.py::test_download_and_extract_movielens", # 0.61s + 3.47s + 8.28s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_item_df", # 0.59s + 3.59s + 8.44s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_pandas_df", # 155.18s + 302.65s

        # Criteo dataset
        "tests/smoke/recommenders/dataset/test_criteo.py::test_download_criteo", # 1.05s
        "tests/smoke/recommenders/dataset/test_criteo.py::test_extract_criteo", # 1.22s
        "tests/smoke/recommenders/dataset/test_criteo.py::test_criteo_load_pandas_df", # 1.73s
        "tests/integration/recommenders/datasets/test_criteo.py::test_criteo_load_pandas_df", # 427.89s

        # MIND dataset
        "tests/smoke/recommenders/dataset/test_mind.py::test_mind_url", # 0.38s
        "tests/smoke/recommenders/dataset/test_mind.py::test_extract_mind", # 10.23s
        "tests/smoke/examples/test_notebooks_python.py::test_mind_utils", # 219.77s
        "tests/integration/recommenders/datasets/test_mind.py::test_download_mind", # 37.63s
        "tests/integration/recommenders/datasets/test_mind.py::test_extract_mind", # 56.30s
        "tests/integration/recommenders/datasets/test_mind.py::test_mind_utils_integration", # 219.26s
    ],
    "group_cpu_002": [ # Total group time: 1444,31s
        "tests/smoke/examples/test_notebooks_python.py::test_lightgbm_quickstart_smoke", # 46.42s

        "tests/smoke/examples/test_notebooks_python.py::test_baseline_deep_dive_smoke", # 15.98s
        "tests/integration/examples/test_notebooks_python.py::test_baseline_deep_dive_integration", # 170.73s

        "tests/smoke/examples/test_notebooks_python.py::test_surprise_svd_smoke", # 45.88s
        "tests/integration/examples/test_notebooks_python.py::test_surprise_svd_integration", # 503.54s

        "tests/smoke/examples/test_notebooks_python.py::test_sar_single_node_smoke", # 12.58s
        "tests/integration/examples/test_notebooks_python.py::test_sar_single_node_integration", # 49.89s + 599.29s
    ],
    "group_cpu_003": [ # Total group time: 1444.26
        "tests/smoke/examples/test_notebooks_python.py::test_cornac_bpr_smoke", # 16.62s
        "tests/integration/examples/test_notebooks_python.py::test_cornac_bpr_integration", # 165.72s

        "tests/integration/examples/test_notebooks_python.py::test_geoimc_integration", # 1006.19s

        "tests/integration/examples/test_notebooks_python.py::test_xlearn_fm_integration", # 255.73s
    ],
    "group_gpu_001":[

    ],
    "group_spark_001":[ # Total group time: 845.16s
        # Movielens dataset
        "tests/smoke/recommenders/dataset/test_movielens.py::test_load_spark_df", # 4.33s
        "tests/integration/recommenders/datasets/test_movielens.py::test_load_spark_df", # 25.58s + 101.99s + 139.23s

        # Criteo dataset
        "tests/smoke/recommenders/dataset/test_criteo.py::test_criteo_load_spark_df", # 6.83s
        "tests/smoke/examples/test_notebooks_pyspark.py::test_mmlspark_lightgbm_criteo_smoke", # 32.45s
        "tests/integration/recommenders/datasets/test_criteo.py::test_criteo_load_spark_df", # 374.64s

        # ALS
        "tests/smoke/examples/test_notebooks_pyspark.py::test_als_pyspark_smoke", # 49.53s
        "tests/integration/examples/test_notebooks_pyspark.py::test_als_pyspark_integration", # 110.58s
    ],
}
#CPU


#GPU
620.13s      tests/smoke/examples/test_notebooks_gpu.py::test_naml_smoke
450.65s      tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_naml
366.22s      tests/smoke/examples/test_notebooks_gpu.py::test_npa_smoke
346.72s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_slirec
246.46s      tests/smoke/examples/test_notebooks_gpu.py::test_lstur_smoke
232.55s      tests/smoke/examples/test_notebooks_gpu.py::test_nrms_smoke
202.61s      tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_npa
194.88s      tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_lstur
188.60s      tests/smoke/recommenders/recommender/test_newsrec_model.py::test_model_nrms
187.20s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_dkn
122.71s      tests/smoke/examples/test_notebooks_gpu.py::test_wide_deep_smoke
114.39s      tests/smoke/examples/test_notebooks_gpu.py::test_ncf_smoke
102.71s      tests/smoke/examples/test_notebooks_gpu.py::test_ncf_deep_dive_smoke
77.93s      tests/smoke/examples/test_notebooks_gpu.py::test_xdeepfm_smoke
67.84s      tests/smoke/examples/test_notebooks_gpu.py::test_cornac_bivae_smoke
33.22s      tests/smoke/examples/test_notebooks_gpu.py::test_fastai_smoke
27.23s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_sum
6.03s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_lightgcn
5.50s      tests/smoke/recommenders/recommender/test_newsrec_utils.py::test_naml_iterator
3.10s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_model_xdeepfm
3.04s      tests/smoke/recommenders/recommender/test_newsrec_utils.py::test_news_iterator
2.63s      tests/smoke/recommenders/recommender/test_deeprec_utils.py::test_DKN_iterator
0.76s      tests/smoke/examples/test_notebooks_gpu.py::test_gpu_vm
0.74s      tests/smoke/recommenders/recommender/test_deeprec_model.py::test_FFM_iterator
0.28s      tests/smoke/recommenders/recommender/test_deeprec_utils.py::test_Sequential_Iterator

2033.85s      tests/integration/examples/test_notebooks_gpu.py::test_naml_quickstart_integration[5-64-42-demo-expected_values0]
1843.29s      tests/integration/examples/test_notebooks_gpu.py::test_wide_deep_integration[1m-50000-expected_values0-42]
1167.93s      tests/integration/examples/test_notebooks_gpu.py::test_dkn_quickstart_integration
1046.97s      tests/integration/examples/test_notebooks_gpu.py::test_ncf_integration[1m-10-expected_values0-42]
857.05s      tests/integration/examples/test_notebooks_gpu.py::test_nrms_quickstart_integration[5-64-42-demo-expected_values0]
810.92s      tests/integration/examples/test_notebooks_gpu.py::test_npa_quickstart_integration[5-64-42-demo-expected_values0]
766.52s      tests/integration/examples/test_notebooks_gpu.py::test_lstur_quickstart_integration[5-64-42-demo-expected_values0]
667.88s      tests/integration/examples/test_notebooks_gpu.py::test_fastai_integration[1m-10-expected_values0]
614.69s      tests/integration/examples/test_notebooks_gpu.py::test_sasrec_quickstart_integration[tests/recsys_data/RecSys/SASRec-tf2/data-1-128-sasrec-expected_values0-42]
470.11s      tests/integration/examples/test_notebooks_gpu.py::test_xdeepfm_integration[15-10-expected_values0-42]
453.21s      tests/integration/examples/test_notebooks_gpu.py::test_cornac_bivae_integration[1m-expected_values0]
448.06s      tests/integration/examples/test_notebooks_gpu.py::test_sasrec_quickstart_integration[tests/recsys_data/RecSys/SASRec-tf2/data-1-128-ssept-expected_values1-42]
351.17s      tests/integration/examples/test_notebooks_gpu.py::test_ncf_deep_dive_integration[100k-10-512-expected_values0-42]
175.00s      tests/integration/examples/test_notebooks_gpu.py::test_slirec_quickstart_integration[recommenders/models/deeprec/config/sli_rec.yaml-tests/resources/deeprec/slirec-10-400-expected_values0-42]
19.45s      tests/integration/examples/test_notebooks_gpu.py::test_lightgcn_deep_dive_integration[recommenders/models/deeprec/config/lightgcn.yaml-tests/resources/deeprec/lightgcn-100k-5-1024-expected_values0-42]
0.82s      tests/integration/examples/test_notebooks_gpu.py::test_gpu_vm

#spark
      
            
      
