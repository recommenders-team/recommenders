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
    "group_gpu_001":[

    ],
    "group_spark_001":[ # Total group time: 
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
      
46.42s      tests/smoke/examples/test_notebooks_python.py::test_lightgbm_quickstart_smoke
45.88s      tests/smoke/examples/test_notebooks_python.py::test_surprise_svd_smoke
16.62s      tests/smoke/examples/test_notebooks_python.py::test_cornac_bpr_smoke
15.98s      tests/smoke/examples/test_notebooks_python.py::test_baseline_deep_dive_smoke
12.58s      tests/smoke/examples/test_notebooks_python.py::test_sar_single_node_smoke
      
      

      
      
1006.19s      tests/integration/examples/test_notebooks_python.py::test_geoimc_integration[expected_values0]
599.29s      tests/integration/examples/test_notebooks_python.py::test_sar_single_node_integration[10m-expected_values1]
503.54s      tests/integration/examples/test_notebooks_python.py::test_surprise_svd_integration[1m-expected_values0]
      
255.73s      tests/integration/examples/test_notebooks_python.py::test_xlearn_fm_integration
      
170.73s      tests/integration/examples/test_notebooks_python.py::test_baseline_deep_dive_integration[1m-expected_values0]
165.72s      tests/integration/examples/test_notebooks_python.py::test_cornac_bpr_integration[1m-expected_values0]
      
      
49.89s      tests/integration/examples/test_notebooks_python.py::test_sar_single_node_integration[1m-expected_values0]
      
      

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

#spark
      
            
      
