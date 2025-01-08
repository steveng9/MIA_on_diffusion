import json

import category_encoders
# from midst_models.single_table_TabDDPM.complex_pipeline import (
#     clava_clustering,
#     clava_training,
#     clava_load_pretrained,
#     clava_synthesizing,
#     load_configs,
# )
# from midst_models.single_table_TabDDPM.pipeline_modules import load_multi_table
#
#
#
# # Load config
# config_path = "configs/trans.json"
# configs, save_dir = load_configs(config_path)
#
# # Display config
# json_str = json.dumps(configs, indent=4)
# print(json_str)
#
#
# # Load  dataset
# # In this step, we load the dataset according to the 'dataset_meta.json' file located in the data_dir.
# tables, relation_order, dataset_meta = load_multi_table(configs["general"]["data_dir"])
# print("")
#
# # Tables is a dictionary of the multi-table dataset
# print(
#     "{} We show the keys of the tables dictionary below {}".format("=" * 20, "=" * 20)
# )
# print(list(tables.keys()))