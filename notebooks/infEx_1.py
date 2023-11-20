from gnn4itk_cf.core.infer_stage import infer
from gnn4itk_cf.core.eval_stage import evaluate

config_dr = "../examples/Example_1/data_reader.yaml"
config_mm = "../examples/Example_1/module_map_infer.yaml"
config_gnn = "../examples/Example_1/gnn_infer.yaml"
config_tbi = "../examples/Example_1/track_building_infer.yaml"
config_tbe = "../examples/Example_1/track_building_eval.yaml"

#infer(config_dr )
#infer(config_mm)
#infer(config_gnn)
#infer(config_tbi)
evaluate(config_tbe)