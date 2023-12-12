sbatch -n1 -t 01:00:00 ./submit_itk.sh g4i-train examples/Example_1/gnn_train.yaml
#sbatch -n 1 -t 20 ./submit_itk.sh g4i-infer examples/Example_1/gnn_infer.yaml
#sbatch -n 1 -t 20 ./submit_itk.sh g4i-eval examples/Example_1/gnn_eval.yaml
#sbatch -n 1 -t 20 ./submit_itk.sh g4i-infer examples/Example_1/track_building_infer.yaml
