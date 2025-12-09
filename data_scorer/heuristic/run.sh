# srun -p raise --gres=gpu:0 python main.py --config configs/NovelSumScorer.yaml --data_ready &
srun -p raise --gres=gpu:0 python main.py --config configs/PureThinkScorer.yaml
