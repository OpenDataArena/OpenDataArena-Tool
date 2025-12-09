source ~/.bashrc
conda activate oda

# DeitaCScorer
python main_para.py --config configs/DeitaCScorer.yaml

# DeitaQScorer
python main_para.py --config configs/DeitaQScorer.yaml

# IFDScorer
python main_para.py --config configs/IFDScorer.yaml

# RewardModel
python main_para.py --config configs/RewardModel.yaml

# ThinkingProbScorer
python main_para.py --config configs/ThinkingProbScorer.yaml

# Multi Scorers Togather
python main_para.py --config configs/MultiScorer.yaml



srun -p raise --gres=gpu:1 python main.py --config configs/DeitaCScorer.yaml