source ~/.bashrc
conda activate oda

# ApjsScorer
python main.py --config configs/ApjsScorer.yaml

# ApsScorer
python main.py --config configs/ApsScorer.yaml

# ClusterInertiaScorer
python main.py --config configs/ClusterInertiaScorer.yaml

# FacilityLocationScorer
python main.py --config configs/FacilityLocationScorer.yaml

# GramEntropyScorer
python main.py --config configs/GramEntropyScorer.yaml

# HddScorer
python main.py --config configs/HddScorer.yaml

# KNNScorer
python main.py --config configs/KNNScorer.yaml

# LogDetDistanceScorer
python main.py --config configs/LogDetDistanceScorer.yaml


# MtldScorer
python main.py --config configs/MtldScorer.yaml

# NovelSumScorer
python main.py --config configs/NovelSumScorer.yaml

# PartitionEntropyScorer
python main.py --config configs/PartitionEntropyScorer.yaml

# PureThinkScorer
python main.py --config configs/PureThinkScorer.yaml

# RadiusScorer
python main.py --config configs/RadiusScorer.yaml

# StrLengthScorer
python main.py --config configs/StrLengthScorer.yaml

# ThinkOrNotScorer
python main.py --config configs/ThinkOrNotScorer.yaml

# TokenEntropyScorer
python main.py --config configs/TokenEntropyScorer.yaml

# TokenLengthScorer
python main.py --config configs/TokenLengthScorer.yaml

# TreeInstructScorer
python main.py --config configs/TreeInstructScorer.yaml

# TsPythonScorer
python main.py --config configs/TsPythonScorer.yaml

# UniqueNgramScorer
python main.py --config configs/UniqueNgramScorer.yaml

# UniqueNtokenScorer
python main.py --config configs/UniqueNtokenScorer.yaml

# VendiScorer
python main.py --config configs/VendiScorer.yaml

# VocdDScorer
python main.py --config configs/VocdDScorer.yaml
