#!/usr/bin/env python
#title           :config.py
#description     :Configuration class for hw2vec
#author          :hw2vec
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
import argparse
import torch
import yaml
from pathlib import Path


class Config:
    """Configuration class for hw2vec that handles command-line arguments and YAML config files."""
    
    def __init__(self, args=None):
        """
        Initialize configuration from command-line arguments.
        
        Args:
            args: List of command-line arguments (similar to sys.argv[1:])
        """
        parser = argparse.ArgumentParser(description='hw2vec Configuration')
        
        # Dataset paths
        parser.add_argument('--raw_dataset_path', type=str, default='',
                            help='Path to raw dataset directory')
        parser.add_argument('--data_pkl_path', type=str, default='',
                            help='Path to cached graph data pickle file')
        
        # Graph type
        parser.add_argument('--graph_type', type=str, default='DFG',
                            choices=['DFG', 'AST', 'CFG'],
                            help='Type of graph to generate: DFG, AST, or CFG')
        
        # YAML configuration
        parser.add_argument('--yaml_path', type=str, default='',
                            help='Path to YAML configuration file')
        
        # Model paths
        parser.add_argument('--model_path', type=str, default='',
                            help='Path to model directory')
        
        # Training hyperparameters
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Initial learning rate')
        parser.add_argument('--seed', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train')
        parser.add_argument('--hidden', type=int, default=200,
                            help='Number of hidden units')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability)')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='Number of graphs in a batch')
        parser.add_argument('--num_layer', type=int, default=2,
                            help='Number of layers in the neural network')
        parser.add_argument('--test_step', type=int, default=10,
                            help='Interval between mini evaluations')
        
        # Graph pooling and readout
        parser.add_argument('--pooling_type', type=str, default='topk',
                            choices=['sagpool', 'topk', 'topkpool'],
                            help='Graph pooling type')
        parser.add_argument('--readout_type', type=str, default='max',
                            choices=['max', 'mean', 'add'],
                            help='Graph readout type')
        parser.add_argument('--poolratio', type=float, default=0.8,
                            help='Ratio for graph pooling')
        
        # Dataset splitting
        parser.add_argument('--ratio', type=float, default=0.8,
                            help='Dataset train/test splitting ratio')
        
        # Embedding
        parser.add_argument('--embed_dim', type=int, default=2,
                            help='Dimension of graph embeddings')
        
        # Device
        parser.add_argument('--device', type=str, default=None,
                            help='Device to use (cuda or cpu)')
        
        # Parse arguments
        if args is None:
            args = []
        parsed_args = parser.parse_args(args)
        
        # Load YAML config if provided
        if parsed_args.yaml_path:
            yaml_path = Path(parsed_args.yaml_path)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        # Update parsed args with YAML values (YAML takes precedence for duplicate keys)
                        for key, value in yaml_config.items():
                            if not hasattr(parsed_args, key) or getattr(parsed_args, key) == parser.get_default(key):
                                setattr(parsed_args, key, value)
        
        # Convert Path strings to Path objects
        if parsed_args.raw_dataset_path:
            parsed_args.raw_dataset_path = Path(parsed_args.raw_dataset_path)
        if parsed_args.data_pkl_path:
            parsed_args.data_pkl_path = Path(parsed_args.data_pkl_path)
        if parsed_args.model_path:
            parsed_args.model_path = Path(parsed_args.model_path)
        
        # Set device
        if parsed_args.device is None:
            parsed_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            parsed_args.device = torch.device(parsed_args.device)
        
        # Initialize num_feature_dim (will be set later by DataProcessor)
        parsed_args.num_feature_dim = None
        
        # Copy all attributes to self
        for key, value in vars(parsed_args).items():
            setattr(self, key, value)
    
    def __repr__(self):
        """String representation of the configuration."""
        items = [f"{k}={v}" for k, v in vars(self).items()]
        return f"Config({', '.join(items)})"
