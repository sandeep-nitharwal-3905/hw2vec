import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *
import torch.nn.functional as F
import numpy as np


def detect_ip(cfg, hw_design_dir_path, reference_designs):
    """
    Detect the IP type of a hardware design using similarity comparison.
    
    Args:
        cfg: Configuration object
        hw_design_dir_path: Path to the hardware design directory to classify
        reference_designs: Dict mapping IP type names to paths of reference designs
        
    Returns:
        predicted_ip: Predicted IP type name
        similarities: Dict of similarities to each IP type
        graph_embed: Graph embedding vector of the test design
    """
    # Convert test hardware design to graph
    hw2graph = HW2GRAPH(cfg)
    test_design_path = hw2graph.preprocess(hw_design_dir_path)
    test_nxgraph = hw2graph.process(test_design_path)

    # Process test graph
    test_data_proc = DataProcessor(cfg)
    test_data_proc.process(test_nxgraph)
    test_loader = DataLoader(test_data_proc.get_graphs(), batch_size=1)
    test_data = next(iter(test_loader)).to(cfg.device)
    
    # Get embedding for test design
    with torch.no_grad():
        test_embed, _ = cfg.model.embed_graph(test_data.x, test_data.edge_index, test_data.batch)
    
    # Compare with reference designs
    similarities = {}
    for ip_type, ref_path in reference_designs.items():
        ref_design_path = hw2graph.preprocess(Path(ref_path))
        ref_nxgraph = hw2graph.process(ref_design_path)
        
        ref_data_proc = DataProcessor(cfg)
        ref_data_proc.process(ref_nxgraph)
        ref_loader = DataLoader(ref_data_proc.get_graphs(), batch_size=1)
        ref_data = next(iter(ref_loader)).to(cfg.device)
        
        with torch.no_grad():
            ref_embed, _ = cfg.model.embed_graph(ref_data.x, ref_data.edge_index, ref_data.batch)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(test_embed, ref_embed, dim=1).item()
            similarities[ip_type] = similarity
    
    # Predict based on highest similarity
    predicted_ip = max(similarities, key=similarities.get)
    
    return predicted_ip, similarities, test_embed


if __name__ == '__main__': 
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect IP type of a hardware design using pre-trained model')
    parser.add_argument('design_path', nargs='?', default="../assets/IP-RTL-toy/adder4bit/adder4bit_1/",
                        help='Path to the hardware design directory to classify')
    parser.add_argument('--graph_type', type=str, default='DFG',
                        help='Graph type to use (DFG, AST, or CFG)')
    
    args, remaining = parser.parse_known_args()
    
    # Add graph_type to remaining args if provided
    if args.graph_type:
        remaining.extend(['--graph_type', args.graph_type])
    
    cfg = Config(remaining)
    
    hw_design_dir_path = Path(args.design_path)
    
    # Load pre-trained model from use_case_1 assets
    pretrained_model_path = Path("../assets/pretrained_DFG_IP_RTL")
    
    if not pretrained_model_path.exists():
        print(f"Error: Pre-trained model not found at {pretrained_model_path}")
        print("\nThis script uses the pre-trained IP similarity model.")
        print("Please ensure the model exists in ../assets/pretrained_DFG_IP_RTL/")
        sys.exit(1)
    
    if not hw_design_dir_path.exists():
        print(f"Error: Hardware design not found at {hw_design_dir_path}")
        sys.exit(1)
    
    # Load the pre-trained model
    model = GRAPH2VEC(cfg)
    model.load_model(
        str(pretrained_model_path/"model.cfg"),
        str(pretrained_model_path/"model.pth")
    )
    model.to(cfg.device)
    model.eval()
    cfg.model = model
    
    # Reference designs for each IP type (using one example from each category)
    reference_designs = {
        'adder4bit': '../assets/IP-RTL-toy/adder4bit/adder4bit_1/',
        'bcdToseg': '../assets/IP-RTL-toy/bcdToseg/bcdToseg_1/',
        'encoder8to3': '../assets/IP-RTL-toy/encoder8to3/encoder8to3_1/'
    }
    
    print(f"Analyzing hardware design: {hw_design_dir_path}")
    print(f"Using pre-trained model from: {pretrained_model_path}")
    print("-" * 60)
    
    predicted_ip, similarities, graph_embed = detect_ip(cfg, hw_design_dir_path, reference_designs)
    
    print(f"\nPredicted IP Type: {predicted_ip}")
    print(f"\nSimilarity scores:")
    for ip_type, similarity in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ip_type:15s}: {similarity:.4f}")
    
    print(f"\nGraph embedding shape: {graph_embed.shape}")
    # print(f"Graph embedding: {graph_embed.cpu().numpy()}")
