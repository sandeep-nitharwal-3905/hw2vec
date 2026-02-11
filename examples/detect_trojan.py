import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *
import torch.nn.functional as F


def detect_trojan(cfg, hw_design_dir_path, trained_model_path):
    """
    Detect if a hardware design contains a trojan.
    
    Args:
        cfg: Configuration object
        hw_design_dir_path: Path to the hardware design directory
        trained_model_path: Path to the trained model directory
        
    Returns:
        prediction: 0 for NON_TROJAN, 1 for TROJAN
        confidence: Probability of the prediction
        graph_embed: Graph embedding vector
    """
    # Convert hardware design to graph
    hw2graph = HW2GRAPH(cfg)
    hw_design_path = hw2graph.preprocess(hw_design_dir_path)
    hardware_nxgraph = hw2graph.process(hw_design_path)

    # Process graph data
    data_proc = DataProcessor(cfg)
    data_proc.process(hardware_nxgraph)
    data_loader = DataLoader(data_proc.get_graphs(), batch_size=1)

    # Load trained model
    model = GRAPH2VEC(cfg)
    model_path = Path(trained_model_path)
    model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))
    model.to(cfg.device)
    model.eval()

    # Get graph data and make prediction
    graph_data = next(iter(data_loader)).to(cfg.device)
    
    with torch.no_grad():
        graph_embed, _ = model.embed_graph(graph_data.x, graph_data.edge_index, graph_data.batch)
        output = model.mlp(graph_embed)
        output = F.log_softmax(output, dim=1)
        
        # Get prediction and confidence
        probabilities = torch.exp(output)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, graph_embed


if __name__ == '__main__': 
    import argparse
    
    # Parse the design path separately
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('design_path', nargs='?', default="../assets/TJ-RTL-toy/TjFree/det_1011/",
                        help='Path to the hardware design directory')
    args, remaining = parser.parse_known_args()
    
    cfg = Config(remaining)
    
    hw_design_dir_path = Path(args.design_path)
    trained_model_path = Path("./my_trained_model")
    
    if not trained_model_path.exists():
        print(f"Error: Trained model not found at {trained_model_path}")
        print("Please train a model first using use_case_2.py with --model_path argument")
        sys.exit(1)
    
    if not hw_design_dir_path.exists():
        print(f"Error: Hardware design not found at {hw_design_dir_path}")
        sys.exit(1)
    
    print(f"Analyzing hardware design: {hw_design_dir_path}")
    print(f"Using trained model from: {trained_model_path}")
    print("-" * 60)
    
    prediction, confidence, graph_embed = detect_trojan(cfg, hw_design_dir_path, trained_model_path)
    
    print(f"\nPrediction: {'TROJAN DETECTED' if prediction == 1 else 'NO TROJAN (Clean)'}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nGraph embedding shape: {graph_embed.shape}")
    # print(f"Graph embedding: {graph_embed.cpu().numpy()}")
