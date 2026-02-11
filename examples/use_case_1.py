import os, sys
from pathlib import Path
from urllib.parse import quote
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.hw2graph import *
from hw2vec.config import Config
from networkx.drawing.nx_pydot import to_pydot


def use_case_1(cfg, hw_design_dir_path, pretrained_model_weight_path, pretrained_model_cfg_path):
    hw2graph = HW2GRAPH(cfg)
    hw_design_path = hw2graph.preprocess(hw_design_dir_path)
    hardware_nxgraph = hw2graph.process(hw_design_path)

    data_proc = DataProcessor(cfg)
    data_proc.process(hardware_nxgraph)
    vis_loader = DataLoader(data_proc.get_graphs(), batch_size=1)

    model = GRAPH2VEC(cfg)
    model.load_model(pretrained_model_cfg_path, pretrained_model_weight_path)
    model.to(cfg.device)
    graph_data = next(iter(vis_loader)).to(cfg.device)
    graph_embed, _ = model.embed_graph(graph_data.x, graph_data.edge_index, graph_data.batch)
    return graph_embed, hardware_nxgraph


if __name__ == '__main__': 
    cfg = Config(sys.argv[1:])
    
    hw_design_dir_path = Path("../assets/TJ-RTL-toy/TjFree/det_1011/") # Change this path to other hardware design folder path.
    pretrained_model_weight_path = "../assets/pretrained_DFG_TJ_RTL/model.pth" # Change this path to your desired pretrained model weight path.
    pretrained_model_cfg_path = "../assets/pretrained_DFG_TJ_RTL/model.cfg" # Change this path to your desired pretrained model config path.
    cfg.graph_type = "DFG" # each pretrained model is bundled with one graph type so you will need to change this to fit the used pretrained model.

    graph_emb, dfg_nx = use_case_1(cfg, hw_design_dir_path,\
                                   pretrained_model_weight_path,\
                                   pretrained_model_cfg_path)

    print("Graph embedding tensor:\n", graph_emb)
    # Export DFG to DOT and create shareable visualization link.
    dot_str = to_pydot(dfg_nx).to_string()
    dot_path = Path("dfg_graph.dot")
    dot_path.write_text(dot_str, encoding="utf-8")

    # Construct online visualization URL (Graphviz Online stores DOT after the hash component)
    graphviz_url = "https://dreampuf.github.io/GraphvizOnline/#" + quote(dot_str)

    print(f"DFG DOT saved to: {dot_path.resolve()}")
    # Write only the visualization URL to a text file (no console output of the link).
    output_file = Path("output_use_case_1.txt")
    output_file.write_text(graphviz_url + "\n", encoding="utf-8")
    print(f"Graphviz URL stored in {output_file.resolve()}.")
    # NOTE: Embedding is a vector of features; perform any downstream inspection as needed.