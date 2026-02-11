import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.hw2graph import *
from hw2vec.config import Config
from networkx.drawing.nx_pydot import to_pydot

if __name__ == '__main__': 
    # Create config with proper arguments
    # Use the example YAML file and specify graph type
    cfg = Config([
        '--yaml_path', './example_gnn4tj.yaml',
        '--graph_type', 'DFG'
    ])
    
    # Use existing test_module directory as input
    test_dir = Path("test_module").resolve()
    
    # Process the hardware design
    hw2graph = HW2GRAPH(cfg)
    hw_design_path = hw2graph.preprocess(test_dir)
    hardware_nxgraph = hw2graph.process(str(hw_design_path))
    
    print("Successfully converted RTL to DFG!")
    print(f"Number of nodes: {hardware_nxgraph.number_of_nodes()}")
    print(f"Number of edges: {hardware_nxgraph.number_of_edges()}")
    print(f"\nNodes in the DFG:")
    for node in hardware_nxgraph.nodes(data=True):
        print(f"  {node}")
    
    print(f"\nEdges in the DFG:")
    for edge in hardware_nxgraph.edges(data=True):
        print(f"  {edge}")
    
    # Export to DOT file for visualization
    dot_str = to_pydot(hardware_nxgraph).to_string()
    dot_path = Path("test_dfg_graph.dot")
    dot_path.write_text(dot_str, encoding="utf-8")
    print(f"\nDFG saved to: {dot_path.resolve()}")
