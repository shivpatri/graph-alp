import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Insert src path to load graphalp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graphalp.label_propagation import HarmonicLabelPropagator
from graphalp.active_learning import HarmonicGreedySampler, S2Sampler, FeatPropSampler, RandomSampler
from graphalp.utils import compute_risk

# Configuration
BG_COLOR = '#ffffff'
EDGE_COLOR = '#cbd5e1'
NODE_BLUE = '#2563eb'
NODE_RED = '#e11d48'
TEXT_COLOR = '#0f172a'

SAMPLERS = {
    'Harmonic Greedy': (HarmonicGreedySampler, '#d97706', 'o', 2.0),
    'S2 Sampler': (lambda G: S2Sampler(G, fallback_sampler=RandomSampler(G)), '#8b5cf6', 's', 1.8),
    'FeatProp': (FeatPropSampler, '#0d9488', 'd', 1.8),
    'Random': (RandomSampler, '#64748b', '^', 1.2)
}

def generate_graph_1():
    # 20x20 Grid: Left vs Right (Columns)
    G = nx.grid_2d_graph(20, 20)
    mapping = {(y, x): y * 20 + x for y, x in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    pos = {}
    gt_labels = {}
    for n in G.nodes():
        x = n % 20
        y = n // 20
        pos[n] = np.array([float(x), float(y)])
        G.nodes[n]['pos'] = pos[n]
        gt_labels[n] = 0 if x < 10 else 1
        
    return G, pos, gt_labels, 0, 399, 200

def generate_graph_2():
    # 20x20 Grid: Top vs Bottom (Rows)
    G = nx.grid_2d_graph(20, 20)
    mapping = {(y, x): y * 20 + x for y, x in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    pos = {}
    gt_labels = {}
    for n in G.nodes():
        x = n % 20
        y = n // 20
        pos[n] = np.array([float(x), float(y)])
        G.nodes[n]['pos'] = pos[n]
        gt_labels[n] = 0 if y < 10 else 1
        
    return G, pos, gt_labels, 0, 399, 200

def generate_graph_3():
    # 20x20 Grid: Top-Left & Bottom-Right Circles
    G = nx.grid_2d_graph(20, 20)
    mapping = {(y, x): y * 20 + x for y, x in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    pos = {}
    gt_labels = {}
    for n in G.nodes():
        x = n % 20
        y = n // 20
        pos[n] = np.array([float(x), float(y)])
        G.nodes[n]['pos'] = pos[n]
        dist1 = np.sqrt(x*x + y*y)
        dist2 = np.sqrt((x-19)*(x-19) + (y-19)*(y-19))
        gt_labels[n] = 1 if (dist1 <= 7.2 or dist2 <= 7.2) else 0
        
    return G, pos, gt_labels, 190, 0, 200

def generate_graph_4():
    # 20x20 Grid: Bottom-Left & Top-Right Circles
    G = nx.grid_2d_graph(20, 20)
    mapping = {(y, x): y * 20 + x for y, x in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    pos = {}
    gt_labels = {}
    for n in G.nodes():
        x = n % 20
        y = n // 20
        pos[n] = np.array([float(x), float(y)])
        G.nodes[n]['pos'] = pos[n]
        dist1 = np.sqrt(x*x + (y-19)*(y-19))
        dist2 = np.sqrt((x-19)*(x-19) + y*y)
        gt_labels[n] = 1 if (dist1 <= 7.2 or dist2 <= 7.2) else 0
        
    return G, pos, gt_labels, 180, 19, 200

def generate_graph_5():
    # 40-Node Clustered Community Graph
    np.random.seed(42)
    G = nx.Graph()
    
    # 20 nodes Cluster 0, 20 nodes Cluster 1
    pos = {}
    gt_labels = {}
    for i in range(40):
        is_c1 = i >= 20
        center_x = 420.0 if is_c1 else 180.0
        center_y = 250.0
        
        angle = np.random.rand() * np.pi * 2
        dist = np.sqrt(np.random.rand()) * 80.0
        x = center_x + np.cos(angle) * dist
        y = center_y + np.sin(angle) * dist
        
        pos[i] = np.array([x / 600.0, y / 500.0]) # Scale coordinates for FeatProp
        G.add_node(i)
        G.nodes[i]['pos'] = pos[i]
        gt_labels[i] = 1 if is_c1 else 0
        
    # Generate SBM connections
    connected = False
    while not connected:
        G.remove_edges_from(list(G.edges()))
        for i in range(40):
            for j in range(i + 1, 40):
                cI = i >= 20
                cJ = j >= 20
                p = 0.28 if (cI == cJ) else 0.02
                if np.random.rand() < p:
                    G.add_edge(i, j)
        
        # Verify connectivity
        if nx.is_connected(G):
            connected = True
        else:
            # Connect components if disconnected
            components = list(nx.connected_components(G))
            for k in range(1, len(components)):
                u = np.random.choice(list(components[0]))
                v = np.random.choice(list(components[k]))
                G.add_edge(u, v)
            connected = True
            
    return G, pos, gt_labels, 0, 39, 35

def get_accuracy(pred_probs, gt):
    correct = 0
    for node, p in pred_probs.items():
        pred_label = 1 if p >= 0.5 else 0
        if pred_label == gt[node]:
            correct += 1
    return (correct / len(gt)) * 100

def draw_graph_structure(G, pos, gt_labels, s0, s1, output_path, title):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_COLOR, width=0.7, alpha=0.5)
    
    # Draw nodes based on labels
    c0_nodes = [n for n in G.nodes() if gt_labels[n] == 0]
    c1_nodes = [n for n in G.nodes() if gt_labels[n] == 1]
    
    # Standard size
    size = 20 if len(G) > 100 else 90
    
    nx.draw_networkx_nodes(G, pos, nodelist=c0_nodes, node_color=NODE_BLUE,
                           node_size=size, edgecolors='none', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=c1_nodes, node_color=NODE_RED,
                           node_size=size, edgecolors='none', ax=ax)
    
    # Draw initial seeds with a solid black border
    nx.draw_networkx_nodes(G, pos, nodelist=[s0], node_color=NODE_BLUE,
                           node_size=size*1.8, edgecolors='#000000', linewidths=2.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[s1], node_color=NODE_RED,
                           node_size=size*1.8, edgecolors='#000000', linewidths=2.0, ax=ax)
    
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)

def run_benchmark(G, gt_labels, s0, s1, max_queries):
    results = {}
    
    for name, (sampler_cls, color, marker, lw) in SAMPLERS.items():
        np.random.seed(42)
        X_labeled = [s0, s1]
        y_labeled = [gt_labels[s0], gt_labels[s1]]
        
        sampler = sampler_cls(G)
        sampler.fit(X_labeled, y_labeled)
        
        prop = HarmonicLabelPropagator(G)
        prop.fit(X_labeled, y_labeled)
        
        pred_probs = dict(zip(G.nodes(), prop.predict_probabilities(list(G.nodes()))))
        initial_risk = compute_risk(np.array([prop.f[n] for n in prop.nodes if n not in prop.labels]))
        initial_acc = get_accuracy(pred_probs, gt_labels)
        
        risks = [initial_risk]
        accuracies = [initial_acc]
        
        for step in range(1, max_queries + 1):
            next_node = sampler.sample()
            if next_node is None:
                break
                
            X_labeled.append(next_node)
            y_labeled.append(gt_labels[next_node])
            
            sampler.fit(X_labeled, y_labeled)
            prop.fit(X_labeled, y_labeled)
            
            pred_probs = dict(zip(G.nodes(), prop.predict_probabilities(list(G.nodes()))))
            current_risk = compute_risk(np.array([prop.f[n] for n in prop.nodes if n not in prop.labels]))
            current_acc = get_accuracy(pred_probs, gt_labels)
            
            risks.append(current_risk)
            accuracies.append(current_acc)
            
        results[name] = {
            'risks': risks,
            'accuracies': accuracies,
            'color': color,
            'marker': marker,
            'lw': lw
        }
        
    return results

def plot_curves(results, max_queries, risk_path, acc_path, graph_num):
    steps = np.arange(0, max_queries + 1)
    
    # 1. Plot Risk Curve
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines['bottom'].set_color('#94a3b8')
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    
    for name, res in results.items():
        # Subsample markers to avoid clutter in 200 steps
        markevery = max(1, len(steps) // 10)
        ax.plot(steps[:len(res['risks'])], res['risks'], color=res['color'],
                marker=res['marker'], markevery=markevery, linewidth=res['lw'], label=name)
        
    ax.set_title(f"Graph {graph_num}: Expected Uncertainty (Risk)", color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel("Query Step", color=TEXT_COLOR)
    ax.set_ylabel("Graph Risk", color=TEXT_COLOR)
    ax.grid(color='#e2e8f0', linestyle=':', linewidth=0.5)
    ax.legend(facecolor=BG_COLOR, edgecolor='#94a3b8', labelcolor=TEXT_COLOR)
    plt.tight_layout()
    fig.savefig(risk_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    
    # 2. Plot Accuracy Curve
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines['bottom'].set_color('#94a3b8')
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    
    for name, res in results.items():
        markevery = max(1, len(steps) // 10)
        ax.plot(steps[:len(res['accuracies'])], res['accuracies'], color=res['color'],
                marker=res['marker'], markevery=markevery, linewidth=res['lw'], label=name)
        
    ax.set_title(f"Graph {graph_num}: Prediction Accuracy", color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel("Query Step", color=TEXT_COLOR)
    ax.set_ylabel("Accuracy (%)", color=TEXT_COLOR)
    ax.grid(color='#e2e8f0', linestyle=':', linewidth=0.5)
    ax.legend(facecolor=BG_COLOR, edgecolor='#94a3b8', labelcolor=TEXT_COLOR)
    plt.tight_layout()
    fig.savefig(acc_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)

def main():
    os.makedirs('../docs/images', exist_ok=True)
    
    graphs = [
        (generate_graph_1, "Graph 1 Structure (Grid Columns)"),
        (generate_graph_2, "Graph 2 Structure (Grid Rows)"),
        (generate_graph_3, "Graph 3 Structure (Diagonal Circles)"),
        (generate_graph_4, "Graph 4 Structure (Off-diagonal Circles)"),
        (generate_graph_5, "Graph 5 Structure (Clustered SBM)")
    ]
    
    for idx, (gen_fn, name) in enumerate(graphs):
        graph_num = idx + 1
        print(f"Generating benchmarks for Graph {graph_num}...")
        G, pos, gt_labels, s0, s1, max_queries = gen_fn()
        
        # Save structure image
        struct_path = f"../docs/images/graph{graph_num}_structure.png"
        draw_graph_structure(G, pos, gt_labels, s0, s1, struct_path, name)
        
        # Run benchmarks
        results = run_benchmark(G, gt_labels, s0, s1, max_queries)
        
        # Save plots
        risk_path = f"../docs/images/graph{graph_num}_risk.png"
        acc_path = f"../docs/images/graph{graph_num}_accuracy.png"
        plot_curves(results, max_queries, risk_path, acc_path, graph_num)
        
    print("All benchmark images generated successfully!")

if __name__ == '__main__':
    main()
