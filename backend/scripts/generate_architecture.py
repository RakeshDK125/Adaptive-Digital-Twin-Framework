import matplotlib.pyplot as plt
import networkx as nx
import os

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

G = nx.DiGraph()

# Add nodes with specific labels
nodes = {
    "Physical": "Physical Industrial\nMachine",
    "Sync": "Digital Twin\nState Synchronizer",
    "ADWIN": "Concept Drift\nDetector (ADWIN)",
    "PID": "Baseline PID\nController",
    "MetaRL": "Meta-RL\nEngine",
    "KG": "Knowledge Graph\nReasoner (Neo4j)",
    "Swarm": "Multi-Agent\nSwarm (RabbitMQ)",
}

for k, v in nodes.items():
    G.add_node(k, label=v)

# Define exact positions for a flowchart look
pos = {
    "Physical": (0, 0.5),
    "Sync": (0.25, 0.5),
    "ADWIN": (0.5, 0.5),
    "PID": (0.5, 0.2),
    "MetaRL": (0.75, 0.8),
    "KG": (0.75, 0.5),
    "Swarm": (0.75, 0.2),
}

# Add edges with labels
edges = [
    ("Physical", "Sync", "Telemetry Data"),
    ("Sync", "ADWIN", "State Updates"),
    ("ADWIN", "PID", "Normal State"),
    ("ADWIN", "MetaRL", "Drift Detected"),
    ("MetaRL", "KG", "Few-Shot Adaptation"),
    ("KG", "Swarm", "Topological Insights"),
    ("Swarm", "Physical", "Self-Healing Commands"),
    ("PID", "Physical", "Standard Commands"),
]

for edge in edges:
    G.add_edge(edge[0], edge[1], label=edge[2])

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_size=6000, node_color='lightblue', edgecolors='black', linewidths=2, node_shape='s')
nx.draw_networkx_edges(G, pos, node_size=6000, arrowstyle='->', arrowsize=25, edge_color='gray', width=2.5, connectionstyle='arc3,rad=0.1')

# Add node labels
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_family='sans-serif')

# Add edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
# Adjust edge label positions slightly so they don't overlap lines too much
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkred', label_pos=0.5, bbox=dict(alpha=0))

plt.title("System Architecture: Adaptive Digital Twin", fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("plots/system_architecture.png", dpi=600, bbox_inches='tight')
plt.close()

print("Architecture diagram generated successfully!")
