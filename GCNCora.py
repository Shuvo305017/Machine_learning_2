import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import networkx as nx

# Load the dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the model
model = GCN(input_dim=dataset.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero out gradients
    out = model(data)  # Forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    return loss.item()

# Testing function
def test():
    model.eval()  # Set the model to evaluation mode
    logits, accs = model(data), []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]  # Get predictions
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()  # Compute accuracy
        accs.append(acc)
    return accs

# Lists to store accuracy values
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Train and evaluate the model
for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# Plot training, validation, and testing accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, 201), val_accuracies, label='Validation Accuracy', color='green')
plt.plot(range(1, 201), test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.legend()
plt.grid()
plt.show()

# Function to visualize the graph
def visualize_graph(edge_index, node_labels=None):
    G = nx.Graph()
    edges = edge_index.t().tolist()  # Convert edge_index to list of tuples
    G.add_edges_from(edges)

    # Add node labels if provided
    if node_labels is not None:
        labels = {i: label for i, label in enumerate(node_labels.tolist())}
        nx.set_node_attributes(G, labels, "label")

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_size=10)
    
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", font_size=8)
    
    plt.title("Graph Visualization")
    plt.show()

# Visualize the graph
visualize_graph(data.edge_index, data.y)

