from torch_geometric.nn import MessagePassing
import torch.nn

class GraphReservoir(MessagePassing):
    def __init__(self):
        pass
    
    def forward(self, x, edge_index):
        pass

    def message(self, x_j):
        pass

    def update(self, aggr_out):
        pass


class GraphESN(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x, edge_index):
        pass