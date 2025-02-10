from torch_geometric.nn import MessagePassing
import torch_geometric.nn
import torch.nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier

import initializers

#TODO: after that check the GES property (Graph Embedding Stability)

class GraphReservoir(MessagePassing):
    def __init__(self, 
                 input_size:int, 
                 hidden_size:int,
                 rho:float=0.9,
                 omhega:float=1.0,
                 input_initializer:str='uniform',
                 recurrent_initializer:str='uniform',
                 ):
        '''
        Reservoir Layer for the Graph Echo State Network

        Parameters
        ----------
        input_size : int
            The number of input features
        hidden_size : int
            The number of hidden units
        rho : float, optional
            The spectral radius of the recurrent weight matrix, by default 0.9
        input_initializer : str, optional
            The initializer for the input weight matrix. It can be 'uniform', 'ring', 'sign', by default 'uniform'
        '''
        super(GraphReservoir, self).__init__(aggr='add')
        
        self.input_size = torch.nn.Parameter(torch.tensor(input_size), requires_grad=False)
        self.hidden_size = torch.nn.Parameter(torch.tensor(hidden_size), requires_grad=False)

        input_initializer = getattr(initializers, input_initializer)
        recurrent_initializer = getattr(initializers, recurrent_initializer)

        self.W_in = torch.nn.Parameter(input_initializer(torch.Size([hidden_size, input_size])), requires_grad=False)
        self.W_h = torch.nn.Parameter(recurrent_initializer(torch.Size([hidden_size, hidden_size])), requires_grad=False)

        # scale the weight matrices
        self.W_in.mul_(omhega).float() 
        self.W_h.div_(torch.linalg.eigvals(self.W_h).abs().max()).mul_(rho).float()


    
    @torch.no_grad()
    def forward(self, x:torch.Tensor, edge_index:torch.Tensor, x_neighbors:torch.Tensor = None, threshold:float=1e-3, max_steps:int=1000) -> torch.Tensor:
        '''
        Forward pass of the Graph Reservoir Network. The formula for the forward pass is given by:
        x = tanh(W_in @ u_in + tanh(W_h @ sum_{j \in N(i)} x_j))
        where:
            x : node embeddings
            u_in : input node features
            W_in : input weight matrix
            W_h : recurrent weight matrix
            N(i) : set of neighbours of node i
            x_j : node embedding of the j-th neighbour

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [N, input_size] where N is the number of nodes in the graph
        edge_index : torch.Tensor
            The edge index tensor of shape [2, E] where E is the number of edges in the graph
        x_neighbors : torch.Tensor, optional
            The initial node embeddings of the neighbours, by default None
        threshold : float, optional
            The threshold for the convergence of the node embeddings, by default 1e-3
        max_steps : int, optional
            The maximum number of steps for the convergence of the node embeddings, by default 100
        
        Returns
        -------
        torch.Tensor
            The node embeddings of the graph of shape [N, hidden_size]
        '''
        u_in = x.clone()
        x_old = torch.zeros(x.shape[0], self.hidden_size) if x_neighbors is None else x_neighbors
        norm = float('inf')

        while (norm > threshold) and (max_steps > 0):
            x = F.linear(x_old, self.W_h) # apply recurrent matrix to the node embeddings
            neighbors = self.propagate(edge_index, x=x) # message passing phase

            x = F.tanh(F.linear(u_in, self.W_in) + neighbors) # update the node embeddings

            norm = torch.norm(x - x_old, p=2)
            x_old = x.clone()
            max_steps -= 1

        return x


# Simple GraphESN classifier with just one reservoir layers
class GraphESN(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:int, 
                 rho:float=0.9, omhega:float=1.0, input_initializer:str='uniform', recurrent_initializer:str='uniform',
                 tikhonov:float=1e-6):
        
        '''
        Graph Echo State Network classifier model

        Parameters
        ----------
        input_size : int
            The number of input features
        hidden_size : int
            The number of hidden units in the reservoir
        rho : float, optional
            The spectral radius of the recurrent weight matrix, by default 0.9
        omhega : float, optional
            The scale of the input weight matrix, by default 1.0
        input_initializer : str, optional
            The initializer for the input weight matrix. It can be 'uniform', 'ring', 'sign', by default 'uniform'
        recurrent_initializer : str, optional
            The initializer for the recurrent weight matrix. It can be 'uniform', 'ring', 'sign', by default 'uniform'
        tikhonov : float, optional
            The regularization parameter for the ridge regression, by default 1e-6
        '''
        
        super(GraphESN, self).__init__()

        self.reservoir = GraphReservoir(input_size, hidden_size, rho, omhega, input_initializer, recurrent_initializer)
        self.readout = RidgeClassifier(alpha=tikhonov)

    def forward(self, x, edge_index, batch=None):
        '''
        Forward pass of the GraphESN model

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [N, input_size] where N is the number of nodes in the graph
        edge_index : torch.Tensor
            The edge index tensor of shape [2, E] where E is the number of edges in the graph
        batch : torch.Tensor, optional
            The batch tensor of shape [N] where N is the number of nodes in the graph, by default None
        
        Returns
        -------
        torch.Tensor
            The predicted labels of the graph of shape [N]
        '''
        x = self.reservoir(x, edge_index)

        if batch:
            x = torch_geometric.nn.global_add_pool(x, batch)
        
        x = self.readout.predict(x)

    def fit(self, x, edge_index, batch, y):
        '''
        Fit the GraphESN model to the data

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [N, input_size] where N is the number of nodes in the graph
        edge_index : torch.Tensor
            The edge index tensor of shape [2, E] where E is the number of edges in the graph
        batch : torch.Tensor
            The batch tensor of shape [N] where N is the number of nodes in the graph
        y : torch.Tensor
            The target tensor of shape [N] where N is the number of nodes in the graph

        Returns
        -------
        None
        '''
        x = self.reservoir(x, edge_index)

        if batch:
            x = torch_geometric.nn.global_add_pool(x, batch)
        
        self.readout.fit(x, y)