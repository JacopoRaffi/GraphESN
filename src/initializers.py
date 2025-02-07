import torch
from math import pi

#TODO: Implement the rescaling of the weights matrices

def uniform(size:torch.Size,
            min_val:float, 
            max_val:float) -> torch.Tensor:
    '''
    Initialize a tensor with uniform distribution.

    Parameters
    ----------
    size : torch.Size
        The size of the tensor
    min_val : float
        The minimum value of the uniform distribution
    max_val : float
        The maximum value of the uniform distribution
    
    Returns
    -------
    torch.Tensor
        The initialized tensor
    '''

    return torch.empty(size).uniform_(min_val, max_val)

def ring(size:torch.Size):
    '''
    Initialize a tensor with the strategies explained in the paper:
        C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
        In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
        https://doi.org/10.1109/IJCNN48605.2020.9206723.

    Parameters
    ----------
    size : torch.Size
        The size of the tensor
    
    Returns
    -------
    torch.Tensor
        The initialized tensor
    '''

    eye = torch.eye(size[0])

    return eye.roll(1, 0)

def sign(size:torch.Size,
         value:float) -> torch.Tensor:
    
    '''
    Initialize a tensor with the strategies explained in the paper:
        C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
        In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
        https://doi.org/10.1109/IJCNN48605.2020.9206723.

    Parameters
    ----------
    size : torch.Size
        The size of the tensor
    value : float
        The value to initialize the tensor

    Returns
    -------
    torch.Tensor
        The initialized tensor
    '''
    
    pi_str = str(torch.tensor(pi).item())[2:2+(size[0]*size[1])]
    pi_matrix = torch.tensor([1 if int(digit) >= 5 else -1 for digit in pi_str])

    return pi_matrix.reshape(size)


if __name__ == '__main__':
    print(sign(torch.Size([3, 3]), 8))

