import torch
import mpmath

def uniform(size:torch.Size,
            min_val:float=-1.0, 
            max_val:float=1.0) -> torch.Tensor:
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

def ring(size:torch.Size) -> torch.Tensor:
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

def sign(size:torch.Size) -> torch.Tensor:
    
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
    num_digits = size.numel() + 2
    mpmath.mp.dps = num_digits
    pi_str = mpmath.nstr(mpmath.pi, num_digits)[2:num_digits]
    pi_matrix = torch.tensor([1 if int(digit) >= 5 else -1 for digit in pi_str], dtype=torch.float32)

    return pi_matrix.reshape(size)

