from managed import ManagedTensor as mt

# Can be put on GPU!!
def solve_pde(x):
    """
    Solves the PDE for the given x.
    """
    neighbors = get_neighbors(x)
    # Goes to GPU device with least memory usage
    pde_matrix = construct_pde_matrix(x, neighbors).as_subclass(mt).cuda(disperse=True)
    # Same device as pde_matrix
    x = x.as_subclass(mt).cuda(pde_matrix.device)
    # That's it! No need to change anything else
    solution = torch.linalg.solve(pde_matrix, x)
    ...
    return solution