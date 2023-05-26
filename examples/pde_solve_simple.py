import torch
from concurrent.futures import ThreadPoolExecutor

# CPU Bound
def get_neighbors(x):
    """
    Returns the neighbors of each element in x.
    """
    ....
    return neighbors

# CPU Bound
def construct_pde_matrix(x, neighbors):
    """
    Constructs the PDE matrix for the given x.
    """
    ...
    return pde_matrix

# Can be put on GPU!!
def solve_pde(x):
    """
    Solves the PDE for the given x.
    """
    neighbors = get_neighbors(x)
    # Goes on the default cuda device
    pde_matrix = construct_pde_matrix(x, neighbors).cuda()
    # Same - default cuda device
    x = x.cuda() 
    solution = torch.linalg.solve(pde_matrix, x)
    ...
    return solution

# CPU Bound
executor = ThreadPoolExecutor(max_workers=4)
for _ in range(ITERATIONS):
    futures = []
    for point in tessalation:
        futures.append(executor.submit(solve_pde, point))
    for future in futures:
        ret = future.result()
        # Do something with ret