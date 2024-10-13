import torch

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available. Using cuda for computations.")
# Check if MPS is available
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using MPS for computations.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU for computations.")

# Example tensor operation to confirm
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print("Tensor on device:", x.device)

