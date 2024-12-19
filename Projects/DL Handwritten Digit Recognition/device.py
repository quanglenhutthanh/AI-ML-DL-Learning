import torch

def get_device():
    if torch.cuda.is_available:
        print('cuda is available')
        return 'cuda'
    elif torch.backends.mps.is_available:
        print('mps is available')
        return 'mps'
    else:
        print('using cpu')
        return 'cpu'

device = get_device()
print(device)
