import torch

# torch.cuda.get_device_name(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
