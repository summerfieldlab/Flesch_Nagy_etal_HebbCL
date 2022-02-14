import torch


def get_device(use_cuda):
    cuda_device = torch.device("cuda" if use_cuda else "cpu")
    cuda_properties = torch.cuda.get_device_properties(cuda_device) if use_cuda else []
    return cuda_device, cuda_properties


def from_gpu(data):
    """
    transfers data from gpu back to cpu and
    converts into Numpy format
    """

    return data.cpu().detach().numpy()
