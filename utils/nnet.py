from typing import Tuple, Union
import torch
import numpy as np


def get_device(
    use_cuda: bool,
) -> Tuple[torch.device, Union[torch._C._CudaDeviceProperties, None]]:
    """retrieves information about device (CPU or CUDA)

    Args:
        use_cuda (bool): if cuda to be used

    Returns:
        Tuple[torch.device, Union[torch._C._CudaDeviceProperties, None]]: info about device
    """
    cuda_device = torch.device("cuda" if use_cuda else "cpu")
    cuda_properties = torch.cuda.get_device_properties(cuda_device) if use_cuda else []
    return cuda_device, cuda_properties


def from_gpu(data: torch.Tensor) -> np.array:
    """transfers torch variables from device to numpy arrays

    Args:
        data (torch.Tensor): torch variable

    Returns:
        np.array: exported variable
    """

    return data.cpu().detach().numpy()
