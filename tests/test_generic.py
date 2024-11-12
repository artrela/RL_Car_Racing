import pytest
import torch


@pytest.mark.gpu
def test_gpu():
    print("Your GPU is not accessable, this will need to be resolved to make local testing feasible!")
    assert torch.cuda.is_available()