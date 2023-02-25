import torch


def pad_mask(value: torch.Tensor, pad_value: int) -> torch.Tensor:
    """
    value has shape (batch_size, seq_len)
    return has shape (batch_size, 1, seq_len)

    """
    """Mask out all padding token"""
    assert value.dim() == 2, "Input must have 2 dimensions, i.e, have a shape of (batch_size, seq_len)"
    # it will be used in a boradcasted way, so we need to add a dimension to fit
    # the shape of (batch_size, 1, seq_len) and (batch_size, y_seq_len , seq_len)
    return value.ne(pad_value).unsqueeze(1).requires_grad_(False).clone().detach()
