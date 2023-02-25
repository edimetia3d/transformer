import torch


def subsequent_mask(seq_len: int) -> torch.Tensor:
    """
    return has shape (1, seq_len, seq_len)
    """
    attn_shape = (seq_len, seq_len)
    # Since this mask will be only applied to the self-attention, so just return a square matrix will be fine
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # unsqueeze to fit the shape of (1, seq_len, seq_len) and (batch_size, seq_len , seq_len)
    return (subsequent_mask == 0).unsqueeze(0).requires_grad_(False).clone().detach()
