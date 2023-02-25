import torch

import transformer.encoder_decoder
from transformer.utils import subsequent_mask
from transformer.utils import pad_mask

SRC_VOCAB_SZ = 1000
TGT_VOCAB_SZ = 2000


def dummy_model() -> transformer.encoder_decoder.EncoderDecoder:
    # just create a un-trained dummy model, you can replace this with trained model
    return transformer.encoder_decoder.EncoderDecoder(SRC_VOCAB_SZ, TGT_VOCAB_SZ)


PAD_TOK = 0
BOS_TOK = 1


def get_input() -> torch.Tensor:
    # We could have many sentences, but 2 is enough to be a batch
    # the value in the sentence is the index of the word in the vocabulary
    sentence_0 = torch.LongTensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    sentence_1 = torch.LongTensor([21, 22, 23, 24, 25, 26, 27, 28])
    sentences = [sentence_0, sentence_1]

    # to get better throughput, we batch these sentences into one tensor

    # but first, we need to pad them to the same length
    max_seq_len = max([len(sentence) for sentence in sentences])
    for i, sentence in enumerate(sentences):
        sentences[i] = torch.cat([sentence, torch.LongTensor([PAD_TOK] * (max_seq_len - len(sentence)))])

    # now we can batch them together
    return torch.stack(sentences, dim=0)


def get_initial_tgt(batch_size: int, bos=BOS_TOK) -> torch.Tensor:
    # we need to get a [batch_size,1] tensor to start the decoding
    return torch.LongTensor([[bos] * batch_size]).transpose(0, 1)


def inference():
    model = dummy_model()
    model.eval()
    src = get_input()
    tgt = greedy_decode(model, src)
    print("Final output: ", tgt)


def greedy_decode(model, src, max_output_len=16, bos=BOS_TOK):
    src_mask = pad_mask.pad_mask(src, PAD_TOK)
    context = model.encode(src, src_mask)
    batch_sz = context.size(0)
    tgt = get_initial_tgt(batch_sz, bos)
    for i in range(max_output_len):
        current_seq_len = tgt.size(1)
        assert current_seq_len == i + 1
        prob = model.decode(
            tgt=tgt,
            tgt_mask=subsequent_mask.subsequent_mask(current_seq_len) & pad_mask.pad_mask(tgt, PAD_TOK),
            context=context,
            src_mask=src_mask)
        _, next_word = torch.max(prob[:, -1, :], dim=1, keepdim=True)
        tgt = torch.cat([tgt, next_word], dim=1)
    # now we can convert tgt back to sentences, note that we may need to remove the BOS token and the padding token
    return tgt


if __name__ == '__main__':
    inference()
