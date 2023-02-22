from collections import namedtuple

_CodexSetting = namedtuple("CodexSetting",
                           ["repeat_count", "d_model", "num_heads", "ffn_hidden_size", "dropout"])


class CodexSetting(_CodexSetting):

    @staticmethod
    def default_setting():
        return CodexSetting(repeat_count=6, d_model=512, num_heads=8, ffn_hidden_size=2048,
                            dropout=0.1)
