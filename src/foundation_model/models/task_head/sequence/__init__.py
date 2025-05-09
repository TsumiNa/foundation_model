"""
Sequence task heads for the FlexibleMultiTaskModel.
"""

from ...model_config import SequenceTaskConfig
from .base import SequenceBaseHead
from .fixed_vec import SequenceHeadFixedVec
from .rnn import SequenceHeadRNN
from .tcn_film import SequenceHeadTCNFiLM


def create_sequence_head(d_in: int, name: str, config: SequenceTaskConfig) -> SequenceBaseHead:
    """
    Factory function to create a sequence head based on configuration.

    Parameters
    ----------
    d_in : int
        Dimension of the latent input.
    name : str
        Name of the sequence task.
    config : SequenceTaskConfig
        Configuration for the sequence task.

    Returns
    -------
    SequenceBaseHead
        The initialized sequence head.

    Raises
    ------
    ValueError
        If the requested sequence head type is not supported.
    """
    subtype = config.subtype.lower()

    if subtype == "rnn":
        return SequenceHeadRNN(
            d_in=d_in,
            name=name,
            hidden=config.hidden,
            cell=config.cell,
        )
    elif subtype == "vec":
        if config.seq_len is None:
            raise ValueError("seq_len must be provided for 'vec' sequence head")
        return SequenceHeadFixedVec(
            d_in=d_in,
            name=name,
            seq_len=config.seq_len,
        )
    elif subtype == "tcn":
        return SequenceHeadTCNFiLM(
            d_in=d_in,
            name=name,
            hidden=config.hidden,
            n_layers=config.n_tcn_layers,
        )
    else:
        raise ValueError(f"Unsupported sequence head type: {subtype}")
