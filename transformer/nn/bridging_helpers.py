from typing import Tuple, Any
import torch
from allennlp.data import Vocabulary
from allennlp.nn import util
from typing import Dict

from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from fairseq.models.fairseq_model import FairseqModel
from fairseq.sequence_generator import SequenceGenerator


"""
Stub classes before wrap allennlp's entities and required minimum of methods
for fairseq's transformer model to work

"""


class Args:
    """Stub class for arguments"""
    def __init__(self):
        pass


class Dictionary:
    """Stub class for dictionary (vocab)"""
    def __init__(self, vocab: Vocabulary, namespace: str,
                 eos: str = '@end@',
                 pad: str = '@@PADDING@@',
                 unk: str = '@@UNKNOWN@@'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos

        self.pad_index = vocab.get_token_index(pad, namespace)
        self.eos_index = vocab.get_token_index(eos, namespace)
        self.unk_index = vocab.get_token_index(unk, namespace)

        self._vocab_size = vocab.get_vocab_size(namespace)

        self._namespace = namespace

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def __len__(self):
        return self._vocab_size

    def __eq__(self, other):
        return self._namespace == other.get_namespace()

    def get_namespace(self):
        return self._namespace


class BeamSearchSequenceGenerator:
    def __init__(self,
                 encoder: FairseqEncoder,
                 decoder: FairseqIncrementalDecoder,
                 target_dictionary: Dictionary,
                 beam_size: int = 8,
                 maximum_decoder_steps: int = 200):
        self._generator = SequenceGenerator(models=[FairseqModel(encoder, decoder)],
                                            tgt_dict=target_dictionary,
                                            beam_size=beam_size,
                                            maxlen=maximum_decoder_steps,
                                            stop_early=True)

    def generate(self, source_tokens: Dict[str, torch.Tensor], beam_size=None):
        encoder_input = prepare_encoder_input(source_tokens)
        list_of_dicts = self._generator.generate(encoder_input, beam_size=beam_size)
        best_predictions = [d[0]["tokens"].detach().cpu().numpy() for d in list_of_dicts]
        return best_predictions


def run_encoder(encoder: FairseqEncoder, source_tokens: Dict[str, torch.Tensor]):
    encoder_input = prepare_encoder_input(source_tokens)
    return encoder(**encoder_input)


def run_decoder(decoder: FairseqIncrementalDecoder, target_tokens: Dict[str, torch.Tensor], encoder_output: Tuple):
    decoder_input = prepare_decoder_input(target_tokens, encoder_output)
    return decoder(**decoder_input)


def prepare_encoder_input(source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    padding_mask = util.get_text_field_mask(source_tokens)
    source_tokens = source_tokens["tokens"]
    source_tokens, padding_mask = remove_eos_from_the_beginning(source_tokens, padding_mask)
    lengths = util.get_lengths_from_binary_sequence_mask(padding_mask)
    return {"src_tokens": source_tokens, "src_lengths": lengths}


def prepare_decoder_input(target_tokens: Dict[str, torch.Tensor], encoder_output: Dict[str, torch.Tensor]):
    target_tokens = target_tokens["tokens"]
    return {"prev_output_tokens": target_tokens, "encoder_out": encoder_output}


def remove_eos_from_the_beginning(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Our source sentences does not need EOS at the beginning while dataset reader appends it.
    """

    return tensor.clone()[:, 1:], mask.clone()[:, 1:]


def move_eos_to_the_end(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assumes EOS is in the beginning. Useful to turn sequence that is an input for teacher forcing (EOS, w1, w2)
    to sequence that is suitable to compute loss (w1, w2, EOS). Takes padding into account.
    """
    batch_size = tensor.size(0)
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    eos_id = tensor[0][0]  # eos is the first symbol in all sequences
    tensor_without_eos, _ = remove_eos_from_the_beginning(tensor, mask)
    padding_column = tensor_without_eos.new_zeros(batch_size, 1).long()
    tensor_with_eos_at_the_end = torch.cat([tensor_without_eos, padding_column], dim=1)
    for i, j in zip(range(batch_size), sequence_lengths):
        tensor_with_eos_at_the_end[i, j - 1] = eos_id

    return tensor_with_eos_at_the_end


def move_eos_to_the_beginning(tensor: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    """
    Mask stays the same
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()

    tensor_with_eos_at_the_beginning = tensor.zeros_like(tensor)
    for i, j in enumerate(sequence_lengths):
        if j > 1:
            tensor_with_eos_at_the_beginning[i, 0] = tensor[i, (j - 1)]
            tensor_with_eos_at_the_beginning[i, 1:j] = tensor[i, :(j - 1)]

    return tensor_with_eos_at_the_beginning


def add_eos_to_the_beginning(tensor: torch.Tensor, eos_index: int):
    eos_column = tensor.new_full((tensor.size(0), 1), eos_index)
    return torch.cat([eos_column, torch.tensor], dim=1)
