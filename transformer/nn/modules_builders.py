from allennlp.common.checks import ConfigurationError
from typing import Tuple

from fairseq.models.transformer import (TransformerEncoder,
                                        TransformerDecoder,
                                        TransformerModel,
                                        Embedding,
                                        base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de
                                        )


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)

    if path:
        raise ConfigurationError("Pretrained embeddings are not implemented yet")
    # if provided, load from preloaded dictionaries
    # if path:
    #    embed_dict = utils.parse_embedding(path)
    #    utils.load_embedding(embed_dict, dictionary, emb)
    return emb


def build_transformer_encoder_and_decoder(args, src_dict, tgt_dict) -> Tuple[TransformerEncoder, TransformerDecoder]:
    """Build a new model instance."""

    # make sure all arguments are present in older models
    base_architecture(args)  # does not override any arguments

    if not hasattr(args, 'max_source_positions'):
        args.max_source_positions = 1024
    if not hasattr(args, 'max_target_positions'):
        args.max_target_positions = 1024

    if args.share_all_embeddings:
        if src_dict != tgt_dict:
            raise ValueError('--share-all-embeddings requires a joined dictionary')
        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError(
                '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
        if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path):
            raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = encoder_embed_tokens
        args.share_decoder_input_output_embed = True
    else:
        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=False)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=False)

    return encoder, decoder
