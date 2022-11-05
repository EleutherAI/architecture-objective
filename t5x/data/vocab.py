# Copyright 2022

"""Defines the vocabulary"""
import seqio


def get_default_vocabulary(mode='gpu'):
  if mode == 'tpu':
    spm_path = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
    extra_ids = 100
  elif mode == 'gpu':
    spm_path = "/fsx/lintangsutawika/t5-tokenizer/spiece.model"
    extra_ids = 0
  else:
    raise NotImplementedError
  return seqio.SentencePieceVocabulary(spm_path, extra_ids)