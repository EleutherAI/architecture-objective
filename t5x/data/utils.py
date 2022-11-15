# Copyright 2022

"""Utilities for data loading and processing."""

import functools
import os
import gin
import seqio

import t5.data.utils
import tensorflow as tf
from typing import Iterable, Mapping, Optional, Union
from t5x.data.vocab import DEFAULT_OUTPUT_FEATURES, DEFAULT_CLM_OUTPUT_FEATURES

from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

@t5.data.utils.map_over_dataset
def extract_text_from_json_tf(json: str):
    output = tf.strings.split(json, '{"text":"', maxsplit=1)[1]
    output = tf.strings.split(output, '",', maxsplit=1)[0]
    return {"text": output}

@t5.data.utils.map_over_dataset
def extract_text_from_jsonl_tf(json: str):
    output = tf.strings.split(json, '{"text": "', maxsplit=1)[1]
    output = tf.strings.split(output, '",', maxsplit=1)[0]
    return {"text": output}


def default_mlm_task(name, split_to_filepattern, jsonl):
    extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
    TaskRegistry.add(
        name,
        source=CustomDataSource(
            split_to_filepattern=split_to_filepattern,
        ),
        preprocessors=[
            extract_text,
            functools.partial(
                preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.span_corruption,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]
    )


def default_clm_task(name, split_to_filepattern, jsonl):
    extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
    TaskRegistry.add(
        name,
        source=CustomDataSource(
            split_to_filepattern=split_to_filepattern,
        ),
        preprocessors=[
            extract_text,
            functools.partial(
                preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.full_lm,
        ],
        output_features=DEFAULT_CLM_OUTPUT_FEATURES,
        metric_fns=[]
    )


class CustomDataSource(seqio.FileDataSource):
    """A `FileDataSource` that reads lines of text from a file as input and takes in _TFDS_DATA_DIR_OVERRIDE"""

    def __init__(self,
                split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
                skip_header_lines: int = 0,
                num_input_examples: Optional[Mapping[str, int]] = None,
                caching_permitted: bool = True,
                file_shuffle_buffer_size: Optional[int] = None):
        """TextLineDataSource constructor.
        Args:
        split_to_filepattern: a mapping from split names to filepatterns to be
            expanded with glob.
        skip_header_lines: int, number of header lines to skip in each source
            file.
        num_input_examples: dict or None, an optional dictionary mapping split to
            its size in number of input examples (before preprocessing). The
            `num_input_examples` method will return None if not provided.
        caching_permitted: indicates whether this data source may be cached.
            Default True.
        file_shuffle_buffer_size: The buffer size to shuffle files when needed. If
            None, the number of files is used as buffer size for a perfect shuffle
            (default and recommended). A value of 16 may be explicitly set to
            replicate earlier behavior.
        """
        # Used during caching.
        self._data_dir = seqio.utils._TFDS_DATA_DIR_OVERRIDE
        if self._data_dir is None:
            self._data_dir = ""
        self._skip_header_lines = skip_header_lines

        def read_file_fn(filepattern):
            return tf.data.TextLineDataset(filepattern).skip(skip_header_lines)


        super().__init__(
            read_file_fn=read_file_fn,
            split_to_filepattern={
                k: [os.path.join(self._data_dir, _v) for _v in v] \
                        if type(v) == list else os.path.join(self._data_dir, v) \
                    for k,v in split_to_filepattern.items()
                    },
            num_input_examples=num_input_examples,
            caching_permitted=caching_permitted,
            file_shuffle_buffer_size=file_shuffle_buffer_size)

