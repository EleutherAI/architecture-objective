"""
To cache tasks before training,

seqio_cache_tasks \
    --tasks=my_task_*,your_task \
    --excluded_tasks=my_task_5 \
    --output_cache_dir=/path/to/cache_dir \
    --module_import=my.tasks \
    --alsologtostderr

For more details, see: seqio/scripts/cache_tasks_main.py

"""

import seqio
import functools

import t5.data.preprocessors

import t5x.data.preprocessors
from t5x.data.vocab import DEFAULT_OUTPUT_FEATURES
from t5x.data.utils import CustomDataSource, extract_text_from_json_tf
from t5x.data.c4 import c4_utils
from t5x.data.utils import default_mlm_task, default_clm_task

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye
TaskRegistry.add(
    '_c4_eye_span_corruption',
    source=CustomDataSource(
        split_to_filepattern=c4_utils.get_c4_files(),
    ),
    preprocessors=[
        extract_text_from_json_tf,
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)

TaskRegistry.add(
    '_c4_fcm_causal_decoder_architecture',
    source=CustomDataSource(
        split_to_filepattern=c4_utils.get_c4_files(),
    ),
    preprocessors=[
        extract_text_from_json_tf,
        functools.partial(
            seqio.preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5x.data.preprocessors.masked_language_modeling,
        seqio.preprocessors.append_eos_after_trim,
        t5x.data.preprocessors.pack_lm_decoder_only,
    ],
    output_features={
        "decoder_target_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_input_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_loss_weights": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_causal_attention": seqio.Feature(vocabulary=vocab, add_eos=False),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=vocab, required=False),
    },
    metric_fns=[]
)

c4_files = c4_utils.get_c4_files()

default_mlm_task('c4_eye_span_corruption', c4_files, jsonl=False)
default_clm_task('c4_eye_full_lm', c4_files, jsonl=False)
