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

from t5x.data import preprocessors
from t5x.data.c4 import c4_utils
from t5x.data.vocab import get_default_vocabulary
from t5x.data.utils import (
                        default_mlm_task,
                        default_clm_task,
                        extract_text_from_json_tf,
                        CustomDataSource
                        )

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye
c4_files = c4_utils.get_c4_files()

default_mlm_task('c4_eye_span_corruption', c4_files, jsonl=False)
default_clm_task('c4_eye_full_lm', c4_files, jsonl=False)

TaskRegistry = seqio.TaskRegistry

TaskRegistry.add(
    'c4_eye_fcm_full_lm',
    source=CustomDataSource(split_to_filepattern=c4_files),
    preprocessors=[
        extract_text_from_json_tf,
        functools.partial(
            seqio.preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.masked_language_modeling,
        seqio.preprocessors.append_eos_after_trim,
        preprocessors.pack_lm_decoder_only,
    ],
    output_features={
        "decoder_target_tokens": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=False),
        "decoder_input_tokens": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=False),
        "decoder_causal_attention": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=False),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=get_default_vocabulary(), required=False),
    },
    metric_fns=[]
)
