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

from t5.data import preprocessors

from t5x.data.vocab import DEFAULT_OUTPUT_FEATURES
from t5x.data.utils import CustomDataSource, extract_text_from_json_tf

from t5x.data.c4 import c4_utils

TaskRegistry = seqio.TaskRegistry

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye
TaskRegistry.add(
    'c4_eye_span_corruption',
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
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)