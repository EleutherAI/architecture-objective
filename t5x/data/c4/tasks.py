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

from t5x.data.c4 import c4_utils
from t5x.data.utils import default_mlm_task, default_clm_task

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye

c4_files = c4_utils.get_c4_files()

default_mlm_task('c4_eye_span_corruption', c4_files, jsonl=False)
default_clm_task('c4_eye_full_lm', c4_files, jsonl=False)