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

from t5x.data.pile import pile_utils
from t5x.data.utils import default_mlm_task, default_clm_task

# ==================================== The Pile ======================================

PILE_SIZES = [(None, None), (20, '2b'), (5, '500m'), (1, '100m')]


def pile_helper(task, name, pile_files, pile_size_name):
    if pile_size_name:
        name += f"_{pile_size_name}"
    task(name, pile_files, jsonl=True)

for num_files, pile_size_name in PILE_SIZES:
    if num_files is None:
        pile_files = pile_utils.get_pile_files()
    else:
        pile_files = pile_utils.get_minipile_files(num_files)
    args = (pile_files, pile_size_name)
    
    name = 'pile_span_corruption'
    pile_helper(default_mlm_task, name, *args)

    name = 'pile_full_lm'
    pile_helper(default_clm_task, name, *args)