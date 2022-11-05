import functools

import seqio

import t5x.data.vocab
import t5x.data.utils
import t5x.data.c4_utils

from flan import tasks as flan_tasks
from flan import utils as flan_utils
from flan import templates as flan_templates
from flan import preprocessors as flan_preprocessors

from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

DEFAULT_OUTPUT_FEATURES = {mode: 
    {
        "inputs": 
            seqio.Feature(
                vocabulary=t5x.data.vocab.get_default_vocabulary(mode),
                add_eos=True,
                required=False),
        "targets":
            seqio.Feature(
                vocabulary=t5x.data.vocab.get_default_vocabulary(mode),
                add_eos=True)
    } for mode in ('gpu', 'tpu')
}

VOCAB_PATHS = [('gpu', None), ('tpu', 'tpu')]

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye

C4_SIZES = [(1022, None), (105, '16b'), (13, '2b'), (3, '500m')]

for num_files, c4_size_name in C4_SIZES:
    for mode, mode_name in VOCAB_PATHS:
        name = 'c4_eye_span_corruption'
        if c4_size_name:
            name += f"_{c4_size_name}"
        if mode_name:
            name += f"_{mode_name}"
        print(name)
        try:
            TaskRegistry.add(
                name,
                source=t5x.data.utils.CustomDataSource(
                    split_to_filepattern=t5x.data.c4_utils.get_c4_files(num_files),
                ),
                preprocessors=[
                    t5x.data.utils.extract_text_from_json_tf,
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
                output_features=DEFAULT_OUTPUT_FEATURES[mode],
                metric_fns=[]
            )
        except Exception:
            pass


# ==================================== Super GLUE ======================================
# Adapted from FLAN

SGLUE_LIST = ['rte', 'wsc', 'wic', 'record', 'multirc', 'copa', 'cb']
SGLUE_SUBSET = []
for task_name in SGLUE_LIST:
    config = flan_tasks.TASK_CONFIGS[task_name]
    flan_name = flan_utils.t_name_to_flan_pattern_name(task_name)
    for idx, pattern in enumerate(flan_templates.PATTERNS[flan_name]):
        inputs_pattern, targets_pattern = pattern
        # task_and_id_name = flan_utils.ZeroshotEvalTaskName.get(task_name, idx)
        task_and_id_name = "{}_prompt_{}".format(task_name, idx)
        SGLUE_SUBSET.append(task_and_id_name)
        for mode, mode_name in VOCAB_PATHS:
            _task_and_id_name = task_and_id_name + f"_{mode_name}" if mode_name else task_and_id_name
            try:
                TaskRegistry.add(
                    _task_and_id_name,
                    source=config.source,
                    preprocessors=config.preprocessors + 
                        flan_preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
                        [
                            seqio.preprocessors.tokenize,
                            seqio.CacheDatasetPlaceholder(),
                            seqio.preprocessors.append_eos_after_trim,
                        ],
                    postprocess_fn=config.postprocess_fn,
                    output_features=DEFAULT_OUTPUT_FEATURES[mode],
                    metric_fns=config.metric_fns
                )
            except Exception:
                pass

MixtureRegistry.add(
  name="sglue_flan_style",
  tasks=SGLUE_SUBSET,
  default_rate=functools.partial(seqio.mixing_rate_num_examples) #, maximum=3000)
  )
