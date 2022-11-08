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

from t5.evaluation import metrics
from t5.data import preprocessors
from t5.data import postprocessors
from t5.data.glue_utils import (
    get_glue_postprocess_fn, 
    get_glue_text_preprocessor, 
    get_super_glue_metric, 
    get_super_glue_weight_mapping, 
    get_super_glue_weight_mapping_sentinel
)

import tensorflow_datasets as tfds

import t5x.data.vocab
import t5x.data.utils

from t5x.data import c4_utils
from t5x.data import p3_utils
# from t5x.data import sglue_utils

# from flan import tasks as flan_tasks
# from flan import utils as flan_utils
# from flan import templates as flan_templates
# from flan import preprocessors as flan_preprocessors


TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": 
        seqio.Feature(
            vocabulary=t5x.data.vocab.get_default_vocabulary(),
            add_eos=True,
            required=False),
    "targets":
        seqio.Feature(
            vocabulary=t5x.data.vocab.get_default_vocabulary(),
            add_eos=True)
}


# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye
TaskRegistry.add(
    'c4_eye_span_corruption',
    source=t5x.data.utils.CustomDataSource(
        split_to_filepattern=c4_utils.get_c4_files(),
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
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)


# ==================================== Super GLUE ======================================
# Original T5 SGLUE
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
    # We use a simplified version of WSC, defined below
    if "wsc" in b.name:
        continue
    if b.name == "axb":
        glue_preprocessors = [
            functools.partial(
                preprocessors.rekey,
                key_map={
                    "premise": "sentence1",
                    "hypothesis": "sentence2",
                    "label": "label",
                    "idx": "idx",
                }),
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ]
    else:
        glue_preprocessors = [
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ]
    TaskRegistry.add(
        "super_glue_%s_v102" % b.name,
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/%s:1.0.2" % b.name,
            splits=["test"] if b.name in ["axb", "axg"] else None),
        preprocessors=glue_preprocessors,
        metric_fns=get_super_glue_metric(b.name),
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=get_glue_postprocess_fn(b))

    # Create SuperGLUE tasks with 1 sentinel token added.
    seqio.experimental.add_task_with_sentinels(
        "super_glue_%s_v102" % b.name, num_sentinels=1
        )

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple",
    source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
    preprocessors=[
        preprocessors.definite_pronoun_resolution_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("dpr_v001_simple", num_sentinels=1)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_train",
                                           num_sentinels=1)

TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            preprocessors.wsc_simple, correct_referent_only=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)
# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_eval",
                                           num_sentinels=1)

_SUPER_GLUE_WEIGHT_MAPPING = get_super_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = get_super_glue_weight_mapping_sentinel()

_super_glue_tasks_with_weight = list(_SUPER_GLUE_WEIGHT_MAPPING.items())
_super_glue_tasks_with_weight_sentinel = list(
    _SUPER_GLUE_WEIGHT_MAPPING_SENTINEL.items())

MixtureRegistry.add(
    "super_glue_v102_proportional",
    _super_glue_tasks_with_weight
)

MixtureRegistry.add(
    "super_glue_v102_proportional_sentinel",
    _super_glue_tasks_with_weight_sentinel
)

# Adapted from FLAN
# SGLUE_LIST = ['rte', 'wsc', 'wic', 'record', 'multirc', 'copa', 'cb']
# SGLUE_SUBSET = []
# for task_name in SGLUE_LIST:
#     config = flan_tasks.TASK_CONFIGS[task_name]
#     flan_name = flan_utils.t_name_to_flan_pattern_name(task_name)
#     for idx, pattern in enumerate(flan_templates.PATTERNS[flan_name]):
#         inputs_pattern, targets_pattern = pattern

#         # task_and_id_name = flan_utils.ZeroshotEvalTaskName.get(task_name, idx)
#         task_and_id_name = "{}_prompt_{}".format(task_name, idx)
#         SGLUE_SUBSET.append(task_and_id_name)
#         TaskRegistry.add(
#             task_and_id_name,
#             source=config.source,
#             preprocessors=config.preprocessors + 
#                 flan_preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
#                 [
#                     seqio.preprocessors.tokenize,
#                     seqio.CacheDatasetPlaceholder(),
#                     seqio.preprocessors.append_eos_after_trim,
#                 ],
#             postprocess_fn=config.postprocess_fn,
#             output_features=DEFAULT_OUTPUT_FEATURES,
#             metric_fns=config.metric_fns
#     )

# MixtureRegistry.add(
#   name="sglue_flan_style",
#   tasks=SGLUE_SUBSET,
#   default_rate=functools.partial(seqio.mixing_rate_num_examples) #, maximum=3000)
#   )


# ==================================== P3 ======================================
# Adapted from T-Zero

# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture = {key: [] for key in p3_utils.t0_train}
t0_eval_mixture = {key: [] for key in p3_utils.t0_eval}
mixture_cap = {}
for dataset_name, subset_name in p3_utils.all_templates.keys:
    if (dataset_name, subset_name) not in p3_utils.all_datasets:
        p3_utils.all_templates.remove(dataset_name, subset_name)
        continue

    cap = p3_utils.get_cap(dataset_name, subset_name)
    dataset = p3_utils.all_templates.get_dataset(dataset_name, subset_name)

    for template_name in dataset.all_template_names:
        # Add train and normal eval tasks
        template = p3_utils.all_templates.get_dataset(dataset_name, subset_name)[template_name]
        task_name = "p3_"+p3_utils.get_task_name(dataset_name, subset_name, template_name)
        TaskRegistry.add(
            name=task_name,
            source=p3_utils.get_p3_source(dataset_name, subset_name, template_name),
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(required=False),
            ],
            postprocess_fn=p3_utils.maybe_get_class_id_postprocessor(template),
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=p3_utils.get_p3_metric(dataset_name, subset_name, template_name)
        )

        # # Add rank classification eval task
        # if template.answer_choices:
        #     rank_classification_preprocessor = functools.partial(
        #         t5.data.preprocessors.rank_classification,
        #         inputs_fn=lambda ex: tf.fill((len(ex["answer_choices"]),), ex["inputs"]),
        #         targets_fn=lambda ex: ex["answer_choices"],
        #         is_correct_fn=lambda ex: tf.equal(ex["answer_choices"], tf.strings.strip(ex["targets"])),
        #         weight_fn=lambda ex: 1.0,
        #     )
        #     fixed_choices = template.get_fixed_answer_choices_list()
        #     num_classes = len(fixed_choices) if fixed_choices else None
        #     seqio.TaskRegistry.add(
        #         task_name + "_score_eval",
        #         data_source,
        #         preprocessors=[rank_classification_preprocessor] + preprocessors,
        #         output_features=output_features,
        #         metric_fns=[functools.partial(t5.evaluation.metrics.rank_classification, num_classes=num_classes)],
        #         postprocess_fn=t5.data.postprocessors.rank_classification,
        #     )

        # Check that the dataset_subset_tuple is in t0_train
        for key, dataset_subset_tuples in p3_utils.t0_train.items():
            if (dataset_name, subset_name) in dataset_subset_tuples:
                t0_train_mixture[key].append(task_name)
                mixture_cap[task_name] = cap


MixtureRegistry.add(
    "t0_train",
    [task for task in t0_train_mixture["BASE"] \
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

MixtureRegistry.add(
    "t0+_train",
    [task for task in t0_train_mixture["BASE"] \
                    + t0_train_mixture["GPT_EVAL"] 
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: p3_utils.mixture_cap[t.name],
)

MixtureRegistry.add(
    "t0++_train",
    [task for task in t0_train_mixture["BASE"] \
                    + t0_train_mixture["GPT_EVAL"] \
                    + t0_train_mixture["SGLUE"] \
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: p3_utils.mixture_cap[t.name],
)