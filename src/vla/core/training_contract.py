"""Versioned training semantics shared by online and offline A/C."""

from __future__ import annotations

import copy

TRAINING_SEMANTICS_VERSION = 3
REFERENCE_POLICY = "per_frame_human_execution_else_vla_v2"


def validate_training_cache(contract):
    cache = contract.get("cache", {})
    if (cache.get("training_semantics_version") != TRAINING_SEMANTICS_VERSION
            or cache.get("annotation_policy", {}).get("ref_chunk") != REFERENCE_POLICY):
        raise ValueError("old A/C cache supervision; rebuild the cache in a new directory")


def validate_cache_upgrade(previous, current):
    raise ValueError('training asset conversion is not supported; create a new schema-5 run')


def supervision_reference(exec_chunk, ref_chunk, annotations, start_frame, actual_steps=None, action_stride=1):
    if exec_chunk.shape != ref_chunk.shape or ref_chunk.ndim != 2:
        raise ValueError("executed/reference chunk dimensions do not match")
    length = ref_chunk.shape[0] if actual_steps is None else actual_steps
    if not 1 <= length <= ref_chunk.shape[0]:
        raise ValueError("invalid supervision chunk length")
    result = ref_chunk.clone()
    for offset in range(length):
        row = (annotations or {}).get(start_frame + offset * action_stride, {})
        if float(row.get("is_intervention", 0)) > 0.5 or int(row.get("collector_policy_id", 1)) == 0:
            result[offset].copy_(exec_chunk[offset])
    result[length:].zero_()
    return result
