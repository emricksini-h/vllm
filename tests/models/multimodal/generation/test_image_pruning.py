# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for position-preserving image token pruning."""

import pytest

from ...utils import check_logprobs_close, check_outputs_equal

MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # has deepstack features
PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>\n<|im_start|>assistant\n"
)


def _generate(vllm_runner, image_assets, logprobs: bool, **kwargs):
    images = [asset.pil_image for asset in image_assets]
    prompts = [PROMPT] * len(images)
    with vllm_runner(
        MODEL, max_model_len=4096, limit_mm_per_prompt={"image": 1}, **kwargs
    ) as llm:
        if logprobs:
            return llm.generate_greedy_logprobs(prompts, 32, 5, images=images)
        return llm.generate_greedy(prompts, 32, images=images)


@pytest.mark.core_model
def test_zero_rate_matches_no_pruning(vllm_runner, image_assets):
    # Rate 0 runs the whole pruning path (selection, packing, position writes)
    # while keeping every token, so outputs must not change.
    check_outputs_equal(
        outputs_0_lst=_generate(vllm_runner, image_assets, False),
        outputs_1_lst=_generate(
            vllm_runner, image_assets, False, image_pruning_rate=0.0
        ),
        name_0="no_pruning",
        name_1="rate_0",
    )


@pytest.mark.core_model
@pytest.mark.parametrize("method", ["cosine", "random"])
def test_pruned_prefill_is_chunk_invariant(vllm_runner, image_assets, method):
    # Small chunks put prefill boundaries inside the pruned images.
    kwargs = dict(image_pruning_rate=0.5, image_pruning_method=method)
    check_logprobs_close(
        outputs_0_lst=_generate(vllm_runner, image_assets, True, **kwargs),
        outputs_1_lst=_generate(
            vllm_runner, image_assets, True, max_num_batched_tokens=128, **kwargs
        ),
        name_0="unchunked",
        name_1="chunked",
    )
