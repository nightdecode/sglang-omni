# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipeline: five stages on one GPU, the LM under SGLang."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.models.personaplex.hf_config import PERSONAPLEX_ARCH

MODEL_STAGES_PREFIX = "sglang_omni.models.personaplex.stages"
PREPROCESSING_STAGE = "preprocessing"
LM_STAGE = "lm"
CODE2WAV_STAGE = "code2wav"


class PersonaPlexCode2WavFactoryArgs(FactoryArgs):
    max_batch_size: int = Field(default=8, ge=1)
    stream_slots: int = Field(default=16, ge=1)


class PersonaPlexCode2WavStageConfig(StageConfig):
    factory: PersonaPlexCode2WavFactoryArgs = Field(
        default_factory=PersonaPlexCode2WavFactoryArgs
    )


def personaplex_stages_factory() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_preprocessing_executor",
            next="mimi_encode",
        ),
        StageConfig(
            name="mimi_encode",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_mimi_encode_executor",
            gpu=0,
            next=LM_STAGE,
        ),
        EngineStageConfig(
            name=LM_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_lm_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            engine=EngineArgs(mem_fraction_static=0.3),
            next=["decode", CODE2WAV_STAGE],
            stream_to=[CODE2WAV_STAGE],
        ),
        StageConfig(
            name="decode",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_decode_executor",
            terminal=True,
        ),
        PersonaPlexCode2WavStageConfig(
            name=CODE2WAV_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory=PersonaPlexCode2WavFactoryArgs(),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class PersonaPlexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = PERSONAPLEX_ARCH
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: EngineStageConfig,
        CODE2WAV_STAGE: PersonaPlexCode2WavStageConfig,
    }

    model_path: str
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=personaplex_stages_factory)

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        self.validate_code2wav_capacity()

    def validate_code2wav_capacity(self) -> None:
        """Every LM request streams into its own code2wav decode slot, so fewer
        slots than LM running requests turns admitted requests into 503s."""
        stages = {stage.name: stage for stage in self.stages}
        lm_stage = stages.get(LM_STAGE)
        code2wav_stage = stages.get(CODE2WAV_STAGE)
        if lm_stage is None or code2wav_stage is None:
            return
        else:
            pass
        engine = getattr(lm_stage, "engine", None)
        max_running = engine.max_running_requests if engine is not None else None
        stream_slots = getattr(code2wav_stage.factory, "stream_slots", None)
        if max_running is None or stream_slots is None:
            return
        else:
            pass
        if stream_slots < max_running:
            raise ValueError(
                f"{CODE2WAV_STAGE}.factory.stream_slots={stream_slots} is below "
                f"{LM_STAGE}.engine.max_running_requests={max_running}: every "
                "running LM request needs its own code2wav decode slot. Raise "
                "stream_slots to at least max_running_requests."
            )
        else:
            pass


EntryClass = PersonaPlexPipelineConfig
