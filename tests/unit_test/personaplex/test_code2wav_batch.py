# SPDX-License-Identifier: Apache-2.0
"""Continuous cross-request batching in the PersonaPlex codec stage."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.personaplex.architecture import (
    MIMI,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.models.personaplex.config import (
    CODE2WAV_STAGE,
    DEFAULT_LM_MAX_RUNNING_REQUESTS,
    PersonaPlexPipelineConfig,
)
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def new_scheduler(
    codec: MimiCodec,
    *,
    num_slots: int,
    max_batch_size: int,
    max_graph_frames: int = 1,
    chunk_ramp_frames: tuple[int, ...] = (1,),
) -> PersonaPlexCode2WavScheduler:
    return PersonaPlexCode2WavScheduler(
        MimiDecodeSlots(
            codec,
            num_slots=num_slots,
            max_batch_size=max_batch_size,
            max_graph_frames=max_graph_frames,
            max_chunk_frames=max(chunk_ramp_frames),
        ),
        compute_fn=lambda payload: payload,
        chunk_ramp_frames=chunk_ramp_frames,
    )


def start_stream(scheduler, request_id: str) -> None:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)


def feed_serial(scheduler, request_id: str, chunk: torch.Tensor) -> torch.Tensor:
    messages = deliver_chunk(
        scheduler, request_id, StreamItem(chunk_id=0, from_stage="lm", data=chunk)
    )
    return torch.cat([decode_waveform(message.data) for message in messages])


def deliver_chunk(
    scheduler, request_id: str, item: StreamItem
) -> list[OutgoingMessage]:
    scheduler.handle_stream_chunk(request_id, item)
    return drain_outbox(scheduler)


def feed_native(
    codec: MimiCodec, state: MimiDecodeState, chunk: torch.Tensor
) -> torch.Tensor:
    return codec.decode_step(chunk.T[None], state)[0, 0].float().cpu()


def feed_batched(
    scheduler, chunks: list[tuple[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    scheduler.on_stream_chunk_batch(
        [
            (
                request_id,
                StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
            )
            for request_id, chunk in chunks
        ]
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    waveforms: dict[str, torch.Tensor] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.type == "stream"
        waveform = decode_waveform(message.data)
        waveforms[message.request_id] = (
            torch.cat([waveforms[message.request_id], waveform])
            if message.request_id in waveforms
            else waveform
        )
    return waveforms


def drain_outbox(scheduler) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def random_codes(frames: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(seed)
    )


def record_decode_steps(scheduler) -> list[int]:
    """Batch size of every decode step the scheduler fires, in order."""
    calls: list[int] = []
    original_decode_step = scheduler.decode_slots.decode_step

    def recording_decode_step(codes_BKF, slot_indices):
        calls.append(codes_BKF.shape[0])
        return original_decode_step(codes_BKF, slot_indices)

    scheduler.decode_slots.decode_step = recording_decode_step
    return calls


def small_context_codec() -> MimiCodec:
    """Random-weight Mimi whose attention ring wraps after a few frames."""
    torch.manual_seed(0)
    codec = MimiCodec(replace(MIMI, context=8)).eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.05)
        for module in codec.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
    return codec


def test_batched_lockstep_matches_serial(random_codec: MimiCodec) -> None:
    codec = random_codec
    frames = 6
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 1), ("b", 2), ("c", 3))
    }
    native_states = {
        request_id: codec.init_decode_state(batch_size=1, max_chunk_frames=2)
        for request_id in codes
    }
    batched = new_scheduler(codec, num_slots=4, max_batch_size=4)
    for request_id in codes:
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(frames):
        expected = {
            request_id: feed_native(
                codec,
                native_states[request_id],
                codes[request_id][frame : frame + 1],
            )
            for request_id in codes
        }
        streamed = feed_batched(
            batched,
            [
                (request_id, codes[request_id][frame : frame + 1])
                for request_id in codes
            ],
        )
        assert set(streamed) == set(codes)
        for request_id in codes:
            # Note (wilsonzheng0327): Matrix kernels may vary with the batch shape.
            torch.testing.assert_close(
                streamed[request_id], expected[request_id], rtol=1e-5, atol=1e-5
            )

    # Every frame of the three lockstep requests decodes in a single step.
    assert batch_sizes == [3] * frames
    for request_id in codes:
        (result,) = batched.on_stream_done(request_id)
        whole = decode_waveform(result.data.data)
        torch.testing.assert_close(
            whole,
            codec.decode(codes[request_id].T[None])[0, 0].cpu(),
            atol=1e-5,
            rtol=1e-5,
        )


def test_survivors_stay_exact_and_batch_after_a_release(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec
    frames = 5
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 4), ("b", 5), ("c", 6))
    }
    batched = new_scheduler(codec, num_slots=3, max_batch_size=3)
    serial = new_scheduler(codec, num_slots=8, max_batch_size=1)
    for request_id in codes:
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(2):
        feed_batched(
            batched,
            [
                (request_id, codes[request_id][frame : frame + 1])
                for request_id in codes
            ],
        )
        for request_id in codes:
            feed_serial(serial, request_id, codes[request_id][frame : frame + 1])

    batched.clear_stream_state("b")
    streamed = feed_batched(
        batched,
        [(request_id, codes[request_id][2 : 2 + 1]) for request_id in ("a", "c")],
    )
    torch.testing.assert_close(
        streamed["a"], feed_serial(serial, "a", codes["a"][2:3]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        streamed["c"], feed_serial(serial, "c", codes["c"][2:3]), rtol=1e-5, atol=1e-5
    )
    # Non-contiguous survivor slots still share one step after the release.
    assert batch_sizes == [3, 3, 2]

    # A late request joins the freed capacity and decodes alone at offset zero.
    start_stream(batched, "d")
    codes_d = random_codes(frames, 7)
    streamed = feed_batched(batched, [("d", codes_d[0:1])])
    torch.testing.assert_close(
        streamed["d"], feed_serial(serial, "d", codes_d[0:1]), rtol=1e-5, atol=1e-5
    )


def test_released_slot_is_reused_without_moving_survivors(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=2, max_batch_size=2)
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    feed_batched(
        scheduler,
        [
            (request_id, random_codes(1, seed))
            for seed, request_id in enumerate(("a", "b"))
        ],
    )
    slot_a = scheduler.stream_states["a"].slot_index
    slot_b = scheduler.stream_states["b"].slot_index
    assert slot_a is not None and slot_b is not None

    scheduler.clear_stream_state("a")
    assert scheduler.stream_states["b"].slot_index == slot_b
    start_stream(scheduler, "c")
    feed_batched(scheduler, [("c", random_codes(1, 3))])
    assert scheduler.stream_states["c"].slot_index == slot_a


def test_ring_wraparound_then_slot_reuse_stays_exact() -> None:
    codec = small_context_codec()  # ring capacity 8 latent frames = 4 codec frames
    frames = 6
    batched = new_scheduler(codec, num_slots=2, max_batch_size=2)
    serial = new_scheduler(codec, num_slots=8, max_batch_size=1)
    codes_a = random_codes(frames, 23)
    start_stream(batched, "a")
    for frame in range(frames):
        feed_batched(batched, [("a", codes_a[frame : frame + 1])])
    batched.clear_stream_state("a")

    # The reused slot held the wrapped ring of its previous owner.
    codes_b = random_codes(frames, 24)
    start_stream(batched, "b")
    for frame in range(frames):
        streamed = feed_batched(batched, [("b", codes_b[frame : frame + 1])])
        torch.testing.assert_close(
            streamed["b"],
            feed_serial(serial, "b", codes_b[frame : frame + 1]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_different_offsets_batch_after_attention_ring_wraparound() -> None:
    codec = small_context_codec()
    batched = new_scheduler(codec, num_slots=2, max_batch_size=2)
    native_states = {
        request_id: codec.init_decode_state(batch_size=1, max_chunk_frames=2)
        for request_id in ("a", "b")
    }
    codes_a = random_codes(6, 25)
    codes_b = random_codes(2, 26)
    for request_id in ("a", "b"):
        start_stream(batched, request_id)
    batch_sizes = record_decode_steps(batched)

    for frame in range(5):
        streamed = feed_batched(batched, [("a", codes_a[frame : frame + 1])])
        torch.testing.assert_close(
            streamed["a"],
            feed_native(codec, native_states["a"], codes_a[frame : frame + 1]),
            rtol=1e-5,
            atol=1e-5,
        )
    streamed = feed_batched(batched, [("b", codes_b[0:1])])
    torch.testing.assert_close(
        streamed["b"],
        feed_native(codec, native_states["b"], codes_b[0:1]),
        rtol=1e-5,
        atol=1e-5,
    )

    streamed = feed_batched(
        batched,
        [("a", codes_a[5:6]), ("b", codes_b[1:2])],
    )
    for request_id, chunk in (("a", codes_a[5:6]), ("b", codes_b[1:2])):
        torch.testing.assert_close(
            streamed[request_id],
            feed_native(codec, native_states[request_id], chunk),
            rtol=1e-5,
            atol=1e-5,
        )
    assert batch_sizes == [1] * 6 + [2]


def test_waiting_streams_take_freed_slots_in_ready_order(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    codes = {request_id: random_codes(2, seed) for seed, request_id in enumerate("abc")}
    for request_id in ("a", "empty", "b", "c"):
        start_stream(scheduler, request_id)
    # c becomes ready before b, although its terminal payload arrived later.
    for request_id in ("a", "c", "b"):
        for chunk in codes[request_id].split(1):
            scheduler.ingest_stream_item(
                request_id,
                StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
            )
        scheduler.handle_stream_done(request_id)
    assert scheduler.stream_states["c"].slot_index is None
    assert scheduler.stream_states["b"].slot_index is None

    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    messages = drain_outbox(scheduler)
    assert [message.request_id for message in messages if message.type == "stream"] == [
        "a",
        "a",
        "c",
        "c",
        "b",
        "b",
    ]
    results = [message for message in messages if message.type == "result"]
    assert [message.request_id for message in results] == ["a", "c", "b"]
    for message in results:
        torch.testing.assert_close(
            decode_waveform(message.data.data),
            random_codec.decode(codes[message.request_id].T[None])[0, 0],
            rtol=1e-5,
            atol=1e-5,
        )
    assert all(message.type != "error" for message in messages)
    assert scheduler.decode_slots.active_slot_count == 0
    assert scheduler.stream_states["empty"].slot_index is None
    # A previously empty waiter must become runnable when its first chunk arrives.
    assert deliver_chunk(
        scheduler,
        "empty",
        StreamItem(chunk_id=0, from_stage="lm", data=codes["a"][:1], metadata=None),
    )
    scheduler.handle_stream_done("empty")
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


@pytest.mark.parametrize("release_reason", ["done", "abort", "error"])
def test_waiters_survive_slot_owner_exit_and_waiting_cancellation(
    random_codec: MimiCodec, release_reason: str, monkeypatch
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    for request_id in ("owner", "cancelled", "survivor"):
        start_stream(scheduler, request_id)
    chunk = random_codes(1, 104)
    feed_serial(scheduler, "owner", chunk)
    for request_id in ("cancelled", "survivor"):
        assert (
            deliver_chunk(
                scheduler,
                request_id,
                StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
            )
            == []
        )
    scheduler.abort("cancelled")
    scheduler.handle_stream_done("survivor")
    assert scheduler.decode_slots.active_slot_count == 1

    if release_reason == "done":
        scheduler.handle_stream_done("owner")
    elif release_reason == "abort":
        scheduler.abort("owner")
    else:
        original_decode_step = scheduler.decode_slots.decode_step

        def fail_once(codes_BKF, slot_indices):
            monkeypatch.setattr(
                scheduler.decode_slots, "decode_step", original_decode_step
            )
            raise RuntimeError("owner failed")

        monkeypatch.setattr(scheduler.decode_slots, "decode_step", fail_once)
        scheduler.on_stream_chunk_batch(
            [
                (
                    "owner",
                    StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
                )
            ]
        )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    messages = drain_outbox(scheduler)
    assert not any(message.request_id == "cancelled" for message in messages)
    assert [message.request_id for message in messages if message.type == "error"] == (
        ["owner"] if release_reason == "error" else []
    )
    result = [
        message
        for message in messages
        if message.type == "result" and message.request_id == "survivor"
    ]
    assert len(result) == 1
    torch.testing.assert_close(
        decode_waveform(result[0].data.data),
        random_codec.decode(chunk.T[None])[0, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0


def test_waiting_stream_finishes_before_its_terminal_payload(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=1, max_batch_size=2)
    start_stream(scheduler, "owner")
    chunk = random_codes(1, 105)
    feed_serial(scheduler, "owner", chunk)
    scheduler.on_stream_chunk_batch(
        [
            (
                "waiting",
                StreamItem(chunk_id=0, from_stage="lm", data=chunk, metadata=None),
            )
        ]
    )
    scheduler.handle_stream_done("waiting")
    scheduler.handle_stream_done("owner")
    scheduler.run_ready_step()
    messages = drain_outbox(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("owner", "result"),
        ("waiting", "stream"),
    ]
    assert scheduler.decode_slots.active_slot_count == 0
    payload = StagePayload(
        "waiting",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=chunk).to_dict(),
    )
    scheduler.handle_streaming_new_request("waiting", payload)
    (result,) = drain_outbox(scheduler)
    assert result.type == "result"
    torch.testing.assert_close(
        decode_waveform(result.data.data),
        random_codec.decode(chunk.T[None])[0, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert scheduler.stream_states == {}
    assert scheduler.pending_done == set()
    assert scheduler.decode_slots.active_slot_count == 0


def test_different_offsets_share_steps_but_different_widths_do_not(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=2, max_batch_size=2, chunk_ramp_frames=(1, 2)
    )
    codes = {"a": random_codes(6, 11), "b": random_codes(5, 12)}
    for request_id in codes:
        start_stream(scheduler, request_id)
    shapes = []
    original = scheduler.decode_slots.decode_step

    def record(codes, slots):
        shapes.append((codes.shape[0], codes.shape[-1]))
        return original(codes, slots)

    scheduler.decode_slots.decode_step = record
    for chunks in [
        [("a", codes["a"][:1])],
        [("a", codes["a"][1:3]), ("b", codes["b"][:1])],
        [("a", codes["a"][3:5]), ("b", codes["b"][1:3])],
        [("a", codes["a"][5:6]), ("b", codes["b"][3:5])],
    ]:
        feed_batched(scheduler, chunks)
    for request_id in codes:
        scheduler.handle_stream_done(request_id)
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    results = [
        message for message in drain_outbox(scheduler) if message.type == "result"
    ]
    assert shapes == [(1, 1), (1, 2), (1, 1), (2, 2), (1, 2), (1, 1)]
    assert len(results) == 2
    for message in results:
        torch.testing.assert_close(
            decode_waveform(message.data.data),
            random_codec.decode(codes[message.request_id].T[None])[0, 0],
            atol=1e-5,
            rtol=1e-5,
        )


def test_per_request_caller_length_trimmed_in_batch(random_codec: MimiCodec) -> None:
    codec = random_codec
    frames = 3
    samples = frames * SAMPLES_PER_FRAME
    scheduler = new_scheduler(codec, num_slots=2, max_batch_size=2)
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    lengths = {"a": samples - 100, "b": samples - SAMPLES_PER_FRAME}
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 16), ("b", 17))
    }

    totals = {"a": 0, "b": 0}
    for frame in range(frames):
        scheduler.on_stream_chunk_batch(
            [
                (
                    request_id,
                    StreamItem(
                        chunk_id=0,
                        from_stage="lm",
                        data=codes[request_id][frame : frame + 1],
                        metadata={"num_samples": lengths[request_id]},
                    ),
                )
                for request_id in ("a", "b")
            ]
        )
        while scheduler.has_ready_work():
            scheduler.run_ready_step()
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            totals[message.request_id] += decode_waveform(message.data).shape[-1]
    assert totals == lengths


def test_a_failing_step_aborts_exactly_its_participants(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=3, max_batch_size=3, chunk_ramp_frames=(1, 2)
    )
    codes = random_codes(5, 33)
    for request_id in ("a", "b", "c"):
        start_stream(scheduler, request_id)
    feed_batched(scheduler, [("c", codes[:1])])
    original = scheduler.decode_slots.decode_step

    def fail_batch_of_two(codes, slots):
        if codes.shape[0] == 2:
            raise RuntimeError("boom")
        return original(codes, slots)

    scheduler.decode_slots.decode_step = fail_batch_of_two
    scheduler.on_stream_chunk_batch(
        [
            (request_id, StreamItem(chunk_id=0, from_stage="lm", data=chunk))
            for request_id, chunk in [
                ("a", codes[:1]),
                ("b", codes[:1]),
                ("c", codes[1:3]),
            ]
        ]
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    messages = drain_outbox(scheduler)
    assert {message.request_id for message in messages if message.type == "error"} == {
        "a",
        "b",
    }
    assert {message.request_id for message in messages if message.type == "stream"} == {
        "c"
    }
    assert set(scheduler.stream_states) == {"c"}
    scheduler.decode_slots.decode_step = original
    feed_batched(scheduler, [("c", codes[3:5]), ("d", codes[:1])])
    scheduler.handle_stream_done("c")
    (result,) = drain_outbox(scheduler)
    torch.testing.assert_close(
        decode_waveform(result.data.data),
        random_codec.decode(codes.T[None])[0, 0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_a_failing_batch_aborts_every_participant(random_codec: MimiCodec) -> None:
    codec = random_codec
    codes = {
        request_id: random_codes(2, seed) for request_id, seed in (("a", 41), ("b", 42))
    }
    batched = new_scheduler(codec, num_slots=4, max_batch_size=4)
    for request_id in codes:
        start_stream(batched, request_id)
    original_decode_step = batched.decode_slots.decode_step

    def always_fail(codes_BKF, slot_indices):
        raise RuntimeError("boom")

    batched.decode_slots.decode_step = always_fail
    batched.on_stream_chunk_batch(
        [
            (
                request_id,
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=codes[request_id][0:1],
                    metadata=None,
                ),
            )
            for request_id in codes
        ]
    )
    batched.run_ready_step()
    messages = drain_outbox(batched)
    assert {m.request_id for m in messages if m.type == "error"} == {"a", "b"}
    assert batched.decode_slots.active_slot_count == 0
    assert batched.stream_states == {}

    # The scheduler keeps serving after the failure.
    start_stream(batched, "e")
    codes_e = random_codes(2, 43)
    batched.decode_slots.decode_step = original_decode_step
    streamed = feed_batched(batched, [("e", codes_e[0:1])])
    reference = new_scheduler(codec, num_slots=8, max_batch_size=1)
    torch.testing.assert_close(
        streamed["e"], feed_serial(reference, "e", codes_e[0:1]), rtol=1e-5, atol=1e-5
    )


def test_multiple_chunks_for_one_request_are_drained_in_order(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=2, max_batch_size=2)
    start_stream(scheduler, "a")
    codes = random_codes(2, 44)

    scheduler.on_stream_chunk_batch(
        [
            (
                "a",
                StreamItem(chunk_id=0, from_stage="lm", data=codes[0:1], metadata=None),
            ),
            (
                "a",
                StreamItem(chunk_id=0, from_stage="lm", data=codes[1:2], metadata=None),
            ),
        ]
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    streamed = [decode_waveform(message.data) for message in drain_outbox(scheduler)]
    expected = random_codec.decode(codes.T[None])[0, 0].cpu()
    torch.testing.assert_close(torch.cat(streamed), expected, rtol=1e-5, atol=1e-5)


def test_backlogged_requests_rotate_between_decode_steps(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=4, max_batch_size=2)
    request_ids = ["a", "b", "c", "d"]
    codes = {
        request_id: random_codes(3, seed)
        for request_id, seed in zip(request_ids, range(51, 55), strict=True)
    }
    for request_id in request_ids:
        start_stream(scheduler, request_id)
    feed_batched(
        scheduler,
        [(request_id, codes[request_id][0:1]) for request_id in request_ids],
    )

    with scheduler.state_lock:
        for request_id in request_ids:
            for frame in (1, 2):
                scheduler.ingest_stream_item(
                    request_id,
                    StreamItem(
                        chunk_id=0,
                        from_stage="lm",
                        data=codes[request_id][frame : frame + 1],
                        metadata=None,
                    ),
                )
    for expected_request_ids in ({"a", "b"}, {"c", "d"}, {"a", "b"}, {"c", "d"}):
        scheduler.run_ready_step()
        assert {
            message.request_id for message in drain_outbox(scheduler)
        } == expected_request_ids


def test_ramp_collects_queued_frames_before_selecting_a_decode_batch(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=8, max_batch_size=8, chunk_ramp_frames=(1, 4)
    )
    codes = {str(index): random_codes(5, index + 100) for index in range(8)}
    for request_id in codes:
        start_stream(scheduler, request_id)
    feed_batched(
        scheduler, [(request_id, frames[:1]) for request_id, frames in codes.items()]
    )
    shapes = []
    original = scheduler.decode_slots.decode_step

    def decode_step(codes_BKF, slot_indices):
        shapes.append((codes_BKF.shape[0], codes_BKF.shape[-1]))
        return original(codes_BKF, slot_indices)

    scheduler.decode_slots.decode_step = decode_step
    for request_id, frames in codes.items():
        for frame in range(1, 5):
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_chunk",
                    data=StreamItem(
                        chunk_id=frame, from_stage="lm", data=frames[frame : frame + 1]
                    ),
                )
            )
    loop = asyncio.new_event_loop()
    try:
        scheduler.handle_message(scheduler.get_batch_message(), loop)
    finally:
        loop.close()
    assert shapes == [(8, 4)]
    assert scheduler.inbox.empty()
    assert {message.request_id for message in drain_outbox(scheduler)} == set(codes)


def test_start_loop_decodes_before_a_continuously_refilled_inbox_drains(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=2, max_batch_size=2)
    start_stream(scheduler, "a")
    frame_codes = random_codes(1, 45)
    processed_message_count = 0
    decoded_step_count = 0
    first_decode_message_count: list[int] = []
    original_handle_message = scheduler.handle_message
    original_decode_step = scheduler.decode_slots.decode_step

    def make_message(message_index: int) -> IncomingMessage:
        return IncomingMessage(
            request_id="a",
            type="stream_chunk",
            data=StreamItem(
                chunk_id=message_index,
                data=frame_codes,
                from_stage="lm",
            ),
        )

    def handle_message(
        message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        nonlocal processed_message_count
        original_handle_message(message, loop)
        processed_message_count += 1
        if processed_message_count < 30:
            scheduler.inbox.put(make_message(processed_message_count))
        elif decoded_step_count == 30:
            scheduler.stop()
        else:
            pass

    def decode_step(codes_BKF: torch.Tensor, slot_indices: list[int]) -> torch.Tensor:
        nonlocal decoded_step_count
        if not first_decode_message_count:
            first_decode_message_count.append(processed_message_count)
        else:
            pass
        waveform = original_decode_step(codes_BKF, slot_indices)
        decoded_step_count += 1
        return waveform

    scheduler.handle_message = handle_message
    scheduler.decode_slots.decode_step = decode_step
    scheduler.inbox.put(make_message(0))
    scheduler.start()

    assert first_decode_message_count == [0]
    assert processed_message_count == 30
    assert decoded_step_count == 30


def test_ready_buckets_rotate_without_waiting(random_codec: MimiCodec) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=3, max_batch_size=2, chunk_ramp_frames=(1, 2)
    )
    for request_id in ("a", "b", "c"):
        start_stream(scheduler, request_id)

    feed_batched(scheduler, [("a", random_codes(1, 44))])
    with scheduler.state_lock:
        scheduler.ingest_stream_item(
            "a",
            StreamItem(
                chunk_id=0, from_stage="lm", data=random_codes(2, 45), metadata=None
            ),
        )
        for request_id, seed in (("b", 46), ("c", 47)):
            scheduler.ingest_stream_item(
                request_id,
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=random_codes(1, seed),
                    metadata=None,
                ),
            )
    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert {message.request_id for message in drain_outbox(scheduler)} == {"a"}

    assert scheduler.has_ready_work()
    scheduler.run_ready_step()
    assert {message.request_id for message in drain_outbox(scheduler)} == {"b", "c"}


def test_stream_capacity_is_independent_from_step_batch_size(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=16, max_batch_size=8)
    request_ids = [f"request-{index}" for index in range(16)]
    for request_id in request_ids:
        start_stream(scheduler, request_id)
    batch_sizes = record_decode_steps(scheduler)

    streamed = feed_batched(
        scheduler,
        [
            (request_id, random_codes(1, index))
            for index, request_id in enumerate(request_ids)
        ],
    )
    assert set(streamed) == set(request_ids)
    assert batch_sizes == [8, 8]


def test_config_enables_batching_by_default(random_codec: MimiCodec) -> None:
    config = PersonaPlexPipelineConfig(model_path="unused")
    factory = config.stage_named(CODE2WAV_STAGE).factory
    assert factory.num_decode_slots is None
    assert factory.max_batch_size == 8
    assert factory.max_graph_frames is None
    assert factory.chunk_ramp_frames == [1, 4]

    scheduler = new_scheduler(
        random_codec,
        num_slots=config.stage_factory_kwargs(CODE2WAV_STAGE)["num_decode_slots"],
        max_batch_size=factory.max_batch_size,
        max_graph_frames=config.stage_factory_kwargs(CODE2WAV_STAGE)[
            "max_graph_frames"
        ],
    )
    assert scheduler.decode_slots.num_slots == DEFAULT_LM_MAX_RUNNING_REQUESTS
    assert scheduler.can_batch_stream_chunks
    assert scheduler.stream_chunk_batch_max == 8


def pending_depth(scheduler) -> int:
    return sum(state.pending_frames for state in scheduler.stream_states.values())


def test_sustained_multi_chunk_batches_do_not_grow_the_backlog(
    random_codec: MimiCodec,
) -> None:
    # Each call stands for one message batch while the inbox never drains, so
    # the ready-work branch never runs between calls. A batch carries several
    # chunks of the same request, including transport messages with two frames.
    codec = random_codec
    rounds = 3
    scheduler = new_scheduler(codec, num_slots=2, max_batch_size=8)
    codes = {"a": random_codes(4 * rounds, 71), "b": random_codes(8 * rounds, 72)}
    for request_id in codes:
        start_stream(scheduler, request_id)

    depths = []
    for round_index in range(rounds):
        items = [
            (
                "a",
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=codes["a"][4 * round_index + i : 4 * round_index + i + 1],
                    metadata=None,
                ),
            )
            for i in range(4)
        ] + [
            (
                "b",
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=codes["b"][
                        8 * round_index + 2 * i : 8 * round_index + 2 * i + 2
                    ],
                    metadata=None,
                ),
            )
            for i in range(4)
        ]
        scheduler.on_stream_chunk_batch(items)
        depths.append(pending_depth(scheduler))
    # One step per message batch leaves frames pending and cannot keep up.
    assert depths == [0] * rounds

    streamed: dict[str, list[torch.Tensor]] = {"a": [], "b": []}
    for message in drain_outbox(scheduler):
        assert message.type == "stream"
        streamed[message.request_id].append(decode_waveform(message.data))
    for request_id, request_codes in codes.items():
        expected = codec.decode(request_codes.T[None])[0, 0].cpu()
        torch.testing.assert_close(
            torch.cat(streamed[request_id]), expected, rtol=1e-5, atol=1e-5
        )
        (result,) = scheduler.on_stream_done(request_id)
        torch.testing.assert_close(
            decode_waveform(result.data.data), expected, rtol=1e-5, atol=1e-5
        )
    assert scheduler.decode_slots.active_slot_count == 0


def test_new_arrivals_cannot_starve_a_backlogged_stream(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=3, max_batch_size=2, chunk_ramp_frames=(2, 1)
    )
    long_codes = random_codes(7, 61)
    start_stream(scheduler, "long")
    feed_batched(scheduler, [("long", long_codes[0:2])])
    with scheduler.state_lock:
        for frame in range(2, 7):
            scheduler.ingest_stream_item(
                "long",
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=long_codes[frame : frame + 1],
                    metadata=None,
                ),
            )

    long_decode_steps: list[int] = []
    short_wait_steps: dict[str, int] = {}
    for step in range(10):
        # A fresh request arrives before every step with a two-frame chunk, so
        # it never shares a step with the one-frame backlog: each step serves
        # one side, and an arrival must not keep winning.
        short_id = f"short-{step}"
        start_stream(scheduler, short_id)
        with scheduler.state_lock:
            scheduler.ingest_stream_item(
                short_id,
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=random_codes(2, 100 + step),
                    metadata=None,
                ),
            )
        short_wait_steps[short_id] = step
        scheduler.run_ready_step()
        for message in drain_outbox(scheduler):
            assert message.type == "stream"
            if message.request_id == "long":
                long_decode_steps.append(step)
            else:
                short_wait_steps[message.request_id] = (
                    step - short_wait_steps[message.request_id]
                )
                scheduler.handle_stream_done(message.request_id)
                (result,) = drain_outbox(scheduler)
                assert result.type == "result"

    # The backlogged stream is served every other step despite the churn.
    assert long_decode_steps == [0, 2, 4, 6, 8]
    assert not scheduler.stream_states["long"].pending_codes
    # Arrivals still get their first step within one step of arriving.
    assert all(wait <= 1 for wait in short_wait_steps.values())


def test_streamed_audio_parts_do_not_pin_the_batch_buffer(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(random_codec, num_slots=4, max_batch_size=4)
    request_ids = ["a", "b", "c"]
    for request_id in request_ids:
        start_stream(scheduler, request_id)
    batch_sizes = record_decode_steps(scheduler)
    scheduler.on_stream_chunk_batch(
        [
            (
                request_id,
                StreamItem(
                    chunk_id=0,
                    from_stage="lm",
                    data=random_codes(1, 80 + index),
                    # a is trimmed to its caller length inside the step.
                    metadata={"num_samples": 1000} if request_id == "a" else None,
                ),
            )
            for index, request_id in enumerate(request_ids)
        ]
    )
    assert batch_sizes == [3]
    for request_id in request_ids:
        (part,) = scheduler.stream_states[request_id].audio_parts
        expected_samples = 1000 if request_id == "a" else SAMPLES_PER_FRAME
        assert part.shape[-1] == expected_samples
        # Each part owns exactly its own samples, not the 3-row batch tensor.
        assert part.untyped_storage().nbytes() == part.numel() * part.element_size()


def test_personaplex_does_not_advertise_realtime_audio_output() -> None:
    # The launcher offers realtime audio output to any pipeline that names a
    # code2wav stage here; PersonaPlex's duplex flow does not serve it.
    assert PersonaPlexPipelineConfig.code2wav_stage() is None


def test_decode_slots_follow_lm_overrides_without_pinning_the_config() -> None:
    config = PersonaPlexPipelineConfig(model_path="unused")
    for max_running_requests in (16, 32, 1):
        config = ConfigManager(config).merge_config(
            {"lm.engine.max_running_requests": max_running_requests}
        )
        stage = config.stage_named(CODE2WAV_STAGE)
        kwargs = resolve_stage_factory_kwargs(stage, config)
        kwargs.update(resolve_stage_typed_kwargs(stage))
        assert kwargs["num_decode_slots"] == max_running_requests
        assert kwargs["max_batch_size"] == 8
        assert stage.factory.num_decode_slots is None
        config = PersonaPlexPipelineConfig.model_validate(config.model_dump())
        assert config.stage_named(CODE2WAV_STAGE).factory.num_decode_slots is None


@pytest.mark.parametrize("num_decode_slots", [16, 17, 24])
def test_explicit_decode_slots_override_derivation(num_decode_slots: int) -> None:
    config = ConfigManager(PersonaPlexPipelineConfig(model_path="unused")).merge_config(
        {
            "lm.engine.max_running_requests": 16,
            "code2wav.factory.num_decode_slots": num_decode_slots,
        }
    )
    stage = config.stage_named(CODE2WAV_STAGE)
    kwargs = resolve_stage_factory_kwargs(stage, config)
    kwargs.update(resolve_stage_typed_kwargs(stage))
    assert kwargs["num_decode_slots"] == num_decode_slots
    with pytest.raises(ValueError, match="Omit num_decode_slots to derive it"):
        ConfigManager(config).merge_config(
            {"lm.engine.max_running_requests": num_decode_slots + 1}
        )


def test_config_rejects_fewer_decode_slots_than_lm_running_requests() -> None:
    def config_data(max_running_requests: int, num_decode_slots: int) -> dict:
        data = PersonaPlexPipelineConfig(model_path="unused").model_dump()
        for stage in data["stages"]:
            if stage["name"] == "lm":
                stage["engine"]["max_running_requests"] = max_running_requests
            elif stage["name"] == CODE2WAV_STAGE:
                stage["factory"]["num_decode_slots"] = num_decode_slots
            else:
                pass
        return data

    with pytest.raises(ValueError, match="num_decode_slots"):
        PersonaPlexPipelineConfig.model_validate(config_data(32, 16))
    PersonaPlexPipelineConfig.model_validate(config_data(16, 16))
    PersonaPlexPipelineConfig.model_validate(config_data(32, 32))


class CountingGraph:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.graph.replay()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="the platform does not capture codec decode graphs",
)
def test_scheduler_batches_replay_device_graphs(random_codec: MimiCodec) -> None:
    codec = random_codec.cuda()
    graphed = new_scheduler(codec, num_slots=4, max_batch_size=4)
    eager = new_scheduler(codec, num_slots=4, max_batch_size=4, max_graph_frames=0)
    assert set(graphed.decode_slots.graphs) == {(1, 1), (2, 1), (3, 1), (4, 1)}
    assert eager.decode_slots.graphs == {}
    for captured_decode in graphed.decode_slots.graphs.values():
        captured_decode.graph = CountingGraph(captured_decode.graph)
    frames = 6
    codes = {
        request_id: random_codes(frames, seed)
        for request_id, seed in (("a", 91), ("b", 92), ("c", 93))
    }
    for scheduler in (graphed, eager):
        for request_id in codes:
            start_stream(scheduler, request_id)
    # c starts two frames late, so every step after the first mixes offsets.
    rounds = [[("a", 0), ("b", 0)], [("a", 1), ("b", 1)]] + [
        [("a", frame), ("b", frame), ("c", frame - 2)] for frame in range(2, frames)
    ]
    for round_frames in rounds:
        chunks = [
            (request_id, codes[request_id][frame : frame + 1])
            for request_id, frame in round_frames
        ]
        replayed = feed_batched(graphed, chunks)
        expected = feed_batched(eager, chunks)
        assert set(replayed) == set(expected) == {rid for rid, _ in round_frames}
        for request_id in replayed:
            assert torch.equal(replayed[request_id], expected[request_id])
    replays = {
        key: captured_decode.graph.replays
        for key, captured_decode in graphed.decode_slots.graphs.items()
    }
    assert replays == {(1, 1): 0, (2, 1): 2, (3, 1): 4, (4, 1): 0}


def test_ramp_catchup_counts_frames_in_burst_messages(random_codec: MimiCodec) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=2, max_batch_size=2, chunk_ramp_frames=(1, 4)
    )
    codes = {"a": random_codes(13, 120), "b": random_codes(9, 121)}
    scheduler.on_stream_chunk_batch(
        [
            (request_id, StreamItem(chunk_id=0, from_stage="lm", data=frames))
            for request_id, frames in codes.items()
        ]
    )
    # No idle-loop pump: two messages contain 22 frames, not two decode steps.
    assert pending_depth(scheduler) == 0
    messages = drain_outbox(scheduler)
    for request_id, frames in codes.items():
        waveform = torch.cat(
            [
                decode_waveform(message.data)
                for message in messages
                if message.request_id == request_id
            ]
        )
        torch.testing.assert_close(
            waveform, random_codec.decode(frames.T[None])[0, 0], atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("max_graph_frames", [0, 4])
@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="requires codec device graphs",
)
def test_ramp_batch_waiting_tail_and_slot_reuse_past_ring(
    random_codec: MimiCodec, max_graph_frames: int
) -> None:
    codec = random_codec.cuda()
    scheduler = new_scheduler(
        codec,
        num_slots=2,
        max_batch_size=2,
        max_graph_frames=max_graph_frames,
        chunk_ramp_frames=(1, 4),
    )
    assert len(scheduler.decode_slots.graphs) == 2 * max_graph_frames
    codes = {
        "a": random_codes(160, 130),
        "b": random_codes(151, 131),
        "waiting": random_codes(10, 132),
    }
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    messages = []
    shapes = []
    original = scheduler.decode_slots.decode_step

    def record(codes, slots):
        shapes.append((codes.shape[0], codes.shape[-1]))
        return original(codes, slots)

    scheduler.decode_slots.decode_step = record
    for tick in range(160):
        items = []
        for request_id, start in (("a", 0), ("b", 8)):
            frame = tick - start
            if 0 <= frame < len(codes[request_id]):
                items.append(
                    (
                        request_id,
                        StreamItem(
                            chunk_id=frame,
                            from_stage="lm",
                            data=codes[request_id][frame : frame + 1],
                        ),
                    )
                )
        if tick == 12:
            items.append(
                (
                    "waiting",
                    StreamItem(chunk_id=0, from_stage="lm", data=codes["waiting"]),
                )
            )
        scheduler.on_stream_chunk_batch(items)
        if tick == 12:
            assert scheduler.stream_states["waiting"].slot_index is None
            assert scheduler.stream_states["waiting"].pending_frames == 10
            scheduler.handle_stream_done("waiting")
        for request_id, start in (("a", 0), ("b", 8)):
            if tick == start + len(codes[request_id]) - 1:
                scheduler.handle_stream_done(request_id)
        while scheduler.has_ready_work():
            scheduler.run_ready_step()
        messages.extend(drain_outbox(scheduler))
    assert (2, 4) in shapes
    assert {width for _, width in shapes} >= {1, 2, 3, 4}
    assert scheduler.decode_slots.active_slot_count == 0
    assert not any(message.type == "error" for message in messages)
    assert not any(
        message.type == "result" and message.request_id == "waiting"
        for message in messages
    )
    payload = StagePayload(
        "waiting",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=codes["waiting"]).to_dict(),
    )
    scheduler.handle_streaming_new_request("waiting", payload)
    messages.extend(drain_outbox(scheduler))
    assert scheduler.stream_states == {}
    for request_id, frames in codes.items():
        ordered = [message for message in messages if message.request_id == request_id]
        assert ordered[-1].type == "result"
        assert all(message.type == "stream" for message in ordered[:-1])
        streamed = torch.cat(
            [decode_waveform(message.data) for message in ordered[:-1]]
        )
        torch.testing.assert_close(
            streamed, decode_waveform(ordered[-1].data.data), atol=0, rtol=0
        )
        # Compare the same fixed codes to a whole-utterance decode, independent of scheduling.
        reference = codec.decode(frames.T[None].cuda())[0, 0].float().cpu()
        assert streamed.shape == reference.shape
        assert (streamed - reference).abs().max() <= 0.05 * reference.abs().max()


def test_ramp_shutdown_and_late_chunks_release_all_slots(
    random_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(
        random_codec, num_slots=1, max_batch_size=2, chunk_ramp_frames=(1, 4)
    )
    codes = random_codes(4, 133)
    scheduler.on_stream_chunk_batch(
        [
            (request_id, StreamItem(chunk_id=0, from_stage="lm", data=codes))
            for request_id in ("a", "b")
        ]
    )
    assert scheduler.decode_slots.active_slot_count == 1
    assert scheduler.stream_states["b"].slot_index is None
    scheduler.stop()
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0

    scheduler = new_scheduler(
        random_codec, num_slots=1, max_batch_size=2, chunk_ramp_frames=(1, 4)
    )
    start_stream(scheduler, "done")
    deliver_chunk(
        scheduler, "done", StreamItem(chunk_id=0, from_stage="lm", data=codes[:1])
    )
    scheduler.handle_stream_done("done")
    drain_outbox(scheduler)
    deliver_chunk(
        scheduler, "done", StreamItem(chunk_id=1, from_stage="lm", data=codes[1:])
    )
    assert scheduler.stream_states == {}
    assert scheduler.decode_slots.active_slot_count == 0
