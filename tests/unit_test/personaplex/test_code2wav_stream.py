# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex import mimi_decode_graph
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def new_scheduler(codec: MimiCodec) -> PersonaPlexCode2WavScheduler:
    return PersonaPlexCode2WavScheduler(
        MimiDecodeSlots(codec, num_slots=2, max_batch_size=1, max_graph_frames=1),
        compute_fn=lambda payload: payload,
    )


def start_stream(scheduler, request_id: str) -> StagePayload:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return payload


def test_interleaved_requests_stream_their_own_waveforms(random_codec):
    codec = random_codec
    scheduler = new_scheduler(codec)
    codes = {
        "a": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(1)),
        "b": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(2)),
    }
    whole = {rid: codec.decode(c.T[None])[0, 0] for rid, c in codes.items()}
    payloads = {rid: start_stream(scheduler, rid) for rid in codes}

    streamed = {rid: [] for rid in codes}

    def push(rid: str, chunk: torch.Tensor) -> None:
        (message,) = scheduler.on_stream_chunk(
            rid, SimpleNamespace(data=chunk, metadata=None)
        )
        assert message.type == "stream"
        streamed[rid].append(decode_waveform(message.data))

    for frame in range(4):
        push("a", codes["a"][frame : frame + 1])
        if frame % 2 == 1:
            push("b", codes["b"][frame - 1 : frame + 1])

    for rid in codes:
        torch.testing.assert_close(
            torch.cat(streamed[rid]), whole[rid], atol=1e-5, rtol=1e-5
        )
        (result,) = scheduler.on_stream_done(rid)
        assert result.type == "result"
        assert result.data.request is payloads[rid].request
        torch.testing.assert_close(
            decode_waveform(result.data.data), whole[rid], atol=1e-5, rtol=1e-5
        )


def test_a_reply_with_frames_streams_whatever_arrives_first(random_codec):
    scheduler = new_scheduler(random_codec)
    empty = StagePayload(
        "a", request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    assert not scheduler.is_streaming_payload(empty)
    with_frames = StagePayload(
        "a",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=torch.zeros(3, 8, dtype=torch.long)).to_dict(),
    )
    assert scheduler.is_streaming_payload(with_frames)


def test_abort_clears_stream_state(random_codec):
    scheduler = new_scheduler(random_codec)
    start_stream(scheduler, "a")
    assert scheduler.on_stream_done("a") != []
    scheduler.clear_stream_state("a")
    assert scheduler.on_stream_done("a") == []
    assert scheduler.on_stream_done("never-started") == []


@pytest.mark.parametrize(
    "num_samples",
    [0, 4 * SAMPLES_PER_FRAME, 3 * SAMPLES_PER_FRAME + 100],
    ids=["unspecified", "whole", "partial"],
)
def test_reply_length_matches_the_caller_in_chunks_and_final_payload(
    random_codec: MimiCodec, num_samples: int
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec)
    frames, samples_per_frame = 4, codec.samples_per_frame
    expected_samples = num_samples or frames * samples_per_frame
    codes = torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(3)
    )
    whole = codec.decode(codes.T[None])[0, 0]
    assert whole.shape[-1] == frames * samples_per_frame

    streamed = []
    for frame in range(frames):
        (message,) = scheduler.on_stream_chunk(
            "a",
            SimpleNamespace(
                data=codes[frame : frame + 1], metadata={"num_samples": num_samples}
            ),
        )
        streamed.append(decode_waveform(message.data))
    torch.testing.assert_close(
        torch.cat(streamed), whole[:expected_samples], atol=1e-5, rtol=1e-5
    )
    assert streamed[-1].shape[-1] == expected_samples - 3 * samples_per_frame

    # Note (wilsonzheng0327): The terminal payload lands after every chunk, as it does
    # when the LM stage finishes.
    start_stream(scheduler, "a")
    (result,) = scheduler.on_stream_done("a")
    reply = decode_waveform(result.data.data)
    torch.testing.assert_close(reply, whole[:expected_samples], atol=1e-5, rtol=1e-5)


def test_platforms_without_codec_decode_graphs_batch_eagerly(random_codec, monkeypatch):
    monkeypatch.setattr(
        mimi_decode_graph,
        "current_platform",
        SimpleNamespace(
            enable_codec_decode_graph=lambda: False,
            get_device_graph_backend=lambda device: SimpleNamespace(),
        ),
    )
    slots = MimiDecodeSlots(
        random_codec, num_slots=8, max_batch_size=8, max_graph_frames=1
    )
    assert slots.graphs == {}
    assert [slots.acquire() for _ in range(8)] == list(range(8))
    assert slots.acquire() is None
    codes = torch.randint(
        0, 2048, (2, 8, 1), generator=torch.Generator().manual_seed(5)
    )
    assert slots.decode_step(codes, [3, 6]).shape == (2, 1, SAMPLES_PER_FRAME)


def decode_plan(
    slots: MimiDecodeSlots,
    codes: torch.Tensor,
    plan: list[tuple[tuple[int, ...], int]],
    slot_of_stream: list[int],
) -> list[torch.Tensor]:
    """Run each (streams, width) step of the plan on the streams' slots."""
    parts: list[list[torch.Tensor]] = [[] for _ in slot_of_stream]
    starts = [0] * len(slot_of_stream)
    for streams, width in plan:
        chunk = torch.stack(
            [
                codes[stream, :, starts[stream] : starts[stream] + width]
                for stream in streams
            ]
        )
        waveform = slots.decode_step(
            chunk, [slot_of_stream[s] for s in streams]
        ).clone()
        for row, stream in enumerate(streams):
            parts[stream].append(waveform[row])
            starts[stream] += width
    return [torch.cat(stream_parts, -1) for stream_parts in parts]


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="the platform does not capture codec decode graphs",
)
def test_batched_graph_replay_matches_eager_decode_across_slot_reuse(random_codec):
    codec = random_codec.cuda()
    graphed = MimiDecodeSlots(codec, num_slots=3, max_batch_size=2, max_graph_frames=2)
    assert set(graphed.graphs) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    eager = MimiDecodeSlots(codec, num_slots=3, max_batch_size=2, max_graph_frames=0)
    assert eager.graphs == {}
    codes = torch.randint(
        0, 2048, (3, 8, 140), generator=torch.Generator().manual_seed(6)
    ).cuda()
    # Note (nightdecode): Streams advance at different rates, so rows in one step
    # sit at different offsets. Width 3 and the batch of three have no graph and
    # run eager on the same slots; stream 2 takes 140 frames, 280 steps, past the
    # transformer's 250-step ring.
    plan = [
        ((0, 1), 1),
        ((0,), 2),
        ((0, 1, 2), 1),
        ((1, 2), 3),
        ((2,), 1),
        ((0, 2), 2),
    ] * 20
    expected = []
    for stream in range(3):
        state = codec.init_decode_state(batch_size=1)
        parts, start = [], 0
        for streams, width in plan:
            if stream in streams:
                chunk = codes[stream : stream + 1, :, start : start + width]
                parts.append(codec.decode_step(chunk, state)[0])
                start += width
            else:
                pass
        expected.append(torch.cat(parts, -1))

    for _ in range(2):
        outputs = {}
        for name, slots in (("graphed", graphed), ("eager", eager)):
            slot_of_stream = [slots.acquire() for _ in range(3)]
            outputs[name] = decode_plan(slots, codes, plan, slot_of_stream)
            # Released in stream order, so the next round maps the streams onto
            # the slots in reverse.
            for slot_index in slot_of_stream:
                slots.release(slot_index)
        for stream in range(3):
            # The same steps replayed and run eagerly agree bit for bit.
            assert torch.equal(outputs["graphed"][stream], outputs["eager"][stream])
            # A batch of one rounds differently (TF32 convolutions pick kernels
            # per shape), so against it only a row decoding the wrong stream,
            # an error the size of the signal, is ruled out.
            error = (outputs["graphed"][stream] - expected[stream]).abs().max()
            assert error <= 0.05 * expected[stream].abs().max()
