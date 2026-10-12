# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal and its chunked path lands on exactly the samples a
whole-sequence decode would, so each frame goes out as soon as its codes
arrive: 1920 samples per 80 ms frame, no holdback. Chunks that are ready
across requests decode together in one step on the shared decode slots; a
stream that finds every slot taken buffers its chunks until one frees.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as emit_event
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

SOURCE_HINT = "PersonaPlex"


def trim_to_caller(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Cut the reply back to the caller recording's own length.

    Frames are whole 80 ms, so a recording that is not a multiple of one is
    padded before encoding; the reference trims that padding off the reply.
    """
    if num_samples and waveform.shape[-1] > num_samples:
        return waveform[..., :num_samples]
    else:
        pass
    return waveform


class AudioPayloadDecoder(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


@dataclass(kw_only=True)
class StreamState:
    # None while waiting for a slot, and again once the stream is done.
    slot_index: int | None
    pending_chunks: deque[torch.Tensor] = field(default_factory=deque)
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    decoded_frames: int = 0
    emitted_samples: int = 0
    num_samples: int = 0
    stream_done: bool = False
    # Position in the ready queue: stamped when the request gains its first
    # pending chunk and again after each decode step that leaves it with
    # pending work. Lower runs first, so a new arrival queues behind every
    # stream already waiting instead of jumping ahead of it.
    ready_order: int = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self, decode_slots: MimiDecodeSlots, *, compute_fn: AudioPayloadDecoder
    ) -> None:
        super().__init__(compute_fn, max_batch_size=decode_slots.max_batch_size)
        self.decode_slots = decode_slots
        self.stream_states: dict[str, StreamState] = {}
        self.next_ready_order = 0
        if self.max_batch_size > 1:
            self.can_batch_stream_chunks = True
            self.stream_chunk_batch_max = self.max_batch_size
        else:
            pass

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Note (wilsonzheng0327): The LM streams every frame it produces, so a reply
        # with frames has chunks on the way whichever channel lands first; only an
        # empty reply is rendered whole.
        codes = PersonaPlexState.from_dict(payload.data).codes
        return codes is not None and codes.shape[0] > 0

    def stream_state(self, request_id: str) -> StreamState:
        state = self.stream_states.get(request_id)
        if state is None:
            # Note (nightdecode): The LM admits its next request as soon as one
            # finishes, while that stream's last chunks may still be decoding
            # here, so for a moment there can be more streams than LM slots.
            # Such a stream buffers its chunks and takes the next free slot.
            state = StreamState(slot_index=self.decode_slots.acquire())
            self.stream_states[request_id] = state
        else:
            pass
        return state

    @staticmethod
    def needs_decode_slot(state: StreamState) -> bool:
        return state.slot_index is None and bool(state.pending_chunks)

    def bind_waiting_streams(self) -> None:
        """Hand free slots to streams in the order their first chunk became ready."""
        waiting = sorted(
            (
                state
                for state in self.stream_states.values()
                if self.needs_decode_slot(state)
            ),
            key=lambda state: state.ready_order,
        )
        for state in waiting:
            state.slot_index = self.decode_slots.acquire()
            if state.slot_index is None:
                return
            else:
                pass

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.stream_state(request_id)

    def release_decode_slot(self, state: StreamState) -> None:
        if state.slot_index is not None:
            self.decode_slots.release(state.slot_index)
            state.slot_index = None
            self.bind_waiting_streams()
        else:
            pass

    def clear_stream_state(self, request_id: str) -> None:
        state = self.stream_states.pop(request_id, None)
        if state is not None:
            self.release_decode_slot(state)
        else:
            pass

    def ingest_chunk(self, request_id: str, stream_item: StreamItem) -> StreamState:
        state = self.stream_state(request_id)
        # Note (wilsonzheng0327): The terminal payload only arrives after the LM finishes,
        # so the caller length travels with each chunk.
        num_samples = int((stream_item.metadata or {}).get("num_samples") or 0)
        if num_samples:
            if state.num_samples not in (0, num_samples):
                raise ValueError(
                    f"PersonaPlex num_samples changed from {state.num_samples} "
                    f"to {num_samples} for request {request_id!r}"
                )
            else:
                pass
            state.num_samples = num_samples
        else:
            pass
        if not state.pending_chunks:
            self.stamp_ready_order(state)
        else:
            pass
        state.pending_chunks.append(
            torch.as_tensor(
                stream_item.data,
                dtype=torch.long,
                device=self.decode_slots.codec.device,
            )
        )
        self.bind_waiting_streams()
        return state

    def stamp_ready_order(self, state: StreamState) -> None:
        state.ready_order = self.next_ready_order
        self.next_ready_order += 1

    def on_stream_chunk(
        self, request_id: str, stream_item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self.ingest_chunk(request_id, stream_item)
        if state.slot_index is None:
            return []
        else:
            return self.decode_slot_batch([(request_id, state)])

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Ingest chunks, then decode at least as many chunks as were
        ingested, so sustained message arrival cannot grow the backlog."""
        failed_request_ids: list[str] = []
        ingested_chunks = 0
        with self.state_lock:
            for request_id, stream_item in items:
                if self.is_aborted(request_id):
                    continue
                else:
                    pass
                try:
                    self.ingest_chunk(request_id, stream_item)
                except Exception as error:
                    self.emit_error(request_id, error)
                    self.abort_state(request_id)
                    failed_request_ids.append(request_id)
                else:
                    ingested_chunks += 1
        for request_id in failed_request_ids:
            self.cleanup_aborted_request(request_id)
        # Note (nightdecode): One step per message batch is not enough: a batch
        # may carry several chunks of one request, and a step serves only one
        # chunk width. Decoding at least the ingested count keeps the total
        # backlog from growing while the inbox stays non-empty; the extra steps
        # are bounded by the batch size.
        decoded_chunks = 0
        while True:
            step_chunks = self.run_decode_step()
            decoded_chunks += step_chunks
            if step_chunks == 0 or decoded_chunks >= ingested_chunks:
                break
            else:
                pass

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return bool(self.select_step_participants())

    def select_step_participants(self) -> list[tuple[str, StreamState]]:
        """The longest-waiting ready request and up to max_batch_size - 1 others
        whose next chunk has its width, longest-waiting first.

        Rows at any decode offset share a step; only the width must match.
        """
        width_groups: dict[int, list[tuple[str, StreamState]]] = {}
        for request_id, state in self.stream_states.items():
            if state.slot_index is not None and state.pending_chunks:
                width_groups.setdefault(state.pending_chunks[0].shape[0], []).append(
                    (request_id, state)
                )
            else:
                pass
        if not width_groups:
            return []
        else:
            pass
        for group in width_groups.values():
            group.sort(key=lambda entry: entry[1].ready_order)
        selected_group = min(
            width_groups.values(), key=lambda group: group[0][1].ready_order
        )
        return selected_group[: self.max_batch_size]

    def run_ready_step(self) -> None:
        self.run_decode_step()

    def run_decode_step(self) -> int:
        """Run one batched decode step; return the number of chunks consumed."""
        failed_request_ids: list[str] = []
        with self.state_lock:
            participants = self.select_step_participants()
            if not participants:
                return 0
            else:
                pass
            try:
                stream_messages = self.decode_slot_batch(participants)
            except Exception as error:
                for request_id, _ in participants:
                    self.emit_error(request_id, error)
                    self.abort_state(request_id)
                    failed_request_ids.append(request_id)
            else:
                for stream_message in stream_messages:
                    if not self.is_aborted(stream_message.request_id):
                        self.outbox.put(stream_message)
                    else:
                        pass
                for request_id, state in participants:
                    if state.stream_done and not state.pending_chunks:
                        self.release_decode_slot(state)
                        if request_id in self.stream_payloads:
                            self.complete_stream_request(
                                request_id,
                                [self.build_result_message(request_id, state)],
                            )
                        else:
                            pass
                    else:
                        pass
        for request_id in failed_request_ids:
            self.cleanup_aborted_request(request_id)
        return len(participants)

    @torch.inference_mode()
    def decode_slot_batch(
        self, participants: list[tuple[str, StreamState]]
    ) -> list[OutgoingMessage]:
        """Decode each participant's next chunk in one step on its own slot."""
        chunk_frames = participants[0][1].pending_chunks[0].shape[0]
        slot_indices = [state.slot_index for _, state in participants]
        assert all(slot_index is not None for slot_index in slot_indices)
        decoded_frames = [state.decoded_frames for _, state in participants]
        batched_codes = torch.stack(
            [state.pending_chunks[0].T for _, state in participants]
        )
        waveform_batch = self.decode_slots.decode_step(batched_codes, slot_indices)
        host_waveforms = waveform_batch[:, 0].float().cpu()
        for _, state in participants:
            state.pending_chunks.popleft()
            state.decoded_frames += chunk_frames
            if state.pending_chunks:
                self.stamp_ready_order(state)
            else:
                pass
        if get_recorder().is_active():
            emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_chunk_batch",
                metadata={
                    "batch_size": len(participants),
                    "chunk_frames": chunk_frames,
                    "graph_replayed": (len(participants), chunk_frames)
                    in self.decode_slots.graphs,
                    "decoded_frames": decoded_frames,
                    "slot_indices": slot_indices,
                    "participant_request_ids": [
                        request_id for request_id, _ in participants
                    ],
                    "active_slots": self.decode_slots.active_slot_count,
                    "waiting_streams": sum(
                        self.needs_decode_slot(state)
                        for state in self.stream_states.values()
                    ),
                    "inbox_depth": self.inbox.qsize(),
                },
            )
        else:
            pass
        return [
            self.build_stream_message(request_id, state, host_waveforms[row_index])
            for row_index, (request_id, state) in enumerate(participants)
        ]

    def build_stream_message(
        self, request_id: str, state: StreamState, waveform: torch.Tensor
    ) -> OutgoingMessage:
        if state.num_samples:
            waveform = waveform[
                ..., : max(state.num_samples - state.emitted_samples, 0)
            ]
        else:
            pass
        # Note (nightdecode): The row is a view into the whole batch's host
        # tensor; clone it so a long stream does not pin every batch it joined.
        waveform = waveform.clone()
        state.emitted_samples += waveform.shape[-1]
        state.audio_parts.append(waveform)
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint=SOURCE_HINT,
            ),
            metadata={"modality": "audio"},
        )

    def finish_stream_decode(self, state: StreamState) -> None:
        """Mark the stream done; its slot frees once its last chunk decodes."""
        state.stream_done = True
        if not state.pending_chunks:
            self.release_decode_slot(state)
        else:
            pass

    def build_result_message(
        self, request_id: str, state: StreamState
    ) -> OutgoingMessage:
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return OutgoingMessage(
            request_id=request_id,
            type="result",
            data=StagePayload(
                request_id=request_id,
                request=self.stream_payloads[request_id].request,
                data=audio_waveform_payload(
                    waveform,
                    sample_rate=SAMPLE_RATE,
                    modality="audio",
                    source_hint=SOURCE_HINT,
                ),
            ),
        )

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is not None:
            self.finish_stream_decode(state)
        else:
            pass
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        self.finish_stream_decode(state)
        if state.pending_chunks:
            return None
        else:
            pass
        return [self.build_result_message(request_id, state)]


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
