"""Batch policy forward requests from independently stepping rollout groups."""

from concurrent.futures import Future
from queue import Empty, Queue
from threading import Thread
from time import monotonic

import torch

from models.v1_spatial_net import SpatialState
from train.train_v1 import precision


class BatchedRolloutPolicy:
    def __init__(self, model, config, device, groups):
        self.model = model
        self.config = config
        self.device = device
        self.groups = groups
        self.wait_seconds = config.get("rollout_batch_wait_ms", 0.5) / 1000
        self.requests = Queue()
        self.thread = Thread(target=self.run, name="rollout-inference")
        self.calls = self.batches = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.requests.put(None)
        self.thread.join()

    def __call__(self, obs, state, privileged):
        result = Future()
        self.requests.put((obs, state, privileged, result))
        return result.result()

    @torch.no_grad()
    def run(self):
        while (first := self.requests.get()) is not None:
            requests = [first]
            deadline = monotonic() + self.wait_seconds
            while len(requests) < self.groups:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self.requests.get(timeout=remaining))
                except Empty:
                    break
            try:
                results = self.forward(requests)
                self.calls += len(requests)
                self.batches += 1
                for request, result in zip(requests, results):
                    request[-1].set_result(result)
            except Exception as error:
                for request in requests:
                    request[-1].set_exception(error)

    def forward(self, requests):
        with precision(self.device, self.config):
            if len(requests) == 1:
                return [self.model(*requests[0][:3])]
            obs, states, privileged, _ = zip(*requests)
            widths = [item.shape[0] for item in obs]
            merged_state = SpatialState(*(torch.cat(items) for items in zip(*states)))
            logits, values, _, next_state = self.model(torch.cat(obs), merged_state, torch.cat(privileged))
        next_states = (SpatialState(*items) for items in zip(*(item.split(widths) for item in next_state)))
        return [(policy, value, None, state) for policy, value, state in
                zip(logits.split(widths), values.split(widths), next_states)]
