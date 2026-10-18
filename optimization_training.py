"""Opt-in read-only training observations; never changes AMP or optimizer state."""
import hashlib
import json
import math
from pathlib import Path

import torch

from adapter_checkpoint_v2 import adapter_state_dict
from optimization_evidence import state_sha, tensor_sha, write_json


class TrainingObserver:
    def __init__(self, out, expected_initial=None, expected_batches=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.handle = (self.out / 'steps.jsonl').open('x', encoding='utf-8')
        self.batch_chain = hashlib.sha256()
        self.pending = {}
        self.expected_initial = expected_initial
        self.expected_batches = expected_batches

    def on_start(self, model):
        initial = {
            'adapter_state_sha256': state_sha(adapter_state_dict(model)),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        write_json(self.out / 'initial_state.json', initial)
        if self.expected_initial is not None and initial != self.expected_initial:
            raise ValueError('paired initial state mismatch BEFORE training')

    def on_batch(self, step, images, labels):
        image_hash, label_hash = tensor_sha(images), tensor_sha(labels)
        if self.expected_batches is not None and (image_hash, label_hash) != self.expected_batches[step]:
            raise ValueError('paired augmented batch mismatch BEFORE optimizer update')
        self.batch_chain.update((image_hash + label_hash).encode())
        self.pending = dict(step=step, image_sha256=image_hash, label_sha256=label_hash,
                            batch_chain_sha256=self.batch_chain.hexdigest())

    def on_backward(self, step, loss, parameters, scale):
        # Divide detached norm values for reporting only. NEVER unscale_ the optimizer.
        if step < 10 or step in (98, 99, 100, 101, 1999) or step % 25 == 0:
            norms = [p.grad.detach().float().norm() / scale
                     for p in parameters if p.grad is not None]
            norm = torch.stack(norms).norm().item() if norms else 0.
            self.pending.update(loss=float(loss.detach()),
                                unscaled_gradient_norm=norm if math.isfinite(norm) else None,
                                gradient_norm_finite=math.isfinite(norm))

    def on_step(self, step, old_scale, new_scale):
        self.pending.update(amp_scale=old_scale, amp_skipped_step=new_scale < old_scale)
        self.handle.write(json.dumps(self.pending, allow_nan=False) + '\n')
        if step % 25 == 0:
            self.handle.flush()

    def close(self):
        self.handle.close()
