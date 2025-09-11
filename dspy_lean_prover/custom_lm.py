from __future__ import annotations

import dspy
from dspy.clients.lm import LM
import litellm

class CustomLM(LM):
    """A custom LM that accepts a logprobs parameter."""

    def __init__(self, *args, **kwargs):
        self.logprobs = kwargs.pop("logprobs", False)
        super().__init__(*args, **kwargs)
        if self.logprobs:
            self.kwargs['logprobs'] = self.logprobs

    def forward(self, prompt=None, messages=None, **kwargs):
        print(f"CustomLM.forward called with logprobs={self.logprobs}")
        # The logprobs are already in self.kwargs, so they will be passed to litellm
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        print(f"CustomLM.aforward called with logprobs={self.logprobs}")
        # The logprobs are already in self.kwargs, so they will be passed to litellm
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)
