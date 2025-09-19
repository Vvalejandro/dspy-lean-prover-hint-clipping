from __future__ import annotations

from dspy.utils.callback import BaseCallback

class PromptCallback(BaseCallback):
    def __init__(self):
        self.prompts = []

    def on_request_start(self, request_data):
        self.prompts.append(request_data["messages"])

    def on_request_end(self, response_data):
        pass
