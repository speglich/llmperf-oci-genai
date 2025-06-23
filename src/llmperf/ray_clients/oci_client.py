import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage

@ray.remote
class OCIGenAIChatCompletionsClient(LLMClient):
    """Client for OCIGenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        max_tokens = request_config.sampling_params.get("max_tokens", 600)
        prompt, prompt_len = prompt

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        character_count = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        self.compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        self.auth_type = os.getenv("OCI_AUTH_TYPE")
        self.config_profile = os.getenv("OCI_CONFIG_PROFILE")
        self.endpoint = os.getenv("OCI_ENDPOINT")

        self.model_id = os.getenv("OCI_MODEL_ID")
        self.provider = os.getenv("OCI_PROVIDER")

        if not self.compartment_id or not self.auth_type or not self.config_profile or not self.endpoint or not self.model_id:
            print(self.compartment_id, self.auth_type, self.config_profile, self.endpoint, self.model_id, self.provider, flush=True)
            raise ValueError(
                "Environment variables OCI_COMPARTMENT_ID, OCI_AUTH_TYPE, "
                "OCI_CONFIG_PROFILE, OCI_ENDPOINT, OCI_MODEL_ID and OCI_PROVIDER must be set."
            )

        self.chat = ChatOCIGenAI(
            model_id=self.model_id,
            service_endpoint=self.endpoint,
            compartment_id=self.compartment_id,
            provider=self.provider,
            is_stream=True,
            model_kwargs={
                "temperature": 1,
                "frequency_penalty": 0,
                "max_tokens": max_tokens,
                "presence_penalty": 0,
                "top_p": 0.75
            },
            auth_type=self.auth_type,
            auth_profile=self.config_profile,
        )

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            messages = [
                HumanMessage(content=prompt),
            ]

            response_stream = self.chat.stream(messages)

            for chunk in response_stream:
                if hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    if not ttft:
                        ttft = time.monotonic() - start_time
                    else:
                        time_to_next_token.append(time.monotonic() - most_recent_received_token_time)
                    most_recent_received_token_time = time.monotonic()

                    generated_text += token
                    tokens_received += 1
                    character_count += len(token)

            total_request_time = time.monotonic() - start_time

            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
