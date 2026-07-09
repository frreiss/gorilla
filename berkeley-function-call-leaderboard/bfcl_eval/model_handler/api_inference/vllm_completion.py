from typing import Any
import json
import os
import time


from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler import utils
from openai import RateLimitError, APITimeoutError

# Limit completion tokens to emulate commercial API behvarior
_DEFAULT_MAX_COMPLETION_TOKENS = 4096

class VLLMCompletionsHandler(OpenAICompletionsHandler):
    """
    Generic handler for vLLM's OpenAI-compatible chat completions endpoint.

    The vLLM implementation diverges somewhat from the commercial OpenAI endpoint
    when dealing with reasoning models and tool calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Optionally read values for the `extra_body` parameter from an environment
        # variable.
        self.extra_body = None
        if extra_body_env := os.getenv("OPENAI_EXTRA_BODY"):
            self.extra_body = json.loads(extra_body_env)

    @utils.retry_with_backoff(error_type=[RateLimitError, APITimeoutError])
    def generate_with_backoff(self, **kwargs):
        """vLLM-specific inner loop for generation.

        This code is identical to the superclass method, with the following changes:
        * additional exceptions that trigger retry.
        """
        start_time = time.time()
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = _DEFAULT_MAX_COMPLETION_TOKENS
        if self.extra_body:
            print(f"Setting extra_body={self.extra_body}")
            kwargs["extra_body"] = self.extra_body
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time
    
    
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse a model response with function calling.
        
        Modified from the base class's version in the following ways:
        * Handle a tool_calls array with zero elements (as produced by vLLM)
        * Add reasoning content if present
        """
        if len(api_response.choices) != 1:
            raise ValueError(f"Response has {len(api_response.choices)}; expected 1")
        choice = api_response.choices[0]
        message = choice.message
        
        if message.tool_calls:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in message.tool_calls
            ]
        else:
            # Not a tool call; look for a message to the user
            if isinstance(message.content, str):
                model_responses = message.content
            elif choice.finish_reason == "length":
                # Ran out of context during chain of thought
                model_responses = []
            else:
                raise TypeError(f"Don't know how to handle content of type "
                                f"{type(message.content)}.\n"
                                f"Response was: {api_response}")
            tool_call_ids = []
        
        # Check for reasoning content
        if hasattr(message, "reasoning_content"):
            reasoning = message.reasoning_content
        elif hasattr(message, "reasoning"):
            reasoning = message.reasoning
        else:
            reasoning = None
            
        #print(f"Input:\n{api_response}")

        result = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": message,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
            "reasoning_content": reasoning
        }
        print(f"Output:\n{result}")
        return result
    
    def decode_execute(self, result, has_tool_call_tag):
        """Handler for decoding raw output for a single step.
        
        Modified form the base class's version in the following ways:
        * Ignore free-text responses with function-calling models
        """
        if self.is_fc_model:
            if isinstance(result, list):
                return utils.convert_to_function_call(result)
            return []
        else:
            return utils.default_decode_execute_prompting(result)
    
    # def _parse_query_response_FC(self, api_response: Any) -> dict:
    #     """Parse a response for a tool-calling chat completion.
        
    #     The base class does not parse reasoning content, so we augment the base class's
    #     result with reasoning content. Behavior is otherwise identical to the base
    #     class."""
    #     response_data = super()._parse_query_response_FC(api_response)
    #     self._add_reasoning_content_if_available_FC(api_response, response_data)
    #     # print(f"Processing API response:\n{api_response}")
    #     # print(f"Processed response data is:\n{response_data}")
    #     return response_data
    
    # def _add_reasoning_content_if_available_FC(
    #     self, api_response: Any, response_data: dict
    # ) -> None:
    #     """
    #     OpenAI models don't show reasoning content in the api response,
    #     but many other models that use the OpenAI interface do, such as
    #     DeepSeek and Grok. This method is included here to avoid code
    #     duplication.

    #     These models often don't take reasoning content in the chat history
    #     for next turn. Thus, this method saves reasoning content to
    #     response_data (for local result file) if present in the response,
    #     but does not include it in the chat history.

    #     This method is a copy of the eponymous method in the superclass,
    #     with a single change: use the key "reasoning" or "reasoning_content"
    #     instead of only looking for the latter. vLLM's API uses "reasoning"
    #     instead of "reasoning_content".
    #     """
    #     # Original assistant message object (contains `reasoning_content` on DeepSeek).
    #     message = api_response.choices[0].message

    #     # Preserve tool_call information but strip the unsupported
    #     # `reasoning_content` field before inserting into chat history.
    #     if getattr(message, "tool_calls", None):
    #         assistant_message = {
    #             "role": "assistant",
    #             "content": message.content,
    #             "tool_calls": [
    #                 {
    #                     "id": tool_call.id,
    #                     "type": tool_call.type,
    #                     "function": {
    #                         "name": tool_call.function.name,
    #                         "arguments": tool_call.function.arguments,
    #                     },
    #                 }
    #                 for tool_call in message.tool_calls
    #             ],
    #         }
    #         response_data[
    #             "model_responses_message_for_chat_history"
    #         ] = assistant_message

    #     # If no tool_calls, we still need to strip reasoning_content.
    #     # (vLLM calls this field "reasoning")
    #     elif hasattr(message, "reasoning_content") or hasattr(
    #         message, "reasoning"
    #     ):
    #         response_data["model_responses_message_for_chat_history"] = {
    #             "role": "assistant",
    #             "content": message.content,
    #         }

    #     # Capture the reasoning trace so it can be logged to the local
    #     # result file. (vLLM calls this field "reasoning")
    #     if hasattr(message, "reasoning_content") or hasattr(
    #         message, "reasoning"
    #     ):
    #         response_data["reasoning_content"] = (
    #             message.reasoning_content
    #             if hasattr(message, "reasoning_content")
    #             else message.reasoning
    #         )
