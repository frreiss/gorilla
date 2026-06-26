from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


class GraniteCompletionsHandler(OpenAICompletionsHandler):
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse a response for a tool-calling chat completion.
        
        The base class does not parse reasoning content, so we augment the base class's
        result with reasoning content. Behavior is otherwise identical to the base
        class."""
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
    
    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as
        DeepSeek and Grok. This method is included here to avoid code
        duplication.

        These models often don't take reasoning content in the chat history
        for next turn. Thus, this method saves reasoning content to
        response_data (for local result file) if present in the response,
        but does not include it in the chat history.

        This method is a copy of the eponymous method in the superclass,
        with a single change: use the key "reasoning" or "reasoning_content"
        instead of only looking for the latter. vLLM's API uses "reasoning"
        instead of "reasoning_content".
        """
        # Original assistant message object (contains `reasoning_content` on DeepSeek).
        message = api_response.choices[0].message

        # Preserve tool_call information but strip the unsupported
        # `reasoning_content` field before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data[
                "model_responses_message_for_chat_history"
            ] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        # (vLLM calls this field "reasoning")
        elif hasattr(message, "reasoning_content") or hasattr(
            message, "reasoning"
        ):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        # Capture the reasoning trace so it can be logged to the local
        # result file. (vLLM calls this field "reasoning")
        if hasattr(message, "reasoning_content") or hasattr(
            message, "reasoning"
        ):
            response_data["reasoning_content"] = (
                message.reasoning_content
                if hasattr(message, "reasoning_content")
                else message.reasoning
            )
