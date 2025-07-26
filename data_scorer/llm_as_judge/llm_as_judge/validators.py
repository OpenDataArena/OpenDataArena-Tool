from typing import Any, Dict, List
import orjson as json


def validate_model_response(
    parsed_result: Dict[str, Any],
    finish_reason: str,
    result_text: str,
    expected_keys: List[str],
):
    """
    Validates the response from the LLM.

    Args:
        parsed_result: The parsed JSON object from the model's response.
        finish_reason: The finish reason from the model's response.
        result_text: The raw text of the response for inclusion in error messages.
        expected_keys: A list of keys expected to be in the response.

    Raises:
        ValueError: If the validation fails for any reason.
    """
    # Check if the output was truncated
    if finish_reason == "length" or finish_reason == "content_filter":
        raise ValueError(f"Output truncated by model (finish_reason='{finish_reason}')")

    # Validate the presence and exclusivity of keys if expected_keys are provided
    if expected_keys:
        response_keys = set(parsed_result.keys())
        expected_keys_set = set(expected_keys)

        if response_keys != expected_keys_set:
            missing = sorted(list(expected_keys_set - response_keys))
            extra = sorted(list(response_keys - expected_keys_set))
            error_msg_parts = []
            if missing:
                error_msg_parts.append(f"missing keys: {missing}")
            if extra:
                error_msg_parts.append(f"extra keys: {extra}")
            
            raise ValueError(f"Response keys do not match expected keys ({'; '.join(error_msg_parts)}). Full response: {result_text}")
    elif not parsed_result: # Fallback check if no keys are expected from prompt
        raise ValueError("The response from the model was an empty JSON object.")

    # Validate that all returned values are scores between 1 and 10.
    for key, value in parsed_result.items():
        try:
            score_val = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"The value for key '{key}' must be a valid integer, but got '{value}'. Full response: {result_text}")
        
        if not 1 <= score_val <= 10:
            raise ValueError(f"The score for key '{key}' must be between 1 and 10, but got {score_val}. Full response: {result_text}") 