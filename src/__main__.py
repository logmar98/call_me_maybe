import argparse
import json
import os
import sys
from time import time
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, ValidationError

from llm_sdk import Small_LLM_Model


class ParameterDef(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class OutputValidJson:
    sdk: Any
    output_path: str
    vocab: Dict[int, str]
    prompts: List[Dict[str, Any]]
    definitions_str: str
    definitions: List[Dict[str, Any]]
    func_names: List[str]
    func_ids: Any
    state: str
    function_name: str
    cache_double_quote: Any
    cache_comma: Any
    cache_parameters_key: Any
    cache_colon_and_brace: Any
    cache_no_end_brace: Any
    cache_valid_closure: Any
    cache_end_brace: Any

    def __init__(
        self, def_path: str, input_path: str, output_path: str
    ) -> None:
        self.sdk = Small_LLM_Model()
        self.output_path = output_path

        func_ids_list: List[int] = []
        self.definitions = []
        self.func_names = []

        try:
            with open(self.sdk.get_path_to_vocab_file()) as file:
                vocab_raw: Dict[str, Any] = json.load(file)
                self.vocab = {int(v): str(k) for k, v in vocab_raw.items()}
        except Exception as e:
            print(f"Error loading vocabulary file: {e}")
            sys.exit(1)

        try:
            with open(input_path) as f_calling:
                self.prompts = json.load(f_calling)
        except Exception as e:
            print(f"Error loading input file '{input_path}': {e}")
            sys.exit(1)

        try:
            with open(def_path) as f_definition:
                self.definitions_str = f_definition.read()
                raw_defs: List[Dict[str, Any]] = json.loads(
                    self.definitions_str
                )

                for d in raw_defs:
                    valid_def = FunctionDef(**d)
                    self.definitions.append(valid_def.model_dump())
                if not self.definitions:
                    print("Error: missing definitions")
                    sys.exit(1)
        except ValidationError as e:
            print(f"Validation error in file '{def_path}':\n{e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading definitions file '{def_path}': {e}")
            sys.exit(1)

        self.func_names = [str(k["name"]) for k in self.definitions]
        for x in self.func_names:
            for i, v in self.vocab.items():
                if v in x:
                    func_ids_list.append(i)

        cache_double_quote_list: List[int] = []
        cache_comma_list: List[int] = []
        cache_parameters_key_list: List[int] = []
        cache_colon_and_brace_list: List[int] = []
        cache_no_end_brace_list: List[int] = []
        cache_valid_closure_list: List[int] = []
        cache_end_brace_list: List[int] = []

        for i, v in self.vocab.items():
            v_lower = v.lower()
            if '"' in v:
                cache_double_quote_list.append(i)
            if ',' in v:
                cache_comma_list.append(i)
            if 'parameter' in v or '"' in v or 's' in v:
                cache_parameters_key_list.append(i)
            if ':' in v or '{' in v or ' ' in v:
                cache_colon_and_brace_list.append(i)
            if '}' not in v:
                cache_no_end_brace_list.append(i)
            if ',' not in v_lower and 'return' not in v_lower:
                cache_valid_closure_list.append(i)
            if '}' in v:
                cache_end_brace_list.append(i)

        self.state = "FUNCTION_NAME"
        self.function_name = ""

        self.func_ids = np.array(func_ids_list, dtype=np.int32)
        self.cache_double_quote = np.array(
            cache_double_quote_list, dtype=np.int32
        )
        self.cache_comma = np.array(cache_comma_list, dtype=np.int32)
        self.cache_parameters_key = np.array(
            cache_parameters_key_list, dtype=np.int32
        )
        self.cache_colon_and_brace = np.array(
            cache_colon_and_brace_list, dtype=np.int32
        )
        self.cache_no_end_brace = np.array(
            cache_no_end_brace_list, dtype=np.int32
        )
        self.cache_valid_closure = np.array(
            cache_valid_closure_list, dtype=np.int32
        )
        self.cache_end_brace = np.array(
            cache_end_brace_list, dtype=np.int32
        )

    def get_allowed_tokens_for_current_state(
        self, current_state_tokens: List[int]
    ) -> Any:
        if self.state == "FUNCTION_NAME":
            return self.func_ids

        elif self.state == "DOUBLE_QUOTE":
            return self.cache_double_quote

        elif self.state == "COMMA":
            return self.cache_comma

        elif self.state == "PARAMETERS_KEY":
            return self.cache_parameters_key

        elif self.state == "COLON_AND_BRACE":
            return self.cache_colon_and_brace

        elif self.state == "PARAM_CONTENT":
            expected_args: List[str] = []
            for x in self.definitions:
                if x.get('name') == getattr(self, 'function_name', ''):
                    expected_args = list(x.get('parameters', {}).keys())
                    break

            output: str = self.sdk.decode(current_state_tokens)
            clean_output: str = output.strip()

            if clean_output.endswith('{') or clean_output.endswith(','):
                return self.cache_double_quote

            if not output.strip().startswith('"'):
                output = '"' + output
                clean_output = output.strip()

            all_args_present: bool = True
            for arg in expected_args:
                if f'"{arg}"' not in output:
                    all_args_present = False
                    break

            is_complete: bool = False
            if all_args_present and expected_args:
                last_arg = expected_args[-1]
                last_arg_idx = output.rfind(f'"{last_arg}"')
                if last_arg_idx != -1:
                    colon_idx = output.find(':', last_arg_idx)
                    if colon_idx != -1 and not clean_output.endswith(':'):
                        is_complete = True
            elif all_args_present and len(expected_args) == 0:
                is_complete = True

            if not is_complete:
                return self.cache_no_end_brace
            else:
                return self.cache_valid_closure

        elif self.state == "END_BRACE":
            return self.cache_end_brace

        return np.array([], dtype=np.int32)

    def transition(self, current_state_tokens: List[int]) -> List[int]:
        old_state: str = self.state
        output: str = self.sdk.decode(current_state_tokens)

        if self.state == "FUNCTION_NAME" and output in self.func_names:
            self.function_name = output
            self.state = "DOUBLE_QUOTE"

        if self.state == "DOUBLE_QUOTE" and '"' in output:
            self.state = "COMMA"

        if self.state == "COMMA" and ',' in output:
            self.state = "PARAMETERS_KEY"

        if self.state == "PARAMETERS_KEY" and 'parameters' in output:
            self.state = "COLON_AND_BRACE"

        if self.state == "COLON_AND_BRACE" and '{' in output:
            self.state = "PARAM_CONTENT"

        if self.state == "PARAM_CONTENT" and '}' in output:
            self.state = "END_BRACE"

        if self.state == "END_BRACE" and '}' in output:
            self.state = "DONE"

        if old_state != self.state:
            return []

        return current_state_tokens

    def get_compact_definitions(self) -> str:
        compact_defs: List[Dict[str, Any]] = []
        for func in self.definitions:
            params = {
                k: v.get("type", "string")
                for k, v in func.get("parameters", {}).items()
            }

            compact_defs.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": params
            })

        return json.dumps(compact_defs, separators=(',', ':'))

    def constrained(self) -> None:
        validated_output_list: List[Dict[str, Any]] = []

        for prompt in self.prompts:
            output_text: List[int] = []
            current_output: List[int] = []
            self.state = "FUNCTION_NAME"
            self.function_name = ""

            p_text = prompt.get('prompt')
            if not p_text:
                break
            print(f"Prompt: {p_text}\n")

            safe_prompt: str = json.dumps(p_text)
            compact_defs_str = self.get_compact_definitions()

            generated_text = (
                "<|im_start|>system\n"
                "You are an AI that translates natural language "
                "into JSON function calls.\n"
                "You must output ONLY a valid JSON object. Use "
                "the EXACT parameter keys from the available "
                "functions.\n\n"
                "Available functions:\n"
                f"{compact_defs_str}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{p_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
                '{"name": "'
            )

            tokens = self.sdk.encode(generated_text)
            current_sequence: List[int] = tokens.tolist()[0]
            running_str = '{"name": "'

            for _ in range(500):
                raw_logits = self.sdk.get_logits_from_input_ids(
                    current_sequence
                )
                raw_logits_np = np.array(raw_logits)
                choose_ids = self.get_allowed_tokens_for_current_state(
                    current_output
                )

                if len(choose_ids) == 0:
                    masked_logits = raw_logits_np
                else:
                    masked_logits = np.full(len(raw_logits_np), -np.inf)
                    masked_logits[choose_ids] = raw_logits_np[choose_ids]

                next_token_id = int(np.argmax(masked_logits))

                current_sequence.append(next_token_id)
                output_text.append(next_token_id)
                current_output.append(next_token_id)

                new_token_str = self.vocab.get(next_token_id, "").replace(
                    "Ġ", " "
                ).replace("Ċ", "\n")
                running_str += new_token_str
                decoded_str = self.sdk.decode(output_text)
                print(
                    '{"prompt": ' + safe_prompt + ', "name": "' + decoded_str
                )
                try:
                    parsed_check = json.loads(running_str)
                    if (
                        isinstance(parsed_check, dict)
                        and "name" in parsed_check
                    ):
                        self.state = "DONE"
                        break
                except json.JSONDecodeError:
                    pass

                current_output = self.transition(current_output)

                if self.state == "DONE":
                    break

            try:
                try:
                    parsed_dict: Dict[str, Any] = json.loads(running_str)
                except json.JSONDecodeError as e:
                    if "Invalid \\escape" in str(e):
                        salvaged_str = running_str.replace('\\', '\\\\')
                        parsed_dict = json.loads(salvaged_str)
                    else:
                        raise e

                parsed_dict["prompt"] = str(p_text)

                func_name = parsed_dict.get("name")
                expected_params: Dict[str, Any] = {}

                for d in self.definitions:
                    if d.get("name") == func_name:
                        expected_params = d.get("parameters", {})
                        break

                for k, v in parsed_dict.get('parameters', {}).items():
                    expected_type = expected_params.get(k, {}).get("type")

                    if expected_type in ["number", "integer"]:
                        try:
                            parsed_val = float(v)
                            if expected_type == "integer":
                                parsed_dict["parameters"][k] = int(parsed_val)
                            else:
                                parsed_dict["parameters"][k] = parsed_val
                        except (ValueError, TypeError):
                            pass
                    elif expected_type == "string":
                        parsed_dict["parameters"][k] = str(v)

                valid_result = FunctionCallResult(**parsed_dict)
                validated_output_list.append(valid_result.model_dump())
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Skipping malformed output for '{p_text}': {e}")
                continue

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self.output_path, "w") as f:
                json.dump(validated_output_list, f, indent=4)
                print(f"Successfully saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving results to '{self.output_path}': {e}")
            sys.exit(1)


def main() -> None:
    start_time: float = time()

    parser = argparse.ArgumentParser(description="Function Calling Tool")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calls.json"
    )

    args = parser.parse_args()
    try:
        print("-------------Call Me Maybe-------------\n")
        ovj = OutputValidJson(args.functions_definition,
                              args.input, args.output)
        ovj.constrained()
    except KeyboardInterrupt:
        print("exit the program")
        sys.exit(0)
    except Exception as e:
        print(e)

    end_time: float = time()
    elapsed: float = end_time - start_time
    print(f"Execution time: {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
