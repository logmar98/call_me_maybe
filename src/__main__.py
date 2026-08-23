import sys
import argparse
import os
import json
import numpy as np
from time import time
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model

# --- Pydantic Models ---

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
    sdk: Small_LLM_Model
    output_path: str
    vocab: Dict[int, str]
    prompts: List[Dict[str, str]]
    definitions_str: str
    definitions: List[Dict[str, Any]]
    func_names: List[str]
    func_ids: List[int]
    state: str
    function_name: str

    def __init__(
        self, def_path: str, input_path: str, output_path: str
    ) -> None:
        self.sdk = Small_LLM_Model()
        self.output_path = output_path
        self.func_ids = []
        self.definitions = []
        self.func_names = []

        # Safely load vocabulary
        try:
            with open(self.sdk.get_path_to_vocab_file()) as file:
                vocab_raw: Dict[str, Any] = json.load(file)
                self.vocab = {int(v): str(k) for k, v in vocab_raw.items()}
        except Exception as e:
            print(f"Error loading vocabulary file: {e}")
            sys.exit(1)

        # Safely load input prompts
        try:
            with open(input_path) as f_calling:
                self.prompts = json.load(f_calling)
        except Exception as e:
            print(f"Error loading input file '{input_path}': {e}")
            sys.exit(1)

        # Safely load and validate definitions
        try:
            with open(def_path) as f_definition:
                self.definitions_str = f_definition.read()
                raw_defs: List[Dict[str, Any]] = json.loads(
                    self.definitions_str
                )

                for d in raw_defs:
                    valid_def = FunctionDef(**d)
                    self.definitions.append(valid_def.model_dump())
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
                    self.func_ids.append(i)
                    
        # --- NEW CACHING LOGIC ---
        self.cache_double_quote = []
        self.cache_comma = []
        self.cache_parameters_key = []
        self.cache_colon_and_brace = []
        self.cache_no_end_brace = []
        self.cache_valid_closure = []
        self.cache_end_brace = []

        for i, v in self.vocab.items():
            v_lower = v.lower()
            if '"' in v:
                self.cache_double_quote.append(i)
            if ',' in v:
                self.cache_comma.append(i)
            if 'parameter' in v or '"' in v or 's' in v:
                self.cache_parameters_key.append(i)
            if ':' in v or '{' in v or ' ' in v:
                self.cache_colon_and_brace.append(i)
            if '}' not in v:
                self.cache_no_end_brace.append(i)
            if ',' not in v_lower and 'return' not in v_lower:
                self.cache_valid_closure.append(i)
            if '}' in v:
                self.cache_end_brace.append(i)
        # -------------------------

        self.state = "FUNCTION_NAME"
        self.function_name = ""
        # --- AT THE END OF YOUR NEW CACHING LOGIC IN __INIT__ ---
        self.func_ids = np.array(self.func_ids, dtype=np.int32)
        self.cache_double_quote = np.array(self.cache_double_quote, dtype=np.int32)
        self.cache_comma = np.array(self.cache_comma, dtype=np.int32)
        self.cache_parameters_key = np.array(self.cache_parameters_key, dtype=np.int32)
        self.cache_colon_and_brace = np.array(self.cache_colon_and_brace, dtype=np.int32)
        self.cache_no_end_brace = np.array(self.cache_no_end_brace, dtype=np.int32)
        self.cache_valid_closure = np.array(self.cache_valid_closure, dtype=np.int32)
        self.cache_end_brace = np.array(self.cache_end_brace, dtype=np.int32)

    def get_allowed_tokens_for_current_state(
        self, current_state_tokens: List[int]
    ) -> List[int]:
        
        # Replace the linear scans with O(1) cache lookups
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
                if x['name'] == getattr(self, 'function_name', ''):
                    expected_args = list(x['parameters'].keys())
                    break

            output: str = self.sdk.decode(current_state_tokens)
            clean_output: str = output.strip()

            # 1. Force opening quotes for keys right after a brace or comma
            if clean_output.endswith('{') or clean_output.endswith(','):
                return self.cache_double_quote

            if not output.strip().startswith('"'):
                output = '"' + output
                clean_output = output.strip()

            # 2. Check if all expected keys have been generated
            all_args_present: bool = True
            for arg in expected_args:
                if f'"{arg}"' not in output:
                    all_args_present = False
                    break

            # 3. Verify that the final value has actually been typed
            is_complete: bool = False
            if all_args_present and expected_args:
                last_arg = expected_args[-1]
                last_arg_idx = output.rfind(f'"{last_arg}"')
                if last_arg_idx != -1:
                    colon_idx = output.find(':', last_arg_idx)
                    # If colon exists, ensure we are not hanging right after it
                    if colon_idx != -1 and not clean_output.endswith(':'):
                        is_complete = True
            elif all_args_present and len(expected_args) == 0:
                is_complete = True

            if not is_complete:
                # We still need args or values, completely block the closing '}'
                return self.cache_no_end_brace
            else:
                # All required args and values are fully present!
                # Block commas AND hallucination words to force a valid closure.
                return self.cache_valid_closure

        elif self.state == "END_BRACE":
            return self.cache_end_brace

        return []

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

    def constrained(self) -> None:
        validated_output_list: List[Dict[str, Any]] = []

        for prompt in self.prompts:
            output_text: List[int] = []
            current_output: List[int] = []
            self.state = "FUNCTION_NAME"
            self.function_name = ""

            p_text = prompt.get('prompt', None)
            if not p_text:
                break
            
            # Using json.dumps ensures any quotes in the prompt are escaped safely
            safe_prompt: str = json.dumps(p_text)
            
            generated_text: str = (
                "You are a helpful assistant that translates natural\n"
                "language into function calls.\n"
                "You must output ONLY a valid JSON object with this format:\n"
                '{"name": "<function_name>", "parameters": {"<p1>": <v1>}}\n\n'
                "Available functions:\n"
                f"{self.definitions_str}\n\n"
                f"User: {p_text}\n"
                "Assistant: "
            )
            generated_text += '{"prompt": ' + safe_prompt + ', "name": "'

            tokens = self.sdk.encode(generated_text)
            current_sequence: List[int] = tokens.tolist()[0]

            for _ in range(500):
                raw_logits = self.sdk.get_logits_from_input_ids(
                    current_sequence
                )
                
                # 1. Convert the raw list to a NumPy array once
                raw_logits_np = np.array(raw_logits)
                
                choose_ids = self.get_allowed_tokens_for_current_state(
                    current_output
                )

                # 2. Vectorized assignment (No more Python 'for' loops!)
                if len(choose_ids) == 0:
                    masked_logits = raw_logits_np
                else:
                    masked_logits = np.full(len(raw_logits_np), -np.inf)
                    masked_logits[choose_ids] = raw_logits_np[choose_ids]

                next_token_id = int(np.argmax(masked_logits))
                
                current_sequence.append(next_token_id)
                output_text.append(next_token_id)
                current_output.append(next_token_id)
                current_output = self.transition(current_output)

                print('{"prompt": ' + safe_prompt + ', "name": "' + self.sdk.decode(output_text))
                if self.state == "DONE":
                    break

            try:
                generated_json_str = (
                    '{"name": "' + self.sdk.decode(output_text)
                )
                
                try:
                    parsed_dict: Dict[str, Any] = json.loads(generated_json_str)
                except json.JSONDecodeError as e:
                    if "Invalid \\escape" in str(e):
                        salvaged_str = generated_json_str.replace('\\', '\\\\')
                        parsed_dict = json.loads(salvaged_str)
                    else:
                        raise e
                
                parsed_dict["prompt"] = p_text

                for k, v in parsed_dict.get('parameters', {}).items():
                    if isinstance(v, int):
                        parsed_dict["parameters"][k] = float(v)

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
        "--output", default="data/output/function_calling_results.json"
    )

    args = parser.parse_args()

    ovj = OutputValidJson(args.functions_definition, args.input, args.output)
    ovj.constrained()

    end_time: float = time()
    elapsed: float = end_time - start_time
    print(f"Execution time: {elapsed / 60.0:.2f} minutes")

if __name__ == "__main__":
    main()