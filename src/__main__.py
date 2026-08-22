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

        self.state = "FUNCTION_NAME"
        self.function_name = ""

    def get_allowed_tokens_for_current_state(
        self, current_state_tokens: List[int]
    ) -> List[int]:
        chosen_ids: List[int] = []

        if self.state == "FUNCTION_NAME":
            return self.func_ids

        elif self.state == "DOUBLE_QUOTE":
            for i, v in self.vocab.items():
                if '"' in v:
                    chosen_ids.append(i)

        elif self.state == "COMMA":
            for i, v in self.vocab.items():
                if ',' in v:
                    chosen_ids.append(i)

        elif self.state == "PARAMETERS_KEY":
            for i, v in self.vocab.items():
                if 'parameter' in v or '"' in v or 's' in v:
                    chosen_ids.append(i)

        elif self.state == "COLON_AND_BRACE":
            for i, v in self.vocab.items():
                if ':' in v or '{' in v or ' ' in v:
                    chosen_ids.append(i)

        elif self.state == "PARAM_CONTENT":
            expected_args: List[str] = []
            for x in self.definitions:
                if x['name'] == getattr(self, 'function_name', ''):
                    expected_args = list(x['parameters'].keys())
                    break

            output: str = self.sdk.decode(current_state_tokens)
            clean_output: str = output.strip()

            if clean_output.endswith('{') or clean_output.endswith(','):
                for i, v in self.vocab.items():
                    if '"' in v:
                        chosen_ids.append(i)
                return chosen_ids

            if not output.strip().startswith('"'):
                output = '"' + output

            all_args_present: bool = all(
                f'"{arg}"' in output for arg in expected_args
            )

            if not all_args_present:
                for i, v in self.vocab.items():
                    if '}' not in v:
                        chosen_ids.append(i)
            else:
                for i, v in self.vocab.items():
                    v_lower: str = v.lower()
                    if ',' not in v_lower and 'return' not in v_lower:
                        chosen_ids.append(i)

            return chosen_ids

        elif self.state == "END_BRACE":
            for i, v in self.vocab.items():
                if '}' in v:
                    chosen_ids.append(i)

        return chosen_ids

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
        all_output: List[str] = []
        current_output: List[int] = []

        for prompt in self.prompts:
            output_text: List[int] = []
            self.state = "FUNCTION_NAME"

            p_text = prompt['prompt']
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
            generated_text += '{"prompt": "' + p_text + '", "name": "'

            tokens = self.sdk.encode(generated_text)
            current_sequence: List[int] = tokens.tolist()[0]

            for _ in range(500):
                raw_logits = self.sdk.get_logits_from_input_ids(
                    current_sequence
                )
                masked_logits = np.full(len(raw_logits), -np.inf)

                choose_ids: List[int] = (
                    self.get_allowed_tokens_for_current_state(
                        current_output
                    )
                )

                if not choose_ids:
                    masked_logits = np.array(raw_logits)

                for i in choose_ids:
                    masked_logits[i] = raw_logits[i]

                next_token_id: int = int(np.argmax(masked_logits))
                current_sequence.append(next_token_id)
                output_text.append(next_token_id)
                current_output.append(next_token_id)
                current_output = self.transition(current_output)

                if self.state == "DONE":
                    all_output.append(
                        '{"prompt": "' + p_text +
                        '", "name": "' + self.sdk.decode(output_text)
                    )
                    break

        # Safely validate and save the output
        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self.output_path, "w") as f:
                all_output_str: str = ", ".join(all_output)
                all_output_str = '[' + all_output_str + ']'

                all_list: List[Dict[str, Any]] = json.loads(all_output_str)
                validated_output_list: List[Dict[str, Any]] = []

                for x in all_list:
                    for k, v in x.get('parameters', {}).items():
                        if isinstance(v, int):
                            x["parameters"][k] = float(v)

                    valid_result = FunctionCallResult(**x)
                    validated_output_list.append(valid_result.model_dump())

                json.dump(validated_output_list, f, indent=4)
                print(f"Successfully saved to {self.output_path}")
        except ValidationError as e:
            print(f"Error: Generated JSON failed schema validation:\n{e}")
            sys.exit(1)
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
