from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
from time import time
import os
# import sys
import argparse


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



class OutputValidJson():
    def __init__(self, def_path: str, input_path: str, output_path: str):
        self.sdk = Small_LLM_Model()
        self.output_path = output_path
        
        with open(self.sdk.get_path_to_vocab_file()) as file:
            vocab = json.load(file)
            self.vocab = {v: k for k, v in vocab.items()}
        
        with open(input_path) as f_calling:
            self.prompts = json.load(f_calling)
            
        with open(def_path) as f_definition:
            self.definitions_str = f_definition.read()
            raw_definitions = json.loads(self.definitions_str)
            
            self.definitions = []
            for d in raw_definitions:
                valid_def = FunctionDef(**d)
                self.definitions.append(valid_def.model_dump())
                
        self.func_names = [k["name"] for k in self.definitions]
        self.func_ids = []
        for x in self.func_names:
            for i, v in self.vocab.items():
                if v in x:
                    self.func_ids.append(i)
        self.state = "FUNCTION_NAME"
        self.function_name = ""
    

    def get_allowed_tokens_for_current_state(self, current_state_tokens):
        chosen_ids = []
        
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
            expected_args = []
            for x in self.definitions:
                if x['name'] == getattr(self, 'function_name', ''):
                    expected_args = list(x['parameters'].keys())
                    break
            
            output = self.sdk.decode(current_state_tokens)
            clean_output = output.strip()
            
        
            if clean_output.endswith('{') or clean_output.endswith(','):
                for i, v in self.vocab.items():
                    if '"' in v:
                        chosen_ids.append(i)
                return chosen_ids
                
        
            if not output.strip().startswith('"'):
                output = '"' + output
            all_args_present = all(f'"{arg}"' in output for arg in expected_args)
            
            if not all_args_present:
            
                for i, v in self.vocab.items():
                    if '}' not in v:
                        chosen_ids.append(i)
            else:
            
            
                for i, v in self.vocab.items():
                    v_lower = v.lower()
                    if ',' not in v_lower and 'return' not in v_lower:
                        chosen_ids.append(i)
                        
            return chosen_ids
            
        elif self.state == "END_BRACE":
            for i, v in self.vocab.items():
                if '}' in v:
                    chosen_ids.append(i)
                    
        return chosen_ids

    def transition(self, current_state_tokens):
        old_state = self.state
        output = self.sdk.decode(current_state_tokens)
        print(output)
        
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
    
    def constrained(self):
        all_output = []
        current_output = []

        for prompt in self.prompts:
            output_text = []
            self.state = "FUNCTION_NAME"
            
        
            print("------------------")
            generated_text = f"""You are a helpful assistant that translates natural
language into function calls.
You must output ONLY a valid JSON object with this exact format:
{{"name": "<function_name>", "parameters": {{"<param1>": <value1>}}}}

Available functions:
{self.definitions_str}

User: {prompt['prompt']}
Assistant: """
            generated_text += '{"prompt": "' + prompt['prompt'] + '", "name": "'
            
        
            tokens = self.sdk.encode(generated_text)
            current_sequence =  tokens.tolist()[0]
            for _ in range(500):

                raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)

                masked_logits = np.full(len(raw_logits), -np.inf)

                choose_ids = self.get_allowed_tokens_for_current_state(current_output)
                
                if not choose_ids:
                    masked_logits = raw_logits

                for i in choose_ids:
                    masked_logits[i] = raw_logits[i]

                next_token_id = np.argmax(masked_logits)
                current_sequence.append(next_token_id)
                output_text.append(next_token_id)
                current_output.append(next_token_id)
                current_output = self.transition(current_output)
                # print('{"prompt": "' + prompt['prompt'] + '", "name": "' + self.sdk.decode(output_text))
                if self.state == "DONE":
                    all_output.append('{"prompt": "' + prompt['prompt'] + '", "name": "' + self.sdk.decode(output_text))
                    break

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open("data/output/function_calling_results.json", "w") as f:
            all_output_str = ", ".join(all_output)
            all_output_str = '[' + all_output_str + ']'
            
            all_output_list = json.loads(all_output_str)
            validated_output_list = []
            
            for x in all_output_list:
                for k, v in x['parameters'].items():
                    if type(v) == int:
                        x["parameters"][k] = float(v)
                
                valid_result = FunctionCallResult(**x)
                validated_output_list.append(valid_result.model_dump())

            print(validated_output_list)
            
            json.dump(validated_output_list, f, indent=4)





def main():
    start = time()
    
    parser = argparse.ArgumentParser(description="Function Calling Tool")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    
    args = parser.parse_args()

    ovj = OutputValidJson(args.functions_definition, args.input, args.output)
    ovj.constrained()
    
    end = time()
    second = end - start
    print(float(second) / 60.0)

        


if __name__ == "__main__":
    main()