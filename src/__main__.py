from llm_sdk import Small_LLM_Model
import json
import numpy as np


class OutputValidJson():
    def __init__(self):
        self.sdk = Small_LLM_Model()
        with open(self.sdk.get_path_to_vocab_file()) as file:
            vocab = json.load(file)
            self.vocab = {v: k for k, v in vocab.items()}
            # print(self.sdk.get_path_to_vocab_file())
        with open("data/input/function_calling_tests.json") as f_calling:
            self.prompts = json.load(f_calling)
        with open("data/input/functions_definition.json") as f_definition:
            self.definitions_str = f_definition.read()
            
            self.definitions = json.loads(self.definitions_str)
            
            # print(self.definitions_str)
    def constrained(self):

        for prompt in self.prompts:
            output_text = []
            
            # generated_text = '{"prompt": "' + prompt['prompt'] + '", "name": "'
            generated_text = f"""You are a helpful assistant that translates natural
language into function calls.
You must output ONLY a valid JSON object with this exact format:
{{"name": "<function_name>", "parameters": {{"<param1>": <value1>}}}}

Available functions:
{self.definitions_str}

User: {prompt['prompt']}
Assistant: """
            generated_text += '{"prompt": "' + prompt['prompt'] + '", "name": "'
            
            # print(generated_text)
            tokens = self.sdk.encode(generated_text)
            current_sequence =  tokens.tolist()[0]
            for _ in range(30):

                raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)

                masked_logits = np.full(len(raw_logits), -np.inf)

                # choose_ids = self.get_allowed_tokens_for_current_state()

                next_token_id = np.argmax(raw_logits)
                current_sequence.append(next_token_id)
                output_text.append(next_token_id)
                if self.sdk.decode(output_text).strip().endswith('}}'):
                    break
            print("------------------")
            print('{"prompt": "' + prompt['prompt'] + '", "name": "' + self.sdk.decode(output_text))
            print("------------------")
            # break





def main():

    ovj = OutputValidJson()
    ovj.constrained()

        


if __name__ == "__main__":
    main()