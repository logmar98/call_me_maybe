*This project has been created as part of the 42 curriculum by [mrbib].*

# Call Me Maybe: Function Calling in LLMs

## Description

**Call Me Maybe** is an AI tool designed to translate natural language requests into structured, machine-executable function calls. Rather than answering a user's question directly (e.g., answering "42" to "What is the sum of 40 and 2?"), the system outputs the exact function name and correctly typed arguments required to solve the task (e.g., `{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}`).

By relying on the lightweight `Qwen/Qwen3-0.6B` model and a custom token-masking engine, this project demonstrates how to guarantee 100% valid, schema-compliant JSON output without relying on massive parameter counts or unpredictable prompting heuristics.

## Instructions

The project includes a `Makefile` to automate environment setup and execution.

* **Installation:**
Run `make install` to set up the virtual environment and install project dependencies (like `numpy` and `pydantic`) using `uv`.


* **Execution:**
Run `make run` to execute the main script.
Alternatively, you can manually run the script and specify custom file paths:


```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

```


*(Note: By default, it reads from `data/input/` and writes to `data/output/`)*.


* **Debug:**
Run `make debug` to execute the script in debug mode.


* **Clean:**
Run `make clean` to remove temporary files and caches.



## Algorithm Explanation (Constrained Decoding)

Small Language Models (SLMs) generate text one token at a time by outputting probability scores (logits) for the next possible token. Left on their own, they frequently produce invalid JSON.
This project implements **Constrained Decoding** to intervene before the token is selected:

1. The script evaluates the current parser state (e.g., waiting for a function name, a double quote, or parameter content).
2. It identifies which tokens in the vocabulary will maintain valid JSON structure and semantic compliance with the provided schema.


3. The logits for all invalid tokens are set to negative infinity (`-np.inf`), forcing the model to select only structurally valid tokens.



## Design Decisions

* **Prompt Engineering (ChatML):** The system prompt was restructured into Qwen's native instruction format (`<|im_start|>system...`). The function definitions were deeply compressed into a minimal JSON array to provide a 1:1 mapping for the attention mechanism, preventing key hallucinations.
* **NumPy Vectorization:** Pre-computed lists of valid tokens were converted into NumPy arrays at initialization, bypassing slow Python `for` loops in favor of highly optimized C-backend masking.
* **O(1) String Building (Bonus):** To avoid the heavy overhead of repeatedly calling the SDK's `decode` method inside the generation loop, the tokenizer was recoded to continuously build a running string using `self.vocab.get()`.


* **Early Exit:** A `json.loads()` check was integrated into the generation loop to break exactly when the JSON becomes valid, preventing the model from wandering aimlessly up to the 500-step limit.

## Performance Analysis

* **Accuracy:** The system achieves near-perfect accuracy, correctly extracting the function name and mapping the right arguments.


* **Reliability:** The output is 100% valid JSON. Schema-based type casting ensures integers and strings are strictly enforced according to the definition before saving.


* **Speed:** Thanks to NumPy masking and the early exit strategy, the solution easily processes all prompts well under the 5-minute requirement on standard hardware.



## Challenges Faced

* **State Machine Infinite Loops:** Initially, if the model hallucinated a wrong parameter key, the state machine would trap the LLM in an infinite loop because it waited for keys that were never generated. *Solution:* Reverted the prompt to a strict, compact JSON schema so the model natively predicted the correct keys, and added an "escape hatch" based on colon counts.
* **Sluggish Generation Times:** The heavy `self.sdk.decode()` method inside a 500-step loop caused severe bottlenecks. *Solution:* Implemented custom string tracking and NumPy arrays to speed up the loop dramatically.

## Testing Strategy

Validation was performed using the provided `data/input/function_calling_tests.json` against the schemas in `functions_definition.json`. Edge cases such as regex patterns and float-to-integer casting were continuously tested to ensure the final `function_calling_results.json` strictly matched expected types.

## Example Usage

**Input Prompt:**

> "What is the sum of 2 and 3?"
> 
> 

**Command:**

```bash
make run

```

**Output File (`data/output/function_calling_results.json`):**

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    }
]

```

*(Reference from the project guidelines)*

## Resources

* https://bitcrowd.dev/logits-processing-and-constrained-sampling-in-bumblebee/
* https://bitcrowd.dev/grammar-constrained-decoding-in-bumblebee/


