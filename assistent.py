"""
Lightweight personal AI assistant starter, tuned for Python 3.11.3.

Features:
 - Uses OpenAI (if OPENAI_API_KEY set) or a small local HF fallback
 - Sandboxed Python execution using the same interpreter (sys.executable)
 - Save files, skeleton image hook
 - Simple REPL interface

Security: running arbitrary code is potentially dangerous. This example uses
a short timeout and runs code in a subprocess; do not run untrusted code on sensitive hosts.
"""

import os
import time
import shlex
import tempfile
import subprocess
import json
import sys
from dataclasses import dataclass
from typing import Optional


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)

LLM_MAX_TOKENS = 512
PY_EXEC_TIMEOUT = 6 

class BaseLLM:
    def generate(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        raise NotImplementedError

if USE_OPENAI:
    class OpenAILLM(BaseLLM):
        def __init__(self, api_key: str):
            import openai
            openai.api_key = api_key
            self.openai = openai

        def generate(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
           
            try:
                resp = self.openai.ChatCompletion.create(
                    model="gpt-4o-mini" if hasattr(self.openai, "ChatCompletion") else "gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
              
                resp = self.openai.Completion.create(engine="text-davinci-003", prompt=prompt,
                                                     max_tokens=max_tokens, temperature=0.7)
                return resp["choices"][0]["text"].strip()

else:
    class LocalLLM(BaseLLM):
        def __init__(self, model_name="gpt2"):
           
            from transformers import pipeline
            self.gen = pipeline("text-generation", model=model_name, max_length=512)

        def generate(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
            out = self.gen(prompt, max_length=max_tokens, do_sample=True, temperature=0.8, num_return_sequences=1)
            return out[0]["generated_text"]

def run_python_code(code: str, timeout: int = PY_EXEC_TIMEOUT) -> dict:
    """
    Run python code in a subprocess using the *same* Python interpreter (sys.executable).
    Returns dict: success, stdout, stderr, runtime.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        fname = f.name

    cmd = [sys.executable, fname]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        runtime = time.time() - start
        result = {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "runtime": runtime
        }
    except subprocess.TimeoutExpired as e:
        result = {"success": False, "stdout": e.stdout or "", "stderr": "TimeoutExpired", "runtime": timeout}
    finally:
        try:
            os.remove(fname)
        except Exception:
            pass
    return result

def save_text_file(filename: str, text: str) -> str:
    path = os.path.abspath(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def generate_image(prompt: str, output_path: str = "output.png") -> Optional[str]:
    """
    Placeholder for image generation. Implement with diffusers or an API if needed.
    Return path string on success, None if not configured.
    """
    return None

@dataclass
class Assistant:
    llm: BaseLLM

    def chat(self, message: str) -> str:
        prompt = f"You are a helpful assistant. {message}"
        return self.llm.generate(prompt)

    def explain(self, topic: str) -> str:
        prompt = f"Explain the following topic clearly with examples and step-by-step instructions: {topic}"
        return self.llm.generate(prompt)

    def code_explain_and_run(self, code: str) -> dict:
        explanation = self.llm.generate("Explain this Python code in detail and mention any issues:\n\n" + code, max_tokens=300)
        run_result = run_python_code(code)
        return {"explanation": explanation, "run_result": run_result}

    def make_file(self, filename: str, prompt_for_content: str) -> str:
        content = self.llm.generate(prompt_for_content, max_tokens=1000)
        return save_text_file(filename, content)

    def create_image(self, prompt: str, output_path: str = "generated.png") -> Optional[str]:
        return generate_image(prompt, output_path)

def build_default_assistant() -> Assistant:
    if USE_OPENAI:
        return Assistant(llm=OpenAILLM(OPENAI_API_KEY))
    else:
        return Assistant(llm=LocalLLM(model_name="gpt2"))

def repl():
    print("Mini-AI Assistant — type 'help', 'exit' to quit.")
    assistant = build_default_assistant()
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not cmd:
            continue
        if cmd in ("exit","quit"):
            print("Goodbye.")
            break
        if cmd == "help":
            print("Commands:\n chat <message>\n explain <topic>\n runcode <paste python code>\n makefile <filename> <prompt>\n createimg <prompt>\n exit")
            continue
        if cmd.startswith("chat "):
            print("Thinking...")
            print(assistant.chat(cmd[len("chat "):]))
            continue
        if cmd.startswith("explain "):
            print("Thinking...")
            print(assistant.explain(cmd[len("explain "):]))
            continue
        if cmd.startswith("runcode "):
            code = cmd[len("runcode "):]
            print("Explaining and running code...")
            res = assistant.code_explain_and_run(code)
            print("=== Explanation ===")
            print(res["explanation"])
            print("=== Run Result ===")
            print(json.dumps(res["run_result"], indent=2))
            continue
        if cmd.startswith("makefile "):
            parts = cmd.split(" ", 2)
            if len(parts) < 3:
                print("Usage: makefile filename prompt")
                continue
            filename, prompt = parts[1], parts[2]
            path = assistant.make_file(filename, prompt)
            print(f"Saved to {path}")
            continue
        if cmd.startswith("createimg "):
            out = assistant.create_image(cmd[len("createimg "):])
            if out:
                print(f"Image saved to {out}")
            else:
                print("Image generation not configured.")
            continue
        print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    repl()
