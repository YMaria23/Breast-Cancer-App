import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import ai_service


def main():
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    missing = [name for name, value in [("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model)]
               if not value]
    if missing:
        print(f"Missing in software-solution/.env: {', '.join(missing)}")
        return 1
    print(f"Endpoint: {base_url}")
    print(f"Model:    {model}")
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        available = [m.id for m in client.models.list(timeout=20).data]
        print(f"Models on the server: {', '.join(available)}")
        if model not in available:
            print(f"Warning: '{model}' is not in that list. Check LLM_MODEL.")
    except Exception as error:
        print(f"Could not list models ({type(error).__name__}: {error}). Trying a request anyway.")
    start = time.perf_counter()
    try:
        reply = ai_service._ask_llm(client, model, [{"role": "user", "content": "Reply with the single word OK."}], 60)
        print(f"Short test: '{(reply.choices[0].message.content or '').strip()}' in {time.perf_counter() - start:.1f} s")
    except Exception as error:
        print(f"Short test failed after {time.perf_counter() - start:.1f} s: {type(error).__name__}: {error}")
    result = ai_service.AnalysisResult(malignant=0.202, benign=0.565, priority="HIGH", model="GMIC (test)",
                                       latency_s=0.0, rois=[{"x": 550, "y": 300, "w": 60, "h": 60,
                                                             "attention": 0.3}])
    study = {"laterality": "L", "view": "MLO", "accession": "llm-check", "_bbox": (0, 0, 1024, 1024)}
    start = time.perf_counter()
    text, source = ai_service.explain(result, study)
    print(f"\nWritten by: {source} ({time.perf_counter() - start:.1f} s)\n")
    print(text)
    if source.startswith("template"):
        print("\nThe LLM call failed; see the last line of logs/ai_calls.jsonl for the error.")
        return 1
    print("\nLLM connection works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
