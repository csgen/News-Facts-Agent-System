"""Langfuse dataset experiment: compare gpt-4o vs gpt-4o-mini on claim extraction.

Runs the Langfuse-hosted `CLAIM_ISOLATION_PROMPT` against every item in the
Langfuse-hosted `CLAIM_ISOLATION` dataset for each model, linking each call
back to the dataset item with a run_name that includes the model name.

After running, in the Langfuse UI → Datasets → CLAIM_ISOLATION → Runs you'll
see two runs side-by-side (one per model) with per-item outputs, tokens, and
latency. You can scroll through the items and eyeball which model gave more
sensible claims.

Run with:
    docker-compose exec -e PYTHONUNBUFFERED=1 app pytest \\
        tests/test_preprocessing/test_claim_isolation_eval.py -s -v

Skipped if OPENAI_API_KEY or LANGFUSE_PUBLIC_KEY is missing.
"""

import os
import time
from datetime import datetime, timezone

import pytest
from langfuse import Langfuse
from openai import OpenAI
from src.config import settings

DATASET_NAME = "CLAIM_ISOLATION"
PROMPT_NAME = "CLAIM_ISOLATION_PROMPT"

pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")),
    reason="OPENAI_API_KEY or LANGFUSE_PUBLIC_KEY missing — skipping live eval",
)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini"])
def test_claim_isolation_eval(model: str):
    """Run CLAIM_ISOLATION_PROMPT against every CLAIM_ISOLATION dataset item
    using `model`, log each call to Langfuse, and link each trace to the
    dataset item under a model-specific run_name."""
    langfuse = Langfuse()
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # Fetch the Langfuse-hosted prompt. production label is the default.
    prompt_obj = langfuse.get_prompt(PROMPT_NAME, version=2)

    # Fetch the dataset and all its items.
    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"\n=== Dataset '{DATASET_NAME}' has {len(dataset.items)} items ===",
          flush=True)

    # Unique run name per model + invocation time, so rerunning doesn't clash.
    run_name = f"{model}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"=== Run: {run_name} ===\n", flush=True)

    for i, item in enumerate(dataset.items):
        title = item.input.get("title", "")
        body_text = item.input.get("body_text", "")

        print(f"--- item {i+1}/{len(dataset.items)}: "
              f"title={title[:60]!r} ---", flush=True)

        # Render the Langfuse-hosted prompt with this item's variables.
        compiled = prompt_obj.compile(title=title, body_text=body_text)

        # Create an explicit trace so we have a stable handle to link to the
        # dataset item. Using an explicit trace (rather than @observe or the
        # langfuse.openai auto-wrapper) gives us deterministic control over
        # the ID that item.link() receives — essential for v2 reliability.
        trace = langfuse.trace(
            name="claim_isolation_eval",
            input={"title": title, "body_text_preview": body_text[:300]},
            metadata={"model": model, "item_id": item.id},
        )

        generation = trace.generation(
            name="openai_claim_extraction",
            model=model,
            input=compiled,
            prompt=prompt_obj,  # links this generation to the prompt version
            model_parameters={
                "temperature": 0.0,
                "response_format": "json_object",
            },
        )

        start = time.perf_counter()
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": compiled}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            elapsed = time.perf_counter() - start

            generation.end(
                output=content,
                usage={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
            )
            trace.update(output=content)

            print(f"    {elapsed:.2f}s | in={usage.prompt_tokens} "
                  f"out={usage.completion_tokens}", flush=True)
            print(f"    output (first 400 chars): {content[:400]}", flush=True)

        except Exception as e:
            elapsed = time.perf_counter() - start
            generation.end(
                level="ERROR",
                status_message=str(e),
            )
            trace.update(output=None, metadata={"error": str(e)})
            print(f"    FAILED after {elapsed:.2f}s: {e}", flush=True)

        # Link this trace to the dataset item under our run_name. This is
        # what makes the run show up in the Datasets → Runs tab.
        item.link(
            trace,
            run_name=run_name,
            run_description=f"Claim extraction eval with {model}",
            run_metadata={
                "model": model,
                "temperature": 0.0,
                "response_format": "json_object",
            },
        )

    # Flush synchronously so the test process doesn't exit before Langfuse
    # has received all traces + link writes.
    langfuse.flush()
    print(f"\n=== Run '{run_name}' complete. "
          f"Check Langfuse → Datasets → {DATASET_NAME} → Runs ===", flush=True)
