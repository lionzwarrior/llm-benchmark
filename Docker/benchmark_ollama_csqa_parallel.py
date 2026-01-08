from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
import re
import asyncio
import aiohttp
import statistics
import pandas as pd
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://ollama-service:11434/api/generate"
MODEL = "llama3.1:8b"
DATA_FILE = "dev_rand_split.jsonl"
NUM_QUESTIONS = 1221
CONCURRENT_USERS = 20
OUTPUT_CSV = f"ollama_async_results_{MODEL.replace(':', '_')}_{int(time.time())}.csv"
OUTPUT_LOG = OUTPUT_CSV.replace(".csv", "_summary.txt")
NS_TO_S = 1e-9
TZ = ZoneInfo("Asia/Jakarta")


async def query_ollama(session, prompt):
    start = time.time()
    try:
        async with session.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
        ) as resp:
            data = await resp.json()

            output = data.get("response", "").strip()

            # Ollama native metrics
            prompt_tokens = data.get("prompt_eval_count", 0)
            gen_tokens = data.get("eval_count", 0)
            gen_eval_ns = data.get("eval_duration", 0)
            total_ns = data.get("total_duration", 0)

    except Exception:
        logger.exception("Ollama request failed")
        return "ERROR", 0, 0, 0, 0, 0

    latency = time.time() - start

    return (
        output,
        latency,
        prompt_tokens,
        gen_tokens,
        gen_eval_ns,
        total_ns,
    )


async def process_question(session, index, data, worker_id):
    question = data["question"]
    labels = data["choices"]["label"]
    choices = data["choices"]["text"]
    answer_key = data["answerKey"].strip().upper()

    # Build prompt
    prompt = (
        "You are answering a multiple-choice question.\n\n" f"Question: {question}\n"
    )
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nRespond with only the letter (A, B, C, D, or E)."

    (
        output,
        latency,
        prompt_tokens,
        gen_tokens,
        gen_eval_ns,
        total_ns,
    ) = await query_ollama(session, prompt)

    match = re.search(r"\b([A-E])\b", output.upper())
    pred = match.group(1) if match else "-"

    result = {
        "worker_id": worker_id,
        "question_id": index + 1,
        "question": question,
        "prediction": pred,
        "answer_key": answer_key,
        "correct": 1 if pred == answer_key else 0,
        "latency_s": round(latency, 3),
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_output": output,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "gen_eval_ns": gen_eval_ns,
        "total_ns": total_ns,
    }

    logger.info(
        "Worker %d | Q%d/%d | Latency=%.2fs | Pred=%s | True=%s",
        worker_id,
        index + 1,
        NUM_QUESTIONS,
        latency,
        pred,
        answer_key,
    )

    return result


async def worker(worker_id, dataset, session, results):
    logger.info("Worker %d starting (%d questions)", worker_id, len(dataset))

    for i, data in enumerate(dataset):
        result = await process_question(session, i, data, worker_id)
        results.append(result)

    logger.info("Worker %d finished", worker_id)


async def benchmark_async():
    logger.info(
        "Launching benchmark | Workers=%d | Questions/worker=%d",
        CONCURRENT_USERS,
        NUM_QUESTIONS,
    )

    base_dataset = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_QUESTIONS:
                break
            base_dataset.append(json.loads(line))

    worker_datasets = [base_dataset[:] for _ in range(CONCURRENT_USERS)]

    start_time = time.time()
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                worker(worker_id, worker_datasets[worker_id], session, results)
            )
            for worker_id in range(CONCURRENT_USERS)
        ]
        await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    total_requests = len(results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_gen_tokens = sum(r["gen_tokens"] for r in results)
    total_gen_time = sum(r["gen_eval_ns"] for r in results) * NS_TO_S
    total_end_time = sum(r["total_ns"] for r in results) * NS_TO_S

    latencies = [r["latency_s"] for r in results]
    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    throughput = total_requests / total_time
    accuracy = sum(r["correct"] for r in results) / total_requests

    generation_tpm = (
        (total_gen_tokens / total_gen_time) * 60 if total_gen_time > 0 else 0
    )
    end_to_end_tpm = (
        ((total_prompt_tokens + total_gen_tokens) / total_end_time) * 60
        if total_end_time > 0
        else 0
    )

    summary = (
        "\n--- Async Benchmark Results ---\n"
        f"Model: {MODEL}\n"
        f"Workers: {CONCURRENT_USERS}\n"
        f"Questions per worker: {NUM_QUESTIONS}\n"
        f"Total requests: {total_requests}\n"
        f"Accuracy: {accuracy:.2%}\n"
        f"Average latency: {avg_latency:.3f}s\n"
        f"95th percentile latency: {p95_latency:.3f}s\n"
        f"Throughput: {throughput:.2f} req/s\n"
        f"Total duration: {total_time:.2f}s\n"
        f"\n--- Token Throughput ---\n"
        f"Prompt tokens: {total_prompt_tokens}\n"
        f"Generated tokens: {total_gen_tokens}\n"
        f"Generation TPM: {generation_tpm:.2f}\n"
        f"End-to-End TPM: {end_to_end_tpm:.2f}\n"
    )

    logger.info(summary)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, sep=";")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info("CSV saved to: %s", OUTPUT_CSV)
    logger.info("Summary saved to: %s", OUTPUT_LOG)


if __name__ == "__main__":
    t0 = datetime.now(TZ)
    asyncio.run(benchmark_async())
    t1 = datetime.now(TZ)

    start_str = t0.strftime("%Y-%m-%d %H:%M:%S")
    end_str = t1.strftime("%Y-%m-%d %H:%M:%S")
    duration = (t1 - t0).total_seconds()

    logger.info("Time start: %s", start_str)
    logger.info("Time end:   %s", end_str)
    logger.info("Total execution time: %.2f seconds", duration)

    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(
            "\n--- Execution Time ---\n"
            f"Start time: {start_str}\n"
            f"End time:   {end_str}\n"
            f"Total execution time: {duration:.2f} seconds\n"
        )
