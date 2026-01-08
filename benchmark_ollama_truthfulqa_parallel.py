import csv
from datetime import datetime
import time
import asyncio
import aiohttp
import statistics

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
DATA_FILE = "TruthfulQA/data/v1/TruthfulQA.csv"
OUTPUT_FILE = f"truthfulqa_benchmark_{MODEL.replace(':', '_')}_{int(time.time())}.csv"
OUTPUT_LOG = OUTPUT_FILE.replace(".csv", "_summary.txt")
NS_TO_S = 1e-9
NUM_QUESTIONS = 817
CONCURRENT_USERS = 20


# --- Query Ollama ---
async def query_ollama(session, prompt):
    start = time.time()
    try:
        async with session.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
        ) as resp:
            data = await resp.json()

            output = data.get("response", "").strip()

            # Token + timing info from Ollama
            prompt_tokens = data.get("prompt_eval_count", 0)
            gen_tokens = data.get("eval_count", 0)
            gen_eval_ns = data.get("eval_duration", 0)
            total_ns = data.get("total_duration", 0)

    except Exception as e:
        print(f"[Error] {e}")
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


# --- Process a question for one worker ---
async def process_question(session, worker_id, idx, question):
    (
        output,
        latency,
        prompt_tokens,
        gen_tokens,
        gen_eval_ns,
        total_ns,
    ) = await query_ollama(session, question)

    print(f"[Worker {worker_id}] Q{idx} | Latency: {latency:.2f}s")

    return {
        "worker_id": worker_id,
        "index": idx,
        "question": question,
        "output": output,
        "latency": round(latency, 3),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "gen_eval_ns": gen_eval_ns,
        "total_ns": total_ns,
    }


# --- Worker function: each worker runs ALL questions sequentially ---
async def worker(worker_id, questions, session, out_list):
    print(f"Worker {worker_id} started with {len(questions)} questions...")

    for idx, q in enumerate(questions, start=1):
        result = await process_question(session, worker_id, idx, q)
        out_list.append(result)

    print(f"Worker {worker_id} finished.")


# --- Main Benchmark ---
async def benchmark_async():
    # --- Load questions ---
    questions = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Question" in row and row["Question"]:
                questions.append(row["Question"])

    questions = questions[:NUM_QUESTIONS]

    print(f"Loaded {len(questions)} TruthfulQA questions.")
    print(f"Launching {CONCURRENT_USERS} workers...")
    print(f"Each worker will process ALL {len(questions)} questions.\n")

    results = []
    start_total = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(worker_id + 1, questions, session, results))
            for worker_id in range(CONCURRENT_USERS)
        ]

        await asyncio.gather(*tasks)

    # --- Compute Metrics ---
    total_time = time.time() - start_total
    latencies = [r["latency"] for r in results]
    total_requests = len(results)

    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_gen_tokens = sum(r["gen_tokens"] for r in results)
    total_gen_time = sum(r["gen_eval_ns"] for r in results) * NS_TO_S
    total_end_time = sum(r["total_ns"] for r in results) * NS_TO_S

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    throughput = total_requests / total_time
    generation_tpm = (
        (total_gen_tokens / total_gen_time) * 60 if total_gen_time > 0 else 0
    )
    end_to_end_tpm = (
        ((total_prompt_tokens + total_gen_tokens) / total_end_time) * 60
        if total_end_time > 0
        else 0
    )

    # --- Save CSV ---
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "worker_id",
                "index",
                "question",
                "model_output",
                "latency_sec",
                "prompt_tokens",
                "gen_tokens",
                "gen_time_sec",
                "total_time_sec",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["worker_id"],
                    r["index"],
                    r["question"],
                    r["output"],
                    r["latency"],
                    r["prompt_tokens"],
                    r["gen_tokens"],
                    r["gen_eval_ns"] * NS_TO_S,
                    r["total_ns"] * NS_TO_S,
                ]
            )

    summary = (
        f"\n--- TruthfulQA Async Benchmark ---\n"
        f"Model: {MODEL}\n"
        f"Workers: {CONCURRENT_USERS}\n"
        f"Questions per worker: {NUM_QUESTIONS}\n"
        f"Total requests: {total_requests}\n"
        f"Average latency: {avg_latency:.2f}s\n"
        f"95th percentile latency: {p95_latency:.2f}s\n"
        f"Throughput: {throughput:.2f} req/s\n"
        f"Total duration: {total_time:.2f}s\n"
        f"\n--- Token Throughput ---\n"
        f"Prompt tokens: {total_prompt_tokens}\n"
        f"Generated tokens: {total_gen_tokens}\n"
        f"Generation TPM: {generation_tpm:.2f}\n"
        f"End-to-End TPM: {end_to_end_tpm:.2f}\n"
    )

    print(summary)

    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log.write(summary)

    print(f"✅ Results saved to: {OUTPUT_FILE}")
    print(f"🧾 Summary saved to: {OUTPUT_LOG}")


# --- Entry Point ---
if __name__ == "__main__":
    t0 = datetime.now()
    asyncio.run(benchmark_async())
    t1 = datetime.now()

    start_str = t0.strftime("%Y-%m-%d %H:%M:%S")
    end_str = t1.strftime("%Y-%m-%d %H:%M:%S")
    duration = (t1 - t0).total_seconds()

    print(f"Time start: {start_str}")
    print(f"Time end:   {end_str}")
    print(f"Total execution time: {duration:.2f} seconds")

    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(
            "\n--- Execution Time ---\n"
            f"Start time: {start_str}\n"
            f"End time:   {end_str}\n"
            f"Total execution time: {duration:.2f} seconds\n"
        )
