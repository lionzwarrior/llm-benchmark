import csv
from datetime import datetime
import time
import requests
import statistics

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
DATA_FILE = "TruthfulQA/data/v1/TruthfulQA.csv"
OUTPUT_FILE = f"truthfulqa_benchmark_{MODEL.replace(':', '_')}_{int(time.time())}.csv"
OUTPUT_LOG = OUTPUT_FILE.replace(".csv", "_summary.txt")

def query_ollama(prompt):
    """Send prompt to Ollama and measure latency."""
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        text = response.json().get("response", "").strip()
    except Exception:
        text = response.text.strip()
    latency = time.time() - start
    return text, latency

def benchmark():
    # --- Load questions ---
    questions = []
    with open(DATA_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Question" in row and row["Question"]:
                questions.append(row["Question"])

    latencies = []
    results = []
    start_total = time.time()

    N = 817  # limit for quick testing; adjust as needed

    # --- Benchmark loop ---
    for i, q in enumerate(questions[:N]):
        output, latency = query_ollama(q)
        latencies.append(latency)
        results.append([i + 1, q, output, round(latency, 3)])
        print(f"[{i+1}] Latency: {latency:.2f}s")

    total_time = time.time() - start_total
    throughput = len(latencies) / total_time if total_time > 0 else 0
    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    # --- Write CSV output (semicolon separated) ---
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["index", "question", "model_output", "latency_sec"])
        writer.writerows(results)

    # --- Write summary ---
    summary = (
        f"\n--- Benchmark Results ---\n"
        f"Model: {MODEL}\n"
        f"Samples tested: {len(latencies)}\n"
        f"Average latency: {avg_latency:.2f}s\n"
        f"95th percentile latency: {p95_latency:.2f}s\n"
        f"Throughput: {throughput:.2f} req/s\n"
        f"Total duration: {total_time:.2f}s\n"
    )
    print(summary)

    with open(OUTPUT_LOG, "w", encoding="utf-8") as logf:
        logf.write(summary)

    print(f"✅ Detailed results saved to: {OUTPUT_FILE}")
    print(f"🧾 Summary saved to: {OUTPUT_LOG}")

if __name__ == "__main__":
    t0 = datetime.now()
    benchmark()
    t1 = datetime.now()

    start_str = t0.strftime('%Y-%m-%d %H:%M:%S')
    end_str = t1.strftime('%Y-%m-%d %H:%M:%S')
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
