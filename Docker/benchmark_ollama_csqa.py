from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
import re
import requests
import statistics
import pandas as pd

# --- Configuration ---
OLLAMA_URL = "http://ollama-service:11434/api/generate"
MODEL = "llama3.1:8b"
DATA_FILE = "dev_rand_split.jsonl"
NUM_QUESTIONS = 1221  # limit for speed test
OUTPUT_CSV = f"ollama_csqa_results_{MODEL.replace(':', '_')}_{int(time.time())}.csv"
OUTPUT_LOG = OUTPUT_CSV.replace(".csv", "_summary.txt")
TZ = ZoneInfo("Asia/Jakarta")


# --- Query Ollama API ---
def query_ollama(prompt):
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        output = data.get("response", "").strip()
    except Exception as e:
        print(f"[Error] {e}")
        output = "ERROR"
    latency = time.time() - start
    return output, latency


# --- Main Benchmark ---
def benchmark():
    total = 0
    correct = 0
    latencies = []
    records = []
    start_total = time.time()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_QUESTIONS:
                break

            data = json.loads(line)
            question = data["question"]
            labels = data["choices"]["label"]
            choices = data["choices"]["text"]
            answer_key = data["answerKey"].strip().upper()

            # --- Stronger prompt ---
            prompt = (
                "You are answering a multiple-choice question. "
                "Choose the best answer based on commonsense knowledge.\n\n"
                f"Question: {question}\n"
            )
            for label, choice in zip(labels, choices):
                prompt += f"{label}. {choice}\n"
            prompt += (
                "\nRespond with only the letter of your final answer (A, B, C, D, or E). "
                "Do not include any explanation.\nYour answer:"
            )

            # --- Query model ---
            output, latency = query_ollama(prompt)

            # --- Extract clean answer (A–E only) ---
            match = re.search(r"\b([A-E])\b", output.upper())
            pred = match.group(1) if match else "-"

            correct_flag = 1 if pred == answer_key else 0
            latencies.append(latency)

            # --- Record result ---
            records.append(
                {
                    "id": i + 1,
                    "question": question,
                    "prediction": pred,
                    "answer_key": answer_key,
                    "correct": correct_flag,
                    "latency_s": round(latency, 2),
                    "model": MODEL,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_output": output,
                }
            )

            if correct_flag:
                correct += 1
            total += 1

            print(
                f"[{i+1}] Latency: {latency:.2f}s | Pred: {pred} | True: {answer_key}"
            )

    # --- Compute metrics ---
    total_time = time.time() - start_total
    throughput = total / total_time if total_time > 0 else 0
    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    summary = (
        f"\n--- Benchmark Results ---\n"
        f"Model: {MODEL}\n"
        f"Samples tested: {total}\n"
        f"Accuracy: {correct / total:.2%}\n"
        f"Average latency: {avg_latency:.2f}s\n"
        f"95th percentile latency: {p95_latency:.2f}s\n"
        f"Throughput: {throughput:.2f} req/s\n"
        f"Total duration: {total_time:.2f}s\n"
    )
    print(summary)

    # --- Export results ---
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, sep=";")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as logf:
        logf.write(summary)

    print(f"✅ Detailed results saved to: {OUTPUT_CSV}")
    print(f"🧾 Summary saved to: {OUTPUT_LOG}")


# --- Entry Point ---
if __name__ == "__main__":
    t0 = datetime.now(TZ)
    benchmark()
    t1 = datetime.now(TZ)

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
