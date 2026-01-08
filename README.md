## Initial setup for running the benchmark
py -V:3.12 -m venv venv

venv/Scripts/activate

pip install -r requirements.txt


## IP for ollama service
gpu1.petra.ac.id: http://172.17.0.1:11434/api/generate

pak carik: http://ollama-service:11434/api/generate


## Open Grafana
kubectl -n remmanuel port-forward svc/prometheus-grafana 3000:80

ssh -L 3000:127.0.0.1:3000 raphael@pakcarik.petra.ac.id


## Test in Linux
source venv/bin/activate

chmod +x monitor.sh

./monitor.sh & MONITOR_PID=$!
python benchmark_ollama_csqa_parallel.py
while read pid; do
    kill "$pid" 2>/dev/null
done < monitor_pids.txt
kill $MONITOR_PID 2>/dev/null


## Build docker image
docker build -t lionzwarrior10/llm-benchmark .

docker push lionzwarrior10/llm-benchmark


# Copy to get data from pod in kubernetes
kubectl -n remmanuel cp llm-benchmark-65b96b594b-xq87f:/app/ollama_async_results_llama3.2_3b_1767251628.csv ./ollama_async_results_llama3.2_3b_1767251628.csv

kubectl -n remmanuel cp llm-benchmark-65b96b594b-xq87f:/app/ollama_async_results_llama3.2_3b_1767251628_summary.txt ./ollama_async_results_llama3.2_3b_1767251628_summary.txt


## Monitor from Grafana
kubectl -n remmanuel port-forward svc/prometheus-grafana 3000:80

ssh -L 3000:127.0.0.1:3000 raphael@pakcarik.petra.ac.id


# Open prometheus
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090

ssh -L 9090:127.0.0.1:9090 raphael@pakcarik.petra.ac.id


## Delete process after stopping port forwarding
ps aux | grep "kubectl port-forward" | grep -v grep

kill 887043


## The start to process cpu.txt
findstr "MiB Mem" cpu.txt | findstr /V "Swap" > cpu_findstr_memory.txt

findstr "%Cpu(s):" .\cpu.txt > cpu_findstr_cpus.txt


## How to get CPU data from cpu.txt in windows
Header: us;sy;ni;id;wa;hi;si;st

cpu usage = 100 - <id value>

=(100-D2)/100


## How to get GPU data from gpu.csv
just do Data > Text to Columns (use ",")

just get average from the columns


## How to get memory data from cpu.txt
Header: total;free;used;buff/cache

used_percent = (1 - ((free + buff/cache) / total)) * 100

=(1-(B2+D2)/A2)


## How to process data from either dataset commonsenseQA or truthfulQA to get avg latency, 95th percentile latency, throughput, gneration TPM, and end-to-end TPM
avg latency = AVERAGE(A1:A100)

95th percentile latency = PERCENTILE.INC(A1:A100, 0.95)

throughput = 1 / avg latency

generation TPM =SUM(gen_tokens) / (SUM(gen_eval_ns) / 1E9 / 60)

end-to-end TPM = SUM(gen_tokens) / (SUM(total_ns) / 1E9 / 60)
