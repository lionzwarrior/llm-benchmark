py -V:3.12 -m venv venv

venv/Scripts/activate

pip install -r requirements.txt


## IP for ollama service
gpu1.petra.ac.id: http://172.17.0.1:11434/api/generate

pak carik: http://ollama-service:11434/api/generate


## Open Grafana
kubectl port-forward svc/prometheus-grafana 3000:80

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
kubectl cp llm-benchmark-9cdcdcbb7-fhj5b:/app/truthfulqa_benchmark_llama3.1_8b_1765967185.csv ./truthfulqa_benchmark_llama3.1_8b_1765967185.csv

kubectl cp llm-benchmark-9cdcdcbb7-fhj5b:/app/truthfulqa_benchmark_llama3.1_8b_1765967185_summary.txt ./truthfulqa_benchmark_llama3.1_8b_1765967185_summary.txt


## Monitor from Grafana
kubectl port-forward svc/prometheus-grafana 3000:80

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
findstr "%Cpu(s):" .\cpu.txt > cpu_findstr_cpus.txt

Header: us;sy;ni;id;wa;hi;si;st

cpu usage = 100 - <id value>

=100-D2


## How to get GPU data from gpu.csv
just do Data > Text to Columns (use ",")

just get average from the columns


## How to get memory data from cpu.txt
findstr "MiB Mem" cpu.txt | findstr /V "Swap" > cpu_findstr_memory.txt

Header: total;free;used;buff/cache

used_percent = (1 - ((free + buff/cache) / total)) * 100

=(1-(B2+D2)/A2)*100
