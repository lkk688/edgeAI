import json
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

RESULTS_FILE = "benchmark_results.json"

def load_results():
    if not Path(RESULTS_FILE).exists():
        print("❌ No benchmark_results.json file found.")
        return []
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def display_table(data):
    table = []
    for entry in data:
        table.append([
            entry.get("timestamp", "-"),
            entry.get("model", "-"),
            entry.get("method", "-"),
            entry.get("tokens", "-"),
            f"{entry.get('time', '-'):0.2f}" if "time" in entry else "-",
            f"{entry.get('speed', '-'):0.2f}" if "speed" in entry else "-",
            entry.get("error", "")
        ])
    print("\n📊 All Benchmark Results:")
    print(tabulate(table, headers=["Time", "Model", "Method", "Tokens", "Time (s)", "Speed (tok/s)", "Error"]))

def plot_results(data):
    df = pd.DataFrame(data)
    df = df[df["error"].isnull()]  # remove failed entries

    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # Plot token speed
    plt.figure(figsize=(10, 6))
    bars = []
    labels = []
    for idx, row in df.iterrows():
        label = f"{row['model']}\n{row['method']}"
        bars.append(row['speed'])
        labels.append(label)
    plt.bar(labels, bars)
    plt.ylabel("Tokens/sec")
    plt.title("LLM Inference Speed by Model and Backend")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("benchmark_speed.png")
    plt.show()

    # Plot time
    plt.figure(figsize=(10, 6))
    bars = []
    labels = []
    for idx, row in df.iterrows():
        label = f"{row['model']}\n{row['method']}"
        bars.append(row['time'])
        labels.append(label)
    plt.bar(labels, bars)
    plt.ylabel("Time to Generate (s)")
    plt.title("LLM Inference Time by Model and Backend")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("benchmark_time.png")
    plt.show()

if __name__ == "__main__":
    data = load_results()
    if data:
        display_table(data)
        plot_results(data)