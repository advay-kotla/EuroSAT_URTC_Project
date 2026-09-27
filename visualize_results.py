import pandas as pd
import matplotlib.pyplot as plt
import os


# Load results
df = pd.read_csv("results/model_results.csv")

print(df)


# Create folder for figures
os.makedirs("results/figures", exist_ok=True)


# ==========================
# Accuracy Comparison
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Test Accuracy Comparison")

plt.ylim(0,1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "results/figures/accuracy_comparison.png",
    dpi=300
)

plt.close()


# ==========================
# All Metrics Comparison
# ==========================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


plt.figure(figsize=(10,6))


x = range(len(df))


width = 0.2


for i, metric in enumerate(metrics):

    plt.bar(
        [j + i*width for j in x],
        df[metric],
        width,
        label=metric
    )


plt.xticks(
    [j + width*1.5 for j in x],
    df["Model"],
    rotation=20
)

plt.ylabel("Score")

plt.ylim(0,1)

plt.title(
    "Performance Comparison Across Models"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results/figures/metric_comparison.png",
    dpi=300
)


plt.close()


print("Figures generated successfully!")