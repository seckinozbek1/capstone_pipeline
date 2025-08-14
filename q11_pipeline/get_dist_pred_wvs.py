import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("predictions/q11_predictions_top_score_filtered.csv")

# Count sentences per predicted label
label_counts = df['predicted_combined_label'].value_counts().sort_index()

# Plot histogram (bar chart) of counts per predicted_combined_label
plt.figure(figsize=(10,6))
label_counts.plot(kind='bar')
plt.title("Sentence Counts per Predicted Combined Label")
plt.xlabel("Predicted Combined Label")
plt.ylabel("Sentence Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Load dataset
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Filter valid values
valid_values = [1, 2]
q11_values = wvs_df['Q11']
q11_filtered = q11_values[q11_values.isin(valid_values)]

# Count frequencies
counts = q11_filtered.value_counts().sort_index()

# Plot histogram
plt.figure(figsize=(8,5))
counts.plot(kind='bar')
plt.title("Distribution of Q11 Values (1 to 2)")
plt.xlabel("Q11 Response")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
