import pandas as pd

# Check both files
df_wide = pd.read_csv('simulation_results_summary_table.csv')
df_long = pd.read_csv('simulation_results_consolidated.csv')

print("--- Data Completeness Check ---")
print(f"Scenarios in Wide: {df_wide['Scenario'].nunique()} (Expected 8)")
print(f"Scenarios in Long: {df_long['Scenario'].nunique()} (Expected 8)")

metrics_in_long = df_long['Metric'].unique()
print(f"Metrics in Long: {metrics_in_long.tolist()}")

# Ensure LLM-specific data (for Fig 5.3 and 5.7) is also preserved
# Wait, the previous consolidated script grouped by Scenario only. 
# Fig 5.3 (Makespan Delta per LLM) and 5.7 (Capability) require 'llm' in the grouping.
# I need to remake the consolidated file to include LLM if the user wants "ALL" data.
