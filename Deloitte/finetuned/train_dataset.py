import pandas as pd
from datasets import load_dataset

train_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="all")
train_df = train_dataset.to_pandas()

train_df.to_csv("train_dataset.csv", index = False)




