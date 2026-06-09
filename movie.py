import os
import pandas as pd
import ast

base_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(base_dir, "static", "movies_metadata.csv")
if not os.path.exists(metadata_path):
    metadata_path = os.path.join(base_dir, "movies_metadata.csv")

df = pd.read_csv(metadata_path, low_memory=False)

# Clean rows where genres is null
df = df[df['genres'].notna()]

unique_genres = set()
for g_str in df['genres']:
    try:
        # Evaluate python literal list of dicts
        genres_list = ast.literal_eval(g_str)
        if isinstance(genres_list, list):
            for g in genres_list:
                if 'name' in g:
                    unique_genres.add(g['name'])
    except Exception:
        pass

print("Unique genres in movies_metadata.csv:")
print(sorted(list(unique_genres)))

