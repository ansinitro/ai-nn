import os
import json
import glob
import subprocess

# 1. Update style specifically for 9.3 Notebooks
def update_notebook_style(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error loading {notebook_path}: {e}")
            return False

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            source_text = ''.join(source)
            if 'import matplotlib.pyplot as plt' in source_text or 'import matplotlib' in source_text:
                new_source = []
                for line in source:
                    # Remove older injected styles if they exist
                    if 'plt.style.use' not in line:
                        new_source.append(line)
                    if 'import matplotlib.pyplot as plt' in line or 'import matplotlib ' in line:
                        # Append new aggressive style overrides
                        new_source.append('\nimport seaborn as sns\n')
                        new_source.append('plt.style.use("seaborn-v0_8-darkgrid")\n')
                        new_source.append('sns.set_palette("husl")\n')
                
                if new_source != source:
                    cell['source'] = new_source
                    modified = True

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

ml_notebooks = glob.glob('/home/titan/Documents/master/ai-nn/midterm_Bekzat/Labs_notebooks/9.3 Machine Learning /**/*.ipynb', recursive=True)

print("--- STYLING NOTEBOOKS ---")
for path in ml_notebooks:
    updated = update_notebook_style(path)
    if updated:
        print(f"[Styled] {os.path.basename(path)}")

print("\n--- EXECUTING NOTEBOOKS ---")
venv_jupyter = "jupyter"

for path in ml_notebooks:
    print(f"Executing: {os.path.basename(path)} ...")
    cmd = ["uv", "run", "--with", "jupyter", "--with", "pandas", "--with", "scikit-learn", "--with", "seaborn", "--with", "matplotlib", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Success: {os.path.basename(path)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {os.path.basename(path)}")
        print(e.stderr.decode('utf-8'))
