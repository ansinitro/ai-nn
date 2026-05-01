import os
import json
import glob

def add_style_to_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error loading {notebook_path}: {e}")
            return

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            # source is a list of strings
            source_text = ''.join(source)
            if 'import matplotlib.pyplot as plt' in source_text or 'import matplotlib' in source_text:
                if 'plt.style.use' not in source_text:
                    new_source = []
                    inserted = False
                    for line in source:
                        new_source.append(line)
                        if ('import matplotlib.pyplot as plt' in line or 'import matplotlib ' in line) and not inserted:
                            new_source.append('plt.style.use("ggplot")\n')
                            inserted = True
                    cell['source'] = new_source
                    modified = True

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {notebook_path}")

print("Scanning for notebooks...")
notebooks = glob.glob('/home/titan/Documents/master/ai-nn/midterm_Bekzat/Labs_notebooks/**/*.ipynb', recursive=True)
for path in notebooks:
    add_style_to_notebook(path)
print("Done!")
