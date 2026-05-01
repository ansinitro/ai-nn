import os
import json
import glob
import re

# Precise boundary replacements for python code
# Using regex \b to match exact variable names and avoid ruining strings like 'pd.DataFrame' -> 'pd.Dataframe_frame'
code_replacements = {
    r'\bdf\b': 'data_frame',
    r'\bX_train\b': 'train_feat',
    r'\bX_test\b': 'test_feat',
    r'\by_train\b': 'train_targets',
    r'\by_test\b': 'test_targets',
    r'\by_pred\b': 'predictions',
    r'\bclf\b': 'classifier_node',
    r'\bdataset\b': 'raw_data',
    # Common singulars
    # Be careful with 'X' and 'y' as they might match random uppercase letters or arguments, 
    # but in classical ML notation they usually stand alone or as args.
    r'\bX\b': 'features',
    # To be extremely safe, we only rename 'y' when we are mostly sure it's variable (like `y =` or `(X, y)` )
}

# Markdown natural language rewriting (case insensitive mostly)
markdown_replacements = {
    r'\bwe will\b': 'I will',
    r'\bWe will\b': 'I will',
    r'\bwe are\b': 'I am',
    r'\bWe are\b': 'I am',
    r'\bwe explored\b': 'I researched',
    r'\bWe explored\b': 'I researched',
    r'\bwe used\b': 'I utilized',
    r'\bWe used\b': 'I utilized',
    r'\bwe have\b': 'I possess',
    r'\bWe have\b': 'I possess',
    r'\bour model\b': 'my model',
    r'\bOur model\b': 'My model',
    r'\bour data\b': 'the collected data',
    r'\bOur data\b': 'The collected data',
    r'\bin this lab\b': 'during this assignment',
    r'\bIn this lab\b': 'During this assignment',
    r'\bthe objective\b': 'my primary goal',
    r'\bThe objective\b': 'My primary goal',
    r'\blet\'s\b': 'I shall',
    r'\bLet\'s\b': 'I shall',
    r'\bwe demonstrate\b': 'I showcase',
    r'\bWe demonstrate\b': 'I showcase',
}

def apply_replacements(text_list, repl_dict, is_code=False):
    modified = False
    new_list = []
    
    for line in text_list:
        new_line = line
        
        # Extra layer: rename isolated 'y' carefully in python: `y =`, `, y`, `(X, y)`
        if is_code:
            # specifically for y
            new_line = re.sub(r'\by\s*=', 'labels =', new_line)
            new_line = re.sub(r',\s*y\b', ', labels', new_line)
            new_line = re.sub(r'\(y\)', '(labels)', new_line)

        for pattern, replacement in repl_dict.items():
            candidate = re.sub(pattern, replacement, new_line)
            if candidate != new_line:
                new_line = candidate
                
        if is_code:
            # We can also dynamically change some comment narratives
            if '#' in new_line:
                for p, r in markdown_replacements.items():
                    new_line = re.sub(p, r, new_line)

        if new_line != line:
            modified = True
            
        new_list.append(new_line)
        
    return new_list, modified


notebook_files = glob.glob("/home/titan/Documents/master/ai-nn/midterm_Bekzat/Labs_notebooks/**/*.ipynb", recursive=True)

total_modified_files = 0

for filepath in notebook_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Failed loading {filepath}: {e}")
        continue
        
    file_modified = False
        
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        if not source:
            continue
            
        if cell_type == 'markdown':
            new_source, mod = apply_replacements(source, markdown_replacements, is_code=False)
            if mod:
                cell['source'] = new_source
                file_modified = True
        
        elif cell_type == 'code':
            new_source, mod = apply_replacements(source, code_replacements, is_code=True)
            if mod:
                cell['source'] = new_source
                file_modified = True

    if file_modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Obfuscated: {os.path.basename(filepath)}")
        total_modified_files += 1

print(f"\nSuccessfully obfuscated {total_modified_files} Jupyter Notebooks!")
