import os
import re

reports = [
    "README.md",
    "Report_9.2_Python.md",
    "Report_9.3_Machine_Learning.md",
    "Report_9.4_Deep_Learning.md"
]

base_dir = "/home/titan/Documents/master/ai-nn/midterm_Bekzat/Labs_notebooks"

typst_content = """#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 12pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 24pt, weight: "bold")[Deep Learning and AI Frameworks]
  #v(1em)
  #text(size: 18pt)[Midterm Implementation Report]
  #v(2em)
  #text(size: 14pt)[*Author:* Bekzat]
  #v(1em)
  #text(size: 14pt)[*Instructor:* Akhmetova Zhanar]
  #v(1em)
  #text(size: 14pt)[Astana IT University]
  #v(2em)
]

#outline(indent: true)
#pagebreak()
"""

def md_to_typst(text):
    # Very basic naive converter
    # Headings
    text = re.sub(r'^#\s+(.*)', r'= \1', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)', r'== \1', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*)', r'=== \1', text, flags=re.MULTILINE)
    
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    
    # Inline code
    text = re.sub(r'`([^`]*)`', r'`\1`', text)
    
    # Code blocks (handling python) -> Typst code blocks are ```python ... ``` which matches markdown mostly!
    # But sometimes ````..```` so we'll leave ``` ... ``` as is, Typst supports it.
    
    # Latex equations from Markdown \( ... \) -> $ ... $ (None in this markdown usually)

    # Images: ![Caption](assets/image.png)
    # We need to replace it with #figure(image("assets/image.png"), caption: [Caption])
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', 
                  r'#figure(image("\2", width: 85%), caption: [\1])', 
                  text)
    
    # LaTeX figures from the original Reports
    # \begin{figure}[H]
    # \centering
    # \includegraphics[width=0.85\linewidth]{assets/9.3.1...png}
    # \caption{...}
    # \end{figure}
    
    def latex_fig_repl(match):
        img_path = match.group(1)
        caption = match.group(2)
        return f'#figure(image("{img_path}", width: 85%), caption: [{caption}])'

    pattern = r'\\begin\{figure\}.*?\\includegraphics.*?\{(.+?)\}.*?\\caption\{(.+?)\}.*?\\end\{figure\}'
    text = re.sub(pattern, latex_fig_repl, text, flags=re.DOTALL)
    
    return text

for r in reports:
    path = os.path.join(base_dir, r)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            typst_text = md_to_typst(md_text)
            typst_content += typst_text + "\n#pagebreak()\n"

typst_file = os.path.join(base_dir, "huawei_midterm.typ")
with open(typst_file, "w", encoding="utf-8") as f:
    f.write(typst_content)

print(f"Typst file written to {typst_file}")
