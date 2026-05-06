import re
from pathlib import Path


AUTHOR = "Sundetkhan Bekzat"
INSTRUCTOR = "Akhmetova Zhanar"
BASE_DIR = Path(__file__).resolve().parent / "Labs_notebooks"

reports = [
    "README.md",
    "Report_9.2_Python.md",
    "Report_9.3_Machine_Learning.md",
    "Report_9.4_Deep_Learning.md",
    "Report_9.5_ModelArts.md",
    "Report_AI_Final_Exam.md",
]

typst_content = f"""#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 12pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 24pt, weight: "bold")[Deep Learning and AI Frameworks]
  #v(1em)
  #text(size: 18pt)[Midterm Implementation Report]
  #v(2em)
  #text(size: 14pt)[*Author:* {AUTHOR}]
  #v(1em)
  #text(size: 14pt)[*Instructor:* {INSTRUCTOR}]
  #v(1em)
  #text(size: 14pt)[Astana IT University]
  #v(2em)
]

#outline(indent: true)
#pagebreak()
"""

def md_to_typst(text):
    text = re.sub(r'^#\s+(.*)', r'= \1', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)', r'== \1', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*)', r'=== \1', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', 
                  r'#figure(image("\2", width: 85%), caption: [\1])', 
                  text)

    def latex_fig_repl(match):
        img_path = match.group(1)
        caption = match.group(2)
        return f'#figure(image("{img_path}", width: 85%), caption: [{caption}])'

    pattern = r'\\begin\{figure\}.*?\\includegraphics.*?\{(.+?)\}.*?\\caption\{(.+?)\}.*?\\end\{figure\}'
    text = re.sub(pattern, latex_fig_repl, text, flags=re.DOTALL)
    
    return text

for r in reports:
    path = BASE_DIR / r
    if path.exists():
        md_text = path.read_text(encoding='utf-8')
        typst_text = md_to_typst(md_text)
        typst_content += typst_text + "\n#pagebreak()\n"

typst_file = BASE_DIR / "huawei_midterm.typ"
typst_file.write_text(typst_content, encoding="utf-8")

print(f"Typst file written to {typst_file}")
