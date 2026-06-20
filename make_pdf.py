from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'SmartHire - Final Project Report', 
                  new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)

def clean_line(text):
    text = text.replace('\u2014', '-')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2018', "'")
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

with open('reports/final_report.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = clean_line(line)
    if line.startswith('# '):
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, line[2:], new_x='LMARGIN', new_y='NEXT')
        pdf.ln(2)
    elif line.startswith('## '):
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, line[3:], new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)
    elif line.startswith('### '):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, line[4:], new_x='LMARGIN', new_y='NEXT')
    elif line.startswith('|'):
        pdf.set_font('Courier', '', 8)
        pdf.cell(0, 6, line[:100], new_x='LMARGIN', new_y='NEXT')
    elif line == '':
        pdf.ln(3)
    else:
        pdf.set_font('Helvetica', '', 10)
        try:
            pdf.multi_cell(0, 6, line)
        except:
            pass

pdf.output('reports/final_report.pdf')
print('✅ PDF saved to reports/final_report.pdf')