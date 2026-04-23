import pypandoc
import os

input_file = "report.md"
output_file = "Report.docx"

print(f"Converting {input_file} to {output_file}...")
pypandoc.convert_file(input_file, 'docx', outputfile=output_file)
print("Conversion successful!")
