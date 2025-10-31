"""
Author: Diego Rosenberg (dr3432)

convertToPdf.py
- A script to convert all Python (.py) files in the current directory to PDF format.

"""

import os
from fpdf import FPDF
from pathlib import Path

# --- Configuration ---
# Source directory (where the script is located)
SOURCE_DIR = Path.cwd() 
# Output directory (a new folder to store the PDFs)
OUTPUT_DIR = SOURCE_DIR / "converted_pdfs"
# ---------------------

def convert_py_to_pdf():
    """
    Finds all .py files in the SOURCE_DIR, converts them to PDF,
    and saves them in the OUTPUT_DIR.
    """
    
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Scanning for .py files in: {SOURCE_DIR}")
    print(f"Saving PDFs to: {OUTPUT_DIR}\n")

    src_dir = SOURCE_DIR / "src"

    # Check if src directory exists
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory '{src_dir}' does not exist.")
        return

    # Find all .py files in the source directory
    py_files = list(src_dir.glob("*.py"))

    if not py_files:
        print("No .py files found to convert.")
        return

    converted_count = 0
    failed_count = 0

    # Loop through each .py file
    for py_file in py_files:
        # Don't convert this script itself!
        if py_file.name == os.path.basename(__file__):
            continue

        try:
            # Read the content of the python file
            with open(py_file, "r", encoding="utf-8") as f:
                code_content = f.read()

            # Setup PDF object
            pdf = FPDF()
            pdf.add_page()
            
            # Set a monospaced font (crucial for code)
            pdf.set_font("Courier", size=10)

            # Add the code content to the PDF
            # w=0 means fill the width, h=5 is the line height
            pdf.multi_cell(0, 5, txt=code_content)

            # Define output path and save the file
            # e.g., "my_script.py" -> "my_script.pdf"
            output_pdf_path = OUTPUT_DIR / f"{py_file.stem}.pdf"
            pdf.output(output_pdf_path)
            
            print(f"[SUCCESS] Converted: {py_file.name}")
            converted_count += 1

        except Exception as e:
            print(f"[FAILURE] Could not convert {py_file.name}: {e}")
            failed_count += 1

    print(f"\n--- Conversion Complete ---")
    print(f"Successfully converted: {converted_count}")
    print(f"Failed to convert:     {failed_count}")

if __name__ == "__main__":
    convert_py_to_pdf()