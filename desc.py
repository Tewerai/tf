import os
import json
import fitz  # PyMuPDF
import re

def extract_question_descriptions(question_pdf_path, output_folder="output"):
    try:
        question_descriptions = extract_data_from_pdf(question_pdf_path)

        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save question descriptions to a JSON file
        output_path = os.path.join(output_folder, "question_descriptions.json")
        with open(output_path, "w") as output_file:
            json.dump(question_descriptions, output_file, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_data_from_pdf(pdf_path):
    data = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()

            # Match patterns like "X.Y" at the beginning of a line
            matches = re.finditer(r"^\s*(\d+\.\d+)(?=\s|$)", text, re.MULTILINE)
            for match in matches:
                number = match.group(1)

                # Find the position of the next "X.Y" pattern at the beginning of a line
                next_pattern_match = re.search(fr"^{re.escape(number)}\.\d+", text[match.end():], re.MULTILINE)
                if next_pattern_match:
                    end_position = match.end() + next_pattern_match.start()
                else:
                    # If no next pattern is found, extract until the end of the text
                    end_position = len(text)

                # Extract the entire sentence until a line is skipped
                description_text = text[match.end():end_position].strip()

                data.append({"number": number, "text": description_text})

    return data

# Example usage:
question_pdf_path = "questions/question_paper.pdf"
extract_question_descriptions(question_pdf_path)
