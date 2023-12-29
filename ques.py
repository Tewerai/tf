import os
import json
import fitz  # PyMuPDF
from PIL import Image
import re

def extract_questions_and_answers(question_pdf_path, answer_pdf_path, annexure_pdf_path, output_folder="output"):
    try:
        detailed_questions = extract_data_from_pdf(question_pdf_path)
        answers = extract_data_from_pdf(answer_pdf_path)

        # Extract data from annexure PDF (including images)
        annexure_data = extract_annexure_data(annexure_pdf_path, output_folder)

        matched_data = match_questions_and_answers(detailed_questions, answers, annexure_data)

        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save questions and answers to a single JSON file
        output_path = os.path.join(output_folder, "questions_answers.json")
        with open(output_path, "w") as output_file:
            json.dump(matched_data, output_file, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_data_from_pdf(pdf_path):
    data = []
    with fitz.open(pdf_path) as pdf:
        current_question = None

        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()

            # Use a regex pattern to capture int.int.int at the beginning of a line
            matches = re.finditer(r"^\s*(\d+\.\d+\.\d+)\s+(.+)$", text, re.MULTILINE)
            for match in matches:
                number = match.group(1)
                question_text = match.group(2).strip()

                # Check if the current question is under a int.int
                if "." in number:
                    current_question = {"number": number, "text": question_text}
                    data.append(current_question)

    return data

def extract_annexure_data(annexure_pdf_path, output_folder):
    annexure_data = {}
    with fitz.open(annexure_pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()
            images = extract_images_from_page(page, output_folder)
            figure_number = extract_figure_number(text)
            related_question = extract_related_question(text)

            if figure_number and related_question:
                annexure_data[figure_number] = {"text": text, "images": images, "related_question": related_question}

    return annexure_data

def extract_images_from_page(page, output_folder):
    images = []
    image_folder = os.path.join(output_folder, "images")
    os.makedirs(image_folder, exist_ok=True)

    for img_index in range(len(page.get_images())):
        try:
            image = page.get_images()[img_index][0]
            base_image = Image.frombytes("RGB", [int(image["width"]), int(image["height"])], image["image"])
            images.append({"image": base_image})

            # Save the image to the images folder
            image_filename = f"image_{img_index + 1}.png"
            image_path = os.path.join(image_folder, image_filename)
            base_image.save(image_path, format="PNG")
        except TypeError:
            # Skip images with no data
            continue

    return images

def extract_figure_number(text):
    match = re.search(r"FIGURE (\d+(\.\d+)*)", text)
    return match.group(1) if match else None

def extract_related_question(text):
    match = re.search(r"(\d+(\.\d+\.\d+)?)", text)
    return match.group(1) if match else None

def match_questions_and_answers(detailed_questions, answers, annexure_data):
    matched_data = {}
    for question in detailed_questions:
        question_number = question["number"]
        answer_info = next((a for a in answers if a["number"] == question_number), None)
        if answer_info:
            matched_data[question_number] = {
                "detailed_question": question["text"],
                "answer": answer_info["text"],
                "related_images": get_related_images(question_number, annexure_data)
            }

    return matched_data

def get_related_images(question_number, annexure_data):
    related_images = []
    for figure_number, data in annexure_data.items():
        if data["related_question"] == question_number:
            related_images.extend(data["images"])
    return related_images

# Example usage:
question_pdf_path = "questions/question_paper.pdf"
answer_pdf_path = "questions/answer_paper.pdf"
annexure_pdf_path = "questions/annexure.pdf"
extract_questions_and_answers(question_pdf_path, answer_pdf_path, annexure_pdf_path)
