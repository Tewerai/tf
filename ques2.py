import os
import json
import fitz  # PyMuPDF
from PIL import Image
import re

def extract_questions_and_answers(question_pdf_path, answer_pdf_path, annexure_pdf_path, output_folder="output"):
    try:
        question_descriptions = extract_data_from_pdf(question_pdf_path)
        detailed_questions = extract_data_from_pdf(question_pdf_path)
        answers = extract_data_from_pdf(answer_pdf_path)

        annexure_data = extract_annexure_data(annexure_pdf_path, output_folder)

        matched_data = match_questions_and_answers(question_descriptions, detailed_questions, answers, annexure_data)

        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, "questions_answers.json")
        with open(output_path, "w") as output_file:
            json.dump(matched_data, output_file, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_data_from_pdf(pdf_path):
    data = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()

            # Use a regex pattern to capture only top-level questions at the beginning of the page
            matches = re.finditer(r"(\d+(\.\d+){2})\s+(.+?)(?=\d+(\.\d+)+|$)", text, re.DOTALL)
            
            for match in matches:
                number = match.group(1)
                text = match.group(3).strip()
                data.append({"number": number, "text": text})
    return data

def extract_annexure_data(annexure_pdf_path, output_folder):
    annexure_data = {}
    with fitz.open(annexure_pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()
            images = extract_images_from_page(page, output_folder, text)
            figure_number = extract_figure_number(text)
            related_question = extract_related_question(text)

            if figure_number and related_question:
                annexure_data[figure_number] = {"text": text, "images": images, "related_question": related_question}

    return annexure_data

def extract_images_from_page(page, output_folder, text):
    images = []
    image_folder = os.path.join(output_folder, "images")
    os.makedirs(image_folder, exist_ok=True)

    for img_index in range(len(page.get_images())):
        try:
            image = page.get_images()[img_index][0]
            base_image = Image.frombytes("RGB", [int(image["width"]), int(image["height"])], image["image"])
            images.append({"image": base_image, "related_question": extract_related_question(text)})

            if check_for_diagrams(text):
                image_filename = f"image_{img_index + 1}.png"
                image_path = os.path.join(image_folder, image_filename)
                base_image.save(image_path, format="PNG")
        except TypeError:
            continue

    return images

def extract_figure_number(text):
    match = re.search(r"FIGURE (\d+(\.\d+){2})", text)
    return match.group(1) if match else None

def extract_related_question(text):
    match = re.search(r"(\d+(\.\d+){2})", text)
    return match.group(1) if match else None

def match_questions_and_answers(question_descriptions, detailed_questions, answers, annexure_data):
    matched_data = {}
    for question in detailed_questions:
        question_number = question["number"]
        answer_info = next((a for a in answers if a["number"] == question_number), None)
        if answer_info:
            matched_data[question_number] = {
                "question_description": next((q["text"] for q in question_descriptions if q["number"] == question_number), ""),
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

def check_for_diagrams(text):
    keywords = ["diagram", "drawing", "figure"]
    return any(keyword in text.lower() for keyword in keywords)

# Example usage:
question_pdf_path = "questions/question_paper.pdf"
answer_pdf_path = "questions/answer_paper.pdf"
annexure_pdf_path = "your_document.pdf"
extract_questions_and_answers(question_pdf_path, answer_pdf_path, annexure_pdf_path)
