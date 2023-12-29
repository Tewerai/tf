import os
import json
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import hashlib
import shutil
import re

def md5_hash(data):
    return hashlib.md5(data).hexdigest()

def extract_toc_and_questions(page_text):
    toc_entries = []
    question_pattern = r'\d+\.\s*[A-Za-z\s]+(\?)'

    toc_entries = re.findall(r'\d+\s+[A-Za-z\s]+', page_text)
    questions = re.findall(question_pattern, page_text)

    return toc_entries, questions

def save_chapter_data(chapter_data, chapter_folder):
    os.makedirs(chapter_folder, exist_ok=True)

    chapter_json_path = os.path.join(chapter_folder, "chapter_data.json")
    with open(chapter_json_path, "w", encoding="utf-8") as json_file:
        json.dump(chapter_data, json_file, ensure_ascii=False, indent=2)

def save_and_console_page_data(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    chapters_found_folder = os.path.join(output_folder, "chapters_found")

    doc = fitz.open(pdf_path)

    toc_found = False  # Flag to track if TOC entries are found

    for page_num in range(doc.page_count):
        page = doc[page_num]

        page_objects = page.get_text("dict")["blocks"]
        page_data = {"Text": [], "Pictures": [], "TOC": [], "Chapters": [], "Questions": []}

        for obj in page_objects:
            if obj["type"] == 0:  # Text
                page_data["Text"].append(obj)
            elif obj["type"] == 1:  # Image
                if isinstance(obj["image"], str):
                    image_data_io = BytesIO(obj["image"].encode())
                else:
                    image_data_io = BytesIO(obj["image"])

                try:
                    image = Image.open(image_data_io)
                    width, height = map(int, obj["bbox"][2:])
                    image_data_io.seek(0)
                    image_hash = md5_hash(image_data_io.read())

                    image_filename = f"image_{page_num + 1}_{image_hash}.png"
                    image_file_path = os.path.join(output_folder, "images", image_filename)

                    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
                    image.save(image_file_path)

                    page_data["Pictures"].append(
                        {"bbox": obj["bbox"], "filename": image_file_path, "width": width, "height": height}
                    )
                except Exception as e:
                    print(f"Error processing image on page {page_num + 1}: {e}")

        page_text = page.get_text()

        # Check if TOC entries and questions are found on the page
        toc_entries, questions = extract_toc_and_questions(page_text)
        if toc_entries or questions:
            toc_found = True

        # Skip processing until TOC is found
        if not toc_found:
            continue

        page_data["TOC"] = toc_entries
        page_data["Questions"] = questions

        if toc_entries:
            for entry in toc_entries:
                chapter_data = {"Title": entry, "Text": []}
                chapter_end = page_text.find(entry, page_text.find(entry) + len(entry))
                chapter_content = page_text[len(entry):chapter_end].strip()

                chapter_data["Text"].append(chapter_content)
                page_data["Chapters"].append(chapter_data)

                chapter_folder = os.path.join(chapters_found_folder, entry)
                save_chapter_data(chapter_data, chapter_folder)

                for picture in page_data["Pictures"]:
                    if picture["bbox"][3] >= chapter_end and picture["bbox"][1] <= len(entry):
                        image_filename = os.path.basename(picture["filename"])
                        image_destination = os.path.join(chapter_folder, "images", image_filename)
                        os.makedirs(os.path.join(chapter_folder, "images"), exist_ok=True)
                        shutil.copy(picture["filename"], image_destination)

        page_data = json.loads(
            json.dumps(page_data, default=lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x)
        )

        print(f"Page {page_num + 1} Data:")
        print(json.dumps(page_data, ensure_ascii=False, indent=2))
        print()

        output_file = f"{output_folder}/page_{page_num + 1}.json"
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(page_data, json_file, ensure_ascii=False, indent=2)

# Replace with your actual PDF file and output folder paths
pdf_path = "books/geo.pdf"
output_folder = "output_pages"

save_and_console_page_data(pdf_path, output_folder)
