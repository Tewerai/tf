import os
import fitz
import json
import re
from PIL import Image
from io import BytesIO
import hashlib
import shutil
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load image processing pipeline from transformers
image_caption_pipeline = pipeline(task="image-captioning")

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

def save_summary(summary_text, summary_folder, chapter_title):
    os.makedirs(summary_folder, exist_ok=True)

    summary_file_path = os.path.join(summary_folder, f"{chapter_title}_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

def text_rank(text):
    # Tokenize and process the text using spaCy
    doc = nlp(text)

    # Calculate the importance score for each sentence based on the number of stop words
    sentence_scores = [len([token.text.lower() for token in sent if token.text.lower() not in STOP_WORDS]) for sent in doc.sents]

    # Select sentences with the highest importance scores as the summary
    num_sentences_in_summary = min(len(sentence_scores), 3)  # You can adjust the number of sentences in the summary
    selected_indices = sorted(range(len(sentence_scores)), key=lambda k: sentence_scores[k], reverse=True)[:num_sentences_in_summary]
    
    # Create the summary by joining selected sentences
    summary = " ".join(doc.sents[i].text for i in selected_indices)

    return summary

def process_chapter_content(chapter_content):
    # Tokenize and process chapter content using spaCy
    doc = nlp(chapter_content)
    
    # Summarize the chapter content using spaCy's extractive summarization
    summarized_text = text_rank(chapter_content)
    
    # Add any additional processing or analysis here
    
    return summarized_text  # You can return the summarized text or the spaCy Doc object

def process_image(image_path):
    # Process image using pre-trained image-captioning model
    result = image_caption_pipeline(images=image_path)
    return result[0]["caption"]

def save_and_console_page_data(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    chapters_found_folder = os.path.join(output_folder, "chapters_found")

    doc = fitz.open(pdf_path)

    toc_found = False  # Flag to track if TOC entries are found

    for page_num in range(doc.page_count):
        page = doc[page_num]

        page_objects = page.get_text("dict")["blocks"]
        page_data = {"Text": [], "Pictures": [], "TOC": [], "Chapters": []}

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
        toc_entries, _ = extract_toc_and_questions(page_text)
        if toc_entries:
            toc_found = True

        # Skip processing until TOC is found
        if not toc_found:
            continue

        page_data["TOC"] = toc_entries

        if toc_entries:
            for entry in toc_entries:
                chapter_data = {"Title": entry, "Text": [], "ImageCaptions": [], "Summary": ""}
                chapter_end = page_text.find(entry, page_text.find(entry) + len(entry))
                chapter_content = page_text[len(entry):chapter_end].strip()

                # Process chapter content and summarize using spaCy
                summarized_chapter = process_chapter_content(chapter_content)
                
                # Add any additional logic based on spaCy analysis
                
                chapter_data["Text"].append(chapter_content)
                chapter_data["Summary"] = summarized_chapter  # Add the summarized text to the chapter data

                for picture in page_data["Pictures"]:
                    try:
                        image = Image.open(picture["filename"])
                        image_caption = process_image(picture["filename"])
                        chapter_data["ImageCaptions"].append({"bbox": picture["bbox"], "caption": image_caption})
                    except Exception as e:
                        print(f"Error processing image caption on page {page_num + 1}: {e}")

                page_data["Chapters"].append(chapter_data)

                chapter_folder = os.path.join(chapters_found_folder, entry)
                save_chapter_data(chapter_data, chapter_folder)

                # Save summary in its own folder
                summary_folder = os.path.join(chapter_folder, "summary")
                save_summary(summarized_chapter, summary_folder, entry)

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
pdf_path = "test.pdf"
output_folder = "output_pages"

save_and_console_page_data(pdf_path, output_folder)
