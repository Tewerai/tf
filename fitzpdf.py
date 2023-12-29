import os
import fitz
import json
from PIL import Image
from io import BytesIO


def save_and_console_page_data(pdf_path, output_folder):
    """
    Extracts text and images from a PDF document and saves them to separate files.

    Args:
        pdf_path: Path to the PDF document.
        output_folder: Path to the output folder where extracted data will be saved.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF document using PyMuPDF
    doc = fitz.open(pdf_path)

    # Loop through each page of the document
    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Extract text and image data from the page
        page_objects = page.get_text("dict")["blocks"]
        page_data = {"Text": [], "Pictures": [], "Diagrams": []}

        for obj in page_objects:
            if obj["type"] == 0:  # Text
                page_data["Text"].append(obj)
            elif obj["type"] == 1:  # Image
                # Check the type of image data and handle accordingly
                if isinstance(obj["image"], str):
                    image_data_io = BytesIO(obj["image"].encode())
                else:
                    image_data_io = BytesIO(obj["image"])

                try:
                    # Open the image using PIL
                    image = Image.open(image_data_io)
                    width, height = map(int, obj["bbox"][2:])  # Ensure width and height are integers

                    # Create image filename
                    image_file = f"{output_folder}/images/image_{page_num + 1}_{page_data['Pictures'].index(obj) + 1}.png"
                    image.save(image_file)

                    # Add image data to page data
                    page_data["Pictures"].append(
                        {"bbox": obj["bbox"], "filename": image_file, "width": width, "height": height}
                    )
                except Exception as e:
                    print(f"Error processing image on page {page_num + 1}: {e}")

        # Convert bytes to strings before saving JSON data
        page_data = json.loads(
            json.dumps(page_data, default=lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x)
        )

        # Print JSON data to the console
        print(f"Page {page_num + 1} Data:")
        print(json.dumps(page_data, ensure_ascii=False, indent=2))
        print()

        # Save page data to a JSON file
        output_file = f"{output_folder}/page_{page_num + 1}.json"
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(page_data, json_file, ensure_ascii=False, indent=2)


# Replace with your actual PDF file and output folder paths
pdf_path = "your_document.pdf"
output_folder = "output_pages"

save_and_console_page_data(pdf_path, output_folder)
