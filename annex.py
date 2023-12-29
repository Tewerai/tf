import fitz  # PyMuPDF
import os
import re

def extract_jpeg_images_based_on_condition(pdf_path, output_folder, condition_pattern):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]

        # Get the images on the page
        images = page.get_images(full=True)

        # Extract figure names from the text
        figure_matches = re.finditer(condition_pattern, page.get_text())
        for match in figure_matches:
            figure_name = match.group(0)
            figure_name = figure_name.rstrip(':')  # Remove trailing colon

            # Find the corresponding image and save it
            for img_index, img_info in enumerate(images):
                img_index += 1
                image_index = img_info[0]

                base_image = pdf_document.extract_image(image_index)

                # Get the image information
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Check if the image extension is 'jpeg' or 'jpg'
                if image_ext.lower() in ['jpeg', 'jpg']:
                    # Save the image to the output folder with the figure name
                    image_filename = f"{figure_name}_img{img_index}.{image_ext}"
                    image_path = os.path.join(output_folder, image_filename)

                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)

    # Close the PDF file
    pdf_document.close()

if __name__ == "__main__":
    # Replace 'your_pdf_file.pdf' with the path to your PDF file
    pdf_file_path = 'questions/annexure.pdf'
    
    # Replace 'newImages' with the desired output folder name
    output_folder_name = 'newImages'

    # Replace the pattern with your specific condition pattern
    condition_pattern = r'FIGURE \d+\.\d+:'

    extract_jpeg_images_based_on_condition(pdf_file_path, output_folder_name, condition_pattern)
    print(f"JPEG images associated with the condition '{condition_pattern}' extracted and saved to the '{output_folder_name}' folder.")
