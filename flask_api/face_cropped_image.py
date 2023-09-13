import os
import xml.etree.ElementTree as ET
from PIL import Image

# İlgili klasörleri ve dosya uzantılarını ayarlayın
dataset_folder = 'C:\\Users\\SELEN\\Desktop\\flaskweb\\flask_api\\dataset'
output_folder = 'C:\\Users\\SELEN\\Desktop\\flaskweb\\flask_api\\cropped_images'
xml_extension = '.xml'

# Klasörleri kontrol edin ve gerekirse oluşturun
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dataset klasöründeki her XML dosyası için işlem yapın
for root_folder, _, xml_files in os.walk(dataset_folder):
    for xml_file in xml_files:
        if xml_file.endswith(xml_extension):
            xml_path = os.path.join(root_folder, xml_file)

            # XML dosyasını analiz edin
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Resim dosyasının adını bulun
            image_filename = root.find('filename').text
            image_path = os.path.join(root_folder, image_filename)

            # Koordinatları bulun
            xmin = int(root.find(".//xmin").text)
            ymin = int(root.find(".//ymin").text)
            xmax = int(root.find(".//xmax").text)
            ymax = int(root.find(".//ymax").text)

            # Resmi kırpın
            image = Image.open(image_path)
            cropped_image = image.crop((xmin, ymin, xmax, ymax))

            # Kırpmış resmi kaydedin
            output_subfolder = root_folder.replace(dataset_folder, output_folder)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, image_filename)
            cropped_image.save(output_path)
