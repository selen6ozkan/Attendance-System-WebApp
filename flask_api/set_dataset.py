import os
import csv

csv_file = open('C:\\Users\\SELEN\\Desktop\\flaskweb\\flask_api\\dataset.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_path', 'label'])

dataset_path = 'C:\\Users\\SELEN\\Desktop\\flaskweb\\flask_api\\cropped_images\\'
for student_folder in os.listdir(dataset_path):
    student_folder_path = os.path.join(dataset_path, student_folder)
    if os.path.isdir(student_folder_path):
        for image_file in os.listdir(student_folder_path):
            if image_file != ".DS_Store":  # .DS_Store dosyasını atla
                image_path = os.path.join(student_folder_path, image_file)
                csv_writer.writerow([image_path, student_folder])

csv_file.close()
