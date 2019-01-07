import os
import csv
import face_recognition
import shutil
import pandas as pd

def face_det():
    path = "/Users/lichenchao/Desktop/AMLSassignment/dataset"
    destination_path = "/Users/lichenchao/Desktop/AMLSassignment/dataset_face1/dataset_face"
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x[:-4]))
    face_det_result = []

    for filename in path_list:
        print(filename)

        image = face_recognition.load_image_file(path + "/" + filename)
        face_locations = face_recognition.face_locations(image)
        # Find all the faces in the image using a pre-trained convolutional neural network.

        if len(face_locations) >=1:
            shutil.copy(path + "/" + filename, destination_path + "/" + filename)
        face_det_result.append(len(face_locations))
    print(face_det_result)


    return face_det_result

def write_csv(file_path, data):
    df = pd.read_csv(file_path, header = 1)
    df = pd.DataFrame(df)
    df.insert(6, 'face', data)
    df_1 = df[-df.face.isin([0])]
    print(df_1)
    df_1.to_csv(file_path, index=False)



if __name__ == '__main__':
    data = face_det()
    print len(data)
    write_csv('attribute_list_face.csv',data)



