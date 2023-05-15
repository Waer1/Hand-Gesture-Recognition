import os

# Get the path to the text file of damaged images
text_file_path = "D:/Material/3rd CMP/Second term/Neural network/project/Hand-Gesture-Recognition/damaged-images/women/0.txt"

# Get the path to the folder of images
damages_images_path = 'D:/Material/3rd CMP/Second term/Neural network/project/Hand-Gesture-Recognition/segmented/women/0/'


def delete_files_from_list(file_list_path):
    with open(file_list_path, 'r') as file:
        file_names = file.readlines()

    # Remove whitespace characters like `\n` at the end of each line
    file_names = [name.strip() for name in file_names]

    deleted_files = []
    not_found_files = []

    for file_name in file_names:
        file_name = damages_images_path+file_name
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                deleted_files.append(file_name)
            except OSError as e:
                print(f"Failed to delete {file_name}: {e}")
        else:
            not_found_files.append(file_name)

    print("Deleted files:")
    for file_name in deleted_files:
        print(file_name)

    if not_found_files:
        print("\nFiles not found:")
        for file_name in not_found_files:
            print(file_name)


delete_files_from_list(text_file_path)
