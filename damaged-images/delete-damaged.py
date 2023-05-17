import os

# Get the path to the text file of damaged images
text_file_paths = ["D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/damaged-images/women/", 
                   'D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/damaged-images/men/'
                   ]


def delete_files_from_list(file_list_path, images_path):
    with open(file_list_path, 'r') as file:
        file_names = file.readlines()

    # Remove whitespace characters like `\n` at the end of each line
    file_names = [name.strip() for name in file_names]

    deleted_files = []
    not_found_files = []

    for file_name in file_names:
        file_name = os.path.join(images_path, file_name)
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



# List of directory paths
directory_paths = ["D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/0",
                   "D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/1",
                   "D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/2",
                   "D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/3",
                   "D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/4",
                   "D:/computer_engineering/Projects/under_development/Hand-Gesture-Recognition/Dataset/5"]


for path in text_file_paths:
    for file_name in os.listdir(path):
        for directory_path in directory_paths:
            delete_files_from_list(path+file_name, directory_path)