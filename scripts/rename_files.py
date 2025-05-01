import os

# rename image function
def rename_jpg_files_in_folder():
    folder_path = os.path.expanduser(input("enter the path to the folder containing the image files: ").strip())
    if not os.path.isdir(folder_path):
        print("FOLDER DOES NOT EXIST!")
        return
    # request new name title
    new_base_name = input("Enter the new base name: ").strip()
    # error handling for invalid entires
    if not new_base_name:
        print("INVALID NAME, FORMAT INCORRECT!")
        return
    # preping images by sorting
    files = sorted(os.listdir(folder_path))
    jpg_files = [f for f in files if f.lower().endswith(".jpg")]
    # no images detected error handling
    if not jpg_files:
        print("NO IMAGE FILES FOUND IN FOLDER!")
        return
    # checking and finalising paths and names
    for i, file_name in enumerate(jpg_files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{i}{new_base_name}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")
    # ssuccess message output
    print("\n All image files renamed successfully!")
# main program loop
if __name__ == "__main__":
    rename_jpg_files_in_folder()
