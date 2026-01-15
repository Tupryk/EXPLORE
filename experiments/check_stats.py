import os

def find_and_print_tree_stats(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "tree_stats_8.txt" in filenames:
            file_path = os.path.join(dirpath, "tree_stats_8.txt")
            print(f"\nFound: {file_path}")
            print("-" * 80)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    print(f.read())
            except Exception as e:
                print(f"Error reading file: {e}")

if __name__ == "__main__":
    root_directory = "multirun/2026-01-14/15-22-11"
    find_and_print_tree_stats(root_directory)
