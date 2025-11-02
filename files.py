import os

def print_tree(start_path, indent=''):
    items = sorted(os.listdir(start_path))
    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = index == len(items) - 1
        connector = '└── ' if is_last else '├── '
        print(indent + connector + item)
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            print_tree(path, indent + extension)

# 🔧 Change this to your project path
project_path = r"D:\AI_Image_Detection_WebApp"
print(f"Project structure for: {project_path}\n")
print_tree(project_path)
