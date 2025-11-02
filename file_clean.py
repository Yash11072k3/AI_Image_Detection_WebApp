import os

def print_tree(startpath, ignore_dirs=None, prefix="", output_file=None):
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.idea', '.vscode'}

    lines = []
    lines.append(f"Project structure for: {startpath}\n")

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        level = root.replace(startpath, "").count(os.sep)
        indent = "│   " * level
        folder_name = os.path.basename(root)
        lines.append(f"{indent}├── {folder_name}")

        subindent = "│   " * (level + 1)
        for f in files:
            lines.append(f"{subindent}├── {f}")

    result = "\n".join(lines)
    print(result)

    # Save output to file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n✅ Project structure saved to: {output_file}")

if __name__ == "__main__":
    project_path = os.getcwd()
    output_path = os.path.join(project_path, "project_structure.txt")
    print_tree(project_path, output_file=output_path)
