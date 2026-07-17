"""
Codebase Bundler

This script traverses the current working directory and concatenates the contents 
of relevant source code files into a single output text file (`codebase.txt`). 
It is particularly useful for extracting a project's context into a single, 
readable document (e.g., for sharing context with Large Language Models).

Key behaviors:
  - Directory Filtering: Skips hidden folders (starting with '.') and heavy/generated 
    directories (like node_modules, venv, __pycache__, build).
  - File Filtering: Skips hidden files, lockfiles, and the script's own output 
    to prevent recursive bundling.
  - Extension Filtering: Only processes files with explicitly approved extensions 
    (e.g., .py, .js, .md, .json).
  - Formatting: Wraps the contents of each bundled file with distinct 'START OF FILE' 
    and 'END OF FILE' markers containing the relative path for easy parsing.
"""

import os

# Files or folders you want to skip completely
EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'dist', 'build', 'venv', '.env'}
EXCLUDE_FILES = {'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock', 'codebase.txt', 'bundle.py'}
VALID_EXTENSIONS = {'.js', '.ts', '.py', '.go', '.rs', '.java', '.cpp', '.h', '.cs', '.html', '.css', '.md', '.json', '.yml', '.yaml'}

def bundle_repo():
    output_file = "codebase.txt"
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk("."):
            # Skip hidden and excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            
            for file in files:
                if file in EXCLUDE_FILES or file.startswith('.'):
                    continue
                    
                _, ext = os.path.splitext(file)
                if ext not in VALID_EXTENSIONS:
                    continue
                    
                relative_path = os.path.relpath(os.path.join(root, file), ".")
                
                outfile.write(f"\n--- START OF FILE: {relative_path} ---\n")
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
                outfile.write(f"\n--- END OF FILE: {relative_path} ---\n")
                
    print(f"✨ Success! Your codebase is bundled into '{output_file}'")

if __name__ == "__main__":
    bundle_repo()
