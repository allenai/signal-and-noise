import os
import zipfile

def get_dir_size(path):
    """Calculate total size of a directory in bytes"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

root_dir = os.getcwd()
zip_path = 'export.zip'

if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    dir_sizes = {}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != 'img' and d != 'data' and d != '__pycache__']
        
        relpath = os.path.relpath(dirpath, root_dir)
        if relpath == '.':
            continue
            
        for filename in filenames:
            if not filename.startswith('.') and filename != 'export.py':
                filepath = os.path.join(dirpath, filename)
                arcname = os.path.join(relpath, filename)
                zipf.write(filepath, arcname)
        
        dir_sizes[relpath] = get_dir_size(dirpath)

sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)
print("\nLargest directories:")
for dir_name, size in sorted_dirs[:10]:
    size_mb = size / (1024 * 1024)
    print(f"{dir_name}: {size_mb:.1f} MB")
