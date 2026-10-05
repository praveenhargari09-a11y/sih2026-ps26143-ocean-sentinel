import os
import zipfile

def create_zip():
    project_dir = r"C:\Users\prave\Desktop\Oil Spill"
    zip_path = r"C:\Users\prave\Desktop\OilSpill_Project.zip"
    
    exclusions = {'node_modules', '.git', '__pycache__', '.venv', 'venv'}
    
    print(f"Creating zip at {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # Modify dirs in-place to prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclusions and d != 'downloads']
            
            for file in files:
                # Exclude large .7z files explicitly just in case
                if file.endswith('.7z'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_dir)
                zipf.write(file_path, arcname)
                
    print("Zip successfully created with contents!")

if __name__ == '__main__':
    create_zip()
