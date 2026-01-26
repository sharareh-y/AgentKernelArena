# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from pathlib import Path

def delete_gpucore_files(root_dir="."):
    """
    delete all the gpucore* files
    
    Args:
        root_dir (str): the root dir to detele, default set to the root path of AgentKernelArena
    """

    deleted_count = 0
    

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:

            if filename.startswith("gpucore"):

                file_path = os.path.join(dirpath, filename)
                try:
 
                    os.remove(file_path)
                    print(f"deleted: {file_path}")
                    deleted_count += 1
                except PermissionError:
                    print(f"permision denied: {file_path}", file=sys.stderr)
                except FileNotFoundError:
                    print(f"file no exist: {file_path}", file=sys.stderr)
                except Exception as e:
                    print(f"error while removing {file_path}: {str(e)}", file=sys.stderr)
    
    print(f"\n delete completed! totaly deleted {deleted_count} gpucore* files")

if __name__ == "__main__":
    
    path = str(Path(__file__).parent.parent.parent)
    confirm = input(f"ready to remove all gpucore* under {path}(y/n): ")
    if confirm.lower() == "y":
        delete_gpucore_files(root_dir=path)
    else:
        print("操作已取消")