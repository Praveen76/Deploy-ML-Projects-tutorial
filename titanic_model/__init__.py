import sys
from pathlib import Path

# Get the absolute path of the current script file and resolve any symbolic links
file = Path(__file__).resolve() #__file__ is a special variable that holds the path of the current script

print("Path of current file :", file) #  D:\IISc_MLOps\MLOps\Practical_MLOps\week2\Practical_MLOps\project_with_test\titanic_model\__init__.py

# Get the parent directory and the grandparent directory(root) of the script file
parent, root = file.parent, file.parents[1]
print("parent :",parent, "root :", root) 
#parent:  D:\IISc_MLOps\MLOps\Practical_MLOps\week2\Practical_MLOps\project_with_test\titanic_model\
#root:  D:\IISc_MLOps\MLOps\Practical_MLOps\week2\Practical_MLOps\project_with_test\
# Append the grandparent directory to the sys.path list
sys.path.append(str(root))

print("sys.path :", sys.path)
from titanic_model.config.core import PACKAGE_ROOT, config

# Open the "VERSION" file located in the PACKAGE_ROOT directory and read its content.
with open(PACKAGE_ROOT / "VERSION") as version_file: # The / operator is used to concatenate the path components.
    __version__ = version_file.read().strip()
    print("__version__ :", __version__) #__version__ : 0.0.1