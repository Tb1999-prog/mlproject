from typing import List
from setuptools import setup,find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    
    Get requirements
    
    '''
    
    requirements = []
    with open(file_path) as file_obj:
          requirements =file_obj.readlines()
          [req.replace("\n", "") for req in requirements]
          
          if HYPEN_E_DOT in requirements:
              requirements.remove(HYPEN_E_DOT)
          
    return requirements



setup(
    name='mlproject',
    version='0.0.1',
    author='Tushar Batra',
    author_email='batra.tushar12dec1999@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)