from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='mlproject_package',
    version='0.1',
    author='Ankitesh Tiwari',
    author_email='official.ankitesh@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
