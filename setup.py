from setuptools import find_packages, setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='namrata',
    author_email='namratasutar1401@gmail.com',
    install_requires=[
        "openai",
        "langchain",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "setuptools",
        "langchain_core",
        "pandas",
        "langchain_community",
        "importlib_resources"
    ],
    packages=find_packages(),
)
