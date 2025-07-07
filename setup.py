from setuptools import setup, find_packages

setup(
    name="chattykg",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "langchain-openai",
        "pydantic",
        "python-dotenv",
        "termcolor",
        "torch",
    ],
) 