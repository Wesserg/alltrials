from setuptools import setup, find_packages

setup(
    name="alltrials",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Data Science
        "numpy",
        "scikit-learn",

        # Utils
        "tqdm",
        "datetime",
        "ipykernel",

        # AI/ML
        "torch",
        "torch_geometric",
        "torchvision",
        "pytorch-lightning",
        "gensim",
        "pomegranate",

        # Data Loading
        "pandas",
        "duckDB",
        "pysqlite3",
        "sqlalchemy",
        "psycopg2-binary",

        # Graphics
        "seaborn",
        "plotly",
        "matplotlib",
    ],
)