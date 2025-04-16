from setuptools import setup, find_packages

setup(
    name = 'memeGPT',
    version = '1.0.0',
    packages = find_packages(include=["memeGPT", "memeGPT.*"]),
    install_requires = [
        "torch>=2.0.0",
        "datasets",
        "transformers",
        "numpy",
        "pyyaml",
        "mlflow",
    ],
    entry_points = {
        "console_scripts": [
            "train=memeGPT.scripts.train:main",
            "evaluate=memeGPT.scripts.evaluate:main",
            "generate=memeGPT.scripts.generate:main"
        ]
    },
    author = 'Kandarpa Sarakar',
    description = None,
    long_description = None,
    long_description_content_type = 'text/markdown',
    python_requires='>=3.8',


)

