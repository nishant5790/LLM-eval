from setuptools import setup, find_packages

setup(
    name='ai_monitor',  # Replace with your package name
    version='0.1.1',
    description='A logging utility for Python',
    long_description=open('README.md').read(),  # Optional
    long_description_content_type='text/markdown',
    author='Kumar Nishant',
    author_email='nkumar@demandbase.com',
    url='https://gitlab.com/demandbase/data-cloud/technographics/data_science/ai-monitor.git',  # Optional
    license='MIT',  # Or the appropriate license
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=['boto3','pytz',
                      'numpy',
                      'scikit-learn',
                      'nltk',
                      'rouge',
                      'torch',
                      'transformers',
                      ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the minimum Python version
)