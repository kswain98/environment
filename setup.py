from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'environment',
  packages = find_packages(),
  version = '1.0.0',
  license='MIT',
  description = 'api',
  author = '',
  author_email = '',
  url = 'https://github.com/kswain55/environment',
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence'
  ],
  install_requires=[
    'python-socketio',
    'python-socketio[client]'
    'eventlet',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.11',
  ],
)