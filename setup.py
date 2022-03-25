from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='CoolMelodyProject',
      version="1.0",
      description="Deep learning RNN project that creates a melody one note at a time with Mozart's help.",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/CoolMelodyProject-run'],
      zip_safe=False)
