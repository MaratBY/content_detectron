from setuptools import setup, find_packages

# Setup declaration
setup(
    name="content detectron",
    description="The package can detect intros, outros, previews in the video file",
    url="https://github.com/Marat-BY/content_detectron",
    author="Marat_BY",
    author_email="none@none.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['dataclasses==0.8',
                      'ffmpeg-python==0.2.0',
                      'Keras==2.2.4',
                      'numpy==1.19.5',
                      'opencv-python==4.1.1.26',
                      'packaging==20.9',
                      'Pillow==8.2.0',
                      'pytest==6.0.2',
                      'PyYAML==5.4.1',
                      'scipy==1.3.1',
                      'tensorflow==2.5.0',
                      'termcolor==1.1.0',
                      'tqdm==4.40.2'
                      ],
    zip_safe=False)
