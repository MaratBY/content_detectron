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
    install_requires=[
                    'Keras==2.2.4',
                    'Pillow==6.1.0',
                    'ffmpeg_python==0.2.0',
                    'matplotlib==3.1.1',
                    'opencv_python==4.1.1.26',
                    'pandas==0.25.0',
                    'scipy==1.3.1',
                    'tqdm==4.40.2',
                    'natsort==6.2.0',
                    'tensorflow==1.14',
                    'numpy==1.16.2',
                    'pytest==6.0.2'
      ],
      zip_safe=False)
