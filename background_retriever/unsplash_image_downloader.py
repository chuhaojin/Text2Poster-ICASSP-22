# -*- encoding: utf-8 -*-
'''
@File    :   unsplash_image_downloader.py
@Time    :   2023/01/15 01:37:53
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import requests as rq
import os
import pathlib


def save_image(folder: str, name: str, url: str):
    # Get the data from the url
    image_source = rq.get(url)

    # If there's a suffix, we will grab that
    suffix = pathlib.Path(url).suffix

    # Check if the suffix is one of the following
    if suffix not in ['.jpg', '.jpeg', '.png', '.gif']:
        # Default to .png
        output = name + '.png'
    else:
        output = name + suffix

    # Check first if folder exists, else create a new one
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create our output in the specified folder (wb = write bytes)
    with open(f'{folder}{output}', 'wb') as file:
        file.write(image_source.content)
        print(f'Successfully downloaded: {output}')
    return None


if __name__ == "__main__":
    url = "https://unsplash.com/photos/ZHS3j0_Y_KM"
    name = "ZHS3j0_Y_KM"
    image_width = 1080
    url = url + "/download?force=true&w={}".format(image_width)
    save_image('./bk_image_folder/', name, url)
