# -*- encoding: utf-8 -*-
'''
@File    :   background_retrieval.py
@Time    :   2023/01/05 16:17:46
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import requests
import os


def bk_img_retrieval(text, local_image_folder = "./bk_image_folder"):
    # The image retrieval API based on our BriVL model and unsplash.com.
    # This API only runs in Chinese Mainland. 
    # You can consider using the BriVL model to build this service yourself, or use VPN to access it.
    # Domain [buling.wudaoai.cn] retired in 2023-01-09.
    # url = "http://buling.wudaoai.cn/t2i_query"
    url = "http://bl.mmd.ac.cn:8889/t2i_query"
    response = requests.get(url, params={"text": text})
    results = response.json()
    # print(results)
    '''
    A results example: 
        {
            'data': 
            [
                {
                    'image_list': 
                    [
                        {'image_path': 'http://buling.wudaoai.cn/image_unzip/mtMFJz071Cs.png',
                        'image_url': 'https://unsplash.com/photos/mtMFJz071Cs'},
                        {'image_path': 'http://buling.wudaoai.cn/image_unzip/ZHS3j0_Y_KM.png',# image_path is the path in our servicer.
                        'image_url': 'https://unsplash.com/photos/ZHS3j0_Y_KM'}, # image_url is the origin path in unsplash.com.
                    ],
                    'text': '下雪啦，一起出去打雪仗吧。'
                },
            ],
            'info': 'sentence',
            'status_code': 2001,
            'work_id': 'bVAUkjVj3AP0A5aWZ7g07oey'
        }
    '''
    # Get the image url 
    net_files = [item["image_path"] for item in results["data"][0]["image_list"]]
    unsplash_files = [item["image_url"] for item in results["data"][0]["image_list"]]
    files = [item["image_path"].split("/")[-1] for item in results["data"][0]["image_list"]]
    image_path_list = [local_image_folder + "/" + file for file in files]
    # If these pictures are not saved locally, they will be downloaded from our server,
    # or you can manually obtain them from unsplash.
    for net_file, file, path, unsplash_path in zip(net_files, files, image_path_list, unsplash_files):
        if not os.path.exists(path):
            print("Download pictiure {} from {}".format(file, net_file))
            print("Pictiure {} Origin UNSPLASH path: {}".format(file, unsplash_path))
            r = requests.get(net_file)
            f = open(path, "wb")
            f.write(r.content)
            f.close()
    return image_path_list


if __name__ == "__main__":
    text = "下雪啦，一起出去打雪仗吧。"
    local_image_folder = "bk_image_folder"
    image_path_list = bk_img_retrieval(text, local_image_folder)
    print(image_path_list)
