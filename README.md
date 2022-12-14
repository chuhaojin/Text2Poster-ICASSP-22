# Text2Poster-ICASSP-22
The inference code of the ICASPP-2022 paper "Text2Poster: Laying Out Stylized Texts on Retrieved Images".

![framework](framework.png)

Paper Link: https://ieeexplore.ieee.org/abstract/document/9747465



## Install

We recommend you use anaconda to run our Text2Poster. Run the following command to install the dependent libraries:

```shell
bash install_package.sh
```

you also can install the dependent libraries manually:

```shell
# using the tsinghua mirror to speed up the install.
conda install pytorch=1.10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install torchvision=0.11.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
pip install opencv_contrib_python
pip install argparse
pip install freetype-py
pip install requests
pip install jsonlines
pip install tqdm
```



# Running

We provide two example, Run the following command to run our Text2Poster:

```sh
bash run.sh
```

Some parameters:

- **input_text_file**: The input text elements, it contains: 1). sentences (phase) and their font size, 2). query used to retrieve background images.
- **output_folder**: The folder to save the output posters and some process figures.
- **background_folder**: The folder to save local background images, If  images are not saved locally, they will be downloaded from remote.
- **top_n**: Arrange the text elements on the top N retrieved images.
- **save_process**: Save the process figure (etc. saliency map) or not.



### **We also provide the following examples:**

- **background image retrieval**

```
python background_retrieval.py
```

- **Layout distribution prediction**

```python
python layout_distribution_predict.py
```

- **Layout refinement**

```python
python layout_refine.py
```



## Tips

### Something about our background image retrieval

- Our background image retrieval is implemented by a Chinese pre-trained  text-image retrieval model [BriVL](https://github.com/BAAI-WuDao/BriVL).
- ~~You can extract text and image embedding by the API of BriVL at  [here](https://github.com/chuhaojin/WenLan-api-document).~~ The API server has been retired in 2023.01.09.
- A text-image retrieval application is provided at [http://1.13.255.9:8889](http://1.13.255.9:8889). The core code of this application at [https://github.com/chuhaojin/BriVL-BUA-applications](https://github.com/chuhaojin/BriVL-BUA-applications).



## News

- **[2023.01.10]** We update the background image retrieval website to [http://1.13.255.9:8889](http://1.13.255.9:8889). The original website *buling.wudaoai.cn* has been retired in 2023.01.09.



# Examples
**input text elements 1**
```json
{
    "sentences": [
        ["??????????????????", 55],
        ["???????????????????????????????????????", 40],
        ["?????????????????????????????????????????????????????????", 30]
    ],
    "background_query": "??????????????????"
}
```

<img src="./example/outputs_1/0/poster.jpg" alt="poster" height="250" /> <img src="./example/outputs_1/1/poster.jpg" alt="poster" height="250" />



**input text elements 2**

```json
{
    "sentences": [
        ["ICASSP 2022", 55],
        ["May 22 - 27, 2022, Singapore", 40]
    ],
    "background_query": "?????????"
}
```

<img src="./example/outputs_2/0/poster.jpg" alt="poster" height="250" /> <img src="./example/outputs_2/2/poster.jpg" alt="poster" height="250" /> 



## Some output during process

we also output some intermediate processing files in `./example/outputs`:

<img src="./bk_image_folder/-SdD0KbD7N0.png" alt="-SdD0KbD7N0" height="250" /> <img src="./example/outputs_1/0/saliency_map_with-smooth.jpg" alt="saliency_map_with-smooth" height="250" /> 

- **Right image**: The original background image.
- **Left image**: Saliency map (**blue**) with smooth region map (**red**).

<img src="./example/outputs_1/0/layout_distribution.jpg" alt="layout_distribution" height="250" /> <img src="./example/outputs_1/0/saliency_map_with-layout-distribution.jpg" alt="saliency_map_with-smooth" height="250" /> 

- **Right image**: The prediction of layout distribution map.
- **Left image**: Saliency map (**blue**) with predicted layout distribution map (**red**). 

<img src="./example/outputs_1/0/initial_layout.jpg" alt="initial_layout" height="250" /> <img src="./example/outputs_1/0/refined_layout.jpg" alt="refined_layout" height="250" /> 

- **Right image**: Initial layout map. 
- **Left image**: Refined layout map. 

**Blue region**: The saliency map;

**Green region**: The predicted layout distribution map;

**Red region**: the predicted layout map.



## Requirements

python==3.7

pytorch=1.10.0

torchvision=0.11.0

freetype-py

opencv_contrib_python

requests

jsonlines

tqdm

argparse



## Citation

If you find this paper and repo useful, please cite us in your work:

```bibtex
@inproceedings{DBLP:conf/icassp/JinXSL22,
  author    = {Chuhao Jin and
               Hongteng Xu and
               Ruihua Song and
               Zhiwu Lu},
  title     = {Text2Poster: Laying Out Stylized Texts on Retrieved Images},
  booktitle = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
               {ICASSP} 2022, Virtual and Singapore, 23-27 May 2022},
  pages     = {4823--4827},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/ICASSP43922.2022.9747465},
  doi       = {10.1109/ICASSP43922.2022.9747465},
  timestamp = {Tue, 07 Jun 2022 17:34:56 +0200},
  biburl    = {https://dblp.org/rec/conf/icassp/JinXSL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact

My Email is: jinchuhao@ruc.edu.cn
