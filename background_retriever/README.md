# Background Retriever for Text2Poster
## Online demo

#### We provide a online background retrieval demo [《布灵的想象世界》](http://1.13.255.9:8889/)。



#### You also can use our background retriever by source code as follows:

## Install

We recommend you use anaconda to run our Text2Poster. Run the following command to install the dependent libraries:

```shell
bash ../install_package.sh
```

you also can install the dependent libraries manually:

```shell
# using the tsinghua mirror to speed up the install.
conda install pytorch=1.10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install torchvision=0.11.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
pip install opencv_contrib_python
pip install transformers==3.2.0
pip install argparse
pip install freetype-py
pip install requests
pip install jsonlines
pip install tqdm
pip install pyyaml
pip install easydict
pip install timm
```



## Download

We provide the following resource:

- Weights of text encoder of BriVL: [brivl-textencoder-weights.pth](https://drive.google.com/drive/folders/18qYZu7TfKpngXuHW9pmkWUn8dK2YPSVW) -> ```./background_retriever/weights/```;
- Unsplash images features (extracted by BriVL): [wenlan_unsplash_feats.npy](https://drive.google.com/drive/folders/18qYZu7TfKpngXuHW9pmkWUn8dK2YPSVW) -> ```./background_retriever/background_feats/```;
- URL of background images: ```./background_retriever/background_feats/unsplash_image_url.jsonl```.



### **We provide the following examples:**

- **background image retrieval （from source code）**

```sh
cd background_retriever
python main.py
```



## Tips

### Something about our background image retrieval

- Our background image retrieval is implemented by a Chinese pre-trained  text-image retrieval model [BriVL](https://github.com/BAAI-WuDao/BriVL).
- ~~You can extract text and image embedding by the API of BriVL at  [here](https://github.com/chuhaojin/WenLan-api-document).~~ The API server has been retired in 2023.01.09.
- A text-image retrieval application is provided at [http://1.13.255.9:8889](http://1.13.255.9:8889). The core code of this application at [https://github.com/chuhaojin/BriVL-BUA-applications](https://github.com/chuhaojin/BriVL-BUA-applications).



## News

- **[2023.01.10]** We update the background image retrieval website to [http://1.13.255.9:8889](http://1.13.255.9:8889). The original website *buling.wudaoai.cn* has been retired in 2023.01.09.



## Requirements

```
python==3.7
pytorch=1.10.0
torchvision=0.11.0
transformers==3.2.0
freetype-py
opencv_contrib_python
requests
jsonlines
tqdm
argparse
pyyaml
easydict
timm
```



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
