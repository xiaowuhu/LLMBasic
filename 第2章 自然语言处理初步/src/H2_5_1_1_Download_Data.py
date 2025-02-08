import os
import hashlib
import requests
import zipfile
import tarfile


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

DATA_HUB['glove.6b.50d'] = (DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

DATA_HUB['glove.6b.100d'] = (DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

DATA_HUB['glove.42b.300d'] = (DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

DATA_HUB['wiki.en'] = (DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')

def download(name, cache_dir=os.path.join('..', 'data')):
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            print("已经下载过该文件:", os.path.abspath(fname))
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def get_data_folder():
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    data_path = os.path.join(current_dir, "data")
    return data_path

# 下载并解压zip/tar文件
def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    # 避免重复解压
    readme_file = os.path.join(base_dir, name, "README")
    if os.path.exists(readme_file) == False:
        if ext == '.zip':
            fp = zipfile.ZipFile(fname, 'r')
        elif ext in ('.tar', '.gz'):
            fp = tarfile.open(fname, 'r')
        else:
            assert False, '只有zip/tar文件可以被解压缩'
        fp.extractall(base_dir)
    print("解压完毕:", os.path.abspath(data_dir), data_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

if __name__=="__main__":
    data_dir = download_extract('aclImdb', 'aclImdb')
    data_dir = download_extract('glove.6b.100d')


