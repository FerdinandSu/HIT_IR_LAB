
import  os

import platform

# 获取Windows或Linux
uname=platform.architecture()[1]

executable_prefix='lib\\svm_rank\\' if uname=='WindowsPE' else 'lib/svm_rank/'

def __wget(url:str,path:str):
    import urllib.request
    urllib.request.urlretrieve(url, path)

def __ensure_svm_rank_bin(force=False):
    if os.path.exists('lib/svm_rank'):
        if not force:
            return
        else:
            os.system('rm -r lib')
    print('Lazy load: Svm_rank Binaries')
    if uname=='WindowsPE':
        os.system('mkdir lib\\svm_rank')
        __wget('https://osmot.cs.cornell.edu/svm_rank/current/svm_rank_windows.zip', "lib/svm_rank/x.zip")
        import zipfile
        archive=zipfile.ZipFile("lib/svm_rank/x.zip")
        for file in archive.namelist():
            archive.extract(file,'lib/svm_rank')
        archive.close()
        os.remove('lib/svm_rank/x.zip')
    elif uname=='ELF': # Linux没有32位发行版了
        os.system('mkdir lib/svm_rank')
        __wget('https://osmot.cs.cornell.edu/svm_rank/current/svm_rank_linux64.tar.gz', "lib/svm_rank/x.tgz")
        import tarfile
        archive=tarfile.open("lib/svm_rank/x.tgz","r:gz")
        for file in archive.getnames():
            archive.extract(file,'lib/svm_rank')
        archive.close()
        os.remove('lib/svm_rank/x.tgz')
    else:
        print('Auto download on Mac OS is not supported. Use Windows or Linux, or compile the binaries manually referring to https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html.')

# 垃圾Windows，CMD里必须用\\分割路径
def __path_fix(path:str):
    return os.path.join(path.split('/'))

def train(train_set:str,model:str):
    __ensure_svm_rank_bin()
    os.system(f'{executable_prefix}svm_rank_learn -c 10 {__path_fix(train_set)} {__path_fix(model)}')

def predict(data_set:str,model:str,output:str):
    __ensure_svm_rank_bin()
    os.system(f'{executable_prefix}svm_rank_classify {__path_fix(data_set)} {__path_fix(model)} {__path_fix(output)}')