# whisper_transcription
whisper transcription pipeline

```python
import os
import whisper
import torch
import codecs
import unicodedata
import json
from tqdm import tqdm
from datetime import timedelta
import multiprocessing
from dataclasses import dataclass


__SB__ = lambda t, d: tqdm(
    total=t,
    desc=d,
    bar_format="{desc}: {percentage:3.0f}%|"
    + "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] "
    + "{rate_fmt}{postfix}]",
)

_f = lambda f: [
    str(f)[: len(f) - (str(f)[::-1].find("/")) :].lower(),
    str(f)[
        len(f)
        - (str(f)[::-1].find("/")) : (len(f))
        - 1
        - len(f[-(str(f)[::-1].find(".")) :])
    ],
    str(f)[-(str(f)[::-1].find(".")) :].lower(),
]

_imr = (
    lambda x: (int(abs(x[2] * (x[0] / x[1]))), int(abs(x[3])))
    if x[0] < x[1]
    else (int(abs(x[2])), int(abs(x[2] // (x[0] / x[1]))))
    if x[0] > x[1]
    else (x[2], x[3])
)


_GET_LIST_ = lambda fp, exts: [
    fp + f
    for f in os.listdir(fp[:-1:])
    if os.path.isfile(fp + f) and str(f[-(f[::-1].find(".")) :]).lower() in exts
]

_ffn = (
    lambda s: str(s).replace(chr(92), chr(47)).replace(chr(34), "")
    if str(s)[-1] == chr(47)
    else str(s).replace(chr(92), chr(47)).replace(chr(34), "") + chr(47)
    if len(_f(str(s).replace(chr(92), chr(47)).replace(chr(34), ""))[2])!=3
    else str(s).replace(chr(92), chr(47)).replace(chr(34), "")
)


def _unique(l):
    s = set()
    n = 0
    for x in l:
        if x not in s:
            s.add(x)
            l[n] = x
            n += 1
    del l[n:]
    return l


@dataclass
class wh:
    model:whisper = None
    tr_models = [
                'tiny.en',  #0
                'tiny',     #1
                'base.en',  #2
                'base',     #3
                'small.en', #4
                'small',    #5
                'medium.en',#6
                'medium',   #7
                'large'     ]#8
    file:str = ""
    file_path:str = ""
    result:dict = None
    def __post_init__(self):
        self.model = wh.model
        self.file = wh.file
        self.file_path = wh.file_path
        self.result = wh.result
        self.tr_models = wh.tr_models
        super().__setattr__("attr_name", self)


print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
cuda_check = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
assert cuda_check == str("cuda")


def seg_proc(file:str,result):
    f = _f(file)
    txt_data = []
    res_dict = dict(result)
    res_segments = res_dict['segments']
    for segment in res_segments:
        text_res = str('')
        segment = dict(segment)
        text_res = text_res + str(f"[{str(float(segment['start']))}-{str(float(segment['end']))}]: {str(codecs.decode(unicodedata.normalize('NFKD', codecs.decode(bytes(segment['text'],encoding='utf-8'))).encode('ascii', 'ignore')))}")+str('\n')
        txt_data.append(text_res)
    print("saving json")
    with open(str(f"{f[0]}{f[1]}.json"),"wt") as fi:
        fi.write(json.dumps(result,indent=4,ensure_ascii=True))
    print("saving txt")
    with open(str(f"{f[0]}{f[1]}.txt"),"wt") as fi:
        for tr in txt_data:
            fi.write(tr) 

def srt_cap_file(file):
    f = _f(file)
    def t_s(i:int,n:int)->str:
        t = t_list[i][n]
        x = str(t).split('.')
        x0=int(x[0])
        x1=int(x[1][:6])
        xMS = str(r'{0:03d}'.format(x1))[:3]
        xH = str(x0 // 60 // 60 % 60).zfill(2)
        xM = str(x0 // 60 % 60).zfill(2)
        xS = str(x0 % 60).zfill(2)
        return f'{xH}:{xM}:{xS},{xMS}'

    txt_data = ""
    with open (f'{f[0]}{f[1]}.txt','rt') as fi:
        txt_data = fi.read()
    txt_list = []
    t_list = []
    txt_list = [x for x in txt_data.replace('\r\n','\n').split('\n') if x != '']
    t_list = [[float(t[0]),float(t[1])] for t in [x[0].replace('[','').replace(']','').split('-') for x in [y.split(':') for y in txt_list]] if t[0]!='' and t[1]!='']
    tx = lambda x: x[-(len(x)-x.find(':  ')-3):]
    srt_list = [str(f'{t_s(i,0)} --> {t_s(i,1)}\n{tx(x)}') for i,x in enumerate(txt_list)]
    srt_write = '\n'.join(f'{str(i)}\n{x}\n' for i,x in enumerate(srt_list))
    print(srt_write)
    with open (f'{f[0]}{f[1]}.srt','wt') as fi:
        fi.write(srt_write)

def transcribe(file:str)->None:
    result = None
    model = whisper.load_model(wh.tr_models[6],device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    result = model.transcribe(
        file,
        verbose=True,
        compression_ratio_threshold=1.4,
        condition_on_previous_text=False,
        temperature=0.8,
        word_timestamps=False
        )
    seg_proc(file,result)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    audio_list:list[str] = []
    audio_list = _GET_LIST_(wh.file_path,['mp4'])
    for file in audio_list:
        wh.file=''
        wh.file = file
        transcribe(wh.file)
```
