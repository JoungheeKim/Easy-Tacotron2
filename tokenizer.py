import torch
import os
from tqdm import tqdm
import re
import logging
from typing import Any, List, Optional
import unicodedata
import string
from g2pk import G2p


def get_abspath(path):
    return os.path.abspath(path)

class dummy_g2p:
    def __init__(self, **kwargs):
        logging.info('dummy g2pk')
    
    def __call__(self, sentence:str, **kwargs):
        return sentence

G2P_LIBRARY_DICT = {
    'g2pk' : G2p,
    'dummy' : dummy_g2p,
}


class TacotronTokenizer(object):
    """
        Tokenizer
        Only support
        - Number
        - English
        - Korean
        - Special Tokens [,.~?!]
    """
    special_regex = re.compile('[%s]' % re.escape(string.punctuation))
    remove_regex = re.compile('[%s]' % re.escape('"#$%&()*+-/:;<=>@[\\]^_{}”“'))

    tokenizer_save_name = 'pretrained_tokenizer.bin'
    pad_token = '<pad>'
    unk_token = '<unk>'
    bos_token = '<bos>'
    eos_token = '<eos>'
    normalize_option = 'NFKD'

    selected_token = {
        '’': "'",
        ",": ","
    }

    g2p_processor = None

    def __init__(self, vocabs={}, normalize_option='NFKD', g2p_lib:str='g2pk'):
        super(TacotronTokenizer, self).__init__()
        self.vocabs_to_ids = vocabs
        
        self.vocabs_to_ids[self.pad_token] = len(self.vocabs_to_ids)
        self.vocabs_to_ids[self.unk_token] = len(self.vocabs_to_ids)
        self.vocabs_to_ids[self.bos_token] = len(self.vocabs_to_ids)
        self.vocabs_to_ids[self.eos_token] = len(self.vocabs_to_ids)

        self.__rebuild()

        self.pad_id = self.vocabs_to_ids[self.pad_token]
        self.unk_id = self.vocabs_to_ids[self.unk_token]
        self.bos_id = self.vocabs_to_ids[self.bos_token]
        self.eos_id = self.vocabs_to_ids[self.eos_token]

        ## add space toekn
        self.add_token(' ')

        self.normalize_option = normalize_option
        NORMALIZE_OPTIONS = ['NFC', 'NFKC', 'NFD', 'NFKD']
        assert self.normalize_option in NORMALIZE_OPTIONS, 'normalize option should either {}. ex) Korean for "NFKD", English for "NFC"'.format(', '.join(NORMALIZE_OPTIONS))

        assert g2p_lib in G2P_LIBRARY_DICT, '{} is not supported g2p library. Please selct one of [{}]'.format(g2p_lib, ', '.join(G2P_LIBRARY_DICT.keys()))
        self.g2p_lib = g2p_lib
        self.g2p_processor = G2P_LIBRARY_DICT[g2p_lib]()


    def __rebuild(self):
        self.ids_to_vocabs = [item for item in self.vocabs_to_ids.keys()]
        self.vocabs_to_ids = {item:idx for idx, item in enumerate(self.ids_to_vocabs)}
        

    def add_token(self, temp_string):
        assert type(temp_string) == str, 'must be string'
        temp_string = unicodedata.normalize(self.normalize_option, temp_string.lower())
        for temp_char in list(temp_string):
            if temp_char not in self.vocabs_to_ids:
                self.vocabs_to_ids[temp_char] = len(self.vocabs_to_ids)
                self.ids_to_vocabs.append(temp_char)
            

    @classmethod
    def from_pretrained(cls, pretrained_path:str):
        pretrained_path = get_abspath(pretrained_path)

        tokenizer_path = os.path.join(pretrained_path, cls.tokenizer_save_name)
        logging.info('load tokenizer from [{}]'.format(pretrained_path))

        state = torch.load(tokenizer_path)
        normalize_option = state['normalize_option']
        vocabs = state['vocabs']
        g2p_lib = state['g2p_lib']

        return cls(vocabs, normalize_option, g2p_lib)

    @classmethod
    def build_tokenizer(cls, transcripts:List[str], normalize_option='NFKD', g2p_lib:str='g2pk'):
        tokenizer = cls({}, normalize_option, g2p_lib)

        for transcript in tqdm(transcripts, desc='build tokenizer'):
            phoneme_transcript = tokenizer.g2p_processor(transcript)
            tokenizer.add_token(phoneme_transcript)

        logging.info('bulild tokenizer with data length [{}]'.format(len(transcripts)))
        return tokenizer

    def save_pretrained(self, save_path:str):
        save_path = get_abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        tokenizer_path = os.path.join(save_path, self.tokenizer_save_name)
        torch.save(
            {
                'vocabs' : self.vocabs_to_ids,
                'normalize_option': self.normalize_option,
                'g2p_lib' : self.g2p_lib
             }, tokenizer_path
        )
        logging.info('save tokenizer to [{}]'.format(save_path))

    def encode(self, transcript:str, add_special_token:bool=True, **kwargs):
        assert type(transcript)==str, 'must be string'
        
        ## convert german accents
        for before, after in self.selected_token.items():
            transcript = transcript.replace(before, after)

        transcript = self.remove_regex.sub('', transcript)
        phoneme_transcript = self.g2p_processor(transcript)

        phoneme_transcript = unicodedata.normalize(self.normalize_option, phoneme_transcript.lower())

        text_ids = list()
        for temp_char in list(phoneme_transcript):
            if temp_char in self.vocabs_to_ids:
                text_ids.append(self.vocabs_to_ids[temp_char])
        
        if add_special_token:
            text_ids = [self.bos_id] + text_ids + [self.eos_id]

        return {
            'text_ids' : text_ids,
        }

    def encode_batch(self, transcripts:List[str], add_special_token:bool=True, **kwargs):
        outputs=list()
        for transcript in tqdm(transcripts, desc='tokenizing'):
            outputs.append(self.encode(transcript, add_special_token)['text_ids'])

        return {
            'text_ids' : outputs if len(outputs) >1 else outputs[0],
        }

    def decode(self, text_ids:list, **kwargs):
        output_string = ''
        for text_id in text_ids:
            if text_id not in  [self.pad_id, self.unk_id, self.eos_id, self.bos_id]:
                output_string += self.ids_to_vocabs[text_id]
            
        return output_string

    def token_to_id(self, temp_str:str):
        return self.vocabs_to_ids(temp_str)

    def tokens_to_ids(self, tokens:List[str]):
        ids = list()
        for token in tokens:
            ids.append(self.vocabs_to_ids(token))
        return ids

    def get_num_labels(self):
        return len(self.vocabs_to_ids)

    def __call__(self, transcript:str, add_special_token:bool=True, **kwargs):
        return self.encode(transcript, add_special_token)['text_ids']

