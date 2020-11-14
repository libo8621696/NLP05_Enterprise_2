import numpy as np
import data_io as pio
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

class Preprocessor:
    #在初始化列表中添加embedding_fp,导入glove.6B文件夹
    def __init__(self, datasets_fp, embedding_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.embedding_fp = embedding_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        # self.build_charset()
        self.embedding_index = {}
        self.concat_cqa_words_list = []  # 去重复后的cqa单词列表
        self.build_gloveset()
        self.tokenize_text()
        self.WordEmbedding()
    

    ## 通过build_gloveset函数导入GLove预训练词向量，形成长度400000的词典embedding_index，其中的键为Glove中的单词，值为该单词的预训练词向量。
    def build_gloveset(self):
        for fp in self.embedding_fp:
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1) # maxsplit=1,只对第一个出现的空格进行分割
                    coefs = np.fromstring(coefs,'f',sep=' ') # fromstring: 将字符串按照分隔符解码成矩阵
                    self.embedding_index[word] = coefs 
        print('Found %s word vectors in GLOVE Embeddings.'%len(self.embedding_index))
       


    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)

    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset
    
    ## 解析上下文、问答形成列表
    def dataset_words_info(self, inn):
        context_words = []
        question_words = []
        answer_words = []
        dataset = pio.load(inn)
        for _, context, question, answer, _ in self.iter_cqa(dataset):
            context_words.extend(context)
            question_words.extend(question)
            answer_words.extend(answer)
        return context_words, question_words, answer_words


    ## 将上下文、问答文本进行分词，其中上下文的分词列表为concat_context_words_list，总共有144707个单词（去重复后）;问句文本的分词列表为concat_question_words_list，总共有48379个单词（去重复后），回答文本的分词列表为concat_answer_words_list, 总共有95061个单词（去重复后），所有上下文和问答综合文本的分词列表为concat_cqa_words_list，总共有220858个单词（去重复后）.
    ## 本次作业使用包含上下文和问答的综合文本列表，对每个词生成相应的词嵌入。
    def tokenize_text(self):
        cqa_chars_list = []
        
        for fp in self.datasets_fp:
            context_chars, question_chars, answer_chars = self.dataset_words_info(fp)
            cqa_chars_list.extend(context_chars+question_chars+answer_chars)
        
        concat_cqa_chars = "".join(cqa_chars_list)

        self.concat_cqa_words_list =list(set(word_tokenize(concat_cqa_chars))) 
        return len(self.concat_cqa_words_list)
    
    # 实现glove加载到Word Embedding层，形成embedding矩阵embedding_matrix

    def WordEmbedding(self):
        embedding_dim = len(list(self.embedding_index.values())[0])
        num_words = min(len(self.concat_cqa_words_list), len(self.embedding_index))
        self.embedding_matrix = np.zeros((num_words, embedding_dim))  ##首先用0初始化嵌入矩阵
        print(self.embedding_index['the'])

        for i, word in enumerate(self.concat_cqa_words_list):
            embedding_vector = self.embedding_index.get(word)# 如果cqa中的词语是收录在glove词表中的词语，直接给该词语附上相应的词向量
             
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
            # 对于cqa中没有收录在glove词表中的词语，我们先默认其词向量为0向量。

        return 0

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)


if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ],['./data/glove.6B/glove.6B.50d.txt'])

    
    num_words = p.embedding_matrix
    print(num_words[:10])
    # print(num_words.shape)
    # print(num_words[:10])
    # print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
