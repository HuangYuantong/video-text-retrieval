"""
自实现分子词BPE；
分子词若以</w>结尾表示该子词只能是结尾，如：st可以是st ar，但st</w>只能是we st</w>
"""
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


# @cache是一项优化技术，把函数的结果保存起来，避免传入相同的参数时重复计算；
# @lru_cache是最近最少使用置换算法，默认最多保存maxsize=128个结果
@lru_cache()
def default_bpe():
    """返回“bpe_simple_vocab_16e6.txt”的绝对路径，中共有262,144个分子词"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings
    TODO others：每个Unicode字符，UTF-8用1~4个字节编码（1Bit=8bit）
     为避免<unk>、减小字典大小，将未见过的UTF-8字符拆碎成字节，{字节值: Unicode字符}长度256，里面是[a A 1 , 乱码]等基本字符。
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.（避免映射到bpe代码所依赖的空白/控制字符）
    """
    # ord()返回一个符号的Unicode值；chr()返回一个Unicode值的符号
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))  # 188个
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    输入('w','o','r','d</w>')，输出set(('w','o'),('a','w'),('o','r'),('r','d</w>'))，无序。
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """格式化字符串：ftfy.fix_text -> html.unescape -> strip"""
    text = ftfy.fix_text(text)  # 把损坏的Unicode还原成正确的字符
    text = html.unescape(html.unescape(text))  # 把一些HTML里的编码替换为Unicode字符
    return text.strip()


def whitespace_clean(text):
    """格式化字符串：去除空白字符 -> strip"""
    text = re.sub(r'\s+', ' ', text)  # \s是空白字符，如\t\n\r\f\v等
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        """
        API：encode：一句话->[单词]->[分子词（以字节划分）]->[index(分子词)]
        API：decoder：[index(分子词)]->[分子词]->[单字节值]->[UTF-8字符]
        """
        self.byte_encoder = bytes_to_unicode()  # {字节值: Unicode字符}长度256
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # {Unicode字符: 字节值}长度256
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]  # [(分子词前段，分子词后段)]长度48894
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]  # vocab=[256个Unicode字符，256个Unicode字符</w>]，</w>表示分子词位于单词末尾
        for merge in merges:
            vocab.append(''.join(merge))  # vocab+=[48894个分子词]
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # vocab+=[<sos>, <eos>]，共49408个分子词

        self.encoder = dict(zip(vocab, range(len(vocab))))  # encoder就是text2index
        self.decoder = {v: k for k, v in self.encoder.items()}  # decoder就是index2text

        # bpe_ranks：dict((元素a 元素b): 使用频率排名)，越常用的排名越小
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # 以空格、标点、数字、<|startoftext|>等为标准分割出单词
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                              re.IGNORECASE)  # 忽略大小写

    def bpe(self, token: str):
        """BPE算法，将未见过的词拆成分子词。例如:输入aword，输出a word"""
        if token in self.cache:  # 如果dict（self.cache）中有token的key，那就返回对应的值(有缓存结果)
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + '</w>',)  # 把word拆成('w','o','r','d</w>')
        pairs = get_pairs(word)  # 生成set(('w','o'),('a','w'),('o','r'),('r','d</w>'))，无序
        if not pairs:  # 词很短拆不了，直接返回成token
            return token + '</w>'

        # 将token拆分成数量尽可能少的分子词（贪心算法）
        while True:
            # 将pairs按照bpe_ranks中的顺序排序，找到最常用的那个pair（在文件中排在最前面的）
            # 找不到的pair返回无穷大float('inf')以免被选上
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 组合不在bpe词表中，pairs不能再合并了，循环结束
            if bigram not in self.bpe_ranks:
                break

            # 每次循环将word中的所有bigram组合
            first, second = bigram  # pairs里在文件中排在最前面的元素对
            new_word = []
            i = 0
            while i < len(word):  # word=('w','o','r','d</w>')
                try:
                    j = word.index(first, i)  # word[i:]中查找first
                    new_word.extend(word[i:j])  # new_word添加 ^xxx 或 (first)xxx
                    i = j  # i：下一轮循环从下一个first在word中位置开始
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:  # new_word添加 (first)(second)
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            # 整个pairs合成一个单词了，循环结束
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str):
        """一句话->[单词]->[分子词（以字节划分）]->[index(分子词)]，单词标志</w>"""
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()  # 不区分大小写（全部小写）
        for token in re.findall(self.pat, text):  # 划分出所有单词
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 将UTF-8字符拆成单字节
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """[index(分子词)]->[分子词]->[单字节值]->[UTF-8字符]，单词标志</w>"""
        print(tokens)
        text = ''.join([self.decoder[token] for token in tokens])  # index(分子词)序列->分子词序列
        text = bytearray([self.byte_decoder[c] for c in text]  # 分子词序列->单字节值序列
                         ).decode('utf-8', errors="replace").replace('</w>', ' ')  # 单字节值序列->UTF-8字符序列
        return text

    # 以下2个是CLIP4Clip中添加的，就是把encode拆开成了2部分
    def tokenize(self, text):
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder[bpe_token] for bpe_token in tokens]
