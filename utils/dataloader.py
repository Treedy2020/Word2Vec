import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103
from utils.cons import (
    CBOW_N,
    SKIP_GRAM_N,
    MIN_WORD_FREQUENCE,
    MAX_LENGTH
)

def getTokenizer(tokenizer_name='basic_english', language='en'):
    """
    getTokenizer for text data
    Doc:
        https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    Args:
        tokenizer_name (str): name of tokenizer
    """
    return get_tokenizer(tokenizer=tokenizer_name, language= language)


def getDataItarator(dataset:str, data_dir:str, ds_type):
    """get map style dataset from Wikitext itarator.

    Args:
        dataset (str): dataset name
        data_dir (str): _description_
        ds_type (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: map style dataset
    """
    if dataset == 'WikiText2':
        data_iterator = WikiText2(root=data_dir, split=(ds_type))
    elif dataset == 'WikiText103':
        data_iterator = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError('Only Support: WikiText2 or WikiText103')
    
    return to_map_style_dataset(data_iterator)

def buildVocab(data_it, tokenizer):
    """Create a torchtext.vocab.Vocab Object
    Doc：
        https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator
    """
    # get vocab
    vocab = build_vocab_from_iterator(
        iterator=map(tokenizer, data_it),
        specials=['<unk>'],
        min_freq=MIN_WORD_FREQUENCE
    )
    # set default index
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def CbowBatch(inputs, token_pipeline):
    """Get batch data of tensor for cbow.

    Args:
        inputs (list[str]): list of text (string).
        token_pipeline (function): function to get Embeddings by tokens.

    Returns:
        batch_input, batch_target: Batch data for input and output.
    """
    batch_input, batch_target = [], []
    
    for text in inputs:
        token_sequence = token_pipeline(text)
        
        if len(token_sequence) < 2*CBOW_N + 1:
            continue
        
        if MAX_LENGTH:
            token_sequence = token_sequence[:MAX_LENGTH]
        
        for ind in range(len(token_sequence) - 2*CBOW_N):
            src_sequence = token_sequence[ind : ind + 2*CBOW_N + 1]
            target = src_sequence.pop(CBOW_N)
            batch_input.append(src_sequence)
            batch_target.append(target)
            
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_target = torch.tensor(batch_target, dtype=torch.long)
    
    return batch_input, batch_target

def skipGramBatch(inputs, token_pipeline):
    """Get batch data of tensor for skip gram model.

    Args:
        inputs (list[str]): list of text (string).
        token_pipeline (function): function to get Embeddings by tokens.

    Returns:
        batch_input, batch_target: Batch data for input and output.
    """
    batch_input, batch_target = [], []
    
    for text in inputs:
        token_sequence = token_pipeline(text)
        
        if len(token_sequence) < 2*SKIP_GRAM_N + 1:
            continue
        
        if MAX_LENGTH:
            token_sequence = token_sequence[:MAX_LENGTH]

        for ind in range(len(token_sequence) - 2*SKIP_GRAM_N):
            src_sequence = token_sequence[ind : ind + 2*SKIP_GRAM_N + 1]
            word = src_sequence.pop(SKIP_GRAM_N)
            
            for src_word in src_sequence:
                batch_input.append(word)
                batch_target.append(src_word)
    
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_target = torch.tensor(batch_target, dtype=torch.long)
    
    return batch_input, batch_target
            
def getDataLoaderAndVocab(
    model_name:str,
    data_set:str,
    data_dir:str,
    ds_type:str,
    batch_size:int,
    shuffle:bool,
    vocab=None,
    ):
    
    # Get collate_fn
    assert model_name in ['SkipGram', 'CBOW'], "Only Support：SkipGram or CBOW"
    
    collate_fn = skipGramBatch if model_name == 'SkipGram' else CbowBatch
    
    # Get Tokenizer
    tokenizer = getTokenizer()
    data_it = getDataItarator(dataset=data_set,
                              data_dir=data_dir,
    
                              ds_type=ds_type)
    # Get Vocab
    if not vocab:
        vocab = buildVocab(data_it=data_it,
                           tokenizer=tokenizer)
        
    # Get Text to Embeddings Pipeline
    token2word_pipeline = lambda x: vocab(tokenizer(x))
    
    data_loader = DataLoader(
        dataset=data_it,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, token_pipeline=token2word_pipeline)
    )
    
    return data_loader, vocab
    

if __name__ == "__main__":
    import random
    inputs = [[random.randint(1,10) for i in range(9)] for j in range(10)]
    pipeline = lambda x: x
    print(skipGramBatch(inputs, pipeline))



    
    
    


