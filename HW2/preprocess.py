from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    
    
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
       
    preprocessed_text = remove_stopwords(text)

    # Begin your code (Part 0)
    def remove_stem(text: str) -> str:
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [PorterStemmer().stem(token) for token in tokens]    
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
    
    def remove_numbers(text: str) -> str:
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token[0] not in string.digits]    
        preprocessed_text = ' '.join(filtered_tokens)
        

        return preprocessed_text
    
    
    def remove_punctuation(text: str) -> str:
        
        reg = re.compile('<[^>]*>')
        text = reg.sub(' ', text).strip()
        reg = re.compile('[^\w\s]')
        preprocessed_text = reg.sub(' ', text).strip()
        
        return preprocessed_text
    
    
    # remove whitespace from text
    def remove_whitespace(text: str) -> str:
        preprocessed_text = ' '.join(text.split())
        return  preprocessed_text
    
    
    
    
 
    preprocessed_text = remove_whitespace(preprocessed_text)
    preprocessed_text = remove_punctuation(preprocessed_text)
    preprocessed_text = remove_numbers(preprocessed_text)
    preprocessed_text = remove_stem(preprocessed_text)
   
    
    # End your code

    return preprocessed_text
# tmp = "   Here is a 111 dog"
# print(tmp)
# print(preprocessing_function(tmp))