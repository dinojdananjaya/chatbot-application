import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = stem_tokens(lemmatized_tokens)
    return ' '.join(stemmed_tokens)

# Function to encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return label_encoder, y
