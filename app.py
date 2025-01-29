from flask import Flask, render_template, request, jsonify
import spacy
from transformers import pipeline
from collections import Counter

from heapq import nlargest

# Load SpaCy and summarization model
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization", model='t5-base', tokenizer='t5-base', framework='pt')
sentiment_analyzer = pipeline("sentiment-analysis")   #sentiiment analysis ek load krl thynne

# Create Flask app
app = Flask(__name__)


# Load the saved model and tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




model = AutoModelForSeq2SeqLM.from_pretrained('model_token_folder')#summarizing wlt gnn model ek
tokenizer = AutoTokenizer.from_pretrained('model_token_folder')

def summarize_text(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1200, truncation=True,padding="max_length")

    # Determine input length and adjust max_length dynamically
    input_length = len(input_text.split())  # Count words in the input
    max_summary_length = min(350, int(input_length * 0.7))  # Set a maximum summary length (70% of input length)
    min_summary_length = min(100, int(input_length * 0.3))  # Set a minimum summary length (30% of input length)

    # Generate the summary
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=max_summary_length, 
        min_length=min_summary_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
import yake
def extract_keywords(text):
   
    max_keywords= 5
    score_threshold=0.001
    # Initialize the keyword extractor
    kw_extractor = yake.KeywordExtractor()

    # Extract keywords from the text
    keywords = kw_extractor.extract_keywords(text)

    # Filter and limit the keywords
    filtered_keywords = [(kw, score) for kw, score in keywords if score > score_threshold]
    top_keywords = sorted(filtered_keywords, key=lambda x: x[1])[:max_keywords]

    return top_keywords

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text from the form
    text = request.form['text']

    # First summarization method (SpaCy)
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text != '\n']
    word_freq = Counter(tokens)
    max_freq = max(word_freq.values())
    
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_token = [sent.text for sent in doc.sents]
    sent_scores = {}
    
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq:
                if sent not in sent_scores:
                    sent_scores[sent] = word_freq[word.lower()]
                else:
                    sent_scores[sent] += word_freq[word.lower()]

    num_sentence = 3
    spacy_summary = " ".join(nlargest(num_sentence, sent_scores, key=sent_scores.get))

    # Second summarization method (T5 Transformer)
    t5_summary = summarize_text(text)

    extracted_keywords = extract_keywords(text)

    topics = get_topic(text)

    # Sentiment Analysis
    def truncate_text(text, max_tokens=512):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

    # Truncate the text before passing it to sentiment analysis
    truncated_text = truncate_text(text)
    sentiment = sentiment_analyzer(truncated_text)[0]

    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)

    # Return the summaries and sentiment analysis
    return jsonify({
        'spacy_summary': summary,
        't5_summary': t5_summary,
        'sentiment': sentiment,
        'extracted_keywords':extracted_keywords,
        'topics':topics
    })




import spacy
from nltk.corpus import stopwords
import gensim #topic modeling walata use karanne.artical ekata adalawa topic ek hoyganna
from gensim import corpora
from flask import Flask, request, jsonify

def get_topic(text):

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Ensure you have the necessary NLTK data
    #nltk.download('stopwords')

    # Sample documents for topic modeling
    documents = [text]# Preprocess the documents
    stop_words = set(stopwords.words('english'))
    dic_of_topics = dict()
    preprocessed_docs = []
    for doc in documents:
        # Use spaCy for tokenization
        doc_nlp = nlp(doc.lower())  # Tokenize and lower case
        tokens = [token.text for token in doc_nlp if token.is_alpha and token.text not in stop_words]  # Remove punctuation and stopwords
        preprocessed_docs.append(tokens)

    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(preprocessed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    # Perform LDA
    num_topics = 2  # You can choose the number of topics you want
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Display the topics in a user-friendly format
    dic_of_topics = dict()
    print("Topics and Keywords:")
    for idx, topic in lda_model.print_topics(-1):
        # Format topic output
        topic_keywords = topic.split(" + ")
        formatted_keywords = []
        for keyword in topic_keywords:
            weight, word = keyword.split("*")
            formatted_keywords.append(f"{word.strip()} (Weight: {float(weight):.3f})")
        
        dic_of_topics[f"Topic {idx + 1}"] = ", ".join(formatted_keywords)
        sentences = []
        for topic, keywords in dic_of_topics.items():
            sentences.append(f'{topic} {keywords}.<br>')

        formatted_text = ' '.join(sentences)
    return formatted_text





if __name__ == "__main__":
    app.run(debug=True)