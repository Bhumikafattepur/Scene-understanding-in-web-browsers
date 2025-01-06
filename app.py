from flask import Flask, render_template, request
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from transformers import BlipProcessor, BlipForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
nltk.download('punkt')
import spacy
import requests
from io import BytesIO
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Initialize the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions for images
def generate_image_caption(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_url}: {e}")
        return None

# Function to extract text and images from a URL
def get_text_and_images(url):
    reqt = Request(url, headers={'User-Agent': "Magic Browser"})
    page = urlopen(reqt)
    soup = BeautifulSoup(page, "html.parser")
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))

    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src and src.startswith(('http://', 'https://')):
            caption = generate_image_caption(src)
            if caption:
                images.append({'src': src, 'caption': caption})
    return fetched_text, images

# Function to summarize text
def lex_summary(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 10)  # Summarize the document with 10 sentences
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# Function to calculate reading time
def readingTime(mytext):
    total_words = len([token for token in mytext.split(" ")])
    estimated_time = total_words / 200.0  # Assuming average reading speed of 200 words per minute
    return estimated_time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_url', methods=['GET', 'POST'])
def process_url():
    start = time.time()
    if request.method == 'POST':
        input_url = request.form['input_url']
        raw_text, images = get_text_and_images(input_url)
        final_reading_time = readingTime(raw_text)
        final_summary = lex_summary(raw_text)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
        return render_template('result_url.html', ctext=raw_text, final_summary=final_summary,
                               images=images, final_reading_time=final_reading_time,
                               summary_reading_time=summary_reading_time, final_time=final_time)

if __name__ == '__main__':
    app.run(debug=True)
