from flask import Flask, request, render_template, redirect, url_for, send_file
import os
from scraibe import Scraibe
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def transcribe_audio(file_path, language="english", num_speakers=1):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")
    
    model = Scraibe()
    transcription = model.autotranscribe(file_path, language=language, num_speakers=num_speakers)
    return transcription

def parse_transcription_to_df(transcription_string):
    pattern = re.compile(r"^(SPEAKER_\d+)\s+\((\d{2}:\d{2}:\d{2})\s+;\s+(\d{2}:\d{2}:\d{2})\):\s*(.*)$")
    transcription_string = str(transcription_string)
    lines = transcription_string.strip().split('\n')
    data_rows = []
    for line in lines:
        match = pattern.match(line)
        if match:
            speaker, start_time, end_time, text = match.groups()
            data_rows.append({
                'speaker': speaker,
                'start_time': start_time,
                'end_time': end_time,
                'text': text.strip()
            })
    df = pd.DataFrame(data_rows)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['audioFile']
        num_speakers = int(request.form['numSpeakers'])
        if file and file.filename.endswith('.wav'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the audio file
            transcription = transcribe_audio(file_path, language="english", num_speakers=num_speakers)
            df = parse_transcription_to_df(transcription)

            # Perform sentiment analysis
            analyzer = SentimentIntensityAnalyzer()
            df['sentiment_score'] = df['text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
            speaker_sentiment = df.groupby('speaker')['sentiment_score'].mean().sort_values()

            # Generate chart
            plt.style.use('ggplot')
            colors = ['g' if x > 0 else 'r' for x in speaker_sentiment.values]
            speaker_sentiment.plot(kind='barh', figsize=(10, 7), color=colors)
            plt.title('Overall Sentiment by Speaker')
            plt.xlabel('Average Sentiment Score (Compound)')
            plt.ylabel('Speaker')
            plt.axvline(x=0, color='k', linestyle='--')
            plt.tight_layout()
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment_chart.png')
            plt.savefig(chart_path)
            plt.close()

            # Return results
            return render_template('results.html', sentiment=speaker_sentiment.to_dict(), chart_path=chart_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)