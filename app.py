import os
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import datetime
import logging
import wave
import numpy as np
# from flask_cors import CORS
import traceback

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from g2p_en import G2p  # Import g2p_en for grapheme to phoneme conversion

from dataclasses import dataclass
import nltk

nltk.download('averaged_perceptron_tagger')



ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'defaultpassword')

# Initialize the Flask application
app = Flask(__name__)
# CORS(app)

# Configure logging to print errors to the console
logging.basicConfig(level=logging.DEBUG)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

# Load Wav2Vec2 ASR BASE 960H for forced alignment
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
alignment_model = bundle.get_model().to(device)
labels = bundle.get_labels()

# G2P converter for phoneme conversion
g2p = G2p()


# Configuration for upload folder
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'}  # Allow webm in addition to wav and mp3
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Predefined dictionary of transcripts based on the last two letters of filenames
predefined_transcripts = {
    '00': 'WE APOLOGIZE FOR THE INCONVENIENCE; WE WILL DEFINITELY RESOLVE THIS ISSUE AS SOON AS POSSIBLE.',
    '01': 'I’M HAPPY TO ASSIST YOU WITH YOUR RESERVATION AND ANSWER ANY QUESTIONS YOU MIGHT HAVE.',
    '02': 'CAN YOU PLEASE PROVIDE YOUR ADDRESS SO WE CAN ENSURE THE PACKAGE IS SENT TO THE CORRECT LOCATION?',
    '03': 'I UNDERSTAND YOUR FRUSTRATION AND WILL DO MY BEST TO HELP YOU FIND A SOLUTION.',
    '04': 'PLEASE CONFIRM YOUR APPOINTMENT BY CALLING OUR OFFICE AT YOUR EARLIEST CONVENIENCE.',
    '05': 'WE’LL NEED TO VERIFY YOUR IDENTITY BEFORE WE CAN PROCESS YOUR REQUEST.',
    '06': 'I’M SORRY FOR THE MISCOMMUNICATION; LET’S CLARIFY THE DETAILS TO AVOID ANY FURTHER ISSUES.',
    '07': 'YOUR FEEDBACK IS VERY VALUABLE TO US AND WILL HELP IMPROVE OUR SERVICES.',
    '08': 'CAN YOU REPEAT THE DETAILS OF YOUR ORDER SO WE CAN ENSURE ACCURACY?',
    '09': 'I’M HERE TO SUPPORT YOU THROUGH THIS PROCESS AND ADDRESS ANY CONCERNS YOU MAY HAVE.',
    '10': 'YOUR SATISFACTION IS OUR TOP PRIORITY; PLEASE LET US KNOW IF THERE’S ANYTHING ELSE WE CAN DO.',
    '11': 'WE’LL SEND YOU A CONFIRMATION EMAIL ONCE YOUR ORDER HAS BEEN PROCESSED.',
    '12': 'COULD YOU PLEASE CLARIFY THE ISSUE SO WE CAN ASSIST YOU MORE EFFECTIVELY?',
    '13': 'I’M HERE TO RESOLVE THE PROBLEM AND ENSURE YOUR EXPERIENCE IS SATISFACTORY.',
    '14': 'PLEASE DESCRIBE THE PROBLEM YOU’RE FACING SO WE CAN BETTER UNDERSTAND AND ASSIST YOU.',
    '15': 'WE APPRECIATE YOUR PATIENCE WHILE WE WORK ON RESOLVING THIS MATTER.',
    '16': 'OUR REPRESENTATIVE WILL FOLLOW UP WITH YOU SHORTLY TO DISCUSS THE NEXT STEPS.',
    '17': 'I’M SORRY FOR THE DELAY; WE ARE WORKING DILIGENTLY TO RECTIFY THE SITUATION.',
    '18': 'PLEASE PROVIDE US WITH THE NECESSARY INFORMATION TO EXPEDITE THE PROCESS.',
    '19': 'WE’LL SCHEDULE A FOLLOW-UP CALL TO ENSURE THAT ALL YOUR CONCERNS ARE ADDRESSED.',
    '20': 'I’LL TRANSFER YOUR CALL TO A SPECIALIST WHO CAN BETTER ASSIST WITH YOUR ISSUE.',
    '21': 'WE NEED TO DOCUMENT THIS INCIDENT TO PREVENT SIMILAR PROBLEMS IN THE FUTURE.',
    '22': 'OUR TEAM IS COMMITTED TO PROVIDING HIGH-QUALITY SERVICE AND SUPPORT.',
    '23': 'PLEASE ACCEPT OUR APOLOGIES FOR THE INCONVENIENCE CAUSED BY THIS OVERSIGHT.',
    '24': 'IF YOU HAVE ANY ADDITIONAL QUESTIONS OR NEED FURTHER ASSISTANCE, FEEL FREE TO ASK.',
    '25': 'THE ISSUE HAS BEEN LOGGED, AND THE JUDGE WILL MAKE A FINAL DECISION SOON.',
    '26': 'OUR TEAM WILL CHEER WHEN WE RESOLVE YOUR CASE SUCCESSFULLY.',
    '27': 'COULD YOU PLEASE CONFIRM IF YOUR PAYMENT WENT THROUGH?',
    '28': 'I’LL ENJOY ASSISTING YOU WITH THAT ORDER TODAY.',
    '29': 'PLEASE GIVE US A FEW MINUTES TO PROCESS THE PAYMENT.',
    '30': 'YOU SHOULD RECEIVE AN EMAIL WITHIN EIGHT BUSINESS HOURS.',
    '31': 'THE EARLY MORNING SHIFT WILL TAKE CARE OF YOUR REQUEST.',
    '32': 'WE APPRECIATE YOUR PATIENCE; THE SYSTEM IS UPDATING NOW.',
    '33': 'LET ME THROW IN A QUICK UPDATE BEFORE WE PROCEED.',
    '34': 'YOU’LL ENJOY THE SERVICE, AND WE’LL ENSURE EVERYTHING IS PERFECT.',
    '35': 'PLEASE VERIFY THE DETAILS, AND I’LL TAKE CARE OF THE REST.',
    '36': 'YOUR REFUND REQUEST HAS BEEN APPROVED AND WILL BE PROCESSED SHORTLY.',
    '37': 'THANK YOU FOR YOUR PATIENCE; WE’LL RESOLVE THIS ISSUE AS SOON AS POSSIBLE.',
    '38': 'I’M HAPPY TO ASSIST YOU WITH ANY QUESTIONS REGARDING YOUR ORDER.',
    '39': 'COULD YOU PLEASE VERIFY YOUR IDENTITY TO PROCEED WITH THE PAYMENT?',
    '40': 'OUR TEAM IS COMMITTED TO PROVIDING THE HIGHEST QUALITY SUPPORT AVAILABLE.',
    '41': 'I’LL NEED TO CHECK WITH THE DEPARTMENT; PLEASE HOLD FOR A MOMENT.',
    '42': 'YOUR REQUEST HAS BEEN RECEIVED AND WILL BE PROCESSED SHORTLY.',
    '43': 'CAN YOU CONFIRM IF YOUR PAYMENT HAS GONE THROUGH SUCCESSFULLY?',
    '44': 'LET ME TRANSFER YOU TO A SPECIALIST WHO CAN ASSIST WITH THIS INQUIRY.',
    '45': 'WE APPRECIATE YOUR FEEDBACK AND WILL USE IT TO IMPROVE OUR SERVICES.',
    '46': 'PLEASE PROVIDE THE NECESSARY DETAILS TO EXPEDITE THE RESOLUTION PROCESS.',
    '47': 'I’M SORRY FOR THE DELAY; WE’RE WORKING TO RESOLVE THE ISSUE IMMEDIATELY.',
    '48': 'CAN YOU PLEASE DESCRIBE THE ISSUE SO WE CAN ASSIST YOU MORE EFFECTIVELY?',
    '49': 'YOU SHOULD RECEIVE A CONFIRMATION EMAIL ONCE THE REQUEST IS PROCESSED.',
    '50': 'THANK YOU FOR CALLING; WE APPRECIATE YOUR PATIENCE DURING THIS PROCESS.',
    '51': 'OUR TEAM WILL FOLLOW UP WITH YOU SHORTLY TO ENSURE EVERYTHING IS RESOLVED.',
    '52': 'CAN YOU REPEAT THE DETAILS OF YOUR INQUIRY SO I CAN ASSIST FURTHER?',
    '53': 'WE’VE RECEIVED YOUR FEEDBACK AND WILL FORWARD IT TO THE APPROPRIATE TEAM.',
    '54': 'WE’LL NEED TO VERIFY SOME ADDITIONAL INFORMATION BEFORE PROCESSING YOUR REQUEST.',
    '55': 'I’M TRANSFERRING YOU TO A HIGHER-LEVEL REPRESENTATIVE WHO CAN HELP WITH THIS MATTER.',
    '56': 'PLEASE DON’T HESITATE TO REACH OUT IF YOU HAVE ANY FURTHER QUESTIONS.',
    '57': 'WE SINCERELY APOLOGIZE FOR ANY INCONVENIENCE CAUSED DURING THE TRANSACTION.',
    '58': 'PLEASE HOLD THE LINE WHILE I CHECK THE AVAILABLE OPTIONS FOR YOU.',
    '59': 'YOUR PACKAGE IS CURRENTLY IN TRANSIT AND SHOULD ARRIVE BY THE END OF THE WEEK.',
    '60': 'COULD YOU CONFIRM THE SPELLING OF YOUR NAME TO AVOID ANY FURTHER CONFUSION?',
    '61': 'WE’RE COMMITTED TO PROVIDING TIMELY AND EFFICIENT SOLUTIONS TO YOUR CONCERNS.',
    '62': 'I’LL ESCALATE THIS ISSUE TO THE CONCERNED DEPARTMENT FOR A QUICKER RESOLUTION.',
    '63': 'LET ME DOUBLE-CHECK THE AVAILABILITY BEFORE PROCEEDING WITH YOUR ORDER.',
    '64': 'YOUR ACCOUNT HAS BEEN SUCCESSFULLY UPDATED; PLEASE VERIFY THE CHANGES.',
    '65': 'I’LL SEND YOU THE CONFIRMATION SHORTLY; PLEASE ENSURE YOU CHECK YOUR INBOX.',
    '66': 'WE’VE ARRANGED A CALLBACK AT YOUR REQUESTED TIME; KINDLY CONFIRM YOUR AVAILABILITY.',
    '67': 'I’M SORRY FOR THE INCONVENIENCE CAUSED BY THIS TECHNICAL GLITCH.',
    '68': 'PLEASE VERIFY YOUR BILLING ADDRESS TO COMPLETE THE TRANSACTION.',
    '69': 'YOUR REQUEST HAS BEEN QUEUED FOR PROCESSING; WE’LL NOTIFY YOU ONCE IT’S COMPLETED.',
    '70': 'KINDLY PROVIDE THE TRANSACTION ID FOR US TO TRACK YOUR PAYMENT STATUS.',
    '71': 'I’LL ENSURE THAT YOUR FEEDBACK REACHES THE APPROPRIATE DEPARTMENT.',
    '72': 'COULD YOU PROVIDE THE REFERENCE NUMBER TO ASSIST US IN TRACKING YOUR CASE?',
    '73': 'PLEASE BEAR WITH US WHILE WE RETRIEVE YOUR ORDER DETAILS.',
    '74': 'I’LL ENSURE THAT THE RESOLUTION MEETS YOUR SATISFACTION, AND WE APPRECIATE YOUR PATIENCE.'
}


# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to create user-specific folder if it doesn't exist
def get_user_folder(speaker_id):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(speaker_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

# Helper function to update or create the user's metadata JSON file
def update_metadata(speaker_id, sentence_index, sentence, file_path, user_metadata):
    try:
        user_folder = get_user_folder(speaker_id)
        metadata_file = os.path.join(user_folder, f"{speaker_id}_metadata.json")

        logging.debug(f"Updating metadata for speaker {speaker_id} at {metadata_file}")

        # Initialize metadata if it doesn't exist
        if not os.path.exists(metadata_file):
            metadata = {
                "speaker_id": speaker_id,
                "state": user_metadata.get("state"),
                "profession": user_metadata.get("profession"),
                "age": user_metadata.get("age"),
                "gender": user_metadata.get("gender"),
                "proficiency": user_metadata.get("proficiency"),
                "test_taken": user_metadata.get("test_taken"),
                "test_name": user_metadata.get("test_name", ""),
                "test_score": user_metadata.get("test_score", ""),
                "recordings": []
            }
        else:
            # Load existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Append new sentence metadata
        metadata["recordings"].append({
            "sentence_index": sentence_index,
            "sentence": sentence,
            "file_path": file_path,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Save the updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.debug(f"Metadata successfully updated for speaker {speaker_id}")
    except Exception as e:
        logging.error(f"Error updating metadata for {speaker_id}: {e}")

# Route to render the login page
@app.route('/')
def login():
    return render_template('login.html')

# Route to handle login and redirect based on user type
@app.route('/login', methods=['POST'])
def handle_login():
    user_type = request.form.get('user_type')
    password = request.form.get('password')

    # Updated password for admin
    ADMIN_PASSWORD = "Admin@247"

    if user_type == 'admin':
        if password == ADMIN_PASSWORD:
            return redirect(url_for('admin'))
        else:
            return jsonify({'error': 'Invalid admin password'}), 403  # Forbidden
    elif user_type == 'normal':
        return redirect(url_for('index'))
    else:
        return jsonify({'error': 'Invalid user type'}), 400

# Route to render index.html for normal users
@app.route('/index')
def index():
    return render_template('index.html')

# Helper function to get all folders and files in the upload directory
def get_folders_and_files():
    folders = []
    for foldername in os.listdir(UPLOAD_FOLDER):
        folder_path = os.path.join(UPLOAD_FOLDER, foldername)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            folders.append({
                'folder_name': foldername,
                'files': files
            })
    return folders

# Route to render the admin.html page
@app.route('/admin')
def admin():
    folders_and_files = get_folders_and_files()
    return render_template('admin.html', folders_and_files=folders_and_files)

# Helper dataclass for storing segment info
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

@dataclass
class WordSegment:
    word: str
    start: int
    end: int
    score: float

# Upload folder configuration
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'webm'}

def transcribe_and_align(file_path, updated_sentence=None):
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample the audio if necessary (Wav2Vec2 model requires 16 kHz)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Step 1: If an updated sentence is provided, use it; otherwise, transcribe the audio
        if updated_sentence:
            transcript = updated_sentence
            logging.info(f"Using given sentence for alignment: {transcript}")
        else:
            # Transcribe the audio using Wav2Vec2
            input_values = processor(waveform[0], return_tensors="pt", sampling_rate=16000).input_values.to(device)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.decode(predicted_ids[0])
            logging.info(f"Transcribed sentence: {transcript}")

        # Step 2: Convert the transcript to phonemes using G2P (Grapheme-to-Phoneme conversion)
        word_list = transcript.split()  # Split transcript into words
        phoneme_list = [g2p(word) for word in word_list]  # Convert words to phonemes
        phoneme_list_flat = [phoneme for sublist in phoneme_list for phoneme in sublist]  # Flatten the phoneme list

        logging.debug(f"Phoneme list (flattened): {phoneme_list_flat}")

        # Step 3: Generate tokens from phonemes for alignment
        # (Using the Wav2Vec2 model's tokenizer or another phoneme-to-token mapping)
        labels = processor.tokenizer.get_vocab()
        phoneme_dict = {p: i for i, p in enumerate(labels)}  # Map phonemes to token ids
        tokens = [phoneme_dict.get(p, phoneme_dict['|']) for p in phoneme_list_flat]  # Convert phonemes to token ids

        # Step 4: Create the trellis matrix for forced alignment
        def get_trellis(emission, tokens, blank_id=0):
            num_frames = emission.size(0)
            num_tokens = len(tokens)
            trellis = torch.zeros((num_frames, num_tokens))
            trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
            trellis[0, 1:] = -float("inf")

            for t in range(num_frames - 1):
                trellis[t + 1, 1:] = torch.maximum(
                    trellis[t, 1:] + emission[t, blank_id],
                    trellis[t, :-1] + emission[t, tokens[1:]]
                )
            return trellis

        # Perform emission generation
        input_values = processor(waveform[0], return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        emissions = torch.log_softmax(logits, dim=-1).cpu().detach()
        trellis = get_trellis(emissions[0], tokens)

        # Step 5: Backtrack through the trellis to find the most likely alignment path
        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emissions, tokens, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1
            path = [Point(j, t, emissions[t, blank_id].exp().item())]

            while j > 0:
                assert t > 0
                p_stay = emissions[t - 1, blank_id]
                p_change = emissions[t - 1, tokens[j]]

                stayed = trellis[t - 1, j] + p_stay
                changed = trellis[t - 1, j - 1] + p_change

                t -= 1
                if changed > stayed:
                    j -= 1

                prob = (p_change if changed > stayed else p_stay).exp().item()
                path.append(Point(j, t, prob))

            while t > 0:
                prob = emissions[t - 1, blank_id].exp().item()
                path.append(Point(j, t - 1, prob))
                t -= 1

            return path[::-1]

        # Step 6: Get the alignment path from the trellis
        path = backtrack(trellis, emissions[0], tokens)

        # Step 7: Merge repeated phonemes and calculate scores
        def merge_repeats(path):
            segments = []
            i1, i2 = 0, 0
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        phoneme_list_flat[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score
                    )
                )
                i1 = i2
            return segments

        phoneme_segments = merge_repeats(path)

        # Step 8: Align phonemes with words
        def merge_words(phoneme_segments, phoneme_list, word_list):
            words = []
            i1 = 0  # Phoneme segment index
            for word_idx, word_phonemes in enumerate(phoneme_list):
                word_start = phoneme_segments[i1].start
                word_end = phoneme_segments[i1].end
                word_score_sum = 0

                phoneme_count = len(word_phonemes)

                for j in range(phoneme_count):
                    if i1 < len(phoneme_segments):
                        word_end = phoneme_segments[i1].end
                        word_score_sum += phoneme_segments[i1].score
                        i1 += 1

                word_score_avg = word_score_sum / phoneme_count
                words.append(WordSegment(word_list[word_idx], word_start, word_end, word_score_avg))

            return words

        word_segments = merge_words(phoneme_segments, phoneme_list, word_list)

        # Step 9: Prepare results with start and end times
        results = {'transcript': transcript, 'word_segments': [], 'segments': []}
        ratio = waveform.size(1) / trellis.size(0)  # Ratio for mapping frames to time

        # Append word segments with calculated times
        for i in range(len(word_segments)):
            word = word_segments[i]
            x0 = int(ratio * word.start)
            if i < len(word_segments) - 1:
                x1 = int(ratio * word.end)
            else:
                x1 = int(ratio * word.end)

            start_time = x0 / sample_rate
            end_time = x1 / sample_rate
            word_data = {
                'word': word.word,
                'start_time': f'{start_time:.3f}',
                'end_time': f'{end_time:.3f}',
                'score': f'{word.score:.2f}'
            }
            results['word_segments'].append(word_data)

        # Append phoneme segments with calculated times
        for segment in phoneme_segments:
            x0 = int(ratio * segment.start)
            x1 = int(ratio * segment.end)
            start_time = x0 / sample_rate
            end_time = x1 / sample_rate
            segment_data = {
                'label': segment.label,
                'start_time': f'{start_time:.3f}',
                'end_time': f'{end_time:.3f}',
                'score': f'{segment.score:.2f}'
            }
            results['segments'].append(segment_data)

        return results

    except Exception as e:
        logging.error(f"Error in transcription and alignment: {e}")
        return None

@app.route('/align_with_text', methods=['POST'])
def align_with_text():
    try:
        if 'audio' not in request.files or 'sentence' not in request.form:
            return jsonify({'error': 'Audio file and sentence text are required'}), 400

        audio_file = request.files['audio']
        updated_sentence = request.form['sentence']  # Get the updated sentence

        file_path = os.path.join("temp_audio", audio_file.filename)
        audio_file.save(file_path)

        # Call the transcribe_and_align function with the updated sentence
        results = transcribe_and_align(file_path, updated_sentence)
        os.remove(file_path)  # Clean up after processing

        if results is None:
            logging.error("Alignment result is None")
            return jsonify({'error': 'Error in processing audio.'}), 500

        return jsonify(results)

    except Exception as e:
        logging.error(f"Exception during alignment: {e}")
        logging.error(traceback.format_exc())  # Log the full error traceback
        return jsonify({'error': str(e)}), 500


@app.route('/align', methods=['POST'])
def align():
    try:
        # Check if the necessary files and form data are provided
        if 'audio' not in request.files or 'last_two_letters' not in request.form:
            return jsonify({'error': 'No audio file or last two letters provided'}), 400

        audio_file = request.files['audio']
        last_two_letters = request.form.get('last_two_letters', '').strip()

        # Define the path to save the audio file temporarily
        file_path = os.path.join("temp_audio", secure_filename(audio_file.filename))
        audio_file.save(file_path)

        # Check if the last two letters are in the predefined dictionary
        if last_two_letters in predefined_transcripts:
            transcript = predefined_transcripts[last_two_letters]
            logging.info(f"Using predefined transcript for {last_two_letters}: {transcript}")
        else:
            # If no predefined transcript is found, use ASR to transcribe the audio
            logging.info(f"No predefined transcript found for {last_two_letters}, using ASR")
            asr_result = transcribe_and_align(file_path)
            if asr_result and 'transcript' in asr_result:
                transcript = asr_result['transcript']
            else:
                logging.error("ASR transcription failed.")
                os.remove(file_path)  # Clean up the temporary file
                return jsonify({'error': 'ASR transcription failed.'}), 500

        # Perform alignment using the transcript (whether predefined or from ASR)
        logging.info(f"Starting alignment with transcript: {transcript}")
        results = transcribe_and_align(file_path, transcript)  # Align with the transcript

        # Clean up the temporary file after transcription
        os.remove(file_path)

        # Ensure the results contain the alignment data
        if results is None:
            logging.error("Error: Alignment results are None")
            return jsonify({'error': 'Error in processing alignment.'}), 500

        return jsonify(results)

    except Exception as e:
        logging.error(f"Exception in /align: {e}")
        return jsonify({'error': 'An error occurred while processing the audio file.'}), 500


# Route to check if the user ID exists (called during metadata form submission)
@app.route('/check_user_id', methods=['GET'])
def check_user_id():
    speaker_id = request.args.get('speaker_id')
    if not speaker_id:
        return jsonify({'error': 'No speaker ID provided'}), 400

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(speaker_id))
    if os.path.exists(user_folder):
        return jsonify({'error': 'User ID already exists'}), 409  # Conflict
    else:
        return jsonify({'message': 'User ID is available'}), 200

# Route to handle the submission of audio files and update metadata
@app.route('/submit_audio', methods=['POST'])
def submit_audio():
    try:
        logging.debug(f"Form Data: {request.form}")
        logging.debug(f"Files: {request.files}")

        required_fields = ['speaker_id', 'sentence_index', 'sentence', 'state', 'profession', 'gender', 'proficiency', 'test_taken']
        missing_fields = [field for field in required_fields if field not in request.form]

        if missing_fields:
            logging.error(f"Missing fields: {', '.join(missing_fields)}")
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        speaker_id = request.form['speaker_id']
        sentence_index = request.form['sentence_index']
        sentence = request.form['sentence']
        user_metadata = {
            "state": request.form['state'],
            "profession": request.form['profession'],
            "age": request.form['age'],
            "gender": request.form['gender'],
            "proficiency": request.form['proficiency'],
            "test_taken": request.form['test_taken'],
            "test_name": request.form.get('test_name', ''),
            "test_score": request.form.get('test_score', '')
        }

        # Handle the uploaded audio file
        audio_file = request.files.get('audio')
        if audio_file and allowed_file(audio_file.filename):
            user_folder = get_user_folder(speaker_id)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_extension = audio_file.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(f"sentence_{sentence_index}_{timestamp}.{file_extension}")
            file_path = os.path.join(user_folder, filename)

            # Save the audio file
            audio_file.save(file_path)
            logging.debug(f"Audio file saved at {file_path} for speaker {speaker_id}.")

            # Update the user's metadata with the new recording and other metadata
            update_metadata(speaker_id, sentence_index, sentence, file_path, user_metadata)

            return jsonify({'message': 'Recording saved successfully', 'filename': filename}), 200
        else:
            logging.error("Invalid file format. Only .wav, .mp3, and .webm are allowed.")
            return jsonify({'error': 'Invalid file format. Only .wav, .mp3, and .webm are allowed.'}), 400
    except Exception as e:
        logging.error(f"Error processing audio upload: {e}")
        return jsonify({'error': 'An error occurred while processing the request.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({'status': 'healthy', 'message': 'Service is running'}), 200
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    try:
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No zip file provided'}), 400

        zip_file = request.files['zip_file']

        if not zip_file.filename.endswith('.zip'):
            return jsonify({'error': 'The uploaded file is not a zip file'}), 400

        filename = secure_filename(zip_file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        zip_file.save(zip_path)

        return jsonify({'message': 'Zip file uploaded and saved successfully', 'file_path': zip_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    temp_audio_folder = os.path.normpath('temp_audio')
    if not os.path.exists(temp_audio_folder):
        os.makedirs(temp_audio_folder)

    app.run(host='0.0.0.0', debug=True, port=8898, use_reloader=False)