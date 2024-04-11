import os
import streamlit as st
import torch
import neural_net
import inference
from collections import defaultdict, Counter

N_TAKEN_AUDIO = 5
K_NEAREST_NEIGHBOURS = 3


# Load pre-trained encoder
encoder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\AI Module\Speaker_Recognition\LSTM\saved_model\nhi_model\models_transformer_mfcc_200k_specaug_batch_8.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)

# Function to recognize speaker from audio file
def recognize_speaker(audio_file_path):
    # Load audio file and get its embedding
    audio_file_embedding = inference.get_embedding(audio_file_path, encoder)

    # Calculate distance between audio embedding and base embeddings
    embedding_vector_distance = [(vector, inference.compute_distance(vector, audio_file_embedding)) for vector in embedding_vectors_data]

    # Sort embedding vectors by distance
    sorted_embedding_vector_distance = sorted(embedding_vector_distance, key=lambda pair: pair[1])

    # Predict speaker using KNN
    speaker_predictions = [speaker_embedding_vector[tuple(vector)] for vector, distance in sorted_embedding_vector_distance[:K_NEAREST_NEIGHBOURS]]

    # Get the most common predicted speaker
    prediction = Counter(speaker_predictions).most_common(1)[0][0]

    return prediction

# Main function
def main():
    st.title("Speaker Recognition App")

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform speaker recognition on the uploaded file
        prediction = recognize_speaker(os.path.join("temp", uploaded_file.name))

        # Display the predicted speaker
        st.write("Predicted Speaker:", prediction)

# Load base embedding vectors
tri_folder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\AI Module\Speaker_Recognition\LSTM\Data Tiếng nói base\Trí"
phat_folder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\AI Module\Speaker_Recognition\LSTM\Data Tiếng nói base\Phát"
dat_folder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\AI Module\Speaker_Recognition\LSTM\Data Tiếng nói base\Đạt"
tuan_folder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\AI Module\Speaker_Recognition\LSTM\Data Tiếng nói base\Tuấn"

tri_audio_files = [file for file in os.listdir(tri_folder_path)[:N_TAKEN_AUDIO] if file.endswith(".wav")]
phat_audio_files = [file for file in os.listdir(phat_folder_path)[:N_TAKEN_AUDIO] if file.endswith(".wav")]
dat_audio_files = [file for file in os.listdir(dat_folder_path)[:N_TAKEN_AUDIO] if file.endswith(".wav")]
tuan_audio_files = [file for file in os.listdir(tuan_folder_path)[:N_TAKEN_AUDIO] if file.endswith(".wav")]

tri_base_embedding_vectors = [inference.get_embedding(os.path.join(tri_folder_path, audio), encoder) for audio in tri_audio_files]
phat_base_embedding_vectors = [inference.get_embedding(os.path.join(phat_folder_path, audio), encoder) for audio in phat_audio_files]
dat_base_embedding_vectors = [inference.get_embedding(os.path.join(dat_folder_path, audio), encoder) for audio in dat_audio_files]
tuan_base_embedding_vectors = [inference.get_embedding(os.path.join(tuan_folder_path, audio), encoder) for audio in tuan_audio_files]

# Create a dictionary to map embedding vectors to speakers
speaker_embedding_vector = defaultdict(lambda: "")
embedding_vectors_data = []

for vector in tri_base_embedding_vectors:
    speaker_embedding_vector[tuple(vector)] = "Trí"
    embedding_vectors_data.append(vector)
for vector in phat_base_embedding_vectors:
    speaker_embedding_vector[tuple(vector)] = "Phát"
    embedding_vectors_data.append(vector)
for vector in dat_base_embedding_vectors:
    speaker_embedding_vector[tuple(vector)] = "Đạt"
    embedding_vectors_data.append(vector)
for vector in tuan_base_embedding_vectors:
    speaker_embedding_vector[tuple(vector)] = "Tuấn"
    embedding_vectors_data.append(vector)

# Run the main function
if __name__ == "__main__":
    main()
