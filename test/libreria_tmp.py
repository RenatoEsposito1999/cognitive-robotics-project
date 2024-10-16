import torch
import cv2
import numpy as np
from moviepy.editor import AudioFileClip
import torchaudio
from torchvision import transforms

def extract_audio_features(audio_path, sample_rate=16000, num_channels=10):
    waveform, sr = torchaudio.load(audio_path)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    waveform = transform(waveform)

    # Convert stereo to mono by averaging channels (using torch.mean)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Adjust number of channels (replicate channels if necessary)
    if waveform.size(0) < num_channels:
        waveform = waveform.repeat(num_channels // waveform.size(0), 1)

    return waveform.unsqueeze(0)  # Add batch dimension

def preprocess_frame(frame, input_size=(224, 224)):
    # Convert BGR (OpenCV) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Ridimensiona l'immagine
    frame = cv2.resize(frame, input_size)
    # Trasformazioni (puoi aggiungere altre se necessarie)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte in tensor [C, H, W]
    ])
    
    return transform(frame)  # Aggiunge dimensione per il batch

def predict_single_video(video_path, model, input_size=(224, 224), device='cpu', frames_per_sample=15):
    model.eval()
    model.to(device)
    
    # Estrarre l'audio dal video
    video_clip = AudioFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.write_audiofile(audio_path, codec='pcm_s16le')
    audio_input = extract_audio_features(audio_path).to(device)

    # Apertura del video
    cap = cv2.VideoCapture(video_path)
    
    predictions = []
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        visual_input = preprocess_frame(frame, input_size).to(device)
        frame_buffer.append(visual_input)

        if len(frame_buffer) == frames_per_sample:
            visual_input_batch = torch.stack(frame_buffer).to(device)
            frame_buffer = []

            with torch.no_grad():
                logits = model(x_visual=visual_input_batch, x_audio=audio_input)
                
                # Stampa i logits per ispezionarli
                print("Logits:", logits)
                
                # Applica softmax e stampa le probabilità
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                print("Probabilità:", probabilities)
                
                predictions.append(probabilities.cpu().numpy())

    if 0 < len(frame_buffer) < frames_per_sample:
        last_frame = frame_buffer[-1]
        while len(frame_buffer) < frames_per_sample:
            frame_buffer.append(last_frame)

        visual_input_batch = torch.stack(frame_buffer).to(device)
        
        with torch.no_grad():
            logits = model(x_visual=visual_input_batch, x_audio=audio_input)
            # Ottieni i logits dal modello

            # Calcola le probabilità (opzionale, se vuoi vedere anche le probabilità)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Trova l'indice della classe con la probabilità più alta
            predicted_class = torch.argmax(logits, dim=-1)

            # Stampa i risultati
            #print(f"Logits: {logits}")
            #print(f"Probabilità: {probabilities}")
            print(f"Classe predetta: {predicted_class.item()}")  # .item() se è un singolo valore

    cap.release()
    
    return np.array(predictions)
