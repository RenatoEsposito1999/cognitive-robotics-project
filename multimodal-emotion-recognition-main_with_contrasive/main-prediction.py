import input_preprocessing as preprocessing
import torch.nn.functional as F
import torch
from opts import parse_opts
from model import generate_model
if __name__ == '__main__':
    #emotion_dict=['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion_dict=['Neutral','Happy','Angry','Sad']
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []

    model, _ = generate_model(opt)
    model.eval()
    model.to(opt.device)

    #PROBLEMA: SU MAC vuole cuda e il modello non funziona senza cuda questa cosa va sistemata forse nel trainig. 
    best_state = torch.load('c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/results/RAVDESS_multimodalcnn_15_best0.pth')
    model.load_state_dict(best_state['state_dict'])
    
    video_audio_path="c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/raw_data/angry_wrong_1.mp4"
    eeg_path="c:/Users/Vince/Desktop/COGNITIVE_ROBOTICS/cognitive-robotics-project/multimodal-emotion-recognition-main/EEGTest.npz"
    audio_var, video_var = preprocessing.preprocessing_sync_source(video_audio_path,video_norm_value=opt.video_norm_value, batch_size=1)
    eeg_var, eeg_label = preprocessing.preprocessing_async_source(eeg_path, batch_size=1)
    with torch.no_grad():
        _ ,_ ,_ ,_ ,_ , output = model(x_audio=audio_var, x_visual=video_var, x_eeg=eeg_var)
    print("[LOGITS] Output: ", output)
    print("Label eeg: ", eeg_label)
    
    
