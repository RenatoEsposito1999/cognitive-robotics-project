import torch
from Data_preprocessing import input_preprocessing_predict
from datasets.synchronized_data import Synchronized_data

label_list = ["Neutral", "Happy", "Angry", "Sad"]

video_audio_path="./raw_data_video/sad_correct.mp4"


def predict(opt, model, test_split):
    model.eval()
    #load best state, there are two file pth for separating if the machine hase cuda or not
    if(opt.device=="cuda"):
        best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+'.pth')
    else:
        best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+'_cpu_.pth', map_location=torch.device("cpu"))
    
    
    #Load the weigths on the model
    model.load_state_dict(best_state['state_dict'])
    
    audio_var, video_var = input_preprocessing_predict.preprocessing_audio_video(video_audio_path,video_norm_value=opt.video_norm_value, batch_size=1)
    
    eeg_test = Synchronized_data(test_split)
    eeg_var, mask = eeg_test.generate_artificial_batch([2])
    
    
    eeg_var_1, mask = eeg_test.generate_artificial_batch([1])
   
    #eeg_var = torch.load("1.pth")
    
    
    
    with torch.no_grad():
        output_logits = model(x_audio=audio_var, x_visual=video_var, x_eeg=eeg_var, device=opt.device)
        output_logits_1 = model(x_audio=audio_var, x_visual=video_var, x_eeg=eeg_var_1, device=opt.device)
    
    
    print(output_logits==output_logits_1)
    softmax_output = torch.nn.functional.softmax(output_logits, dim=1)
    max_value, max_index = torch.max(softmax_output, dim=1)
    print(max_value, max_index)
    print(f"Max Value: {max_value}, Label: {label_list[max_index.item()]}")
    
    
    