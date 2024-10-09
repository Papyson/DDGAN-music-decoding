import os

#Process Data
cmd = 'python process_data.py --bids_root /path/to/bids --stimulus_folder /path/to/stimuli --train_save_folder /path/to/train --val_save_folder /path/to/val'
os.system(cmd)


#Train model
cmd = 'python dual-dualgan-main/train.py'
os.system(cmd)

#Test model
cmd = 'python dual-dualgan-main/test.py'
os.system(cmd)

# Process Images (Output of model)
cmd = 'python process_images.py --source_folder /path/to/source_folder --destination_folder /path/to/destination_folder'
os.system(cmd)

#Original Audio reconstruction
cmd = 'python reconstruct_audio.py --generated_folder /path/to/generated_folder --metadata_file /path/to/metadata.json --output_folder /path/to/output_folder --segment_prefix C'
os.system(cmd)

#GAN generated Audio reconstruction
cmd = 'python reconstruct_audio.py --generated_folder /path/to/generated_folder --metadata_file /path/to/metadata.json --output_folder /path/to/output_folder --segment_prefix C'
os.system(cmd)

#Evaluation
cmd = 'python evaluate_audio.py --original_folder /path/to/original_folder --gan_folder /path/to/gan_folder'
os.system(cmd)
