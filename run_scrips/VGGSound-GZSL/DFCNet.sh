






python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --DFCNet --lr 0.001 --n_batches 100 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.1 --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --reg_loss --momentum 0.1 --bs 64 --num_classes 310 --exp_name attention_vgg_val_main
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --DFCNet --lr 0.001 --retrain_all --n_batches 100 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.1 --save_checkpoints --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --reg_loss --momentum 0.1 --bs 64 --num_classes 310 --exp_name attention_vgg_all_main
python get_evaluation.py --load_path_stage_A runs/attention_vgg_val_main --load_path_stage_B runs/attention_vgg_all_main  --dataset_name VGGSound --DFCNet

