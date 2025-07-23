import sys
#import torch.nn as nn
from torch import optim
from torch.utils import data

from src.args import args_main
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.metrics import MeanClassAccuracy
from src.sampler import SamplerFactory
from src.model import DCFNet
from src.loss import AVGZSLLoss, L2Loss, SquaredL2Loss, ClsContrastiveLoss, APN_Loss, CJMELoss
from src.utils_improvements import get_model_params
from src.train import train
from src.utils import fix_seeds, setup_experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    args = args_main()

    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size

    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")

    if args.dataset_name == "AudioSetZSL":
        train_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="seen",
        )
        val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="seen",
        )
        train_val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode="seen",
        )
        val_all_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="all",
        )

    elif args.dataset_name == "VGGSound":
        train_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
        train_val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )
        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

    elif args.dataset_name == "UCF":
        train_dataset = UCFDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
        train_val_dataset = UCFDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )
        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

    elif args.dataset_name == "ActivityNet":
        train_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
        train_val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )
        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

    else:
        raise NotImplementedError()


    contrastive_train_dataset = ContrastiveDataset(train_dataset)
    contrastive_val_dataset = ContrastiveDataset(val_dataset)
    contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    train_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    train_val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    train_loader = data.DataLoader(
        dataset=contrastive_train_dataset,
        batch_sampler=train_sampler,
        num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        batch_sampler=val_sampler,
        num_workers=8
    )

    train_val_loader = data.DataLoader(
        dataset=contrastive_train_val_dataset,
        batch_sampler=train_val_sampler,
        num_workers=8
    )

    val_all_loader = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        num_workers=8
    )


    if args.DFCNet == True:
        model_params = get_model_params(args.lr, args.first_additional_triplet, args.second_additional_triplet, \
                                        args.reg_loss, args.additional_triplets_loss, args.embedding_dropout, \
                                        args.decoder_dropout, args.additional_dropout, args.embeddings_hidden_size, \
                                        args.decoder_hidden_size, args.depth_transformer, args.momentum,args.num_classes)


    if args.DFCNet == True:
        model = DCFNet(model_params, input_size_audio=args.input_size_audio, input_size_video=args.input_size_video)



    model.to(args.device)
    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()



    if args.DFCNet == True:
        criterion = None


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True) if args.lr_scheduler else None


    if args.DFCNet == True:
        metrics = [
            MeanClassAccuracy(model=model, dataset=val_all_dataset, device=args.device, distance_fn=distance_fn,
                              model_devise=args.ale or args.sje or args.devise,
                              new_model_attention=args.DFCNet,
                              apn=args.apn,
                              args=args)
        ]

    logger.info(model)
    logger.info(criterion)
    logger.info(optimizer)
    logger.info(lr_scheduler)
    logger.info([metric.__class__.__name__ for metric in metrics])


    if args.val_all_loss:
        v_loader = val_all_loader
    elif args.retrain_all:
        v_loader = train_val_loader
    else:
        v_loader = val_loader



    if args.DFCNet == True:
        best_loss, best_score = train(
        train_loader=train_val_loader if args.retrain_all else train_loader,
        val_loader=v_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        new_model_attention=args.DFCNet,
        val_stats=val_stats,
        log_dir=log_dir,
        model_devise=args.ale or args.sje or args.devise,
        apn=args.apn,
        cjme=args.cjme,
        args=args
    )
    # 记录完成信息
    logger.info(f"FINISHED. Run is stored at {log_dir}")


if __name__ == '__main__':
    # 运行主函数
    main()
