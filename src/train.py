import logging
import os

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 导入必要的模块和工具函数
from .utils import check_best_loss, check_best_score, evaluate_dataset, save_best_model
from sklearn.preprocessing import StandardScaler


import time




def get_feature( epoch, model, device, dataloader):
    model.eval()
    all_audio = []
    all_visual = []
    all_label = []
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            p = data["positive"]
            q = data["negative"]

            audio = p["audio"].to(device)
            video = p["video"].to(device)
            label = target["positive"].to(device)

            a, v = model.get(audio, video)

            all_audio.append(a.data.cpu())
            all_visual.append(v.data.cpu())
            all_label.append(label.data.cpu())

    all_audio = torch.cat(all_audio)
    all_visual = torch.cat(all_visual)
    all_label = torch.cat(all_label)

    return all_audio, all_visual, all_label



def cluster_centers(audio, visual, n_clusters=310):

    scaler_audio = StandardScaler()
    norm_audio = scaler_audio.fit_transform(audio)
    kmeans_audio = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(norm_audio)
    audio_centers = kmeans_audio.cluster_centers_

    # 处理视觉特征
    scaler_visual = StandardScaler()
    norm_visual = scaler_visual.fit_transform(visual)
    kmeans_visual = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(norm_visual)
    visual_centers = kmeans_visual.cluster_centers_

    return audio_centers, visual_centers






def calculate_class(model, dataloader, device,num_class, epoch, a_class=None, v_class=None):
    n_classes = num_class
    momentum = 0.2

    audio, visual, label = get_feature(epoch, model, device, dataloader)

    audio_centers, visual_centers = cluster_centers(
        audio.cpu().numpy(),
        visual.cpu().numpy(),
        n_clusters=n_classes
    )

    audio_centers = torch.from_numpy(audio_centers).float().to(device)
    visual_centers = torch.from_numpy(visual_centers).float().to(device)


    if epoch <= 0:
        audio_class = audio_centers
        visual_class = visual_centers
    else:

        a_class = a_class.to(device)
        v_class = v_class.to(device)
        audio_class = (1 - momentum) * audio_centers + momentum * a_class
        visual_class = (1 - momentum) * visual_centers + momentum * v_class

    return audio_class, visual_class


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
          train_stats, val_stats, log_dir, new_model_attention=False, model_devise=False, apn=False, cjme=False, args=None):


    num_class = args.num_classes
    best_loss = None
    best_score = None

    train_loss = 0.0
    val_loss = 0.0
    val_hm = 0.0


    audio_class, visual_class = calculate_class(model, train_loader, device, num_class, epoch=0)
    for epoch in range(epochs):
        train_loss = train_step(train_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics,
                            train_stats, new_model_attention, model_devise, apn, cjme, audio_class, visual_class,
                            args)
        torch.cuda.empty_cache()
        audio_class, visual_class = calculate_class(model, train_loader, device, num_class, epoch, audio_class,
                                                    visual_class)
        
        torch.cuda.empty_cache()
        val_loss, val_hm, zsl_score = val_step(val_loader, model, criterion, epoch, epochs, writer, device, metrics, val_stats,
                                new_model_attention, model_devise, apn, cjme, audio_class, visual_class, args)
        
        test_inference_speed(val_loader, model, device, args)

        torch.cuda.empty_cache()
        best_loss = check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir)
        best_score = check_best_score(epoch, best_score, val_hm, model, optimizer, log_dir)

        if args.save_checkpoints:
            # save_best_model(epoch, val_loss, model, optimizer, log_dir / "checkpoints", metric="loss", checkpoint=True)
            save_best_model(epoch, val_hm, model, optimizer, log_dir / "checkpoints", metric="score", checkpoint=True)



        if lr_scheduler:
            lr_scheduler.step(val_hm)
        if new_model_attention == True:
            model.optimize_scheduler(val_hm)



    return best_loss, best_score



def train_step(data_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics, stats,
               new_model_attention, model_devise, apn, cjme, audio_class,visual_class,args):

    logger = logging.getLogger()
    model.train()

    print("Start training ... ")
    batch_counter = 0

    # 重置所有评估指标
    for metric in metrics:
        metric.reset()

    batch_loss = 0
    batch_loss_ad = 0
    beta = 0
    lam = 0
    # print(f"当前模型参数量: {sum(p.numel() for p in model.parameters())}")
    for batch_idx, (data, target) in enumerate(data_loader):
        model.train()
        batch_counter += 1
        optimizer.zero_grad()

        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(device)
        x_p_v = p["video"].to(device)
        x_p_t = p["text"].to(device)
        x_p_num = target["positive"].to(device)

        x_q_a = q["audio"].to(device)
        x_q_v = q["video"].to(device)
        x_q_t = q["text"].to(device)


        inputs = (
            x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t
        )

        if args.z_score_inputs:
            inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])


        loss, loss_details= model.optimize_params(*inputs, audio_class,visual_class,epoch, label=x_p_num, optimize=True)


        batch_loss += loss.item()

        p_target = target["positive"].to(device)
        q_target = target["negative"].to(device)

        iteration = len(data_loader) * epoch + batch_idx

    batch_loss /= (batch_idx + 1)
    stats.update((epoch, batch_loss, None))


    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss



# def test_inference_speed(data_loader, model, device, args):
 
#     # 初始化日志记录器
#     logger = logging.getLogger()
#     # 设置模型为评估模式
#     model.eval()
    
#     # 初始化总推理时间和总样本数
#     total_inference_time = 0.0
#     total_samples = 0
    
#     # 开始总计时
#     start_total_time = time.time()
    
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(data_loader):
#             # 记录批次开始时间
#             batch_start_time = time.time()
            
#             # 获取输入数据并移动到设备
#             p = data["positive"]
#             x_p_a = p["audio"].to(device)
#             x_p_v = p["video"].to(device)
#             x_p_t = p["text"].to(device)
            
#             # 准备输入元组（根据模型需求调整）
#             inputs = (x_p_a, x_p_v, x_p_t)
            
#             # 如果需要标准化输入
#             if args.z_score_inputs:
#                 inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])
            
#             # 模型推理（假设模型通过get_embeddings获取输出）
#             _, _, _ = model.get_embeddings(*inputs)
            
#             # 记录批次结束时间并累加推理时间
#             batch_end_time = time.time()
#             total_inference_time += (batch_end_time - batch_start_time)
            
#             # 累加当前批次的样本数（假设每个批次的样本数为x_p_a的第一个维度）
#             total_samples += x_p_a.shape[0]
    
#     # 计算总耗时（包括数据加载等 overhead）
#     total_time = time.time() - start_total_time
#     print("Total time:", total_time, "seconds")
#     # 计算推理速度（平均每个样本的推理时间，单位：毫秒）
#     avg_inference_time = (total_inference_time / total_samples) * 1000
#     print("Average inference time per sample:", avg_inference_time, "ms")
#     # 计算吞吐量（每秒处理的样本数）
#     throughput = total_samples / total_time
#     print("throughput:", throughput)
#     # 日志输出结果
#     logger.info(
#         f"INFERENCE SPEED REPORT\t"
#         f"Total samples: {total_samples}\t"
#         f"Average inference time per sample: {avg_inference_time:.2f} ms\t"
#         f"Throughput: {throughput:.2f} samples/sec"
#     )
    
#     return avg_inference_time, throughput


def val_step(data_loader, model, criterion, epoch, epochs, writer, device, metrics, stats,             new_model_attention, model_devise, apn, cjme, audio_class,visual_class,args=None):
    logger = logging.getLogger()
    model.eval()
    batch_counter = 0
    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        batch_loss = 0
        hm_score = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(device)
            x_p_v = p["video"].to(device)
            x_p_t = p["text"].to(device)
            x_p_num = target["positive"].to(device)

            x_q_a = q["audio"].to(device)
            x_q_v = q["video"].to(device)
            x_q_t = q["text"].to(device)


            inputs = (
                x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t
            )


            if args.z_score_inputs:
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])




            loss, loss_details= model.optimize_params(*inputs, audio_class,visual_class,epoch,label=x_p_num, optimize=False)
            audio_emb, video_emb, emb_cls = model.get_embeddings(inputs[0], inputs[1], inputs[3])
            outputs = (video_emb, emb_cls)

            batch_loss += loss.item()

            p_target = target["positive"].to(device)
            q_target = target["negative"].to(device)

            # 更新指标并记录
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                for metric in metrics:
                    metric(outputs, (p_target, q_target), (loss, loss_details))
                    for key, value in metric.value().items():
                        if "recall" in key:
                            continue
                        if "both_hm" in key:
                            hm_score = value
                        if "both_zsl" in key:
                            zsl_score = value
                        writer.add_scalar(
                            f"val_{key}", value, iteration
                        )


        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, zsl_score))

        # 日志记录
        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
            f"ZSL score: {zsl_score:.4f}\t"
            f"HM: {hm_score:.4f}"
        )
    return batch_loss, hm_score,zsl_score

