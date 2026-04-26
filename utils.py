"""
Utility functions for interpretation generation.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torch
import torch.nn.functional as F
import os
import cv2
import time
import sys
import requests

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from scipy.interpolate import interp1d

from PIL import Image
from textwrap import fill
import yake
import pdb
import torch.utils.checkpoint as checkpoint
from io import BytesIO

# mean and standard deviation for the imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

special_ids = [12, 13, 29871, 29889, 29892, 1919, 869, 29899, 29897, 29898, 278, 322, 310, 263, 385, 445, 393, 338, 411]
CHOICES = ['A','B','C','D','E','F']

def eprint(*args, **kwargs):
    """
        Prints to the std.err

    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def get_data(args, row):
    if args.dataset == 'MMVP':
        photo_id = row['index']
        image_path = os.path.join(args.image_folder, f"{photo_id}.jpg")
        image = Image.open(image_path).convert('RGB')

        if args.choices:
            opts = row['Options'].split(' ')
            cur_prompt = f"{row.get('Question', row.get('question', 'What is in this image?'))}\nA. {opts[1]}\nB. {opts[3]}\n"
            qs = cur_prompt + "Answer only with the option's letter A or B from the given choices directly."
        else:
            qs = row.get('Question', row.get('question', 'What is in this image?'))
            cur_prompt = qs
    elif args.dataset == 'cvbench':
        image_path = os.path.join(args.image_folder, row["filename"])
        image = Image.open(image_path).convert('RGB')
        if args.choices:
            choices = row['choices']
            choices_text = ''
            for j,item in enumerate(choices):
                choices_text += f'\n{CHOICES[j]}. {item}'

            cur_prompt = row['question'] + choices_text
            qs = cur_prompt + '\nAnswer with the option\'s letter from the given choices directly.'
        else:
            qs = row['question']
            cur_prompt = qs
    elif args.dataset == 'mmstar':
        image_stream = BytesIO(row["image"]['bytes'])
        image = Image.open(image_stream).convert('RGB')
        if args.choices:
            cur_prompt = row['question']
            qs = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        else:
            qs = row['question'].split('?')[0] + '?'
            cur_prompt = qs
            
    elif args.dataset == 'llava-bench':
        image_path = os.path.join(args.image_folder, row["image"])
        image = Image.open(image_path).convert('RGB')
        cur_prompt = row['text']
        qs = cur_prompt
    return image, qs, cur_prompt
    

def get_kernel_size(image_size):
    if image_size[0] >=2000:
        kernel_size = 501
    elif image_size[0] >= 800:
        kernel_size = 301
    elif image_size[0] >= 500:
        kernel_size = 201
    else:
        kernel_size = 101
    return kernel_size

def save_heatmaps(masks, images, size, index, index_o, outdir, model_name, box, classes, labels, out=224):
    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)
    masks = masks.view(-1, 1, size, size)
    up = torch.nn.UpsamplingBilinear2d(size=images.shape[-2:]).cuda()

    u_mask = up(masks)
    u_mask = u_mask.permute((0,2, 3, 1))

    # Normalize the mask
    u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))
    u_mask = u_mask.cpu().detach().numpy()

    # deprocess images
    images = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    images = images.numpy()

    for i, (image, u_mask) in enumerate(zip(images, u_mask)):

        # get the color map and normalize to 0-1
        heatmap = cv2.applyColorMap(np.uint8(255 * u_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap / 255)
        # overlay the mask over the image
        #overlay = (u_mask ** 0.8) *0.5* image + (1 - u_mask ** 0.8) * heatmap
        overlay = 0.5 * heatmap + 0.5 * image
        cv2.normalize(overlay.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        overlay[overlay < 0] = 0

        plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_heatmap.jpg'), heatmap)
        plt.imsave(os.path.join(outdir, f'{index+i}_{index_o}_overlay.jpg'), overlay)


def save_masks(masks, index, categories, mask_name, outdir):
    """
        Saves the generated masks as numpy.ndarrays.

    :param masks:
    :param index:
    :param categories:
    :param mask_name:
    :param outdir:
    :return:
    """
    masks = masks.cpu().detach().numpy()
    for i, (mask, category) in enumerate(zip(masks, categories), start=index):
        np.save(os.path.join(outdir, f'{mask_name}_{i+1}_mask_{category}.npy'), mask)

def save_loss(loss_del, loss_ins, loss_l1, loss_tv, loss_l2, index, index_o, outdir, loss_comb_del=None, loss_comb_ins=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    iterations = np.arange(len(loss_del))

    if loss_comb_del:
        ax1.plot(iterations, loss_comb_del, label='loss comb del', marker='^')
    if loss_comb_ins:
        ax1.plot(iterations, loss_comb_ins, label='loss comb ins', marker='<')
    ax1.plot(iterations, loss_del, label='loss del', marker='o')
    ax1.plot(iterations, loss_ins, label='loss ins', marker='s')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(iterations, loss_tv, label='loss TV', marker='o')
    ax2.plot(iterations, loss_l2, label='loss L2', marker='+')
    ax2.plot(iterations, loss_l1, label='loss L1', marker='x')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True)
    ax2.legend()

    plt.savefig(os.path.join(outdir, f'{index}_{index_o}_losses.jpg'), bbox_inches='tight', pad_inches = 0)
    plt.close()

def save_curves(del_curve, ins_curve, index_curve, index, index_o, outdir):
    """
        Save the deletion/insertion curves for the generated masks.

    :param del_curve:
    :param ins_curve:
    :param index_curve:
    :param index:
    :param index_o:
    :param outdir:
    :return:
    """
    for i in range(len(del_curve)):
        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(index_curve, del_curve[i], color='r', label='deletion')
        ax.fill_between(index_curve, del_curve[i], facecolor='maroon', alpha=0.4)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=14)
        ax.set_yticks(np.arange(0, 1.01, 1))
        ax.legend(['Deletion'], fontsize='x-large')
        ax.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(del_curve[i])),  fontsize=14, horizontalalignment='center', verticalalignment='center')

        ax1.plot(index_curve, ins_curve[i], color='b', label='Insertion')
        ax1.fill_between(index_curve, ins_curve[i], facecolor='darkblue', alpha=0.4)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=14)
        ax1.set_yticks(np.arange(0, 1.01, 1))
        ax1.legend(['Insertion'], fontsize='x-large')
        ax1.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(ins_curve[i])), fontsize=14, horizontalalignment='center', verticalalignment='center')

        # save the plot
        plt.savefig(os.path.join(outdir, f'{index}_{index_o}_curves.jpg'), bbox_inches='tight', pad_inches = 0)
        plt.close()


def save_images(images, index, index_o, outdir, classes, labels, pred_data, text=None):
    """
        saves original images into output directory

    :param images:
    :param index:
    :param index_o:
    :param outdir:
    :param classes:
    :param labels:
    :return:
    """
    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)
    images_ = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    for i, image in enumerate(images_):
        wrapped_text = fill(text, width=60)
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.subplots_adjust(top=0.8)
        ax.imshow(image.numpy())
        ax.axis('off')
        fig.text(0.5, 0.9, wrapped_text, ha='center', va='top', wrap=True, fontsize=10)
        plt.savefig(os.path.join(outdir, f'{index+i}_{index_o}_image.jpg'), bbox_inches='tight', pad_inches=0.2)


def auc(array):
    """
        calculates area under the curve (AUC)

    :param array:
    :return:
    """
    return (sum(array) - array[0]/2 - array[-1]/2)/len(array)


def get_initial(pred_data, k, init_posi, init_val, input_size, out_size):
    """
        filter the detection results by the threshold (predicted score)

    :param pred_data:
    :param k:
    :param initial_posi:
    :param init_val:
    :param input_size:
    :param out_size:
    :return:
    """
    interval_r = (pred_data['boxes'][:,2] - pred_data['boxes'][:,0]) / k
    interval_c = (pred_data['boxes'][:,3] - pred_data['boxes'][:,1]) / k
    num_row = init_posi // k
    num_col = init_posi - num_row * k
    init_boxes = np.concatenate([
                            [pred_data['boxes'][:,0] + interval_r * num_row], # x1
                            [pred_data['boxes'][:,1] + interval_c * num_col], # y1
                            [pred_data['boxes'][:,0] + interval_r * (num_row + 1)], # x2
                            [pred_data['boxes'][:,1] + interval_c * (num_col + 1)], # y2
                            ], axis=0).T 

    pred_data['init_masks'] = []
    down = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    for ith, box in enumerate(init_boxes):
        init_mask = torch.zeros((input_size, input_size)).unsqueeze(0)
        init_mask[int(box[0]) : int(box[2]), int(box[1]) : int(box[3])] = 1

        if 'masks' in pred_data.keys():
            init_mask = init_mask * pred_data['masks'][ith]
        
        init_mask = down(init_mask.unsqueeze(0)) * init_val
        pred_data['init_masks'].append(1 - init_mask)
    return pred_data


def generate(args, model, input_ids, image, image_size):
    output_ids = model.generate(
            input_ids,
            images=image,
            #image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    output_ids = output_ids[:,1:-1]
    return output_ids

def match_keywords(lst, sublist):
    lst_str = ','.join(map(str, lst))
    sublist_str = ','.join(map(str, sublist))
    start_idx = lst_str.find(sublist_str)

    if start_idx == -1:
        return None

    start = lst_str[:start_idx].count(',')
    end = start + len(sublist) - 1
    return [start, end]

def pred_probs(args, model, prompt, output_ids, image, image_size):
    if args.model == 'llava' or args.model == 'llava_next':
        def custom_forward(input_ids, images, image_size):
            (inputs, _, _, _, inputs_embeds, labels, _) = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, images, image_sizes=image_size)
            return inputs, inputs_embeds, labels
        inputs, inputs_embeds, labels = checkpoint.checkpoint(custom_forward, prompt, image, image_size)
        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
    
    elif args.model == 'mgm':
        (inputs, _, _, _, inputs_embeds, labels) = model.prepare_inputs_labels_for_multimodal(prompt, None, None, None, None, image)
        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
    
    elif args.model == 'cambrian':
        (
            inputs,
            _,
            _,
            _,
            inputs_embeds,
            labels,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = model.prepare_inputs_labels_for_multimodal(
            prompt,
            None,
            None,
            None,
            None,
            image,
            image_sizes=image_size
        )
        model.vision_tower_aux_feature_list = vision_tower_aux_feature_list
        model.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
        model.final_vision_feature_size = final_vision_feature_size
        model.global_context_feature = global_context_feature

        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
        outputs = model(inputs, attention_mask, position_ids, None, inputs_embeds, labels, True, False, False, return_dict=True)

    else:
        raise ValueError('Model not supported: {}'.format(args.model))
    logits = outputs['logits']
    #logits.requires_grad_(True)
    logits_pred = logits[:,-output_ids.shape[-1]-1:-1,:]
    probs_pred = F.softmax(logits_pred, dim=-1) # (1, len, 32000)
    probs_pred = torch.gather(probs_pred, 2, output_ids.unsqueeze(-1)).squeeze()
    if probs_pred.dim() == 0:
        probs_pred = probs_pred.unsqueeze(0)
    return probs_pred

def find_keywords(args, model, input_ids, output_ids, image, blur_image, image_size, output_text, tokenizer=None, use_yake=False):
    if output_ids.shape[-1] == 1:
        full_prompt = torch.cat((input_ids, output_ids), dim=1)
        probs = pred_probs(args, model, full_prompt, output_ids, image, image_size)
        probs_blur = pred_probs(args, model, full_prompt, output_ids, blur_image, image_size)
        if probs_blur.squeeze().item() > 0.5*probs.squeeze().item():
            positions = []
            keywords = []
        else:
            positions = [0]
            keywords = [output_text]

    else:
        if use_yake:
            # select keywords using nlp tool
            num_words = len(output_text.split())
            if num_words <= 10:
                keywords_num = 3
            else:
                keywords_num = num_words // 4

            kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.2, top=keywords_num, features=None)
            keywords = kw_extractor.extract_keywords(output_text)
            keywords = [kw[0] for kw in keywords]
            positions = []
            for kw in keywords:
                kw_ids = tokenizer.encode(kw, add_special_tokens=False)
                position = match_keywords(output_ids[0].tolist(), kw_ids)
                if position:
                    positions.extend(position)
        else:
            # select keywords according to logits drop
            full_prompt = torch.cat((input_ids, output_ids), dim=1)
            probs = pred_probs(args, model, full_prompt, output_ids, image, image_size)
            probs_blur = pred_probs(args, model, full_prompt, output_ids, blur_image, image_size)
            # condition = (probs_blur <= 0.4*probs) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
            condition = (torch.log(probs)-torch.log(probs_blur) > 1.0)& (probs>=0.0) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
            positions = torch.where(condition)[0].tolist()
            keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
    
    return positions, keywords
