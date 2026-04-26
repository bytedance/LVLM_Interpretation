"""
main file to call the explanations methods and run experiments to generate saliency heatmap
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torchvision.models as models
from torch.autograd import Variable
from datasets import load_dataset
from args import init_args
from utils import *
from methods_helper import *
from methods import iGOS_p, iGOS_pp
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_vicuna_v1
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from cambrian.conversation import conv_llama_3, conv_phi3
from cambrian.mm_utils import process_images_cambrian
from cambrian.model.builder import load_pretrained_model_cambrian
from llava_next.model.builder import load_pretrained_model_next
from llava_next.conversation import conv_qwen
from llava_next.mm_utils import process_images_next
from mgm.model.builder import load_pretrained_model_mgm
from mgm.mm_utils import process_images_mgm
import pandas as pd
import cv2
import json
import os
import random
import pickle
from PIL import Image

join = os.path.join

def gen_explanations_llava(model, image_processor, data, args):
    setting = f'{args.method}_L1_{args.L1}_L2_{args.L2}_L3_{args.L3}_momentum_{args.momentum}_igiter_{args.ig_iter}_lr_{args.lr}_iter_{args.iterations}'
    out_dir = os.path.join(args.output_dir, setting)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.method == "iGOS+":
        method = iGOS_p
    elif args.method == "iGOS++":
        method = iGOS_pp
    else:
        raise ValueError("the method does not exist.")

    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0
    all_del_scores = []
    all_ins_scores = []
    save_list = []
    for i_img in range(len(data)):
        row = data[i_img]
        image, qs, cur_prompt = get_data(args, row)

        image_size = [image.size]
        kernel_size = get_kernel_size(image.size)
        
        if args.ablation_noise:
            noise = np.random.randint(0, 256, size=np.asarray(image).shape, dtype=np.uint8)
            blur = Image.fromarray(noise)
        elif args.ablation_zero:
            blur = np.zeros_like(np.asarray(image))
            blur = Image.fromarray(blur.astype(np.uint8))
        else:
            blur = cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size-1)
            blur = Image.fromarray(blur.astype(np.uint8))

        if args.model == 'cambrian':
            image_tensor = process_images_cambrian([image], image_processor, model.config)
            blur_tensor = process_images_cambrian([blur], image_processor, model.config)
            # use different conversation template for different model size
            # conv = conv_vicuna_v1.copy() # cambrian-13b
            conv = conv_llama_3.copy() # cambrian-8b
            # conv = conv_phi3.copy() # cambrian-phi3-3b
        elif args.model == 'llava':
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).detach().half().cuda()
            blur_tensor = process_images([blur], image_processor, model.config)[0]
            blur_tensor = blur_tensor.unsqueeze(0).detach().half().cuda()
            conv = conv_vicuna_v1.copy()
        elif args.model == 'llava_next':
            image_tensor, resolution = process_images_next([image], image_processor, model.config)
            image_tensor = image_tensor[0].to(model.device).half()
            blur_tensor, _ = process_images_next([blur], image_processor, model.config)
            blur_tensor = blur_tensor[0].to(model.device).half()
            conv = conv_qwen.copy()
        elif args.model == 'mgm':
            image_tensor = process_images_mgm([image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).detach().half().cuda()
            blur_tensor = process_images_mgm([blur], image_processor, model.config)[0]
            blur_tensor = blur_tensor.unsqueeze(0).detach().half().cuda()
            conv = conv_vicuna_v1.copy()
        #print('original size:', image.size, 'processed:', image_tensor.shape)
        #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device='cuda', dtype=torch.float16)

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        if input_ids is not None: input_ids = input_ids.to('cuda')
        output_ids = generate(args, model, input_ids, image_tensor, image_size)
        output_text = tokenizer.decode(output_ids[0])
        # output_ids_blur = generate(args, model, input_ids, blur_tensor, image_size)
        # output_text_blur = tokenizer.decode(output_ids_blur[0])
        # # print('output text blurred:', output_text_blur)
        positions, keywords = find_keywords(args, model, input_ids, output_ids, image_tensor, blur_tensor, 
                                            image_size, output_text, tokenizer, args.use_yake)
        
        if len(positions) == 0:
            continue
        # print('----------------')
        # print(cur_prompt)
        # print(output_text)
        # print('detected keywords:', keywords)

        pred_data=dict()
        pred_data['labels'] = output_ids.detach()
        pred_data['keywords'] = positions
        pred_data['boxes'] = np.array([[0, 0, args.input_size, args.input_size]])
        pred_data['no_res'] = False
        pred_data['pred_text'] = output_text
        pred_data['keywords_text'] = keywords

        # calculate init area
        pred_data = get_initial(pred_data, args.diverse_k, args.init_posi, 
                                args.init_val, args.input_size, args.size)

        for l_i, label in enumerate(pred_data['labels']):
        #for l_i, keyword in enumerate(pred_data['keywords']):
            label = label.unsqueeze(0)
            keyword = pred_data['keywords']
            now = time.time()
            masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
                    args,
                    model=model,
                    model_name=args.model,
                    init_mask=pred_data['init_masks'][0],
                    image=image_tensor,
                    baseline=blur_tensor,
                    label=label,
                    size=args.size,
                    iterations=args.iterations,
                    ig_iter=args.ig_iter,
                    L1=args.L1,
                    L2=args.L2,
                    L3=args.L3,
                    lr=args.lr,
                    opt=args.opt,
                    prompt=input_ids,
                    image_size=image_size,
                    positions=keyword,
                    resolution=resolution if args.model=='llava_next' else None
                )
            total_time += time.time() - now

            # Calculate the scores for the masks
            del_scores, ins_scores, del_curve, ins_curve, index = metric(
                    args,
                    image_tensor,
                    blur_tensor,
                    masks.detach(),
                    model,
                    args.model,
                    label,
                    l_i,
                    pred_data,
                    size=args.size,
                    prompt=input_ids,
                    image_size=image_size,
                    positions=keyword,
                    resolution=resolution if args.model=='llava_next' else None,
            )
            # save heatmaps, images, and del/ins curves
            classes=None
            cur_prompt = cur_prompt.strip().replace('\n','_')
            save_heatmaps(masks, image_tensor, args.size, i_img, l_i, out_dir, args.model, None, classes, label)
            save_curves(del_curve, ins_curve, index, i_img, l_i, out_dir)
            save_images(image_tensor, i_img, l_i, out_dir, classes, label, pred_data, text=cur_prompt)
            save_loss(loss_del, loss_ins, loss_l1, loss_tv, loss_l2, i_img, l_i, out_dir, loss_comb_del, loss_comb_ins)

            # log info
            all_del_scores.append(del_scores.sum().item())
            all_ins_scores.append(ins_scores.sum().item())
            i_obj += 1

            with open(join(out_dir, f'{i_img}_{l_i}_output.txt'), 'w', encoding='utf-8') as f:
                f.write('prompt: '+ cur_prompt + '\n')
                f.write('answer: '+ pred_data['pred_text'] + '\n')
                f.write('keywords: ' + ','.join(pred_data['keywords_text']) + '\n')
                # f.write('answer with blurred image: ' + output_text_blur + '\n')
                f.write(f'Deletion (Avg.): {del_scores.sum().item():.05f}' + '\n')
                f.write(f'Insertion (Avg.): {ins_scores.sum().item():.05f}' + '\n')
                f.write(f'Time (Avg.): {total_time / i_obj:.03f}' + '\n')

            eprint(
                    f' Deletion (Avg.): {del_scores.sum().item():.05f}'
                    f' Insertion (Avg.): {ins_scores.sum().item():.05f}'
                    f' Time (Avg.): {total_time / i_obj:.03f}'
            )


if __name__ == "__main__":

    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)

    disable_torch_init()
    if args.model == 'cambrian':
        model_path = "nyu-visionx/cambrian-8b"
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model_cambrian(model_path, args.model_base, model_name)
    elif args.model == 'llava':
        model_path = 'liuhaotian/llava-v1.5-7b'
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_4bit=True)
        for name, buffer in model.named_buffers():
            buffer.data = buffer.data.to('cuda')
    elif args.model == 'llava_next':
        model_path = "lmms-lab/llava-onevision-qwen2-72b-ov"
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model_next(model_path, args.model_base, model_name, device_map='auto')
    elif args.model == 'mgm':
        model_path = 'mgm/work_dirs/mgm13bhd'
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model_mgm(model_path, args.model_base, model_name)

    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing = True
    if args.data_path.endswith('csv'):
        df = pd.read_csv(args.data_path)
        data = df.to_dict(orient='records')
    elif args.data_path.endswith('jsonl'):
        data = [json.loads(q) for q in open(args.data_path, "r")]
    elif args.data_path.endswith('json'):
        data = json.load(open(args.data_path))
    elif args.data_path.endswith('pkl'):
        data = pickle.load(open(args.data_path, 'rb'))
    else:
        data = load_dataset(args.data_path, "val")["val"].to_pandas()
        data = data.to_dict(orient="records")

    print('total number of data samples:', len(data))
    gen_explanations_llava(model, image_processor, data, args)
