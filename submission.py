from tabnanny import check
import torch
import pandas as pd
from data import RoadAllDataTest, build_transform
import argparse
import models
from tqdm import tqdm

parser = argparse.ArgumentParser(description='submission')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data_path', default='/data2/zrliu/dataset_cmp/dataset/test_images', type=str, 
                    help='data path')
parser.add_argument('-b', '--batch-size', default=24, type=int, metavar='N',help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--old_version', action='store_true', help='to support old checkpoint.')
parser.add_argument('--multi_cls', action='store_true', help='whether to use multi-classifier')
parser.add_argument('--img_size', nargs=2, type=int, default=[2400,1080], help='image size')

def main():
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    arch = checkpoint['arch'] if 'arch' in checkpoint else 'resnet18'
    # delete renamed or unused k
    for k in list(state_dict.keys()):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    if 'args' in checkpoint:
        print('\n''\n')
        print("config of checkpoint: ")
        print('--------------------------------------------------------------------------------------------------')
        # Print arguments
        for k in checkpoint['args'].__dict__:
            print("{:40}{}".format(k, str(checkpoint['args'].__dict__[k])))
        print('--------------------------------------------------------------------------------------------------')
    if args.old_version:
        for k in list(state_dict.keys()):
            if k.startswith("backbone."):
                state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
                
    args_ = checkpoint['args']  
    model = models.__dict__[arch](multi_cls=args_.multi_cls if hasattr(args_, 'multi_cls') else False,
                multi_scale=args_.multi_scale if hasattr(args_, 'multi_scale') else False).cuda()
    model.load_state_dict(state_dict, strict=False)

    transform = build_transform(size=(args.img_size[0], args.img_size[1]), rate_for_crop=None, rand_filp=False, to='Tensor', norm=True, 
                    canny=args_.canny if hasattr(args_, 'canny') else False,
                    hist=args_.hist if hasattr(args_, 'hist') else False, 
                    hist_level=args_.hist_level if hasattr(args_, 'hist_level') else 16)
    dataset = RoadAllDataTest(data_path=args.data_path, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    model.eval()
    with torch.no_grad():
        total_preds = []
        name_list = []
        for images, names in tqdm(dataloader):
            images = images.cuda()
            out, _ = model(images)
            preds = out[0].softmax(dim=1)[:, 0]
            total_preds.extend(preds.detach().cpu().numpy())
            name_list.extend(names)
    
    submission = pd.DataFrame({'imagename':name_list, 'defect_prob':total_preds})
    submission.to_csv('./submission.csv', index=False, encoding='utf-8')
    

if __name__ == '__main__':
    main()


