import torch
import numpy as np
import argparse
import os
from more_itertools import pairwise

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module

def check_path(path):
    """
    Checks if path is valid and returns the absolute path
    :param path: Path obtained from user
    :return: absolute path if it exists
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if os.path.exists(path):
        return path
    else:
        raise Exception("Path does not exist.")

# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument("--input_dir", help='Path to folder with input sequence of images')
    parser.add_argument("--output_dir", help='Path to save output images')
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    net.eval()

    # load the image pair, you can find this operation in dataset.py
    input_dir = check_path(args.input_dir)
    output_dir = check_path(args.output_dir)
    filelist = sorted(os.listdir(input_dir))
    for i, (file1, file2) in enumerate(pairwise(filelist)):
        pim1 = read_gen(os.path.join(input_dir, file1))
        pim2 = read_gen(os.path.join(input_dir, file1))
        images = [pim1, pim2]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # process the image pair to obtian the flow
        result = net(im).squeeze()

        data = result.data.cpu().numpy().transpose(1, 2, 0)
        output_file_name = file1.split(".")[0] + ".flo"
        writeFlow(os.path.join(output_dir, output_file_name), data)
        
        # For pool boiling experimental data, which has 2184 images per input directory
        if (i+1)%91 == 0:
            print(f"Step {i//91} of {len(filelist)//91} completed")
    print(f"Inference for folder {input_dir} saved in folder {output_dir}")
    print("\n")
    print("#"*40)
    print("\n\n")