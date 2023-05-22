import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import cv2
from tqdm import tqdm

from config import *
from misc import *
from TCGNet import TCGNet
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE

torch.manual_seed(2021)


results_path = './result'
check_mkdir(results_path)
record_file = './record.txt'
exp_name = 'PFNet'
args = {
    'scale': 352,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    # ('CHAMELEON', chameleon_path),
    ('ECSSD', ecssd_path)
    # ('COD10K', cod10k_path),
    # ('NC4K', nc4k_path)
])

results = OrderedDict()


def main_infer(model_dir):
    net = TCGNet(backbone_path).cuda()

    net.load_state_dict(torch.load(model_dir))
    print('Load {} succeed!'.format('PFNet.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root)

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

                start_each = time.time()
                prediction = net(img_var)
                prediction = torch.sigmoid(prediction)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(results_path, exp_name, name, img_name + '.png'))
            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

    #  eval
    gt_path = "/home/liuy/workspace/dataset/test/gt/ECSSD/"
    predict_path = './result/PFNet/ECSSD/'

    mae = MAE()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()

    images = os.listdir(predict_path)
    for image in tqdm(images):
        gt = cv2.imread(os.path.join(gt_path, image), 0)
        predict = cv2.imread(os.path.join(predict_path, image), 0)

        h, w = gt.shape
        predict = cv2.resize(predict, (w, h))

        mae.step(predict, gt)
        wfm.step(predict, gt)
        sm.step(predict, gt)
        em.step(predict, gt)

    print('mae: %.4f' % mae.get_results()['mae'])
    print('wfm: %.4f' % wfm.get_results()['wfm'])
    print('em: %.4f' % em.get_results()['em']['curve'].mean())
    print('sm: %.4f' % sm.get_results()['sm'])

    file = open(record_file, 'a')
    file.write("pth: %s" % model_dir + '\n')
    file.write("mae: %.4f" % mae.get_results()['mae'] + '\n')
    file.write('wfm: %.4f' % wfm.get_results()['wfm'] + '\n')
    file.write("em: %.4f" % em.get_results()['em']['curve'].mean() + '\n')
    file.write('sm: %.4f' % sm.get_results()['sm'] + '\n\n')

    file.close()


if __name__ == '__main__':
    model_dir = './ckpt/PFNet/55.pth'
    main_infer(model_dir=model_dir)
