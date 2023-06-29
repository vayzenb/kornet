curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'
import sys
vone_dir = '/user_data/vayzenbe/GitHub_Repos/vonenet'
cornet_dir = '/user_data/vayzenbe/GitHub_Repos/CORnet'

sys.path.insert(1, vone_dir)
sys.path.insert(1, cornet_dir)
sys.path.insert(1,curr_dir)
import vonenet
import cornet
from torchvision.models import convnext_large, ConvNeXt_Large_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import alexnet, AlexNet_Weights, vgg19, VGG19_Weights
import torch


import torchvision


weights_dir = f'/lab_data/behrmannlab/vlad/kornet/modelling/weights'
def load_model(model_arch):    
    """
    load model
    """
    

    if model_arch == 'vonenet_r_ecoset' or model_arch =='vonenet_r_stylized-ecoset':
        model = vonenet.get_model(model_arch='cornets', pretrained=False).module
        layer_call = "getattr(getattr(getattr(getattr(model,'module'),'model'),'decoder'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])
        

    elif model_arch == 'vonenet_ff_ecoset' or model_arch =='vonenet_ff_stylized-ecoset':
        model = vonenet.get_model(model_arch='cornets_ff', pretrained=False).module
        layer_call = "getattr(getattr(getattr(getattr(model,'module'),'model'),'decoder'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])


    elif model_arch == 'vonenet_r_stylized-ecoset':

        model = vonenet.get_model(model_arch='cornets_ff', pretrained=False).module
        layer_call = "getattr(getattr(getattr(getattr(model,'module'),'model'),'decoder'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])
       

    

    if model_arch == 'vonecornet_s':
        model = vonenet.get_model(model_arch='cornets', pretrained=True).module
        layer_call = "getattr(getattr(getattr(getattr(model,'module'),'model'),'decoder'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])

    elif model_arch == 'cornet_s':
        model = cornet.get_model('s', pretrained=True).module
        layer_call = "getattr(getattr(getattr(model,'module'),'decoder'),'avgpool')"

        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
        

    if model_arch == 'voneresnet':
        model = vonenet.get_model(model_arch='resnet50', pretrained=True).module
        layer_call = "getattr(getattr(getattr(model,'module'),'model'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])

    elif model_arch == 'convnext':
        model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        transform = ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
        layer_call = "getattr(getattr(model,'module'),'avgpool')"

    elif model_arch == 'vit':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        transform = ViT_B_16_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'encoder'),'ln')"
        #layer_call = "getattr(getattr(getattr(getattr(getattr(getattr(model,'module'),'encoder'),'layers'),'encoder_layer_11'),'mlp'),'3')"

    elif model_arch == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        transform = ResNet50_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(model,'module'),'avgpool')"
    
    elif model_arch == 'resnext50':
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        transform = ResNeXt50_32X4D_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(model,'module'),'avgpool')"
    
    elif model_arch == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        transform = AlexNet_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'classifier'),'5')"

    elif model_arch == 'vgg19':
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        transform = VGG19_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'classifier'),'5')"

    elif model_arch == 'ShapeNet':
        model = resnet50(weights=None)
        checkpoint = torch.load(f'{weights_dir}/ShapeNet_ResNet50_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        transform = ResNet50_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(model,'module'),'avgpool')"

    elif model_arch == 'SayCam':
        model = resnext50_32x4d(weights=None)
        transform = ResNeXt50_32X4D_Weights.DEFAULT.transforms()
        
        checkpoint = torch.load(f'{weights_dir}/SayCam_ResNext_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        layer_call = "getattr(getattr(model,'module'),'avgpool')"

        

    model = torch.nn.DataParallel(model).cuda()

    
    if model_arch == 'vonenet_r_ecoset' or model_arch =='vonenet_r_stylized-ecoset' or model_arch =='vonenet_ff_ecoset' or model_arch =='vonenet_ff_stylized-ecoset':
        checkpoint = torch.load(f'{weights_dir}/{model_arch}_best_1.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

    return model, transform, layer_call