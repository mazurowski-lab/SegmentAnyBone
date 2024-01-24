from sam import sam_model_registry
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class SAM_LST(nn.Module):

    def __init__(self,args,sam):
        super(SAM_LST, self).__init__()
        self.args = args
        self.args.if_LST_CNN = True

        self.sam  = sam

        self.CNN_encoder = resnet18(pretrained=True)
        
        self.sam_encoder = self.sam.image_encoder
        self.sam_decoder = self.sam.mask_decoder
        self.sam_prompt_encoder = self.sam.prompt_encoder
        
        self.alpha =self.sam.alpha


        for n, p in self.sam.named_parameters():
            p.requires_grad = False

        for n, p in self.sam.named_parameters():
            if "alpha" in n:
                p.requires_grad = True

            if "output_upscaling" in n: # the output upscaling part of SAM decoder
                p.requires_grad = True



    def forward(self, input_images, points = None, boxes = None, masks = None, multimask_output = None):


        cnn_out = self.CNN_encoder.conv1(input_images)
        cnn_out = self.CNN_encoder.bn1(cnn_out)
        cnn_out = self.CNN_encoder.relu(cnn_out)
        cnn_out = self.CNN_encoder.maxpool(cnn_out)

        cnn_out = self.CNN_encoder.layer1(cnn_out)
        cnn_out = self.CNN_encoder.layer2(cnn_out)
        CNN_input = self.CNN_encoder.layer3(cnn_out)

        image_embeddings = self.image_encoder(input_images)
        
        if self.args.if_LST_CNN:
            gate = torch.sigmoid(self.alpha)
            image_embeddings = gate*image_embeddings + (1-gate) * CNN_input
            
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )
        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
        return low_res_masks





if __name__ == "__main__":


    net = SAM_LST().cuda()
    out = net(torch.rand(1, 3, 512, 512).cuda(), 1, 512)
    parameter = 0
    select = 0
    for n, p in net.named_parameters():

        parameter += len(p.reshape(-1))
        if p.requires_grad == True:
            select += len(p.reshape(-1))
    print(select / parameter * 100)

    print(out['masks'].shape)