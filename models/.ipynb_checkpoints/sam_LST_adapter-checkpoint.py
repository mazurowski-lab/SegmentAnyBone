from .sam import sam_model_registry
import torch
import torch.nn as nn
from .sam.modeling.common import LayerNorm2d, MLPBlock, Adapter

class LST_Adapter(nn.Module):
    def __init__(self, D_features, D_out, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.D_out = nn.Linear(D_features,D_out)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            xs = x + xs
        xs = self.act(xs)
        x = self.D_out(xs)
        return x

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LST_adapter_layer(nn.Module):
    def __init__(self, dim_in=64, dim_out=128, pre_resolution=(128,128), new_resolution=(64,64)):
        super().__init__()
        self.LST_gate = nn.Parameter(torch.zeros(1))
        self.LST_adapter = LST_Adapter(dim_in,dim_out)

        self.pre_resolution = pre_resolution
        self.new_resolution = new_resolution
         
        if pre_resolution!=new_resolution:
            self.downsample = Downsample(pre_resolution,dim_in,dim_in)
            
        
    def forward(self, x_pre=None, x_new=None):
        if self.pre_resolution!=self.new_resolution:
            x_pre = self.downsample(x_pre) 
        x_up = self.LST_adapter(x_pre)
        x_gate = self.LST_gate*x_up+ (1-self.LST_gate) * x_new
        return x_gate
        
class Downsample(nn.Module):   
    def __init__(self, input_resolution, D_in, D_out):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim =  D_in
        self.out_dim = D_out
        self.conv = nn.Sequential(Conv2d_BN(D_in, D_out, 1, 1, 0),
                                   nn.GELU(),
                                   Conv2d_BN(D_out, D_out, 3, 2, 1),
                                  )
    def forward(self,x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
                           

class SAM_LST(nn.Module):

    def __init__(self,args,sam,embed_dims=[64, 128, 160, 320]):
        super(SAM_LST, self).__init__()
        self.args = args
        self.args.if_LST_CNN = True

        self.sam  = sam
        self.LST_encoder_blocks = []
        self.LST_decoder_blocks = []
        
        if self.args.if_LST_encoder_adapter:
            if len(self.args.encoder_LST_adapter_layers):
                self.LST_adapter = self.args.encoder_LST_adapter_layers
            else:
                self.LST_adapter = list(
                    range(len(sam.image_encoder.blocks)))
    
            for n, p in self.sam.image_encoder.named_parameters():
                p.requires_grad = False
                
            self.if_LST_encoder_blocks = []
            self.sam.LST_encoder_blocks = []

            for t_layer_i, layer in enumerate(sam.image_encoder.layers[1:]):
                # If we only want few LST_adapter instead of all
                if t_layer_i not in self.LST_adapter:
                    self.if_LST_encoder_blocks.append(False)
                    self.sam.LST_encoder_blocks.append([])
                else:
                    dim_in = embed_dims[t_layer_i+1]
                    if t_layer_i<len(embed_dims)-2:
                        dim_out = embed_dims[t_layer_i+2]
                    else:
                        dim_out = embed_dims[-1]
                    input_resolution = layer.input_resolution
                    self.if_LST_encoder_blocks.append(True)
                    self.sam.LST_encoder_blocks.append(LST_adapter_layer(dim_in,dim_out,input_resolution,(64,64)).cuda())
                
            self.sam.LST_encoder_blocks = nn.ModuleList(self.sam.LST_encoder_blocks)

        if self.args.if_LST_decoder_adapter:
            self.sam.LST_decoder_blocks = []
            for n, p in self.sam.mask_decoder.named_parameters():
                p.requires_grad = False
            
            for t_layer_i, layer in enumerate(sam.mask_decoder.transformer.layers):
                dim_in = 256
                dim_out = 256
                self.sam.LST_decoder_blocks.append(LST_adapter_layer(dim_in,dim_out,input_resolution,(64,64)).cuda())
            self.sam.LST_decoder_blocks = nn.ModuleList(self.sam.LST_decoder_blocks)
        
        for n, p in self.sam.named_parameters():
            if 'LST' in n:
                p.requires_grad = True

            if "output_upscaling" in n: # the output upscaling part of SAM decoder
                p.requires_grad = True
        
                



    def forward_encoder(self, input_images):
        # have no gradient for sam encoder blocks
        x_pre = self.sam.image_encoder.patch_embed(input_images)
        with torch.no_grad():
            x_pre = self.sam.image_encoder.layers[0](x_pre) # b*16384*128
            x_new = self.sam.image_encoder.layers[1](x_pre)
        if self.if_LST_encoder_blocks[0]:
            x_pre = self.sam.LST_encoder_blocks[0](x_pre,x_new)
        with torch.no_grad():   
            x_new = self.sam.image_encoder.layers[2](x_new)
        if self.if_LST_encoder_blocks[1]:
            x_pre =  self.sam.LST_encoder_blocks[1](x_pre,x_new)
        with torch.no_grad(): 
            x_new = self.sam.image_encoder.layers[3](x_new)
        if self.if_LST_encoder_blocks[2]:
            x_final = self.sam.LST_encoder_blocks[2](x_pre,x_new)
            
        B,_,C = x_final.size()
        x_final = x_final.view(B, 64, 64, C)
        x_final=x_final.permute(0, 3, 1, 2)
        x_final=self.sam.image_encoder.neck(x_final)
        
        return x_final

    def forward_decoder(self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ):
        
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.sam.mask_decoder.iou_token.weight, self.sam.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # apply as new image embedding, image_pe, and point_embeddings
        image_embedding = src
        image_pe = pos_src
        point_embedding = tokens

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        pre_keys = image_embedding
        queries = point_embedding

        # run the side network

        with torch.no_grad():
            queries, new_keys = self.sam.mask_decoder.transformer.layers[0](queries=queries,
                keys=pre_keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        pre_keys = self.sam.LST_decoder_blocks[0](pre_keys,new_keys)

        with torch.no_grad():
            queries, new_keys = self.sam.mask_decoder.transformer.layers[1](queries=queries,
                keys=pre_keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        keys = self.sam.LST_decoder_blocks[1](pre_keys,new_keys)


        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.sam.mask_decoder.transformer.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.sam.mask_decoder.transformer.norm_final_attn(queries)

        # go back to the main network
        
        hs = queries
        src = keys
            
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.sam.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.sam.mask_decoder.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.sam.mask_decoder.num_mask_tokens):
            hyper_in_list.append(self.sam.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.sam.mask_decoder.iou_prediction_head(iou_token_out)
        return masks, iou_pred

        

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