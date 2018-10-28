import argparse
import torch
import torch.utils.data
from torch import nn
from torch.autograd import grad
from train import prepare_dataloaders
from dataset import collate_fn, TranslationDataset
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
from transformer.Models import Transformer
from transformer.Beam import Beam
# The first challenge is to handle beam search, this of course means for seq2seq problems we
# apply IG in train phase 
# Another challenge is the encoder decoder problem, i think can be solve by chain rule
# Chain rule 
# ddecoder_output/dinput = ddecoder_output/dencoder_output * dencoder_output/dinput
# model output.backward() -> find gradient with respect to input 
# I don't know why but model() immediately triggers the foward method, 
# pytorch weirdness 1
# class -> object -> object(forward pass params) -> generates forward pass
class Attribution(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self,opt,training_data):
        #opt is from argprass 
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        #opt.model is the model path 
        checkpoint = torch.load(opt.model)
        #model_opt is the model hyper params
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        #Load the actual model weights 
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def attribute_batch(self):
        ''' Attribute in one batch '''
        #-- Encode
        for batch in tqdm(training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

            # forward
            pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
            pred.backward()


if __name__ == "__main__":
    # Prepare DataLoader
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-data', required=True)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #========= Loading Dataset =========#
    data = torch.load(opt.data)

    training_data, validation_data = prepare_dataloaders(data, opt)
    translator = Attribution(opt,training_data)
