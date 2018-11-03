import argparse
import pickle
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable, grad
from train import prepare_dataloaders
from dataset import collate_fn, TranslationDataset
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
from transformer.Models import Transformer

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

    def __init__(self,opt):
        #opt is from argprass 
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        self.m = opt.m 
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

    def attribute_batch(self,training_data,dev=False):

        # LongTensor cannot be backpropogated
        def f(x):
            x.to(self.device)
            return x
        ''' Attribute in one batch '''
        #-- Encode
        # Understand the correctness of the code
        # Understand the output -> Visualisation 
        F = []
        for batch in tqdm(training_data, mininterval=2,
            desc='  - (Attributing)   ', leave=False):
            src_seq, src_pos, tgt_seq, tgt_pos = map(f, batch)
            IG = []
            for k in range(1,self.m+1):

                pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos,alpha=k/self.m)
                
                val,translated_sentence = torch.max(pred,1)
                for id_,translated_word in enumerate(val):
                    #Finds the gradient of a single sentence

                    if k == 1: IG.append(torch.sum(1/self.m*self.model.encoder.difference*grad(translated_word, self.model.encoder.emb, retain_graph=True,allow_unused=True)[0],2))
                    IG[id_] += torch.sum(1/self.m*self.model.encoder.difference*grad(translated_word, self.model.encoder.emb, retain_graph=True,allow_unused=True)[0],2)
            
            F.append({
                    "IG":IG,
                    "src_seq":src_seq,
                    "translated_sentence":translated_sentence,
                })

            if dev:
                IG = torch.squeeze(torch.stack(IG)).detach().numpy().T
                return IG,src_seq,translated_sentence
        return F 

if __name__ == "__main__":
    # Prepare DataLoader
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-data', required=True)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-m',type=int, default=10,
                        help='Resolution of the integrated gradient')
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-out',help='Path to output file of ')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-dev', action='store_true')

    opt = parser.parse_args()
    assert opt.dev or opt.out!=None, "You are not in dev mode but you don't have an output file"
    opt.cuda = not opt.no_cuda

    #========= Loading Dataset =========#
    data = torch.load(opt.data)

    training_data, validation_data = prepare_dataloaders(data, opt)
    attributor = Attribution(opt)
    if opt.dev : 
        IG,src_seq,translated_sentence  = attributor.attribute_batch(validation_data,dev=True)

        original_line = [validation_data.dataset.src_idx2word[idx.item()] for idx in src_seq[0]]
        pred_line = [validation_data.dataset.tgt_idx2word[idx.item()] for idx in translated_sentence]

        fig = plt.figure(figsize=(8, 8.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(IG, interpolation='nearest', cmap='gray')

        ax.set_yticks(range(len(original_line)))
        ax.set_yticklabels(original_line)

        ax.set_xticks(range(len(pred_line)))
        ax.set_xticklabels(pred_line, rotation=45)

        ax.set_xlabel('Output Sequence')
        ax.set_ylabel('Input Sequence')
        fig.show()
        plt.show()

    else: 
        outfile = open(opt.out,'wb')
        F = attributor.attribute_batch(validation_data)
        pickle.dump(F,outfile)
        outfile.close()
