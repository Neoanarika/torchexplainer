import torch
from torch.autograd import Variable, grad


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab,m):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.m = m 
        self.IG = []
        self.tgt_IG = []

    def get_decoder_features(self, src_seq,no_grad=True,tgt_seq=[]):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        #print("no_grad",no_grad)
        if no_grad:
            with torch.no_grad():
                softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        else:
            for k in range(1,self.m+1):
                softmax_list, _, other = self.model(src_id_seq, [len(src_seq)],k=k,m=self.m)
                length = other['length'][0]
                for i in range(length):
                    if k == 1 : 
                        if len(tgt_seq) > 0: 
                            if i == length-1: 
                                 self.tgt_IG.append(1/self.m * self.model.encoder.diff * grad(torch.max(softmax_list[i]), self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0]) 
                            else:
                                self.tgt_IG.append(1/self.m * self.model.encoder.diff * grad(softmax_list[i][0][int(tgt_seq[i])-1], self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0])
                        self.IG.append(1/self.m * self.model.encoder.diff * grad(torch.max(softmax_list[i]), self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0])
                    else:
                        if i == length-1: 
                            self.tgt_IG[i]+=(1/self.m * self.model.encoder.diff * grad(torch.max(softmax_list[i]), self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0]) 
                        else:
                            self.tgt_IG[i]+=(1/self.m * self.model.encoder.diff * grad(softmax_list[i][0][int(tgt_seq[i])-1], self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0])
                        self.IG[i] += 1/self.m * self.model.encoder.diff * grad(torch.max(softmax_list[i]), self.model.encoder.embedded, retain_graph=True,allow_unused=True)[0][0]
                        
        return other

    def predict(self, src_seq,no_grad=True,tgt_seq=[]):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        self.IG = []
        self.tgt_IG = []
        other = self.get_decoder_features(src_seq,no_grad,tgt_seq=tgt_seq)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq

    def predict_n(self, src_seq, n=1,no_grad=True):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq,no_grad)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
