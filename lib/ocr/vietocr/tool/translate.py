import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax

from ..model.transformerocr import VietOCR
from ..model.vocab import Vocab
from ..model.beam import Beam

def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        memories = model.transformer.forward_encoder(src)
        for i in range(memories.size(1)):
            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            sent = beamsearch(memory, model, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents
   
def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    
    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src) #TxNxE

        sent = beamsearch(memory, model, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
    # memory: Tx1xE
    model.eval()
    if len(memory) == 2:
        memory = memory[0]
    device = memory.device
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)
    print(memory)
    with torch.no_grad():
        memory = memory.repeat(1, beam_size, 1) # TxNxE
         
        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
            decoder_outputs = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
    
    return translated_sentence

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def resize_padding(w, h, expected_height, image_fix_width=128):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    if new_w < image_fix_width:
        return new_w, expected_height
    else:
        return image_fix_width, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.Resampling.LANCZOS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img

def process_image_fix(image, image_height, image_width=128):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize_padding(w, h, image_height,image_width)

    if new_w < image_width:
        img = img.resize((new_w, image_height), Image.Resampling.LANCZOS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (image_width, image_height))
        new_im.paste(img, ((image_width - new_w) // 2, 0))
        img = new_im

    else:
        img = img.resize((image_width, image_height), Image.Resampling.LANCZOS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img

def process_batch_input(batch_image, image_height, standard_size=65):

    batch_image_processed = [process_image_fix(img, image_height, image_width=standard_size) for img in batch_image]
    print("Batch shape: ", batch_image_processed[0].shape)
    batch_image_processed = np.stack(batch_image_processed)
    # print(batch_image_processed.shape)
    batch_image_processed = np.array(batch_image_processed)
    batch_image_processed = torch.FloatTensor(batch_image_processed)
    return batch_image_processed

def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s

