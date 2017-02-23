import os
from os.path import join
import json
import argparse
import errno
from productdict.createdict import DocDictionary
from conv_net import simple_conv_word_embedding
from conv_net.simple_conv_word_embedding_train import train,test_label_from_ckpt,write_features,train_vgg,compare_models_tsne

class meta_data():
    def __init__(self):
        self.data_sources = []
        self.data_sources.append(data_source('title', 40, [3, 4, 5]))
        # self.data_sources.append(data_source('description',300,[3, 4, 5]))
        # self.data_sources.append(data_source('image_features', 4096, []))
        self.data_sources.append(data_source('image', 4096, []))
        # self.data_sources.append(data_source('image_logits', 2890, []))


        self.Nlayers = 0
        self.penalty = [] #either empty or two values
        self.lr = 0.001 #0.001
        self.dropout = 0.5 #0.5
        self.labelmap=[]
        self.train_batch_size= 2
        self.validation_batch_size= 2
        self.validation_iters = 2
        self.pos_coeff = 30
        self.num_filters = 128
        self.imsize = 224
        self.vgg_weights = 'util/vgg16_weights.npz'
        self.image_flag = False
        self.imagefeatures_flag = False
        self.text_flag = False
        self.logits_flag = False
        self.gate = 'none'
        self.gate_size = 1000
        self.optimizer = 'adam' #'adam', 'mom', by def sgd
        self.dic_f  = []
        self.trainpolicy=True
        self.policy_pos_coeff = 15
        self.policy_logits_bias = 0
        for ds in self.data_sources:
            if ds.name == 'image':
                self.image_flag = True
            if ds.name == 'image_logits':
                self.logits_flag = True
            if ds.name == 'image_features':
                self.imagefeatures_flag = True
            if ds.name == 'title' or ds.name == 'description':
                self.text_flag = True

class data_source():
    def __init__(self,name,size,filters):
        self.name = name
        self.size = size
        self.filters = filters


def write_params_file_to_disk(params):
    params_filename = join(params.get('train_dir'), 'params.txt')

    if not os.path.exists(os.path.dirname(params_filename)):
        try:
            os.makedirs(os.path.dirname(params_filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(join(params.get('train_dir'), 'params.txt'), 'w') as params_file:
        json.dump(params, params_file)


def main():
    #DataDir = 'shelf_model_20160721'
    DataDir = 'shelf_model_20160802' #_features
    parser = argparse.ArgumentParser(description='Script to train a convolutional neural net.')
    #parser.add_argument('--data_folder', help='Directory where the training and validation data are present.', default='/Users/tzahavy/git/models/shelf_model_03_30_16_1')
    parser.add_argument('--data_folder', help='Directory where the training and validation data are present.',
                        default='/Users/tzahavy/git/models/'+DataDir)
    #parser.add_argument('--label_mapping', help='Location of the label mapping file to be used for training.', default='/Users/tzahavy/git/models/shelf_model_03_30_16_1/label_mapping.json')
    parser.add_argument('--label_mapping', help='Location of the label mapping file to be used for training.',
                        default='/Users/tzahavy/git/models/'+DataDir+'/label_mapping.json')

    parser.add_argument('--dictionary_file', help='Location of the dictionary file to be used for training.', default='/Users/tzahavy/git/code/product_dictionary_embedding/data/full_dictionary_100k.msgpack')
    #parser.add_argument('--train_dir', help='Directory to which the output of the training phase is written to.', default='/Users/tzahavy/git/models/shelf_model_03_30_16_1/train_64filters_title_nce')
    parser.add_argument('--train_dir', help='Directory to which the output of the training phase is written to.',
                        default='/Users/tzahavy/git/models/save_dir/image_featuresonly_test7')

    parser.add_argument('--restore_from_ckpt', help='Provide this argument to restore from checkpoint directory.', action='store_true', default=True)
    parser.add_argument('--checkpoint_dir', help='Directory from which a checkpoint model can be restored.',
                        default='/Users/tzahavy/git/models/'+DataDir+'/train_title_batch300') #train_title_batch300 train_f_batch128_imageonly vgg_mom_16_lr001
    parser.add_argument('--embedding_size', type=int, help='Size of the word embedding vectors to be used.', default=100)
    parser.add_argument('--max_iterations', type=int, help='Maximum number of iterations to run.', default=50000)

    args = parser.parse_args()
    params = {}
    params['data_folder'] = args.data_folder
    params['train_filenames'] = []
    params['validation_filenames'] = []

    count = 1
    while os.path.isfile(join(params.get('data_folder'), 'features_train_{}.bin'.format(count))):
        params.get('train_filenames').append(join(params.get('data_folder'), 'features_train_{}.bin'.format(count)))
        count += 1

    count = 1
    while os.path.isfile(join(params.get('data_folder'), 'features_validation_{}.bin'.format(count))):
        params.get('validation_filenames').append(join(params.get('data_folder'), 'features_validation_{}.bin'.format(count)))
        count += 1

    print 'There are {} training files'.format(len(params.get('train_filenames')))
    print 'There are {} validation files'.format(len(params.get('validation_filenames')))

    params['train_dir'] = args.train_dir
    params['checkpoint_dir'] = args.checkpoint_dir

    params['label_embedding_file'] = args.label_mapping
    params['dictionary_file'] = args.dictionary_file

    dictionary = DocDictionary()

    with open(params.get('dictionary_file'), 'rb') as f:
        dictionary.load(f)

    vocab_size = dictionary.n_index
    print 'vocab size: {}'.format(vocab_size)

    with open(params.get('label_embedding_file'), 'rb') as f:
         label_embedding = json.load(f)


    network_spec = simple_conv_word_embedding

    params['size_embedding'] = args.embedding_size
    params['max_iterations'] = args.max_iterations
    params['restore_from_ckpt'] = args.restore_from_ckpt
    params['MOVING_AVERAGE_DECAY'] = network_spec.MOVING_AVERAGE_DECAY

    write_params_file_to_disk(params)
    metadata = meta_data()
    metadata.dic_f = params.get('dictionary_file')
    metadata.labelmap = label_embedding
    metadata.num_classes = len(label_embedding)
    mode = 3
    metadata.vgg_checkpoint = '/Users/tzahavy/git/models/'+DataDir+'/vgg_mom_16_lr001'
    use_old  = False
    if mode == 0:
        train(params.get('max_iterations'), train_dir=params.get('train_dir'), restore_from_ckpt=args.restore_from_ckpt, checkpoint_dir=params.get('checkpoint_dir'),
          train_filenames=params.get('train_filenames'), validation_filenames=params.get('validation_filenames'), network_spec=network_spec,
          size_embedding=params.get('size_embedding'), vocab_size=vocab_size, metadata=metadata,dictionary=dictionary)
    elif mode == 1:
        test_label_from_ckpt(checkpoint_dir=params.get('checkpoint_dir'),val_size=500,metadata=metadata,fromfile=use_old,
                             validation_filenames=params.get('validation_filenames'), network_spec=network_spec,
                             size_embedding=params.get('size_embedding'), vocab_size=vocab_size, dictionary=dictionary)
    elif mode == 2:
        write_features(train_dir=params.get('train_dir'),output_dir=params['data_folder']+'_predictions', train_filenames=params.get('train_filenames'),
                       validation_filenames=params.get('validation_filenames'), network_spec=network_spec,
                       metadata=metadata,dictionary=dictionary)
    elif mode ==3 :
        train_vgg(params.get('max_iterations'), train_dir=params.get('train_dir'),
                  restore_from_ckpt=args.restore_from_ckpt, checkpoint_dir=params.get('checkpoint_dir'),
                  train_filenames=params.get('train_filenames'),
                  validation_filenames=params.get('validation_filenames'), network_spec=network_spec,
                  size_embedding=params.get('size_embedding'), vocab_size=vocab_size, metadata=metadata,
                  dictionary=dictionary)
    else:
        compare_models_tsne('50000title','50000image_90')
if __name__ == "__main__":
    main()

