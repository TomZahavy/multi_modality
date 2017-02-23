from productdict.createdict import DocDictionary
from parsing.parse import parse_sentence_padd
import tensorflow as tf
import urllib
import json


ATTRIBUTE_TYPE = 'attr_type'
TEXT = 'text'
IMAGE = 'image'

TRANSFORM_TYPE = 'transform_type'
RAW = 'raw'
EMBEDDING_PADDED = 'embedding_padded'
FLIPPED_EMBEDDING_PADDED = 'flipped_embedding_padded'
PRESENCE = 'presence'
EXTRA_EMBEDDING_FEATURES = 'embedding_extra'

JOIN_VALUES = 'join_values'
FIELDS = 'attr_ids'
SIZE = 'size'
DICTIONARY_SIZE = 'dictionary_size'
NAME = 'name'


class DataPreparation(object):
    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _get_field(fields, product, join_values):
        values = []
        for field in fields:
            d = product.get(field)
            if d is not None:
                values.append(d)
        if not values:
            data = product.get('data')
            if data is None:
                return None
            for entry in data:
                if entry.get('attr_id') in fields:
                    values.append(entry.get('value'))
        if values:
            if join_values:
                return ' '.join(values)
            else:
                return values[0]
        return None

    @staticmethod
    def new_resolution_preserving_aspect_ratio(image_height, image_width, target_height, target_width):
        image_resolution = float(image_width)/image_height
        target_resolution = float(target_width)/target_height

        if target_resolution > image_resolution:
            new_width = (image_width*target_height)/image_height
            new_height = target_height
        else:
            new_width = target_width
            new_height = (image_height*target_width)/image_width

        return new_height, new_width

    @staticmethod
    def resize_image_preserving_aspect_ratio(image, target_height, target_width):
        image_height = image.shape[0]
        image_width = image.shape[1]
        new_height, new_width = DataPreparation.new_resolution_preserving_aspect_ratio(image_height, image_width, target_height, target_width)

        resized_image = tf.image.resize_images(tf.convert_to_tensor(image), new_height, new_width)
        resized_image = tf.image.resize_image_with_crop_or_pad(resized_image, target_height, target_width)

        with tf.Session() as sess:
            resized_image_data = sess.run([resized_image])

        return resized_image_data

    def __init__(self, dictionary_file=None, image_coder=None):
        assert dictionary_file
        assert image_coder

        self.dictionary_file = dictionary_file
        self.image_coder = image_coder

        self.dictionary = DocDictionary()
        with open(self.dictionary_file, 'rb') as f:
            self.dictionary.load(f)
        self.dictionary_size = self.dictionary.n_index

    def get_raw_text_feature(self, features, spec):
        raw_text = DataPreparation._get_field(fields=spec.get(FIELDS), product=features, join_values=spec.get(JOIN_VALUES))
        name = spec.get(NAME)
        assert name

        out = {
            '{}/{}/{}'.format(TEXT, name, RAW): (DataPreparation._bytes_feature(str(raw_text.encode('ascii', 'ignore').decode('ascii')) if raw_text else ''))
        }

        return out

    def get_embedding_padded_features(self, features, spec):
        raw_text = DataPreparation._get_field(fields=spec.get(FIELDS), product=features, join_values=spec.get(JOIN_VALUES))

        name = spec.get(NAME)
        assert name
        size = spec.get(SIZE)
        assert size > 0

        encoded_sentence, extra_features = parse_sentence_padd(self.dictionary, raw_text, size=size)
        tot_size_extra = extra_features.shape[0] * extra_features.shape[1]

        out = {
            '{}/{}/{}'.format(TEXT, name, SIZE): DataPreparation._int64_feature(size),
            '{}/{}/{}'.format(TEXT, name, DICTIONARY_SIZE): DataPreparation._int64_feature(self.dictionary_size),
            '{}/{}/{}'.format(TEXT, name, EMBEDDING_PADDED): tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_sentence)),
            '{}/{}/{}'.format(TEXT, name, EXTRA_EMBEDDING_FEATURES): tf.train.Feature(float_list=tf.train.FloatList(value=extra_features.reshape((tot_size_extra,))))
        }

        return out

    def get_flipped_embedding_padded_features(self, features, spec):
        name = spec.get(NAME)
        assert name
        size = spec.get(SIZE)
        assert size > 0

        raw_text = DataPreparation._get_field(fields=spec.get(FIELDS), product=features, join_values=spec.get(JOIN_VALUES))

        encoded_sentence, extra_features = parse_sentence_padd(self.dictionary, raw_text, size=size, pad_in_front=True)
        tot_size_extra = extra_features.shape[0] * extra_features.shape[1]

        out = {
            '{}/{}/{}'.format(TEXT, name, SIZE): DataPreparation._int64_feature(size),
            '{}/{}/{}'.format(TEXT, name, DICTIONARY_SIZE): DataPreparation._int64_feature(self.dictionary_size),
            '{}/{}/{}'.format(TEXT, name, EMBEDDING_PADDED): tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_sentence)),
            '{}/{}/{}'.format(TEXT, name, EXTRA_EMBEDDING_FEATURES): tf.train.Feature(float_list=tf.train.FloatList(value=extra_features.reshape((tot_size_extra,))))
        }

        return out

    def get_presence_features(self, features, spec):
        name = spec.get(NAME)
        assert name

        raw_text = DataPreparation._get_field(fields=spec.get(FIELDS), product=features, join_values=True)

        value = 0
        if raw_text is not None:
            value = 1

        out = {
            '{}/{}/{}'.format(TEXT, name, PRESENCE): DataPreparation._int64_feature(value)
        }

        return out

    def get_image_features(self, features, spec):
        image_file = DataPreparation._get_field(fields=spec.get(FIELDS), product=features, join_values=True)
        assert isinstance(spec.get(SIZE), list)
        name = spec.get(NAME)
        size = spec.get(SIZE)
        response = urllib.urlopen(image_file)
        code = response.getcode()
        if code != 200:
            print 'Could not find image for product {}'.format(features.get('product_ids')[0])
            image_data = ''
            # raise Exception('wrong return code: {}'.format(code))
        else:
            image_data = response.read()
            image = self.image_coder.decode_jpeg(image_data)
            assert len(image.shape) == 3
            # print 'Height = {}'.format(image.shape[0])
            # print 'Width = {}'.format(image.shape[1])
            assert image.shape[2] == 3
            resized_image_data = DataPreparation.resize_image_preserving_aspect_ratio(image, size[0], size[1])

            # print 'Original shape = {}x{}, Resized shape = {}x{}'.format(image.shape[0], image.shape[1], resized_image_data.shape[0], resized_image_data.shape[1])
            image_data = self.image_coder.encode_jpeg(resized_image_data)

        out = {
            '{}/{}/{}'.format(IMAGE, name, 'encoded'): (DataPreparation._bytes_feature(image_data))
        }

        return out

    def build_example(self, features, label, data_spec):
        feature_data = dict()
        feature_data['label'] = DataPreparation._int64_feature(int(label))

        for spec in data_spec:
            attr_type = spec.get(ATTRIBUTE_TYPE)

            if attr_type == TEXT:
                tform_type = spec.get(TRANSFORM_TYPE)
                if tform_type == RAW:
                    feature_data.update(self.get_raw_text_feature(features, spec))
                elif tform_type == EMBEDDING_PADDED:
                    feature_data.update(self.get_embedding_padded_features(features, spec))
                elif tform_type == FLIPPED_EMBEDDING_PADDED:
                    feature_data.update(self.get_flipped_embedding_padded_features(features, spec))
                elif tform_type == PRESENCE:
                    feature_data.update(self.get_presence_features(features, spec))
                else:
                    raise Exception('unsupported {}: {}'.format(TRANSFORM_TYPE, tform_type))
            elif attr_type == IMAGE:
                feature_data.update(self.get_image_features(features, spec))
            else:
                raise Exception('unsupported {}: {}'.format(ATTRIBUTE_TYPE, attr_type))
        example = tf.train.Example(features=tf.train.Features(feature=feature_data))
        return example

    def build_example_with_multiple_labels(self, features, labels, data_spec):
        feature_data = dict()
        feature_data['label'] = DataPreparation._bytes_feature(json.dumps(labels))

        for spec in data_spec:
            attr_type = spec.get(ATTRIBUTE_TYPE)

            if attr_type == TEXT:
                tform_type = spec.get(TRANSFORM_TYPE)
                if tform_type == RAW:
                    feature_data.update(self.get_raw_text_feature(features, spec))
                elif tform_type == EMBEDDING_PADDED:
                    feature_data.update(self.get_embedding_padded_features(features, spec))
                elif tform_type == FLIPPED_EMBEDDING_PADDED:
                    feature_data.update(self.get_flipped_embedding_padded_features(features, spec))
                elif tform_type == PRESENCE:
                    feature_data.update(self.get_presence_features(features, spec))
                else:
                    raise Exception('unsupported {}: {}'.format(TRANSFORM_TYPE, tform_type))
            elif attr_type == IMAGE:
                feature_data.update(self.get_image_features(features, spec))
            else:
                raise Exception('unsupported {}: {}'.format(ATTRIBUTE_TYPE, attr_type))
        example = tf.train.Example(features=tf.train.Features(feature=feature_data))
        return example

