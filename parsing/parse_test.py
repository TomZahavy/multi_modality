__author__ = 'akrish6'

import os
from os.path import join
import tensorflow as tf
import numpy as np
from productdict.createdict import DocDictionary
import util
import data.preparation as prep
import conv_net.simple_conv_word_embedding as scwe


class PBParseTest(tf.test.TestCase):

    def setUp(self):
        self.product_name = {"data": [{"locale": "en_US", "attr_id": "product_name", "attr_name": "Product Name", "value": "NEXTE Jewelry Red, Green and Yellow CZ 3-piece Stackable Ring Set Goldtone Size 7"}], "product_ids": ["4XKG72BK9D05"]}

    def get_ground_truth(self, product_name, label, spec, dictionary, size):
        raw_text = prep._get_field(fields=spec.get(prep.FIELDS), product=product_name, join_values=spec.get(prep.JOIN_VALUES))
        encoded_sentence, extra_features = prep.parse_sentence_padd(dictionary, raw_text, size=size)
        sentence = tf.constant(encoded_sentence, shape=(size,))
        extra = tf.constant(extra_features, shape=(size, 3))
        label = tf.constant(label, shape=(1,))
        sentence = tf.cast(sentence, tf.int32)
        extra = tf.cast(extra, tf.float32)
        label = tf.cast(label, tf.int32)
        return sentence, label, extra

    def test_data_encode_and_decode(self):
        with self.test_session() as sess:
            tmpdir = tf.test.get_temp_dir()
            dictionary = DocDictionary()
            dictionary_file = util.get_full_filepath('full_dictionary_100k.msgpack')
            with open(dictionary_file, 'rb') as f:
                dictionary.load(f)
            dictionary_size = dictionary.n_index
            data_spec = [{"attr_ids": ["product_name"], "transform_type": prep.EMBEDDING_PADDED, "name": "title", "size": 40}]
            output_file_template = join(tmpdir, 'features_{}.bin')
            processed_ids = set()
            processed_ids.add("4XKG72BK9D05")
            processed_ids_file = join(tmpdir, 'processed_ids.txt')
            label = 1

            # Prepare example and write to protocol buffer
            example = prep.build_example(self.product_name, label, data_spec, dictionary, dictionary_size)
            prep.write_batch([example], output_file_template, processed_ids, processed_ids_file)

            filenames = []

            count = 1
            while os.path.isfile(join(tmpdir, 'features_{}.bin'.format(count))):
                filenames.append(join(tmpdir, 'features_{}.bin'.format(count)))
                count += 1

            filename_queue = tf.train.string_input_producer(filenames)
            sentence_dec, label_dec, extra_dec = scwe.read_and_decode_sentence(filename_queue, size=40)
            sentence, label, extra = self.get_ground_truth(self.product_name, label, data_spec[0], dictionary, 40)
            sentence_eq = tf.reduce_sum(tf.cast(tf.equal(sentence, sentence_dec), tf.float32))
            label_eq = tf.reduce_sum(tf.cast(tf.equal(label, label_dec), tf.float32))
            extra_eq = tf.reduce_sum(tf.cast(tf.equal(extra, extra_dec), tf.float32))

            tf.train.start_queue_runners(sess=sess)

            sentence_eq, label_eq, extra_eq = sess.run([sentence_eq, label_eq, extra_eq])

            np.testing.assert_equal(sentence_eq, np.asscalar(np.array([40], dtype=np.float32)))
            np.testing.assert_equal(label_eq, np.asscalar(np.array([1], dtype=np.float32)))
            np.testing.assert_equal(extra_eq, np.asscalar(np.array([120], dtype=np.float32)))

    def tearDown(self):
        tmpdir = tf.test.get_temp_dir()

        for the_file in os.listdir(tmpdir):
            file_path = os.path.join(tmpdir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e

    if __name__ == "__main__":
        tf.test.main()
