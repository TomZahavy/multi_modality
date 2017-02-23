from os.path import join
import tensorflow as tf
import json
from psclient import has_required_data
import os
import random
from data.data_preparation import DataPreparation
from data.image_coder import ImageCoder

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
TEXT = 'text'
IMAGE = 'image'

PROB_VALIDATION = 0.1


def random_toss(probability):
    return random.random() < probability


def _json_from_file_generator(filename):
    with open(filename, 'rb') as f:
        for line in f:
            if line:
                yield json.loads(line)


def write_batch(examples, output_file_template, processed_ids, processed_ids_file):
    count = 1
    while os.path.isfile(output_file_template.format(count)):
        count += 1

    writer = tf.python_io.TFRecordWriter(output_file_template.format(count))
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()

    with open(processed_ids_file, 'wb') as f:
        for id in processed_ids:
            f.write('{}\n'.format(id))


def main():
    features_file = '/home/akrish6/data/shelf_data_20160708/products.json'
    tags_file = '/home/akrish6/data/shelf_data_20160708/tags.json'
    dictionary_file = '/home/akrish6/code/product_dictionary_embedding/data/full_dictionary_100k.msgpack'
    output_folder = '/home/akrish6/models/shelf_model_20160727'

    batch_size = 1000

    write_tf_features_from_raw_json(features_file=features_file, tags_file=tags_file, dictionary_file=dictionary_file,
                                    output_folder=output_folder, batch_size=batch_size)


def write_tf_features_from_raw_json( features_file, tags_file, dictionary_file, output_folder,
                                     data_spec=[{"attr_ids": ['product_name'], "attr_type": TEXT, "transform_type": RAW, "name":"title", SIZE: 40},
                                                {"attr_ids": ["product_long_description", "product_short_description", "product_medium_description", 'product_name', 'Product Name'], "attr_type": TEXT, "transform_type": RAW, "name":"description", SIZE: 300, JOIN_VALUES: True},
                                                {"attr_ids": ["actual_color"], "attr_type": TEXT, "transform_type": RAW, "name": "color", "required": False},
                                                {"attr_ids": ["item_class_id"], "attr_type": TEXT, "transform_type": RAW, "name": "item_class_id", "required": False},
                                                {"attr_ids": ["ISBN"], "attr_type": TEXT, "transform_type": RAW, "name": "isbn", "required": False},
                                                {"attr_ids": ["genre_id"], "attr_type": TEXT, "transform_type": RAW, "name": "genre_id", "required": False},
                                                {"attr_ids": ["artist_id"], "attr_type": TEXT, "transform_type": RAW, "name": "artist_id", "required": False},
                                                {"attr_ids": ["publisher"], "attr_type": TEXT, "transform_type": RAW, "name": "publisher", "required": False},
                                                {"attr_ids": ["literary_genre"], "attr_type": TEXT, "transform_type": RAW, "name": "literary_genre", "required": False},
                                                {"attr_ids": ["mpaa_rating"], "attr_type": TEXT, "transform_type": RAW, "name": "mpaa_rating", "required": False},
                                                {"attr_ids": ["actors"], "attr_type": TEXT, "transform_type": RAW, "name": "actors", "required": False},
                                                {"attr_ids": ["aspect_ratio"], "attr_type": TEXT, "transform_type": RAW, "name": "aspect_ratio", "required": False},
                                                {"attr_ids": ["synopsis"], "attr_type": TEXT, "transform_type": RAW, "name": "synopsis", "required": False},
                                                {"attr_ids": ["recommended_use"], "attr_type": TEXT, "transform_type": RAW, "name": "recommended_use", "required": False},
                                                {"attr_ids": ["recommended_room"], "attr_type": TEXT, "transform_type": RAW, "name": "recommended_room", "required": False},
                                                {"attr_ids": ["recommended_location"], "attr_type": TEXT, "transform_type": RAW, "name": "recommended_location", "required": False},
                                                {"attr_ids": ['image_url_primary'], "attr_type": IMAGE, "name": "Primary Image", SIZE: [500, 500]}
                                                ],
                                     batch_size=1000, output_training_file_template='features_train_{}.bin', output_validation_file_template='features_validation_{}.bin'):

    generator_tags = _json_from_file_generator(tags_file)

    image_coder = ImageCoder()
    preparer = DataPreparation(dictionary_file, image_coder)

    data_spec_file = join(output_folder, 'data_spec.json')
    if os.path.isfile(data_spec_file):
        with open(data_spec_file, 'rb') as f:
            data_spec_load = json.load(f)
        if data_spec_load != data_spec:
            raise Exception('mismatch with stored and given dataspec: {}, {}'.format(data_spec, data_spec_load))
    else:
        with open(data_spec_file, 'wb') as f:
            json.dump(data_spec, f)

    label_mapping_file = join(output_folder, 'label_mapping.json')
    if os.path.isfile(label_mapping_file):
        with open(label_mapping_file, 'rb') as f:
            label_encoding = json.load(f)
    else:
        label_encoding = dict()
        for tags in generator_tags:
            found_tag = False
            for tag in tags:
                if tag.get('decision') == 'is':
                    value_id = tag.get('value_id')
                    found_tag = True
                    break
            if not found_tag:
                continue
            if value_id not in label_encoding:
                label_encoding[value_id] = len(label_encoding)
        with open(label_mapping_file, 'wb') as f:
            json.dump(label_encoding, f)
    print 'there are {} unique labels'.format(len(label_encoding))

    processed_ids = set()
    processed_ids_file = join(output_folder, 'processed_ids.txt')
    if os.path.isfile(processed_ids_file):
        with open(processed_ids_file, 'rb') as f:
            for line in f:
                id = line.strip()
                if id:
                    processed_ids.add(id)
    print 'there are {} ids already processed'.format(len(processed_ids))

    generator_tags = _json_from_file_generator(tags_file)
    generator_features = _json_from_file_generator(features_file)

    count = 0
    exception_count = 0
    stored_train = 0
    batch_examples_train = []
    stored_validation = 0
    batch_examples_validation = []
    for features in generator_features:
        count += 1
        ids = features.get('product_ids')
        if count % 1000 == 0:
            print 'processed {} products, stored as train: {}, failed due to exceptions: {}'.format(count, stored_train, exception_count)
            print 'Ids: {}'.format(ids)

        tags = generator_tags.next()

        found_id = False
        for id in ids:
            if id in processed_ids:
                found_id = True
                break
        if found_id:
            continue
        processed_ids.update(ids)

        if not has_required_data(features, data_spec):
            continue

        found_tag = False
        for tag in tags:
            if tag.get('decision') == 'is':
                value_id = tag.get('value_id')
                found_tag = True
                break
        if not found_tag:
            continue

        try:
            example = preparer.build_example_with_multiple_labels(features, tags, data_spec)
        except Exception as e:
            print e
            exception_count += 1
            continue

        is_val = random_toss(PROB_VALIDATION)

        if is_val:
            stored_validation += 1
            batch_examples_validation.append(example)
        else:
            stored_train += 1
            batch_examples_train.append(example)

        if len(batch_examples_train) >= batch_size:
            write_batch(batch_examples_train, join(output_folder, output_training_file_template), processed_ids, processed_ids_file)
            batch_examples_train = []

        if len(batch_examples_validation) >= batch_size:
            write_batch(batch_examples_validation, join(output_folder, output_validation_file_template), processed_ids, processed_ids_file)
            batch_examples_validation = []

    if len(batch_examples_train) >= 0:
        write_batch(batch_examples_train, join(output_folder, output_training_file_template), processed_ids, processed_ids_file)

    if len(batch_examples_validation) >= 0:
        write_batch(batch_examples_validation, join(output_folder, output_validation_file_template), processed_ids, processed_ids_file)

    print 'processed {} products, stored {} training products, stored {} validation products, {} products encountered exceptions'.format(count, stored_train, stored_validation, exception_count)

if __name__ == "__main__":
    main()
