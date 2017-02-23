import json
import os
from os.path import join
import random
import tensorflow as tf
import urllib
from psclient import has_required_data
from image_coder import ImageCoder


FIELDS = 'attr_ids'
SIZE = 'size'

PROB_VALIDATION = 0.1


def random_toss(probability):
    return random.random() < probability


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _json_from_file_generator(filename):
    with open(filename, 'rb') as f:
        for line in f:
            if line:
                yield json.loads(line)


def _get_field(fields, product):
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
    if len(values) > 0:
        return values[0]
    return None


def get_image_features(features, spec, coder):
    image_file = _get_field(fields=spec.get(FIELDS), product=features)
    response = urllib.urlopen(image_file)
    code = response.getcode()
    if code != 200:
        raise Exception('wrong return code: {}'.format(code))
    print 'code: {}'.format(code)
    image_data = response.read()

    assert isinstance(spec.get(SIZE), list)
    size = spec.get(SIZE)

    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    # print 'Height = {}'.format(image.shape[0])
    # print 'Width = {}'.format(image.shape[1])
    assert image.shape[2] == 3
    resized_image = tf.image.resize_image_with_crop_or_pad(tf.convert_to_tensor(image), size[0], size[1])
    print 'Original shape = {}x{}, Resized shape = {}x{}'.format(image.shape[0], image.shape[1], tf.shape(resized_image)[0], tf.shape(resized_image)[1])
    image_data = coder.encode_jpeg(resized_image)
    return image_data


def build_example(features, label, data_spec, coder):
    feature_data = dict()
    feature_data['label'] = _int64_feature(int(label))

    for spec in data_spec:
        attr_type = spec.get("attr_type")

        if attr_type == "image":
            image = get_image_features(features=features, spec=spec, coder=coder)

            if image:
                feature_data.update({'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))})
                example = tf.train.Example(features=tf.train.Features(feature=feature_data))
                return example

    return None


def write_tf_features_from_image(features_file, tags_file, output_folder,
                                 data_spec=[{"attr_ids": ['image_url_primary'], "attr_type": "image", "name": "Primary Image", SIZE: [500, 500]}],
                                 batch_size=1000, output_training_file_template='features_images_train_{}.bin', output_validation_file_template='features_images_validation_{}.bin'):
    generator_tags = _json_from_file_generator(tags_file)
    coder = ImageCoder()

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
    stored_train = 0
    batch_examples_train = []
    stored_validation = 0
    batch_examples_validation = []
    for features in generator_features:
        count += 1
        if count % 1000 == 0:
            print 'processed {} products, stored: {}'.format(count, stored_train)

        tags = generator_tags.next()

        ids = features.get('product_ids')
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

        example = build_example(features, label_encoding[value_id], data_spec, coder)

        if not example:
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

    print 'processed {} products, stored {} training products, stored {} validation products'.format(count, stored_train, stored_validation)


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
    features_file = '/home/akrish6/data/shelf_data_04_08_16/products.json'
    tags_file = '/home/akrish6/data/shelf_data_04_08_16/tags.json'
    output_folder = '/home/akrish6/models/shelf_model_05_18_16'

    batch_size = 10000

    write_tf_features_from_image(features_file=features_file, tags_file=tags_file, output_folder=output_folder, batch_size=batch_size)


if __name__ == "__main__":
    main()
