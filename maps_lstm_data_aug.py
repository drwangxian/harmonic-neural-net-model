"""
"Harmonic Structure-Based Neural Network Model for Music Pitch Detection," Submitted to
IEEE ICMLA 2020 for review.

This is the accompanying code for the above paper.
Train and test the harmonic acoustic model on the MAPS dataset.
Data augmentation and LSTM are used.

Configure the code by setting the variables in class Config.

Abbreviations:
tb = tensorboard
tf = tensorflow
"""


DEBUG = True  # True to run in debug mode that uses less recordings and only runs few epochs.
GPU_ID = 0  # If you have multiple GPUs, you can choose which one to use. The gpu index starts from 0 rather than 1.


import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import re

from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import datetime
import magenta.music
import soundfile
import mido


class MiscFns(object):
    """Miscellaneous functions"""

    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id."""
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def times_to_frames_fn(sr, spec_stride, start_time, end_time):
        """start and times in seconds to start and end frames"""

        assert sr in (16000, 44100)
        spec_stride = int(spec_stride)
        assert spec_stride == 512 if sr == 16000 else 22 * 64
        assert spec_stride & 1 == 0
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride
        return start_frame, end_frame + 1

    @staticmethod
    def label_fn(sr, mid_file_name, num_frames, spec_stride):
        """construct ground-truth labels from midi file"""
        assert sr in (16000, 44100)
        if sr == 16000:
            assert spec_stride == 512
        else:
            assert spec_stride == 22 * 64
        spec_stride = int(spec_stride)
        frame_matrix = np.zeros((num_frames, 88), dtype=np.bool_)
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)
        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            note_start_frame, note_end_frame = MiscFns.times_to_frames_fn(
                sr=sr,
                spec_stride=spec_stride,
                start_time=note.start_time,
                end_time=note.end_time
            )
            frame_matrix[note_start_frame:note_end_frame, note.pitch - 21] = True

        return frame_matrix

    @staticmethod
    def harmonic_acoustic_model(spec_batch, is_training, trainable, use_feature):
        """the harmonic acoustic model"""

        assert tf.get_variable_scope().name != ''
        spec_batch.set_shape([None, None, 336])
        assert all(isinstance(v, bool) for v in (is_training, trainable, use_feature))

        outputs = spec_batch[..., None]
        outputs = slim.conv2d(
            scope='C_0',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )

        outputs = slim.conv2d(
            scope='C_1',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_1', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.conv2d(
            scope='C_2',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_2', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.conv2d(
            scope='C_3',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_3', inputs=outputs, keep_prob=.8, is_training=is_training)

        _num_hd_features = 256
        outputs = slim.harmonic_dense(
            scope='HD_4',
            inputs=outputs,
            harmonic_bins=MiscFns.harmonic_bins_fn(),
            num_outputs=_num_hd_features,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs.set_shape([None, None, 88 * 3, _num_hd_features])
        outputs = slim.max_pool2d(scope='MP_4', inputs=outputs, kernel_size=[1, 3], stride=[1, 3])
        outputs = slim.dropout(scope='DO_4', inputs=outputs, keep_prob=.8, is_training=is_training)
        outputs.set_shape([None, None, 88, _num_hd_features])

        outputs = slim.fully_connected(
            scope='FC_5',
            inputs=outputs,
            num_outputs=64,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_5', inputs=outputs, keep_prob=.8, is_training=is_training)
        outputs.set_shape([None, None, 88, 64])

        if not use_feature:
            outputs = slim.fully_connected(
                scope='FC_6',
                inputs=outputs,
                num_outputs=1,
                activation_fn=None,
                trainable=trainable
            )
            outputs = tf.squeeze(outputs, axis=-1)
            outputs.set_shape([None, None, 88])

        return outputs

    @staticmethod
    def split_train_valid_and_test_files_fn():
        """partition the MAPS dataset into training, validation and test splits"""

        test_dirs = ['ENSTDkCl_2/MUS', 'ENSTDkAm_2/MUS']
        train_dirs = ['AkPnBcht_2/MUS', 'AkPnBsdf_2/MUS', 'AkPnCGdD_2/MUS', 'AkPnStgb_2/MUS',
                      'SptkBGAm_2/MUS', 'SptkBGCl_2/MUS', 'StbgTGd2_2/MUS']
        maps_dir = os.environ['maps']

        test_files = []
        for directory in test_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            test_files += wav_files

        test_ids = set([MiscFns.filename_to_id(wav_file) for wav_file in test_files])
        assert len(test_ids) == 53

        training_files = []
        validation_files = []
        for directory in train_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            for wav_file in wav_files:
                me_id = MiscFns.filename_to_id(wav_file)
                if me_id not in test_ids:
                    training_files.append(wav_file)
                else:
                    validation_files.append(wav_file)

        assert len(training_files) == 139 and len(test_files) == 60 and len(validation_files) == 71

        return dict(training=training_files, test=test_files, validation=validation_files)

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):
        """cut the given number of frames into mini-examples of length snippet_len frames"""
        split_frames = range(0, num_frames + 1, snippet_len)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])

        return start_end_frame_pairs

    @staticmethod
    def load_np_array_from_file_fn(file_name):
        """load the vqt from a file"""
        with open(file_name, 'rb') as fh:
            first_line = str(fh.readline()).split()
            rec_name = first_line[0]
            dtype = first_line[1]
            dim = first_line[2:]
            dim = [int(_item) for _item in dim]
            output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)
            return rec_name, output

    @staticmethod
    def to_db_scale_fn(sg):
        """convert VQT to dB scale"""
        return 20. * np.log10(sg + 1e-10) + 200.

    @staticmethod
    def from_db_scale_fn(sg):
        """convert VQT from dB scale to normal value"""
        return 10. ** ((sg - 200.) / 20.) - 1e-10

    @staticmethod
    def num_samples_to_num_frames_fn(num_samples):
        """convert number of samples to number of frames at a hop size of 22 * 64 samples"""
        assert isinstance(num_samples, (int, long))
        num_frames = (num_samples + 63) // 64
        num_frames = (num_frames + 21) // 22

        return num_frames

    @staticmethod
    def array_to_table_tf_fn(tf_array, header, scope, title, names, precision=None):
        """tensorboard proto that displays a tensorflow array as a table"""
        tf_array = tf.convert_to_tensor(tf_array)
        assert tf_array._rank() == 2
        num_examples = tf_array.shape[0].value
        num_fields = tf_array.shape[1].value
        assert num_examples is not None
        assert num_fields is not None
        assert isinstance(header, list)
        assert len(header) == num_fields
        header = ['id', 'name'] + header
        header = tf.constant(header)
        assert isinstance(names, list)
        assert len(names) == num_examples
        names = tf.constant(names)[:, None]
        assert names.dtype == tf.string
        ids = [str(i) for i in range(1, num_examples + 1)]
        ids = tf.constant(ids)[:, None]
        if precision is None:
            if tf_array.dtype in (tf.float32, tf.float64):
                precision = 4
            else:
                precision = -1
        tf_array = tf.as_string(tf_array, precision=precision)
        tf_array = tf.concat([ids, names, tf_array], axis=1)
        tf_array.set_shape([num_examples, num_fields + 2])
        tf_array = tf.strings.reduce_join(tf_array, axis=1, separator=' | ')
        tf_array = tf.strings.reduce_join(tf_array, separator='\n')
        header = tf.strings.reduce_join(header, separator=' | ')
        sep = tf.constant(['---'])
        sep = tf.tile(sep, [num_fields + 2])
        sep = tf.strings.reduce_join(sep, separator=' | ')
        tf_array = tf.strings.join([header, sep, tf_array], separator='\n')
        assert isinstance(title, str)
        tf_array = tf.strings.join([tf.constant(title), tf_array], separator='\n\n')
        assert isinstance(scope, str)
        op = tf.summary.text(scope, tf_array)

        return op

    @staticmethod
    def cal_tps_fps_tns_fns_tf_fn(pred, target):
        """calculate numbers of true positives, false positives, true negatives and false negatives
        from predictions and targets"""
        assert pred.dtype == tf.bool and target.dtype == tf.bool
        npred = tf.logical_not(pred)
        ntarget = tf.logical_not(target)
        tps = tf.logical_and(pred, target)
        fps = tf.logical_and(pred, ntarget)
        tns = tf.logical_and(npred, ntarget)
        fns = tf.logical_and(npred, target)
        tps, fps, tns, fns = [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
        inc_tps_fps_tns_fns = tf.convert_to_tensor([tps, fps, tns, fns], dtype=tf.int64)

        return inc_tps_fps_tns_fns

    @staticmethod
    def cal_prf_tf_fn(tps, fps, fns):
        """calculate precision, recall and f-measure"""
        assert tps.dtype == tf.float64
        p = tps / (tps + fps + 1e-7)
        r = tps / (tps + fns + 1e-7)
        f = 2. * p * r / (p + r + 1e-7)
        return p, r, f

    @staticmethod
    def harmonic_bins_fn():
        """generate the list of 79 harmonic bins"""

        coord_list = [
            [-2830, -2826],
            [-2718, -2714],
            [-2664, -2660],
            [-2648, -2644],
            [-2597, -2593],
            [-2503, -2499],
            [-2213, -2207],
            [-2162, -2158],
            [-2119, -2115],
            [-1976, -1972],
            [-1783, -1779],
            [-1606, -1602],
            [-1568, -1564],
            [-1534, -1530],
            [-1511, -1508],
            [-1411, -1407],
            [-1249, -1245],
            [-1229, -1225],
            [-1193, -1189],
            [-951, -947],
            [-763, -759],
            [-711, -707],
            [-592, -588],
            [-543, -539],
            [-516, -512],
            [-503, -499],
            [-397, -393],
            [-276, -270],
            [-245, -241],
            [-173, -169],
            [-135, -131],
            [-117, -111],
            [-83, -80],
            [-52, -46],
            [0, 3],
            [81, 84],
            [141, 145],
            [262, 266],
            [365, 369],
            [392, 396],
            [406, 409],
            [457, 461],
            [538, 541],
            [549, 553],
            [612, 615],
            [711, 714]
        ]

        f0_coord = -1247.
        s = 538
        freq_bin = lambda c: int(np.round((c - f0_coord) / s * 240))

        accum_id = 0
        harmonic_bins = []
        for cor_idx in xrange(len(coord_list)):
            cor_pair = coord_list[cor_idx]
            if cor_pair[1] - cor_pair[0] == 4:
                c = cor_pair[0] + 2
                h_bin = freq_bin(c)
                harmonic_bins.append(h_bin)
                logging.debug('{} - {}'.format(accum_id + 1, h_bin))
                accum_id += 1
            elif cor_pair[1] - cor_pair[0] == 3:
                c = cor_pair[0] + 1.5
                h_bin = freq_bin(c)
                harmonic_bins.append(h_bin)
                logging.debug('{} - {}'.format(accum_id + 1, h_bin))
                accum_id += 1
            elif cor_pair[1] - cor_pair[0] == 6:
                b1 = freq_bin(cor_pair[0] + 2)
                b2 = freq_bin(cor_pair[1] - 2)
                harmonic_bins.append(b1)
                harmonic_bins.append(b2)
                logging.debug('{}&{} - {}&{}'.format(accum_id + 1, accum_id + 2, b1, b2))
                accum_id += 2
            else:
                raise ValueError('unknown distance')

        assert len(harmonic_bins) == 50

        harmonic_set = []
        for bin in harmonic_bins:
            b_floor = int(np.floor(1. * bin * 3 / 20))
            b_ceil = int(np.ceil(1. * bin * 3 / 20))
            harmonic_set.extend([b_floor, b_ceil])
        harmonic_set = set(harmonic_set)
        harmonic_set = sorted(harmonic_set)
        assert len(harmonic_set) == 79

        return harmonic_set

    @staticmethod
    def unstack_88_into_batch_dim_fn(note_dim, inputs):
        """reshape a tensor of shape (batch,  frames, 88, channels) to shape (batch * 88, frames, channels)"""
        outputs = inputs
        input_dims = outputs._rank()
        assert outputs.shape[note_dim].value == 88
        outputs = tf.unstack(outputs, axis=note_dim)
        assert len(outputs) == 88
        outputs = tf.concat(outputs, axis=0)
        output_dims = outputs._rank()
        assert input_dims - output_dims == 1

        return outputs

    @staticmethod
    def split_batch_dim_into_88_fn(note_dim, inputs):
        """reshape a tensor of shape (batch * 88, frames, channels) to shape (batch,  frames, 88, channels)"""
        outputs = inputs
        input_dims = outputs._rank()
        outputs = tf.split(value=outputs, num_or_size_splits=88, axis=0)
        assert len(outputs) == 88
        outputs = tf.stack(outputs, axis=note_dim)
        assert outputs.shape[note_dim].value == 88
        output_dims = outputs._rank()
        assert output_dims - input_dims == 1

        return outputs

    @staticmethod
    def frame_label_detector_fn(inputs, is_training):
        """the frame label detector that tops the harmonic acoustic model with an LSTM layer"""

        def _rnn_layer_fn(inputs):
            inputs.set_shape([1, None, 88, 64])
            lstm_cell = tf.nn.rnn_cell.LSTMCell(name='lstm_cell', num_units=64, dtype=tf.float32)
            outputs = MiscFns.unstack_88_into_batch_dim_fn(note_dim=2, inputs=inputs)
            outputs, _ = tf.nn.dynamic_rnn(
                scope='dy_rnn',
                cell=lstm_cell,
                inputs=outputs,
                dtype=tf.float32
            )
            outputs = MiscFns.split_batch_dim_into_88_fn(note_dim=2, inputs=outputs)
            outputs.set_shape([1, None, 88, 64])

            return outputs

        inputs.set_shape([1, None, 336])
        assert isinstance(is_training, bool)
        assert tf.get_variable_scope().name == ''

        with tf.variable_scope('frame_label_detector', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('cnn_layers'):
                cnn_features = MiscFns.harmonic_acoustic_model(
                    spec_batch=inputs,
                    is_training=is_training,
                    use_feature=True,
                    trainable=True
                )
                cnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('rnn_layer'):
                rnn_features = _rnn_layer_fn(inputs=cnn_features)
                rnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('output_layer'):
                logits = slim.fully_connected(
                    scope='FC',
                    inputs=rnn_features,
                    num_outputs=1,
                    activation_fn=None
                )
                logits = tf.squeeze(logits, axis=-1)
                logits.set_shape([1, None, 88])

        return logits

    @staticmethod
    def get_me_id_and_ins_fn(wav_or_mid_file):
        """
        example (input, output) pair:
        MAPS_MUS-alb_esp2_AkPnStgb.wav -> alb_esp2_AkPnStgb
        """
        p = re.match(r'MAPS_MUS-(.+).\w{3}$', os.path.basename(wav_or_mid_file))
        assert p is not None

        return p.group(1)


class Config(object):
    """all setting goes here"""

    def __init__(self):
        self.debug_mode = DEBUG
        self.gpu_id = GPU_ID
        self.snippet_len = 600  # number of frames per mini-example

        self.num_epochs = 20
        self.batches_per_epoch = 5000

        self.learning_rate = 1e-4

        """
        Instructions on configuring the running mode of the code.
        
        1. The code can run in inference mode or training mode.
        2. If you want to run in inference mode, set the variable inference to point to a saved model, e.g., inference = 
           os.path.join('saved_model', 'frame_model'), and set the variables from_saved and model_prefix both to None.
        3. If you want to run in training mode, set the variable inference to None and set the variable from_saved to 
           point to a saved model if you want to continue training from the saved model or otherwise set from_saved to 
           None to train from scratch.
        4. In training mode, if you want to save the trained models, you can specify a prefix by setting the variable 
           model_prefix. The trained models will be saved in ./saved_model/. In training mode, if the variable 
           model_prefix is None, the trained models will not be saved.
        5. All statistics are saved as tensorboard summaries. Set the variable tb_dir to naming the folder for storing 
           the data for tensorboard.
           
        """
        self.train_or_inference = Namespace(
            inference=None,
            from_saved=None,
            model_prefix='d0'
        )
        self.tb_dir = 'tb_d0'

        # check if tb_dir exists. In non-debug mode, this folder cannot exist beforehand, because the the data in this
        # folder may be overwritten.
        if not self.debug_mode:
            # check if tb_dir exists
            assert self.tb_dir is not None
            tmp_dirs = glob.glob('./*/')
            tmp_dirs = [s[2:-1] for s in tmp_dirs]
            assert self.tb_dir not in tmp_dirs

            # check if model exists
            if self.train_or_inference.inference is None and self.train_or_inference.model_prefix is not None:
                if os.path.isdir('./saved_model'):
                    tmp_prefixes = glob.glob('./saved_model/*')
                    prog = re.compile(r'./saved_model/(.+?)_')
                    tmp = []
                    for file_name in tmp_prefixes:
                        try:
                            prefix = prog.match(file_name).group(1)
                        except AttributeError:
                            pass
                        else:
                            tmp.append(prefix)
                    tmp_prefixes = set(tmp)
                    assert self.train_or_inference.model_prefix not in tmp_prefixes

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.gpu_config = gpu_config

        # partition the dataset into three splits
        self.tvt_split_dict = MiscFns.split_train_valid_and_test_files_fn()

        # for debug mode we use less recordings
        if self.debug_mode:
            np.random.seed(100)
            for tvt in self.tvt_split_dict.keys():
                _num = len(self.tvt_split_dict[tvt])
                _sel = np.random.choice(_num, min(5, _num), replace=False)
                self.tvt_split_dict[tvt] = [self.tvt_split_dict[tvt][ii] for ii in _sel]

            self.num_epochs = 2
            self.batches_per_epoch = 50
            self.gpu_id = 0

        if self.train_or_inference.inference is not None:
            for tvt in ('training', 'validation'):
                del self.tvt_split_dict[tvt][1:]


class Model(object):
    """
    input pipeline, neural network model, loss function, store and display statistics
    """

    def __init__(self, config, name):
        assert name in ('validation', 'training', 'test')
        self.name = name
        logging.debug('{} - model - initialize'.format(self.name))
        self.is_training = True if self.name == 'training' else False
        self.config = config
        self._gen_dataset_fn()
        self.batch = self._gen_batch_fn()

        with tf.name_scope(self.name):
            logits = self._nn_model_fn()
            logits.set_shape([1, None, 88])
            self.logits = logits
            self.loss = self._loss_fn()
            if self.is_training:
                self.training_op = self._training_op_fn()
            self.stats = self._stats_fn()
            self.tb_proto = self._tb_summary_fn()

    def _dataset_iter_fn(self):
        """dataset iterator"""

        if self.is_training:
            logging.debug('{} - enter generator'.format(self.name))
            assert hasattr(self, 'dataset')

            self._power_variation_fn()

            logging.debug('{} - generator begins'.format(self.name))
            np.random.shuffle(self.rec_shift_ch_start_end_list)

            for rec_idx, shift_idx, ch_idx, start, end in self.rec_shift_ch_start_end_list:
                rec_dict = self.dataset[rec_idx]
                yield dict(
                    spectrogram=rec_dict['sg'][start:end, :, ch_idx, shift_idx],
                    label=rec_dict['label'][start: end],
                    num_frames=end - start
                )

            logging.debug('{} - generator ended'.format(self.name))

        if not self.is_training:
            logging.debug('{} - enter generator'.format(self.name))
            assert hasattr(self, 'dataset')

            logging.debug('{} - generator begins'.format(self.name))
            for rec_idx, rec_dict in enumerate(self.dataset):
                split_list = rec_dict['split_list']
                for start_frame, end_frame in split_list:
                    yield dict(
                        spectrogram=rec_dict['sg'][start_frame:end_frame],
                        label=rec_dict['label'][start_frame:end_frame],
                        num_frames=end_frame - start_frame,
                        me_id=rec_idx
                    )
            logging.debug('{} - generator ended'.format(self.name))

    def _gen_batch_fn(self):
        """construct input pipeline with tf.data.Dataset"""

        with tf.device('/cpu:0'):
            if self.is_training:
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32),
                    output_shapes=dict(spectrogram=[None, 336], label=[None, 88], num_frames=[])
                )

                dataset = dataset.repeat()

                dataset = dataset.batch(1)

                dataset = dataset.prefetch(5)

                dataset_iter = dataset.make_one_shot_iterator()
                element = dataset_iter.get_next()

                return element
            else:  # not self.is_training
                _sg_shape = [None, 336, 2]
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(
                        spectrogram=tf.float32,
                        label=tf.bool,
                        num_frames=tf.int32,
                        me_id=tf.int32),
                    output_shapes=dict(
                        spectrogram=_sg_shape,
                        label=[None, 88],
                        num_frames=[],
                        me_id=[]
                    )
                )

                dataset = dataset.batch(1)

                dataset = dataset.prefetch(5)

                reinitializabel_iter = dataset.make_initializable_iterator()
                self.reinitializable_iter_for_dataset = reinitializabel_iter
                element = reinitializabel_iter.get_next()
                element['spectrogram'].set_shape([1] + _sg_shape)
                element['label'].set_shape([1, None, 88])
                element['num_frames'].set_shape([1])
                element['me_id'].set_shape([1])

                return element

    def _nn_model_fn(self):
        """tf function for predicting frame labels """

        inputs = self.batch['spectrogram']
        _nn_fn = MiscFns.frame_label_detector_fn

        if self.is_training:
            inputs.set_shape([1, None, 336])
        else:
            inputs.set_shape([1, None, 336, 2])

        if self.is_training:
            outputs = _nn_fn(inputs=inputs, is_training=self.is_training)
            outputs.set_shape([1, None, 88])
        else:
            inputs_list = tf.unstack(inputs, axis=-1)
            outputs_list = []
            for inputs in inputs_list:
                outputs = _nn_fn(inputs=inputs, is_training=self.is_training)
                outputs.set_shape([1, None, 88])
                outputs_list.append(outputs)
            outputs = tf.stack(outputs_list, axis=-1)
            outputs.set_shape([1, None, 88, 2])
            outputs = tf.reduce_mean(outputs, axis=-1)
            outputs.set_shape([1, None, 88])

        return outputs

    def _maps_vqt_sg_and_label_fn(self, wav_file):
        """read VQT from file and construct target labels"""

        rec_name = os.path.basename(wav_file)[:-4]
        vqt_file = os.path.join(os.environ['maps_vqt'], self.name, rec_name + '.vqt')
        _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
        # for maps, vqt is not in dB scale
        assert _rec_name == rec_name
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        num_frames = MiscFns.num_samples_to_num_frames_fn(wav_info.frames)
        if self.is_training:
            assert vqt.shape == (num_frames, 336, 2, 5)
        else:
            assert vqt.shape == (num_frames, 336, 2)
        assert vqt.dtype == np.float32

        mid_file = wav_file[:-3] + 'mid'
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * wav_info.samplerate))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames = min(num_frames, num_frames_from_midi)
        vqt = vqt[:num_frames]

        vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])

        label = MiscFns.label_fn(
            sr=wav_info.samplerate,
            mid_file_name=mid_file,
            num_frames=num_frames,
            spec_stride=64 * 22
        )
        assert label.dtype == np.bool_
        label.flags['WRITEABLE'] = False

        return dict(sg=vqt, label=label)

    def _gen_dataset_fn(self):
        """generate dataset"""

        if self.is_training:
            assert not hasattr(self, 'dataset')
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            logging.debug('{} - generate spectrograms and labels'.format(self.name))
            dataset = []

            for file_idx, wav_file_name in enumerate(file_names):
                me_id_and_ins = MiscFns.get_me_id_and_ins_fn(wav_file_name)
                logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, me_id_and_ins))
                sg_label_dict = self._maps_vqt_sg_and_label_fn(wav_file_name)  # spectrogram and labels
                sg = sg_label_dict['sg']
                assert sg.shape[1:] == (336, 2, 5)  # 336 freq. bins, 2 sound channels, 5 pitch shifts
                label = sg_label_dict['label']
                dataset.append(dict(sg=sg, label=label))

            self.dataset = dataset

            rec_shift_ch_start_end_list = []
            for rec_idx in xrange(num_recs):
                split_list = MiscFns.gen_split_list_fn(
                    num_frames=len(self.dataset[rec_idx]['sg']),
                    snippet_len=self.config.snippet_len
                )
                for shift_idx in xrange(5):
                    for ch_idx in xrange(2):
                        l = [[rec_idx, shift_idx, ch_idx, s[0], s[1]] for s in split_list]
                        rec_shift_ch_start_end_list.extend(l)
            self.rec_shift_ch_start_end_list = rec_shift_ch_start_end_list

            num_batches_per_iter = len(self.rec_shift_ch_start_end_list)
            logging.info('number of batches per iteration over the dataset - {}'.format(num_batches_per_iter))

        if not self.is_training:
            assert not hasattr(self, 'dataset')
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            logging.debug('{} - generate spectrograms and labels'.format(self.name))
            dataset = []
            me_ids = []

            for file_idx, wav_file_name in enumerate(file_names):
                me_id_and_ins = MiscFns.get_me_id_and_ins_fn(wav_file_name)
                me_ids.append(me_id_and_ins)
                logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, me_id_and_ins))
                sg_label_dict = self._maps_vqt_sg_and_label_fn(wav_file_name)
                sg = sg_label_dict['sg']
                assert sg.shape[1:] == (336, 2)  # 336 feq. bins, 2 sound channels
                sg = MiscFns.to_db_scale_fn(sg)
                label = sg_label_dict['label']
                dataset.append(dict(
                    sg=sg,
                    label=label
                ))
            self.dataset = dataset
            self.num_frames_vector = np.asarray([len(rec_dict['sg']) for rec_dict in self.dataset], dtype=np.int64)
            self.me_ids = tuple(me_ids)

            for rec_dict in self.dataset:
                split_list = MiscFns.gen_split_list_fn(
                    num_frames=len(rec_dict['sg']),
                    snippet_len=self.config.snippet_len
                )
                rec_dict['split_list'] = split_list

    def _power_variation_fn(self):
        """data augmentation by power variation"""

        assert hasattr(self, 'dataset')
        assert hasattr(self, 'rec_shift_ch_start_end_list')
        num_recs = len(self.dataset)
        num_shifts = 5
        num_chs = 2
        if not hasattr(self, 'previous_power_levels'):
            self.previous_power_levels = None
        power_range = np.linspace(.9, 1.1, 11, dtype=np.float32)
        current_power_levels = np.random.choice(power_range, [num_recs, 336, num_chs, num_shifts])
        for rec_idx in xrange(num_recs):
            if self.previous_power_levels is not None:
                sg = self.dataset[rec_idx]['sg']
                sg = MiscFns.from_db_scale_fn(sg)
                pg = current_power_levels[rec_idx:rec_idx + 1] / self.previous_power_levels[rec_idx:rec_idx + 1]
                sg = sg * pg
                sg = MiscFns.to_db_scale_fn(sg)
                self.dataset[rec_idx]['sg'] = sg
            else:
                sg = self.dataset[rec_idx]['sg']
                sg = sg * current_power_levels[rec_idx:rec_idx + 1]
                sg = MiscFns.to_db_scale_fn(sg)
                self.dataset[rec_idx]['sg'] = sg
        self.previous_power_levels = current_power_levels

    def _loss_fn(self):
        """calculate loss"""
        logits = self.logits
        logits.set_shape([1, None, 88])
        logits = tf.squeeze(logits, axis=0)
        labels = self.batch['label']
        labels.set_shape([1, None, 88])
        labels = tf.squeeze(labels, axis=0)
        labels = tf.cast(labels, tf.float32)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

        return loss

    def _training_op_fn(self):
        """generate training operator"""
        loss = self.loss
        if self.is_training:
            _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if _update_ops:
                with tf.control_dependencies(_update_ops):
                    training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
            else:
                training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
            return training_op
        else:
            return None

    def _stats_fn(self):
        """calculate statistics"""

        if not self.is_training:
            assert tf.get_variable_scope().name == ''
            num_recs = len(self.dataset)
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_ind_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_ind_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[num_recs, 4],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels = tf.greater(tf.squeeze(self.logits, axis=0), 0.)
                    target_labels = self.batch['label']
                    target_labels.set_shape([1, None, 88])
                    target_labels = tf.squeeze(target_labels, axis=0)
                    inc = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels,
                        target=target_labels
                    )
                    assert inc.dtype == tf.int64

                    num_labels = tf.cast(self.batch['num_frames'][0], tf.int64) * tf.constant(88, dtype=tf.int64)
                    _num_labels = tf.reduce_sum(inc)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ind_update_op = tf.scatter_add(
                            var_int64_ind_tps_fps_tns_fns,
                            self.batch['me_id'][0],
                            inc
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(self.loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, dtype=tf.int64))

                    # ind and average stats
                    tps, fps, _, fns = tf.unstack(tf.cast(var_int64_ind_tps_fps_tns_fns, tf.float64), axis=1)
                    ps, rs, fs = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ind_prfs = tf.stack([ps, rs, fs], axis=1)
                    ind_prfs.set_shape([num_recs, 3])
                    _num_labels = self.num_frames_vector
                    assert isinstance(_num_labels, np.ndarray) and _num_labels.dtype == np.int64
                    _num_labels = tf.constant(_num_labels) * tf.constant(88, dtype=tf.int64)
                    _num_labels_p = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=1)
                    _assert_op = tf.assert_equal(_num_labels, _num_labels_p)
                    with tf.control_dependencies([_assert_op]):
                        ave_prf = tf.reduce_mean(ind_prfs, axis=0)

                    # ensemble stats
                    ensemble = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=0)
                    tps, fps, _, fns = tf.unstack(tf.cast(ensemble, tf.float64))
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    with tf.control_dependencies([_assert_op]):
                        en_prf_and_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ind_update_op, batch_counter_update_op, acc_loss_update_op)

                    stats = dict(
                        individual_tps_fps_tns_fns=var_int64_ind_tps_fps_tns_fns,
                        individual_prfs=ind_prfs,
                        average_prf=ave_prf,
                        ensemble_tps_fps_tns_fns=ensemble,
                        ensemble_prf_and_loss=en_prf_and_loss
                    )
            return dict(update_op=update_op, value=stats)

        if self.is_training:
            logits = self.logits
            loss = self.loss
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[4],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels_flattened = tf.squeeze(logits, axis=0)
                    pred_labels_flattened.set_shape([None, 88])
                    pred_labels_flattened = tf.greater(pred_labels_flattened, 0.)

                    target_labels_flattened = self.batch['label']
                    target_labels_flattened.dtype == tf.bool
                    target_labels_flattened = tf.squeeze(target_labels_flattened, axis=0)
                    target_labels_flattened.set_shape([None, 88])
                    inc_tps_fps_tns_fns = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels_flattened,
                        target=target_labels_flattened
                    )
                    assert inc_tps_fps_tns_fns.dtype == tf.int64
                    num_labels = tf.reduce_sum(self.batch['num_frames'])
                    num_labels = tf.cast(num_labels, tf.int64) * tf.constant(88, tf.int64)
                    _num_labels = tf.reduce_sum(inc_tps_fps_tns_fns)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ensemble_tps_fps_tns_fns_update_op = tf.assign_add(
                            var_int64_tps_fps_tns_fns,
                            inc_tps_fps_tns_fns
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, tf.int64))

                    en_tps_fps_tns_fns_float64 = tf.cast(var_int64_tps_fps_tns_fns, tf.float64)
                    tps, fps, _, fns = tf.unstack(en_tps_fps_tns_fns_float64)
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    prf_and_ave_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ensemble_tps_fps_tns_fns_update_op, acc_loss_update_op, batch_counter_update_op)
                    stats = dict(tps_fps_tns_fns=var_int64_tps_fps_tns_fns, prf_and_loss=prf_and_ave_loss)

            return dict(update_op=update_op, value=stats)

    def _tb_summary_fn(self):
        """store statistics as tb summaries"""

        if self.is_training:
            scalar_summaries = []
            with tf.name_scope('statistics'):
                stats = self.stats['value']
                p, r, f, l = tf.unstack(stats['prf_and_loss'])
                summary_dict = dict(precision=p, recall=r, f1=f, loss=l)
                for sum_name, sum_value in summary_dict.iteritems():
                    scalar_summaries.append(tf.summary.scalar(sum_name, sum_value))
                scalar_summaries = tf.summary.merge(scalar_summaries)
            return scalar_summaries

        if not self.is_training:
            num_recs = len(self.dataset)
            tb_table_and_scalar_protos = []
            with tf.name_scope('statistics'):
                stats = self.stats['value']
                ind_prfs = stats['individual_prfs']
                ave_prf = stats['average_prf']
                en_prf, ave_loss = tf.split(stats['ensemble_prf_and_loss'], [3, 1])
                ave_loss = ave_loss[0]

                assert ind_prfs.dtype == ave_prf.dtype == en_prf.dtype == tf.float64
                prfs = tf.concat([ind_prfs, ave_prf[None, :], en_prf[None, :]], axis=0)
                prfs.set_shape([num_recs + 2, 3])
                names = list(self.me_ids) + ['average', 'ensemble']
                prf_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=prfs,
                    header=['precision', 'recall', 'f1'],
                    scope='ind_ave_en_prf',
                    title='individual with their average and ensemble prfs',
                    names=names
                )
                tb_table_and_scalar_protos.append(prf_tb_table_proto)

                ind_tps_fps_tns_fns = stats['individual_tps_fps_tns_fns']
                en_tps_fps_tns_fns = stats['ensemble_tps_fps_tns_fns']
                en_tps_fps_tns_fns = en_tps_fps_tns_fns[None, :]
                assert ind_tps_fps_tns_fns.dtype.base_dtype == en_tps_fps_tns_fns.dtype == tf.int64
                ind_en = tf.concat([ind_tps_fps_tns_fns, en_tps_fps_tns_fns], axis=0)
                names = list(self.me_ids) + ['ensemble']
                tps_fps_tns_fns_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=ind_en,
                    header=['TP', 'FP', 'TN', 'FN'],
                    scope='ind_en_tps_fps_tns_fns',
                    title='individual and their ensemble tps, fps, tns and fns',
                    names=names
                )
                tb_table_and_scalar_protos.append(tps_fps_tns_fns_tb_table_proto)

                prf_loss_tb_scalar_proto = []
                for ave_or_en in ('average', 'ensemble'):
                    with tf.name_scope(ave_or_en):
                        p, r, f = tf.unstack(ave_prf if ave_or_en == 'average' else en_prf)
                        prfl_protos = []
                        items_for_summary = dict(precision=p, recall=r, f1=f)
                        for item_name, item_value in items_for_summary.iteritems():
                            prfl_protos.append(tf.summary.scalar(item_name, item_value))
                        if ave_or_en == 'average':
                            prfl_protos.append(tf.summary.scalar('loss', ave_loss))
                        prf_loss_tb_scalar_proto.append(tf.summary.merge(prfl_protos))
                prf_loss_tb_scalar_proto = tf.summary.merge(prf_loss_tb_scalar_proto)

                tb_table_and_scalar_protos.append(prf_loss_tb_scalar_proto)
                tb_table_and_scalar_protos = tf.summary.merge(tb_table_and_scalar_protos)

            return tb_table_and_scalar_protos


def main():
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    # construct models for training, validation and test
    for name in ('training', 'validation', 'test'):
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)
    with tf.Session(config=MODEL_DICT['config'].gpu_config) as sess:
        # summary writer
        summary_writer_dict = {}
        for training_valid_or_test in ('training', 'validation', 'test'):
            summary_writer_dict[training_valid_or_test] = tf.summary.FileWriter(
                os.path.join(MODEL_DICT['config'].tb_dir, training_valid_or_test)
            )

        aug_info = []

        aug_info.append('frame label detection')

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            aug_info.append('inference with {}'.format(MODEL_DICT['config'].train_or_inference.inference))
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            aug_info.append('continue training from {}'.format(MODEL_DICT['config'].train_or_inference.from_saved))

        if MODEL_DICT['config'].train_or_inference.inference is None:
            _model_prefix = MODEL_DICT['config'].train_or_inference.model_prefix
            if _model_prefix is not None:
                aug_info.append('model prefix {}'.format(_model_prefix))
            else:
                aug_info.append('model will not be saved')

        aug_info.append('learning rate - {}'.format(MODEL_DICT['config'].learning_rate))
        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
        aug_info.append('snippet length - {}'.format(MODEL_DICT['config'].snippet_len))
        aug_info.append('batch size - 1')
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of epochs - {}'.format(MODEL_DICT['config'].num_epochs))
        aug_info.append('training start time - {}'.format(datetime.datetime.now()))
        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        logging.info('global vars -')
        for idx, var in enumerate(tf.global_variables()):
            logging.info("{}\t{}\t{}".format(idx, var.name, var.shape))

        logging.info('local vars -')
        for idx, var in enumerate(tf.local_variables()):
            logging.info('{}\t{}'.format(idx, var.name))

        logging.info('trainable vars -')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
            logging.info('{}\t{}'.format(idx, var.op.name))

        OP_DICT = {}
        for training_valid_or_test in ('training', 'validation', 'test'):
            m = MODEL_DICT[training_valid_or_test]

            if m.is_training:
                tmp = dict(
                    batch=[m.training_op, m.stats['update_op']],
                    epoch=m.tb_proto
                )
            else:
                tmp = dict(
                    batch=dict(
                        me_id=m.batch['me_id'],
                        update_op=m.stats['update_op']
                    ),
                    epoch=m.tb_proto
                )

            OP_DICT[training_valid_or_test] = tmp

        # function for test or validation
        def test_or_validate_fn(valid_or_test, global_step=None):
            assert valid_or_test in ('test', 'validation')

            ops_per_batch = OP_DICT[valid_or_test]['batch']
            ops_per_epoch = OP_DICT[valid_or_test]['epoch']

            batch_idx = 0
            _dataset_test = MODEL_DICT[valid_or_test].dataset
            total_num_snippets = sum(len(rec_dict['split_list']) for rec_dict in _dataset_test)
            num_recs = len(_dataset_test)

            for rec_idx in xrange(num_recs):
                rec_dict = _dataset_test[rec_idx]
                split_list = rec_dict['split_list']
                num_snippets = len(split_list)
                num_frames = len(rec_dict['sg'])
                assert num_frames == MODEL_DICT[valid_or_test].num_frames_vector[rec_idx]
                for snippet_idx in xrange(num_snippets):
                    logging.debug('batch {}/{}'.format(batch_idx + 1, total_num_snippets))
                    tmp = sess.run(ops_per_batch)
                    _rec_idx = tmp['me_id'][0]
                    assert _rec_idx == rec_idx
                    batch_idx += 1

            summary_writer_dict[valid_or_test].add_summary(sess.run(ops_per_epoch), global_step)

        def check_all_global_vars_initialized_fn():
            tmp = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
            assert not tmp

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            save_path = MODEL_DICT['config'].train_or_inference.inference
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            logging.info('do inference ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            sess.run(MODEL_DICT['test'].reinitializable_iter_for_dataset.initializer)

            test_or_validate_fn('test')
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            save_path = MODEL_DICT['config'].train_or_inference.from_saved
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            logging.info('reproduce results ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            for valid_or_test in ('validation', 'test'):
                sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)
            for valid_or_test in ('validation', 'test'):
                logging.info(valid_or_test)
                test_or_validate_fn(valid_or_test, 0)
        else:  # neither inference or from saved
            logging.info('train from scratch')
            sess.run(tf.initializers.variables(tf.global_variables()))
            check_all_global_vars_initialized_fn()

        # train the model
        if MODEL_DICT['config'].train_or_inference.inference is None:
            check_all_global_vars_initialized_fn()
            if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                assert 'model_saver' not in MODEL_DICT
                MODEL_DICT['model_saver'] = tf.train.Saver(max_to_keep=200)

            for training_valid_test_epoch_idx in xrange(MODEL_DICT['config'].num_epochs):
                logging.info('\n\ncycle - {}/{}'.format(training_valid_test_epoch_idx + 1, MODEL_DICT['config'].num_epochs))

                sess.run(tf.initializers.variables(tf.local_variables()))

                # to enable prefetch
                for valid_or_test in ('validation', 'test'):
                    sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)

                for training_valid_or_test in ('training', 'validation', 'test'):
                    logging.info(training_valid_or_test)

                    if training_valid_or_test == 'training':
                        ops_per_batch = OP_DICT[training_valid_or_test]['batch']
                        ops_per_epoch = OP_DICT[training_valid_or_test]['epoch']
                        for batch_idx in xrange(MODEL_DICT['config'].batches_per_epoch):
                            sess.run(ops_per_batch)
                            logging.debug('batch - {}/{}'.format(batch_idx + 1, MODEL_DICT['config'].batches_per_epoch))
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(ops_per_epoch),
                            training_valid_test_epoch_idx + 1
                        )

                        if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                            save_path = MODEL_DICT['config'].train_or_inference.model_prefix + \
                                        '_' + 'epoch_{}_of_{}'.format(training_valid_test_epoch_idx + 1,
                                                                      MODEL_DICT['config'].num_epochs)
                            save_path = os.path.join('saved_model', save_path)
                            save_path = MODEL_DICT['model_saver'].save(
                                sess=sess,
                                save_path=save_path,
                                global_step=None,
                                write_meta_graph=False
                            )
                            logging.info('model saved to {}'.format(save_path))

                    else:
                        test_or_validate_fn(training_valid_or_test, training_valid_test_epoch_idx + 1)

        msg = 'training end time - {}'.format(datetime.datetime.now())
        logging.info(msg)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: msg}))

        for training_valid_or_test in ('training', 'validation', 'test'):
            summary_writer_dict[training_valid_or_test].close()


if __name__ == '__main__':
   main()


















