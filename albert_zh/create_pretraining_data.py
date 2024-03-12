# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import jieba
import re
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_bool("non_chinese", False,"manually set this to True if you are not doing chinese pre-train task.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        strings=reader.readline()
        strings=strings.replace("   "," ").replace("  "," ") #If there are two or three spaces, replace them with one
        line = tokenization.convert_to_unicode(strings)
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
        create_instances_from_document_albert( # change to albert style for sentence order prediction(SOP), 2019-08-28, brightmart
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances

def get_new_segment(segment):  # new method ####
    """
    Enter a sentence and return a processed sentence: to support the full Chinese name mask, the words that will be separated will be specially marked (”#”)
    so that the subsequent processing module can know which words belong to the same word.
    :param segment:  e.g.  [ 'hanging' , 'Moxibustion' , 'technique' , 'training' , 'training' , 'special' , 'home' , 'teaching' , 'You' , 'Ai' , 'Moxibustion' , 'descending' , 'blood' , 'sugar' , ' , 'for' , 'Dad' , 'Mum' , 'Harvest' , 'good' , 'gone' , ' ! ']
    :return: A processed word e.g.  ['hanging' , '##Moxibustion' , 'technique' , 'training' , 'training' , 'training' , '##home' , 'teaching' , 'You' , 'Ai' , '##Moxibustion' , 'down' , '##blood' , '##sugar' , ' , 'for' , 'Dad' , '##Mum' , 'harvest' , '##good' , 'gone' , ' ! ']
    """
    seq_cws = jieba.lcut("".join(segment)) # Participle
    seq_cws_dict = {x: 1 for x in seq_cws} # The word after the participle is added to the dictionary dict
    new_segment = []
    i = 0
    while i < len(segment): # start with the first word of the sentence and work your way through it
      if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # If Chinese is not found, the original text is added without special treatment.
        new_segment.append(segment[i])
        i += 1
        continue

      has_add = False
      for length in range(3, 0, -1):
        if i + length > len(segment):
          continue
        if ''.join(segment[i:i + length]) in seq_cws_dict:
          new_segment.append(segment[i])
          for l in range(1, length):
            new_segment.append('##' + segment[i + l])
          i += length
          has_add = True
          break
      if not has_add:
        new_segment.append(segment[i])
        i += 1
    # print("get_new_segment.wwm.get_new_segment:",new_segment)
    return new_segment

def create_instances_from_document_albert(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document.
     This method is changed to create sentence-order prediction (SOP) followed by idea from paper of ALBERT, 2019-08-28, brightmart
  """
  document = all_documents[document_index] # get a document


  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob: # With a certain proportion, such as a 10% probability,
                                    # we used shorter sequence lengths to mitigate the inconsistency between the pretrained long sequence and the (possibly) short sequence of the tuning phase
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # Try to use actual sentences instead of arbitrarily truncating sentences to better construct the task of predicting sentence coherence
  instances = []
  current_chunk = [] # The text segment being processed contains multiple sentences
  current_length = 0
  i = 0
  while i < len(document): # Start at the first position of the document and click down
    segment = document[i]
    if FLAGS.non_chinese==False: # if non chinese is False, that means it is chinese, then do something to make chinese whole word mask works.
      segment = get_new_segment(segment)  # whole word mask for chinese: the Chinese whole mask setting with the participle is added where needed“##”

    current_chunk.append(segment) # Adds a separate sentence to the current text block
    current_length += len(segment) # The total length of a sentence that accumulates until the position is reached
    if i == len(document) - 1 or current_length >= target_seq_length:
      # If the cumulative sequence length reaches the target length, or is currently at the end of the document==>constructs and adds to a and B in“A [ SEP ] B”;
      if current_chunk: # If the current block is not empty
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2: # The current block, if it contains more than two sentences, takes part of the current block as part A in“A [ SEP ] B”
          a_end = rng.randint(1, len(current_chunk) - 1)
        # Assign the first half of the current text segment to A, or tokens_a
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])
        # Construct the b part of“A [ SEP ] B”(part of which is the second half of the normal current document; part of the original Bert implementation is randomly selected from another document,)
        tokens_b = []
        for j in range(a_end, len(current_chunk)):
          tokens_b.extend(current_chunk[j])

        # There's a 50% chance tokens and tokens will switch places
        # print("tokens_a length1:",len(tokens_a))
        # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0

        if len(tokens_a)==0 or len(tokens_b)==0: continue
        if rng.random() < 0.5: # Switch tokens_a and tokens_b
          is_random_next=True
          temp=tokens_a
          tokens_a=tokens_b
          tokens_b=temp
        else:
          is_random_next=False

        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        #Add tokens_a & tokens_b in bert's style, that is, in the form of [CLS]tokens_a[SEP]tokens_b[SEP] , together as the final tokens; also with the segment_ids, with the value 0 for the first segment and 1 for the second segment.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # build masked data for the tasks of the LM ：Creates the predictions for the masked LM objective
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance( #  An object that creates a training instance
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = [] # clears the current block
      current_length = 0 # resets the length of the current text block
    i += 1 # then look back at the contents of the document

  return instances


def create_instances_from_document_original( # THIS IS ORIGINAL BERT STYLE FOR CREATE DATA OF MLM AND NEXT SENTENCE PREDICTION TASK
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index] # get a document

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob: # With a certain proportion, such as a 10% probability,
                                    # we used shorter sequence lengths to mitigate the inconsistency between the pretrained long sequence and the (possibly) short sequence of the tuning phase
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # Try to use actual sentences instead of arbitrarily truncating sentences to better construct the task of predicting sentence coherence
  instances = []
  current_chunk = [] # The text segment being processed contains multiple sentences
  current_length = 0
  i = 0

  while i < len(document): # Start at the first position of the document and click down
    segment = document[i] # segment is a list, representing a complete sentence separated by words, such as segment=['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。']
    # print("###i:",i,";segment:",segment)
    current_chunk.append(segment) #  Adds a separate sentence to the current text block
    current_length += len(segment) # The total length of a sentence that accumulates until the position is reached
    if i == len(document) - 1 or current_length >= target_seq_length:
      # If the cumulative sequence length reaches the target length, or is currently at the end of the document==>constructs and adds to a and B in“A [ SEP ] B”;
      if current_chunk: # If the current block is not empty
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2: # The current block, if it contains more than two sentences, how to take part of the current block as part A in“A [SEP] B”
          a_end = rng.randint(1, len(current_chunk) - 1)
        # Assign the first half of the current text segment to A, or tokens_a
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        #Construct the b part of“A [ SEP ] B”(part of which is the second half of the normal current document; part of the original Bert implementation is randomly selected from another document,)
        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5: # There is a 50% chance that a random selection of documents from other documents, and get the latter half of the document as B, that is, tokens_b
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          random_document_index=0
          for _ in range(10): # Randomly select an index for a document that is different from the current document
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index] #  Select this document
          random_start = rng.randint(0, len(random_document) - 1) # Select the beginning of a paragraph from this document
          for j in range(random_start, len(random_document)): # From the beginning to the end of this document, as the B in our“A[sep]B”, orTokens_b
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste. Here's a tip to avoid wasting text
          num_unused_segments = len(current_chunk) - a_end # e.g. 550-200=350
          i -= num_unused_segments # i=i-num_unused_segments, e.g. i=400, num_unused_segments=350, 那么 i=i-num_unused_segments=400-350=50
        # Actual next
        else: # An additional 50% , almost, is populated from the back segment in the current text block (length Max) to B in tokens, “A[SEP]B”
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        #Add tokens_a & tokens_b in bert's style, that is, in the form of [CLS]tokens_a[SEP]tokens_b[SEP] , together as the final tokens; also with the segment_ids, with the value 0 for the first segment and 1 for the second segment.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # build data for the tasks of the masked LM： Creates the predictions for the masked LM objective
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance( #  An object that creates a training instance
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = [] # Clears the current block
      current_length = 0 # Resets the length of the current text block
    i += 1 # Then look back at the contents of the document

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
            token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  if FLAGS.non_chinese==False: # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
    output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]  # Take it off"##"
  else: # english and other language, which is not chinese
    output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          if FLAGS.non_chinese == False: # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
            masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 else tokens[index]  # Take it off"##"
          else:
            masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  # tf.logging.info('%s' % (tokens))
  # tf.logging.info('%s' % (output_tokens))
  return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_masked_lm_predictions_original(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()