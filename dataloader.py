import copy
import numpy as np

class DataIter(object):
    def __init__(self, document_list, label_list, tfidf_list,batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = self._batch_starting_point_list()
        self.tfidf_list = tfidf_list

    def _batch_starting_point_list(self):
        num_turn_list = [len(document) for document in self.document_list]
        batch_starting_list = []
        previous_turn_index=-1
        previous_num_turn=-1
        for index,num_turn in enumerate(num_turn_list):
            if num_turn != previous_num_turn:
                if index != 0:
                    assert num_turn > previous_num_turn 
                    num_batch = (index-previous_turn_index) // self.batch_size
                    for i in range(num_batch):
                        batch_starting_list.append(previous_turn_index + i*self.batch_size)
                previous_turn_index = index
                previous_num_turn = num_turn
        if previous_num_turn != len(self.document_list):
            num_batch = (index - previous_turn_index) // self.batch_size
            for i in range(num_batch):
                batch_starting_list.append(previous_turn_index + i * self.batch_size)
        return batch_starting_list

    def sample_document(self,index):
        return self.document_list[index]

    def __iter__(self):
        self.current_batch_starting_point_list = copy.copy(self.batch_starting_point_list)
        self.current_batch_starting_point_list = np.random.permutation(self.current_batch_starting_point_list) 
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.current_batch_starting_point_list):
            raise StopIteration
        batch_starting = self.current_batch_starting_point_list[self.batch_index]
        batch_end = batch_starting + self.batch_size
        raw_batch = self.document_list[batch_starting:batch_end]
        label_batch = self.label_list[batch_starting:batch_end]
        tfidf_batch = self.tfidf_list[batch_starting:batch_end]
        transeposed_batch = map(list, zip(*raw_batch))
        transeposed_tfidf_batch = map(list,zip(*tfidf_batch))
        padded_batch = []
        length_batch = []
        padded_tfidf_batch = []
        for transeposed_doc,transeposed_tfidf in zip(transeposed_batch,transeposed_tfidf_batch): #padding for each batch data.
            length_list = [len(sent) for sent in transeposed_doc]
            max_length = max(length_list)
            new_doc = [sent+[self.padded_value]*(max_length-len(sent)) for sent in transeposed_doc]
            new_tfidf = [sent+[1e-5]*(max_length-len(sent)) for sent in transeposed_tfidf]
            padded_batch.append(np.asarray(new_doc, dtype=np.int32).transpose(1,0))
            padded_tfidf_batch.append(np.asarray(new_tfidf, dtype=np.float32).transpose(1,0))
            length_batch.append(length_list)
        padded_length = np.asarray(length_batch)
        padded_label = np.asarray(label_batch, dtype=np.int32) -1
        original_index =  np.arange(batch_starting,batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, padded_tfidf_batch,padded_length ,original_index
