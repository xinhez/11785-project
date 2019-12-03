class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '-padding-'}
        self.num_words = 1

    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word 
            self.num_words += 1
    
    def getIndex(self, word):
        if word not in self.word2index:
            self.addWord(word)
        return self.word2index[word]

def new_exercise(use_all_features=False):
    exercise = dict()
    exercise['token'] = []
    exercise['instance_id'] = []
    if use_all_features:
        exercise['part_of_speech'] = []
        exercise['morphological_features'] = []
        exercise['dependency_label'] = []
        exercise['dependency_edge_head'] = []
    return exercise

def load_data(filename, lang, dbg=False, use_all_features=False):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of excercises
    data = []
    labels = []

    exercise = new_exercise(use_all_features)
    label = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    num_exercises = 0
    dbg_count = 0

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:

                num_exercises += 1
                data.append(exercise)
                labels.append(label)
                if dbg:
                    dbg_count += 1
                    if dbg_count > 10000:
                        break
                exercise = new_exercise(use_all_features)
                label = []
                
            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if use_all_features:
                    if 'prompt' in line:
                        exercise['prompt'] = line.split(':')[1]
                    else:
                        list_of_exercise_parameters = line[2:].split()
                        for exercise_parameter in list_of_exercise_parameters:
                            [key, value] = exercise_parameter.split(':')
                            if key == 'countries':
                                value = value.split('|')
                            elif key == 'days':
                                value = float(value)
                            elif key == 'time':
                                if value == 'null':
                                    value = None
                                else:
                                    assert '.' not in value
                                    value = int(value)
                            exercise[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                # add the work to word2vec dictionary
                lang.addWord(line[1])

                exercise['instance_id'].append(line[0])
                exercise['token'].append(line[1])
                if training:
                   label.append(float(line[6]))

                if use_all_features:
                    exercise['part_of_speech'].append(line[2])
                    morphological_features = dict()
                    for l in line[3].split('|'):
                        [key, value] = l.split('=')
                        if key == 'Person':
                            value = int(value)
                        morphological_features[key] = value
                    exercise['morphological_features'].append(morphological_features)
                    exercise['dependency_label'].append(line[4])
                    exercise['dependency_edge_head'].append(line[5])

    if training:
        return data, labels
    else:
        return data