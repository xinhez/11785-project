import copy

def new_instance():
    instance = dict()
    instance['token'] = '-padding-'
    return instance

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
    
    instance = new_instance()

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
                if dbg:
                    dbg_count += 1
                    if dbg_count > 10000:
                        break
                instance = new_instance()
                
            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if 'prompt' in line:
                    instance['prompt'] = line.split(':')[1]
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
                        instance[key] = value

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
                instance['last_token'] = instance['token']

                instance['instance_id'] = line[0]
                instance['token'] = line[1]
                if training:
                   labels.append(float(line[6]))

                instance['part_of_speech'] = line[2]
                morphological_features = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    morphological_features[key] = value
                instance['morphological_features'] = morphological_features
                instance['dependency_label'] = line[4]
                instance['dependency_edge_head'] = line[5]

                data.append(copy.deepcopy(instance))

    if training:
        return data, labels
    else:
        return data