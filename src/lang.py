class Lang:
    def __init__(self):
        self.word2index = {'-padding-': 0}
        self.word2count = {'-padding-': 0}
        self.index2word = {0: '-padding-'}
        self.letters = ['', '<sos>', '<eos>', ' ']
        self.letter2Index = dict()
        self.num_words = 1

    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            for c in word:
                if c not in self.letters:
                    self.letters.append(c)
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word 
            self.num_words += 1
    
    def getIndex(self, word):
        if word not in self.word2index:
            self.addWord(word)
        return self.word2index[word]

    def addUsers(self, users):
        self.num_users = len(users)
        self.user2Index = dict()
        for i in range(len(users)):
            self.user2Index[users[i]] = i
        
    def getUserIndex(self, user):
        return self.user2Index[user]