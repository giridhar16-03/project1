from typing import Dict, List

class CharacterWordPreprocessor:
    def __init__(self,max_word_length : int = 10):
        self.max_word_length = max_word_length

        #Defining special control tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bow_id = 2
        self.eow_id = 3

        #Offset parameter to separate characters from contol tokens
        self.char_offset = 4

        #Empty dictionary to store character to index mapping
        self.char_to_id : Dict[str, int] = {}

    def _get_char_id(self, char : str) -> int:
        """Returns the id of a character, if not present returns unk_id"""
        if char not in self.char_to_id:
            return ord(char) + self.char_offset
        return self.char_to_id.get(char, self.unk_id)

    def encode_sentence(self, text : str) -> List[List[int]]:
        """Transforms a raw string into 2d list of characters id's with shape:
        [number_of_words, max_word_length]"""

        #word-split
        words = text.split()

        #handling empty strings 
        if not words:
            return [[self.pad_id] * self.max_word_length]
        
        sentence_matrix = []

        for word in words:
            #Map characters to their respective ids
            char_ids = [self._get_char_id(c) for c in word]

            #Handle truncation or padding 
            max_chars_allowed = self.max_word_length - 2  # Reserving space for BOW and EOW

            if(len(char_ids) > max_chars_allowed):
                #Truncate the content to fit, leaving space for boundary tokens
                char_ids = char_ids[:max_chars_allowed]
                wrapped_word = [self.bow_id] + char_ids + [self.eow_id]
            else:
                #Wrap with boundary tokens
                wrapped_word = [self.bow_id] + char_ids + [self.eow_id]
                #Pad out the remaining slots to the right with 0s
                padding_needed = self.max_word_length - len(wrapped_word)
                wrapped_word += [self.pad_id] * padding_needed
            
            sentence_matrix.append(wrapped_word)

        return sentence_matrix



if __name__ == "__main__":
    #Initialize the preprocessor with example max length of 10
    preprocessor = CharacterWordPreprocessor(max_word_length = 10)

    preprocessor.char_to_id = {
        'H' : 12, 'e' : 9, 'y' : 21,
        'm' : 17, 'a' : 5, 'c' : 7, 'h': 12, 'i' : 13, 'n' : 18
    }

    raw_input = "Hey machine"
    output_matrix = preprocessor.encode_sentence(raw_input)
    print(output_matrix)